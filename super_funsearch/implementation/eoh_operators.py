"""EoH-style multi-operator prompt builders for Super-FunSearch.

This module ports the five evolution operators from
  Liu et al., "Evolution of Heuristics: Towards Efficient Automatic Algorithm
  Design Using Large Language Model" (ICML 2024)
into the Super-FunSearch sampling pipeline. Each operator builds a *single*
prompt that asks the LLM to emit both a one-sentence strategy description
(wrapped in `{...}`) AND a Python function. Compared with our previous
two-step "thought then code" generation, this:

  * cuts LLM calls roughly in half,
  * forces thought and code to come from the same forward pass (no semantic
    drift between them),
  * exposes five complementary mutation/crossover modes for diversity:
        i1  – initialisation from scratch
        e1  – exploration: "totally different form" from given parents
        e2  – exploration: extract common backbone, then differ
        m1  – mutation: modified version of a single parent
        m2  – mutation: tweak parameters/coefficients of a single parent
        m3  – mutation: simplify components to fight overfitting

The operator selection itself lives in `sampler.py`; this module only knows
how to (a) build prompts, (b) parse LLM responses back into
`(thought, code_body)` tuples that downstream code can feed into
`evaluator._validate_sample_body`.
"""

from __future__ import annotations

import logging
import random
import re
import textwrap
from dataclasses import dataclass
from typing import Iterable

# ---------------------------------------------------------------------------
# Operator catalogue & default weights
# ---------------------------------------------------------------------------

OPERATORS = ('i1', 'e1', 'e2', 'm1', 'm2', 'm3')

# Mirrors EoH's Table 2 default — equal-weighted exploration & mutation
# operators; i1 is *not* sampled by `pick_operator` (it is only used as a
# fallback when the population is too small for the selected operator).
DEFAULT_OPERATOR_WEIGHTS: dict[str, float] = {
    'e1': 1.0,
    'e2': 1.0,
    'm1': 1.0,
    'm2': 1.0,
    'm3': 1.0,
}

# Number of parents each operator consumes.
PARENTS_PER_OPERATOR: dict[str, int] = {
    'i1': 0,
    'e1': 2,
    'e2': 2,
    'm1': 1,
    'm2': 1,
    'm3': 1,
}


@dataclass(frozen=True)
class TaskSpec:
    """Problem-level constants injected into every operator prompt."""

    task_description: str
    function_name: str           # e.g. 'priority'
    function_inputs: tuple       # e.g. ('item', 'bins')
    function_outputs: tuple      # e.g. ('priorities',)
    inout_inf: str               # input/output meaning notes
    other_inf: str               # any extra constraints
    interface_spec: str = ''     # full interface block (rendered above prompt)


BIN_PACKING_TASK = TaskSpec(
    # ------------------------------------------------------------------
    # Mirrors EoH's vanilla `prompt_task` verbatim (see
    # EoH/eoh/src/eoh/problems/optimization/bp_online/prompts.py).
    # We deliberately KEEP the wording neutral — describing the physical
    # fact ("will not be used") instead of issuing an instruction ("should
    # be discouraged"). The previous instruction-style wording was traced
    # to collapsing all i1 init samples into a single "fresh-bin penalty"
    # basin (see logs/reohprompt_150.log), which we want to avoid.
    # ------------------------------------------------------------------
    task_description=(
        "I need help designing a novel score function that scoring a set "
        "of bins to assign an item. In each step, the item will be "
        "assigned to the bin with the maximum score. If the rest "
        "capacity of a bin equals the maximum capacity, it will not be "
        "used. The final goal is to minimize the number of used bins."
    ),
    function_name='priority',
    function_inputs=('item', 'bins'),
    function_outputs=('priorities',),
    # Mirrors EoH's `prompt_inout_inf`, only the variable names are
    # renamed (priorities vs scores, item-as-float vs item-as-int) to
    # match our evaluator interface.
    inout_inf=(
        "'item' and 'bins' are the size of the current item and the "
        "rest capacities of feasible bins, which are larger than the "
        "item size. The output named 'priorities' is the scores for "
        "the bins for assignment."
    ),
    # Mirrors EoH's `prompt_other_inf` plus bare interface constraints needed
    # to make the LLM emit code that runs inside our evaluator. Search-shaping
    # constraints live in `_implementation_constraints()` so they apply
    # uniformly to all operators without injecting a target formula.
    other_inf=(
        "Note that 'item' is a float scalar, while 'bins' and "
        "'priorities' are both 1-D numpy arrays of the same shape. "
        "`numpy` is available as `np` — do NOT add `import` lines. "
        "AVOID 2-D indexing or `axis=1` reductions — the input is "
        "strictly 1-D. The novel function should be sufficiently "
        "complex in order to achieve better performance. It is "
        "important to ensure self-consistency."
    ),
)


# ---------------------------------------------------------------------------
# Operator selection
# ---------------------------------------------------------------------------

def pick_operator(
        weights: dict[str, float] | None = None,
        rng: random.Random | None = None,
) -> str:
    """Weighted-random pick from the e1/e2/m1/m2/m3 catalogue.

    `i1` is intentionally excluded — Sampler is responsible for falling back
    to i1 when the chosen operator needs more parents than the population can
    currently supply.
    """
    rng = rng or random
    weights = weights or DEFAULT_OPERATOR_WEIGHTS
    keys = list(weights.keys())
    ws = [weights[k] for k in keys]
    return rng.choices(keys, weights=ws, k=1)[0]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _format_inputs(spec: TaskSpec) -> str:
    if len(spec.function_inputs) > 1:
        return ', '.join(f"'{s}'" for s in spec.function_inputs)
    return f"'{spec.function_inputs[0]}'"


def _format_outputs(spec: TaskSpec) -> str:
    if len(spec.function_outputs) > 1:
        return ', '.join(f"'{s}'" for s in spec.function_outputs)
    return f"'{spec.function_outputs[0]}'"


def _format_example(spec: TaskSpec) -> str:
    """Concrete output template the LLM should mirror.

    Showing a 1-shot exemplar massively cuts the chance the model writes
    a discussion section without ever emitting `def {function_name}(...):`.
    """
    inputs = ', '.join(spec.function_inputs)
    return (
        "**Output format (mandatory — copy this layout exactly):**\n"
        "```text\n"
        "{One-sentence description of your algorithm.}\n"
        "```python\n"
        f"def {spec.function_name}({inputs}):\n"
        "    # ... your implementation, single-line `def`, no decorators, "
        "no surrounding class ...\n"
        "    return ...\n"
        "```\n"
        "```\n"
    )


def _implementation_constraints() -> str:
    """Shared A1 code-generation boundaries.

    These constraints keep the generated heuristic in a runnable, online
    scoring space without revealing a target formula.
    """
    return (
        "\nImplementation constraints for this online scorer:\n"
        "- Return one vectorized 1-D numpy score array with the same shape as "
        "`bins`; do not return a single index, sorted array, or chosen bin.\n"
        "- Use O(n) numpy operations only; avoid nested loops, per-bin full "
        "simulation, recursion, or recomputing global packing states.\n"
        "- Keep post-placement residual (`bins - item`) or an equivalent "
        "tight-fit signal as the primary signal.\n"
        "- Score direction must be correct: higher priority should prefer "
        "smaller post-placement residuals. Do not use positive residual "
        "(`bins - item`) as the primary score.\n"
        "- Additional terms should be bounded secondary perturbations "
        "(piecewise weights, small rank/tie-break bias, bounded "
        "multipliers); they must not dominate the tight-fit signal.\n"
        "- Deterministic only: do not use `np.random`, random noise, "
        "stochastic tie-breakers, or seeding inside the function.\n"
        "- Avoid pure monotone transformations of residual that preserve "
        "exactly the same Best-Fit ranking; any secondary term should be "
        "deterministic and bounded.\n"
        "- Avoid global balance objectives such as mean-target, variance "
        "minimization, or fragmentation profile as the main score; if used, "
        "use only as a small tie-breaker.\n"
    )


def _common_tail(spec: TaskSpec) -> str:
    """The shared closing block reused by i1 / e1 / e2 / m1 / m2.

    Asks for `{algorithm}` description first, then a Python function with the
    expected name & signature. Mirrors EoH's wording so that we inherit their
    well-tested response format.
    """
    return (
        "First, describe your new algorithm and main steps in **one "
        "sentence**. The description must be inside a brace `{...}`. "
        "Next, implement it in Python as a function named "
        f"`{spec.function_name}`. The function should accept "
        f"{len(spec.function_inputs)} input(s): {_format_inputs(spec)}. "
        f"It should return {len(spec.function_outputs)} output(s): "
        f"{_format_outputs(spec)}. {spec.inout_inf} {spec.other_inf}\n"
        + _implementation_constraints()
        + "**Hard requirements**: your reply must contain BOTH the "
        "`{description}` brace AND a fenced ```python``` code block "
        "containing a single-line `def "
        f"{spec.function_name}(...):` header (no class wrapper, no "
        "multi-line signature, no trailing comment after `:`). "
        "Do not give additional explanations.\n\n"
        + _format_example(spec)
    )


def _format_parents(parents: Iterable[dict]) -> str:
    """Render a list of {algorithm, code} dicts as numbered prose."""
    parts = []
    for i, p in enumerate(parents, start=1):
        algo = (p.get('algorithm') or '').strip() or '(no description)'
        code = (p.get('code') or '').strip()
        parts.append(
            f"No.{i} algorithm and the corresponding code are:\n"
            f"{algo}\n{code}\n"
        )
    return ''.join(parts)


def _format_reflection_block(reflection: str) -> str:
    """Render a ReEvo-style verbal-gradient hint as its own labelled section.

    Kept as a single helper so the formatting stays identical between the
    `head`-using operators (e1/e2/m1/m2) and the bespoke m3 prompt.
    """
    if not reflection:
        return ''
    return '\n[Reflection from prior comparisons]\n' + reflection.strip() + '\n'


def build_prompt(
        operator: str,
        parents: list[dict],
        spec: TaskSpec,
        error_avoidance: str = '',
        reflection: str = '',
        init_diversity: str = '',
) -> str:
    """Build the full LLM prompt for the given operator.

    Args:
        operator: one of i1/e1/e2/m1/m2/m3.
        parents: list of dicts with keys 'algorithm' and 'code'. Length
            must match PARENTS_PER_OPERATOR[operator] (pass an empty list
            for i1).
        spec: problem-level constants.
        error_avoidance: optional pre-rendered "avoid these failures" block
            from ``error_memory.ErrorMemory.render_for_prompt()``. If
            non-empty it is injected immediately after the task
            description so the LLM sees it before being asked to write
            code. The decision of *which* operators receive this hint
            lives in ``sampler.py`` (see ``_eoh_generate``); this builder
            simply renders whatever was passed in.
        reflection: optional ReEvo-style verbal-gradient hint, either a
            short-term pairwise reflection (for crossover operators e1/e2)
            or the accumulated long-term reflection (for mutation
            operators m1/m2/m3). Splicing logic also lives in
            ``sampler.py``; this builder renders it verbatim under a
            ``[Reflection from prior comparisons]`` header so the LLM
            sees it before any parent code.
        init_diversity: optional initialization-only instruction.  Used to
            prevent the first population from collapsing into Best-Fit clones;
            ignored by non-i1 operators.
    """
    expected = PARENTS_PER_OPERATOR[operator]
    if len(parents) < expected:
        raise ValueError(
            f'operator {operator} needs {expected} parents, got {len(parents)}'
        )

    head = (spec.interface_spec + '\n\n') if spec.interface_spec else ''
    head += spec.task_description + '\n'
    if error_avoidance:
        head += '\n' + error_avoidance.rstrip() + '\n'
    head += _format_reflection_block(reflection)

    if operator == 'i1':
        if init_diversity:
            head += '\n[Initialization diversity constraint]\n'
            head += init_diversity.strip() + '\n'
        return head + _common_tail(spec)

    if operator == 'e1':
        return (
            head
            + f"I have {len(parents)} existing algorithms with their codes "
              f"as follows:\n"
            + _format_parents(parents)
            + "Please help me create a new algorithm with a meaningfully "
              "different but bounded form from the given ones. Prefer a "
              "near-tight-fit perturbation over global balancing.\n"
            + _common_tail(spec)
        )

    if operator == 'e2':
        return (
            head
            + f"I have {len(parents)} existing algorithms with their codes "
              f"as follows:\n"
            + _format_parents(parents)
            + "Please help me create a new algorithm that has a meaningfully "
              "different but bounded form from the given ones and can be "
              "motivated from them.\n"
              "Internally identify the common backbone idea, then design a "
              "new algorithm. **Do NOT include your reasoning steps in the "
              "reply.** Only emit the final result in the format below.\n"
              f"The function must be named `{spec.function_name}`, accept "
              f"{_format_inputs(spec)} and return {_format_outputs(spec)}. "
              f"{spec.inout_inf} {spec.other_inf}\n"
            + _implementation_constraints()
            +
              "Do not give additional explanations.\n\n"
            + _format_example(spec)
        )

    if operator == 'm1':
        p = parents[0]
        return (
            head
            + "I have one algorithm with its code as follows:\n"
              f"Algorithm description: {p.get('algorithm', '').strip()}\n"
              f"Code:\n{p.get('code', '').strip()}\n"
              "Please assist me in creating a new algorithm that has a "
              "**different but bounded form** and can be a modified version "
              "of the algorithm provided.\n"
            + _common_tail(spec)
        )

    if operator == 'm2':
        p = parents[0]
        return (
            head
            + "I have one algorithm with its code as follows:\n"
              f"Algorithm description: {p.get('algorithm', '').strip()}\n"
              f"Code:\n{p.get('code', '').strip()}\n"
              "Please identify the **main parameters / coefficients** in "
              "the score function and assist me in creating a new algorithm "
              "that uses **different parameter settings** of the same "
              "structure.\n"
            + _common_tail(spec)
        )

    if operator == 'm3':
        p = parents[0]
        # m3 is a refactor / generalisation operator — no parents in the
        # usual sense, just transform the single given code. m3 does NOT
        # use `head`, so we splice the avoidance + reflection blocks in
        # by hand to keep parity with all other operators.
        avoidance = (error_avoidance.rstrip() + '\n\n') if error_avoidance else ''
        reflection_block = _format_reflection_block(reflection)
        if reflection_block:
            reflection_block = reflection_block.lstrip('\n') + '\n'
        return (
            avoidance
            + reflection_block
            + "First, you need to identify the main components in the "
            "function below. Next, analyze whether any of these components "
            "can be **overfit** to the in-distribution instances. Then, "
            "based on your analysis, **simplify the components to enhance "
            "generalization** to potential out-of-distribution instances. "
            "Finally, provide the revised code, keeping the function name, "
            f"inputs and outputs unchanged. Wrap a one-sentence description "
            f"of the simplification inside a brace `{{...}}` BEFORE the "
            f"code.\n\n{p.get('code', '').strip()}\n\n"
            f"{spec.inout_inf}\n{_implementation_constraints()}"
            "Do not give additional explanations.\n\n"
            + _format_example(spec)
        )

    raise ValueError(f'Unknown operator: {operator}')


# ---------------------------------------------------------------------------
# Response parsing: (raw LLM text) -> (thought, code_body)
# ---------------------------------------------------------------------------

# Used only in the slow fallback path (single-line {...} with no nesting).
_FALLBACK_THOUGHT_RE = re.compile(r'\{([^{}]+)\}', re.DOTALL)
# Loose `def NAME(` matcher — we then walk forward to find the matching `)`
# and the trailing `:`, so the regex itself need not be airtight. Allow
# leading indentation so we still catch `def`s nested inside a `class`
# wrapper that the LLM produced (we'll just extract the inner body).
_DEF_HEAD_ANY_RE = re.compile(r'^[ \t]*def\s+\w+\s*\(', re.MULTILINE)


def _outermost_braced(text: str) -> str | None:
    """Return content of the outermost `{...}` block (handles nesting).

    Returns None if no balanced top-level pair is found.
    """
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}' and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                return text[start + 1:i]
    return None


def _find_def_signature_end(text: str, function_name: str) -> int | None:
    """Locate where the body of `def function_name(...):` begins.

    Returns the offset of the *first character of the line after the
    closing `:`*, or None if no such header exists.

    Robust to:
      - multi-line `def name(\n  arg1,\n  arg2,\n) -> Ret:` signatures,
      - trailing comments after the closing `:` (e.g. `: # comment`),
      - return-type annotations,
      - any def appearing inside a class body (still picked up).
    """
    targeted = re.compile(
        rf'^[ \t]*def\s+{re.escape(function_name)}\s*\(', re.MULTILINE)
    m = targeted.search(text)
    if m is None:
        m = _DEF_HEAD_ANY_RE.search(text)
    if m is None:
        return None

    # Walk forward from just-after `(` to the matching ')'.
    i = m.end()
    depth = 1
    while i < len(text) and depth > 0:
        c = text[i]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        i += 1
    if depth != 0:
        return None

    # Find the next `:` (skipping `->` annotation).
    colon = text.find(':', i)
    if colon == -1:
        return None

    eol = text.find('\n', colon)
    if eol == -1:
        return len(text)
    return eol + 1


def _strip_leading_docstring(body: str) -> str:
    stripped = body.lstrip('\n').lstrip()
    for triple in ('"""', "'''"):
        if stripped.startswith(triple):
            end = stripped.find(triple, 3)
            if end != -1:
                return stripped[end + 3:].lstrip('\n')
    return body


def parse_response(raw: str, function_name: str) -> tuple[str, str]:
    """Parse the LLM's single-shot response into (thought, code_body).

    `code_body` is what `evaluator._validate_sample_body` expects — i.e. the
    indented body of a function (no `def` header, no docstring, no
    surrounding markdown).

    On parse failure either field may be empty; downstream
    `_validate_sample_body` will then reject the sample cleanly.
    """
    text = (raw or '').strip()

    # Strip ```python ... ``` fences but keep any thought paragraphs that
    # appear outside the fence.
    cleaned = re.sub(r'```[a-zA-Z]*\s*', '\n', text)
    cleaned = cleaned.replace('```', '\n').strip()

    # 1) Thought: prefer the outermost balanced {...}; fall back to the
    #    last line before the first `def`/`class`/`import`.
    thought = (_outermost_braced(cleaned) or '').strip()
    if not thought:
        m_inner = _FALLBACK_THOUGHT_RE.search(cleaned)
        if m_inner:
            thought = m_inner.group(1).strip()
    if not thought:
        before_def = re.split(
            r'\n\s*(?:def|class|import)\s', cleaned, maxsplit=1)[0]
        if before_def.strip():
            thought = before_def.strip().splitlines()[-1].strip()

    # 2) Locate the function header (multi-line aware).
    body_start = _find_def_signature_end(cleaned, function_name)
    if body_start is None:
        return thought, ''

    after_def = cleaned[body_start:]

    # 3) Capture body lines: everything indented at >= the first non-empty
    #    line's indent. Stop at the first strictly less-indented line.
    body_lines: list[str] = []
    base_indent: int | None = None
    for line in after_def.split('\n'):
        if not line.strip():
            body_lines.append(line)
            continue
        indent = len(line) - len(line.lstrip(' \t'))
        if base_indent is None:
            if indent == 0:
                # No indented body at all — bail.
                break
            base_indent = indent
        if indent < base_indent:
            break
        body_lines.append(line)
    body = '\n'.join(body_lines).rstrip()

    body = _strip_leading_docstring(body)

    # Re-indent uniformly to 4 spaces — `_trim_function_body` will then wrap
    # the body in a `def fake_function_header():` and parse via AST.
    body = textwrap.dedent(body)
    if body.strip():
        out = []
        for line in body.split('\n'):
            out.append('    ' + line if line.strip() else '')
        body = '\n'.join(out)

    return thought, body


# ---------------------------------------------------------------------------
# Plan-B: format-fix retry prompt
# ---------------------------------------------------------------------------

def build_format_fix_prompt(raw: str, spec: TaskSpec) -> str:
    """Build a one-shot 'fix your formatting' prompt.

    Used after `parse_response` returns an empty body. Crucially we only
    ask the model to RE-EMIT the same algorithm in the strict format —
    not to invent a new one — so the call is cheap and (almost always)
    converges in one shot.
    """
    inputs = ', '.join(spec.function_inputs)
    return (
        "Your previous reply could not be parsed: I could not find a "
        f"valid `def {spec.function_name}({inputs}):` block followed by "
        "an indented body, OR you wrote analysis prose without any "
        "actual Python code.\n\n"
        "Re-emit the SAME algorithm strictly in the format below — no "
        "other text, no analysis, no extra prose. If your previous reply "
        "had no algorithm at all, invent the simplest non-trivial one "
        "(e.g. negative squared waste) and emit it now.\n\n"
        "```text\n"
        "{One-sentence description of your algorithm.}\n"
        "```python\n"
        f"def {spec.function_name}({inputs}):\n"
        "    # implementation here, single-line def, no class wrapper\n"
        "    return ...\n"
        "```\n"
        "```\n\n"
        "Your previous (un-parseable) reply for reference:\n"
        "<<<\n"
        f"{(raw or '').strip()[:1500]}\n"
        ">>>"
    )


# ---------------------------------------------------------------------------
# Helpers for converting code_manipulation.Function into operator parents
# ---------------------------------------------------------------------------

def function_to_parent_dict(fn) -> dict:
    """Render a `code_manipulation.Function` as `{algorithm, code}`.

    `algorithm` falls back to a placeholder string if the parent has no
    `.thought` attached (e.g. the seed program).
    """
    algorithm = getattr(fn, 'thought', None) or '(no description available)'
    return {'algorithm': algorithm.strip(), 'code': str(fn).strip()}


def _select_parents_for_operator(
        operator: str,
        parents: list,
        rng: random.Random | None = None,
) -> list[dict]:
    """Pick the right number of parents for the given operator.

    Returns dicts ready to be fed into `build_prompt`. If the population is
    smaller than required, returns an empty list — the caller should fall
    back to `i1`.
    """
    rng = rng or random
    needed = PARENTS_PER_OPERATOR[operator]
    if needed == 0:
        return []
    if len(parents) < needed:
        logging.debug(
            'EoH operator %s needs %d parents but only %d available — '
            'falling back', operator, needed, len(parents))
        return []
    chosen = rng.sample(list(parents), k=needed)
    return [function_to_parent_dict(p) for p in chosen]
