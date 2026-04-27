# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for sampling new programs.

v3.5 architecture: three agents, three memories, ONE coordinator.
=================================================================

The ``Sampler.sample()`` loop is a thin coordinator. It does not own any
algorithm-level state; it routes work between three independent agents
that each manage their own memory:

    ┌──────────────────────────────────────────────────────────────────┐
    │  A1  Coder Agent          ─ writes thought + code                │
    │      memory: none across samples (parents come from DB)          │
    │      methods: _a1_eoh_write_code(), _a1_two_step_write_code()    │
    │                                                                  │
    │  A2  Bug-Fixer Agent      ─ patches runtime crashes              │
    │      memory: BugFixMemory (known code-level repair recipes)      │
    │      methods: _a2_fix_runtime_bug()                              │
    │                                                                  │
    │  A3  Algorithm-Guide Agent ─ verbal-gradient hints               │
    │      memory: ST buffer + LT summary + past-outcomes (FIFO)       │
    │      methods: _a3_pre_generation_hint(), _a3_observe_outcome(),  │
    │               _a3_maybe_update_long_term()                       │
    │                                                                  │
    │  L2  ErrorMemory          ─ cross-sample crash avoidance         │
    │      memory: FIFO of recent runtime errors                       │
    │      data structure (NOT an agent), informs A1's next prompt     │
    │                                                                  │
    │  Dispatcher  (_dispatch)  ─ classifies eval results and routes:  │
    │      • valid           → A3.observe(child_score)                 │
    │      • runtime_crash   → L2.record + A2.fix + A3.observe         │
    │      • format_fail     → counter only (A1 already retried once)  │
    └──────────────────────────────────────────────────────────────────┘

Memory MUST stay separate. A2 never writes to L2 or A3; A3 never reads
ErrorMemory; the dispatcher is the only place where eval results
fan out to memories. This is exactly the boundary the user asked for:
"三个 agent 各司其职，各自的记忆自己弄，不要混". The legacy V2
Reflector path (``triage`` / ``diagnose_symptom``) is preserved
behind ``_is_legacy_v2_reflector()`` and is NEVER mixed with the new
A1/A2/A3 path.
"""
from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Collection, Optional, Sequence, Type
import dataclasses
import numpy as np
import re
import textwrap
import time
import logging

from implementation import evaluator
from implementation import programs_database
from implementation import eoh_operators
from implementation import error_memory as error_memory_lib
from implementation import bug_fix_memory as bug_fix_memory_lib
from implementation import reevo_reflector as reevo_reflector_lib
from implementation import search_controller as search_controller_lib


# ---------------------------------------------------------------------------
# Shared bag passed from A1 (writer) through the evaluator into _dispatch().
# Bundling these lets us add new fields without churning every signature.
# ---------------------------------------------------------------------------
def _safe_float(x) -> Optional[float]:
    """Best-effort numeric coercion. Returns ``None`` when the value is
    missing, NaN, or non-numeric — useful for parent-score lookups
    where a function may not have been scored yet."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


@dataclasses.dataclass
class _Generation:
    """Output of a single A1 write call.

    Attributes:
        thought: natural-language strategy A1 emitted (may be empty).
        code: function body string A1 produced (may be empty if both
            the initial parse and the format-fix retry failed).
        operator: which EoH operator was selected
            ('i1' / 'e1' / 'e2' / 'm1' / 'm2' / 'm3' / 'two_step').
        hint: the A3 reflection text that fed this prompt (may be ''
            when A3 was disabled or had nothing to say). The
            dispatcher uses this verbatim when calling
            ``A3.observe_outcome``.
        parent_score: score of the better parent the child should
            beat. Used for Δ in A3's self-feedback FIFO. ``None``
            means the operator had no parent (e.g. i1 init phase).
    """
    thought: str
    code: str
    operator: str
    hint: str = ''
    parent_score: Optional[float] = None


class LLM(ABC):
    """Language model that predicts continuation of provided source code.

    Subclasses must implement _call_api() as the single point of LLM invocation.
    Three generation modes are provided:
      - draw_samples(): original FunSearch code continuation (backward compatible)
      - generate_thought(): produce a natural-language heuristic strategy
      - generate_code_from_thought(): translate a thought into pure Python code
    """

    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    # ------------------------------------------------------------------
    # Core API – subclasses MUST override
    # ------------------------------------------------------------------
    @abstractmethod
    def _call_api(self, prompt: str) -> str:
        """Send a single prompt to the LLM and return the raw response text."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Original FunSearch interface (kept for backward compatibility)
    # ------------------------------------------------------------------
    def _draw_sample(self, prompt: str) -> str:
        """Returns a predicted continuation of `prompt`."""
        return self._call_api(prompt)

    def draw_samples(self, prompt: str) -> Collection[str]:
        """Returns multiple predicted continuations of `prompt`."""
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]

    # ------------------------------------------------------------------
    # Two-step thought-guided generation
    # ------------------------------------------------------------------
    def generate_thought(self, context: str, interface_spec: str = '') -> str:
        """Step 1: evolve / create a heuristic strategy in natural language.

        Args:
            context: assembled string containing parent thoughts, diagnosis
                     symptoms, RAG knowledge, etc.
            interface_spec: function interface specification to ground the LLM.
        Returns:
            A 2-3 sentence natural-language strategy description.
        """
        prompt = (
            f"{interface_spec}\n\n"
            f"{context}\n\n"
            "Based on the information above, propose an improved heuristic strategy "
            "in 2-3 sentences. Describe the core idea, what mathematical signals "
            "derived ONLY from `item` (float) and `bins` (1-D remaining-capacity "
            "array) to use, and how to combine them into a score for each bin. "
            "Do NOT output any code."
        )
        response = self._call_api(prompt)
        return response.strip()

    def generate_code_from_thought(
            self,
            thought: str,
            function_header: str,
            extra_context: str = '',
            interface_spec: str = '',
    ) -> str:
        """Step 2: translate a thought into pure Python code.

        Args:
            thought: the natural-language strategy.
            function_header: the function signature the LLM must implement.
            extra_context: optional extra info (e.g. error trace for local fix).
            interface_spec: function interface specification to ground the LLM.
        Returns:
            Raw Python code string (will be trimmed by evaluator).
        """
        baseline = (
            "### Current baseline (score ≈ -212.75):\n"
            "```python\n"
            "def priority(item: float, bins: np.ndarray) -> np.ndarray:\n"
            "    ratios = item / bins\n"
            "    log_ratios = np.log(ratios)\n"
            "    priorities = -log_ratios\n"
            "    return priorities\n"
            "```\n"
            "Your code MUST beat this baseline. Lower (more negative) score is "
            "worse; higher is better."
        )
        parts = [interface_spec] if interface_spec else []
        parts.append(baseline)
        parts.append(f"Heuristic strategy to implement:\n{thought}")
        if extra_context:
            parts.append(extra_context)
        parts.append(
            "Now write the COMPLETE Python function implementing the strategy. "
            "Requirements:\n"
            "1. Start with `def priority(item: float, bins: np.ndarray) -> np.ndarray:`\n"
            "2. Include a `return` statement that returns a 1-D np.ndarray "
            "of the same shape as `bins`.\n"
            "3. Do NOT include `import` statements — `numpy` is already "
            "available as `np`.\n"
            "4. Output ONLY the function definition, no explanation."
        )
        prompt = '\n\n'.join(parts)
        response = self._call_api(prompt)
        return self._extract_function_body(response)

    @staticmethod
    def _extract_function_body(raw_code: str) -> str:
        """Post-process LLM output: strip markdown fences, imports, def line,
        and docstring — return only the indented function body."""
        code = raw_code.strip()

        # 1. Strip markdown code fences
        code = re.sub(r'^```[a-zA-Z]*\n?', '', code)
        code = re.sub(r'\n?```$', '', code)
        code = code.strip()

        # 2. Remove top-level import lines
        lines = code.split('\n')
        lines = [l for l in lines if not re.match(
            r'^(import |from \S+ import )', l.strip())]
        code = '\n'.join(lines).strip()

        # 3. If there's a def line, extract only the body after it
        def_match = re.search(r'^def \w+\(.*?\).*?:\s*$', code, re.MULTILINE)
        if def_match:
            after_def = code[def_match.end():]
            body_lines = after_def.split('\n')

            # Skip docstring if present
            stripped = after_def.lstrip('\n')
            if stripped.lstrip().startswith('"""') or stripped.lstrip().startswith("'''"):
                quote = '"""' if '"""' in stripped else "'''"
                first_quote = stripped.find(quote)
                second_quote = stripped.find(quote, first_quote + 3)
                if second_quote != -1:
                    after_docstring = stripped[second_quote + 3:]
                    body_lines = after_docstring.split('\n')

            code = '\n'.join(body_lines)

        # 4. Ensure body has proper indentation (at least 4 spaces)
        if code.strip():
            dedented = textwrap.dedent(code)
            indented_lines = []
            for line in dedented.split('\n'):
                if line.strip():
                    indented_lines.append('    ' + line)
                else:
                    indented_lines.append('')
            code = '\n'.join(indented_lines)

        # NOTE: we DO NOT inject a fallback `return ...` here anymore.
        # If the LLM failed to produce a body with an explicit `return`,
        # let `evaluator._validate_sample_body()` reject it cleanly so the
        # sample is excluded from the evolutionary database instead of being
        # silently rewritten to a default heuristic that pollutes the search.
        return code


class Sampler:
    """Node that samples program continuations and sends them for analysis."""

    _global_samples_nums: int = 1

    FUNCTION_INTERFACE_SPEC = (
        "## Function interface (MUST follow strictly)\n"
        "```\n"
        "def priority(item: float, bins: np.ndarray) -> np.ndarray\n"
        "```\n"
        "- `item`: a **float scalar** — the size of the item to pack.\n"
        "- `bins`: a **1-D np.ndarray** — each element is the **remaining capacity** "
        "of that bin. The array has already been filtered to only include bins where "
        "the item fits (i.e. every element >= item). There is NO other information "
        "about the bins (no 2-D structure, no history of packed items).\n"
        "- **Return**: a **1-D np.ndarray of the same shape as `bins`**. "
        "Each element is the priority score for the corresponding bin. "
        "The bin with the **highest** score will be chosen.\n\n"
        "### Critical constraints\n"
        "- Do NOT use `axis=1`, `.sum(axis=1)`, `.shape[1]`, or treat `bins` as 2-D.\n"
        "- Do NOT return indices or sorted arrays — return a **score array**.\n"
        "- Do NOT filter or mask bins (they are already valid).\n"
        "- Only use `item` (scalar) and `bins` (1-D array) as inputs.\n"
        "- You may use numpy operations.\n"
    )

    def __init__(
            self,
            database: programs_database.ProgramsDatabase,
            evaluators: Sequence[evaluator.Evaluator],
            samples_per_prompt: int,
            max_sample_nums: int | None = None,
            llm_class: Type[LLM] = LLM,
            # --- new components (optional, for backward compat) ---
            reflector=None,
            knowledge_base=None,
            extractor=None,
            reflector_config=None,
            domain_id: str = 'PROB_BIN_PACKING_1D',
            # --- EoH multi-operator mode ---
            use_eoh_operators: bool = True,
            eoh_task_spec: 'eoh_operators.TaskSpec | None' = None,
            eoh_operator_weights: dict | None = None,
            # --- EoH-style explicit init phase ---
            # When > 0, the first `init_population_size` LLM samples are
            # FORCED to use the i1 (initialisation) prompt, regardless of
            # what `pick_operator` would have chosen. This mirrors EoH's
            # vanilla flow: generate N diverse heuristics from scratch via
            # `get_prompt_i1` BEFORE crossover/mutation operators (e1/e2/m1/
            # m2/m3) are allowed to fire. Without this, the initial
            # population is dominated by the single hand-coded seed in
            # `specification`, which biases everything downstream.
            init_population_size: int = 0,
            # --- LLAMEA-style error memory ---
            # When non-None, recent runtime errors are stored and rendered
            # as an "## Recent failures to avoid" block at the top of every
            # EoH operator prompt. This dramatically cuts the rate at which
            # the LLM repeats the same mistake (e.g. `axis=1` on a 1-D
            # `bins` array) across consecutive samples. Pass None to
            # disable for ablation runs. Structural ``InvalidSampleError``
            # records are NOT recorded — only sandbox / runtime failures,
            # because telling the LLM "your body was empty" doesn't teach
            # anything strategic and is already handled by format-fix
            # retry.
            error_memory: 'error_memory_lib.ErrorMemory | None' = None,
            # --- ReEvo-style verbal reflection ("verbal gradient") ---
            # When non-None, before every crossover-flavoured operator
            # (e1/e2) we ask the reflector to produce a < 20-word
            # short-term hint comparing the worst vs. best parent shown
            # to the generator. For mutation operators (m1/m2/m3) we
            # inject the accumulated long-term reflection. Long-term
            # updates are triggered every ``reevo_lt_update_period``
            # registered samples — mirroring ReEvo's "once per
            # generation" cadence with a sample-count proxy because we
            # have no explicit generation boundary. Pass None to disable
            # cleanly for ablation; format/fallback paths still work.
            reevo_reflector: 'reevo_reflector_lib.ReevoReflector | None' = None,
            reevo_lt_update_period: int = 10,
            # --- A4 Search-Controller ---
            search_controller: 'search_controller_lib.SearchController | None' = None,
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums

        # ─── A2: Bug-Fixer Agent ──────────────────────────────────────
        # In the v3.5/EoH path this is a dedicated bug-fix lane:
        # deterministic known recipes first, LLM fallback second.  The
        # legacy V2 triage path is retained only for non-EoH old runs.
        self._a2_bug_fixer = reflector
        self._a2_config = reflector_config
        self._a2_bug_memory = bug_fix_memory_lib.BugFixMemory()

        self._knowledge_base = knowledge_base
        self._extractor = extractor
        self._domain_id = domain_id
        self._best_score: float = -float('inf')
        self._invalid_sample_count: int = 0
        self._runtime_error_count: int = 0
        # EoH config
        self._use_eoh_operators = use_eoh_operators
        self._eoh_task_spec = eoh_task_spec or eoh_operators.BIN_PACKING_TASK
        self._eoh_operator_weights = (
            eoh_operator_weights or eoh_operators.DEFAULT_OPERATOR_WEIGHTS)
        # Per-operator counters (visibility into which mutation modes work).
        self._operator_counts: dict[str, int] = {
            op: 0 for op in eoh_operators.OPERATORS
        }
        self._operator_valid_counts: dict[str, int] = {
            op: 0 for op in eoh_operators.OPERATORS
        }
        # A1 format-fix retry counters (parse failed -> retry once
        # *inside A1*). Counted here purely for run-summary logging.
        self._format_fix_attempts: int = 0
        self._format_fix_successes: int = 0
        # EoH-style init phase: force i1 for the first N samples.
        self._init_population_size: int = max(0, int(init_population_size))

        # ─── L2: ErrorMemory (cross-sample runtime-crash avoidance) ───
        # Owned by the Sampler; written by the dispatcher; READ by A1
        # when assembling its prompt. A2 and A3 do NOT read L2.
        self._error_memory = error_memory

        # ─── A3: Algorithm-Guide Agent (verbal gradient) ──────────────
        # Owns its own memories (ST buffer, LT summary, past-outcomes
        # FIFO). The dispatcher feeds it eval outcomes; nobody else
        # reaches into A3's internals.
        self._a3_algo_guide = reevo_reflector
        self._a3_lt_update_period: int = max(1, int(reevo_lt_update_period))
        self._a3_samples_since_lt: int = 0

        # ─── A4: Search-Controller Agent (operator/parent scheduling) ─
        # Owns trajectory-level scheduling memory. It does not write code,
        # repair code, or produce algorithm formulas; its JSON policy only
        # tilts operator-group and parent-source sampling probabilities.
        self._a4_search_controller = search_controller

    # ==================================================================
    # Coordinator (sample) — keeps each agent in its own lane.
    #
    #   while not done:
    #       prompt   = ProgramsDatabase.get_prompt()
    #       gen      = A1.write_code(prompt)            # A1 owns A3.hint
    #       result   = Evaluator.analyse(gen.code)      # sandbox
    #       _dispatch(result, prompt, gen)              # routes to A2/A3/L2
    #       _a3_maybe_update_long_term()                # A3 housekeeping
    #
    # The coordinator NEVER touches A3's buffers, NEVER renders L2's
    # avoidance block, and NEVER decides to retry. Routing is the
    # dispatcher's job; memory is each agent's own.
    # ==================================================================
    def sample(self, **kwargs):
        """Continuously gets prompts, samples programs, sends them for analysis."""
        while True:
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                break
            try:
                reset_time = time.time()

                for _ in range(self._samples_per_prompt):
                    next_global = self._get_global_sample_nums() + 1
                    init_island = self._init_target_island(next_global)
                    search_policy = self._a4_current_policy(
                        active=(init_island is None))
                    prompt = self._database.get_prompt(
                        island_id=init_island,
                        search_policy=search_policy)
                    self._global_sample_nums_plus_one()
                    cur_global = self._get_global_sample_nums()

                    # === A1: Coder Agent — write code ====================
                    if self._use_eoh_operators:
                        gen = self._a1_eoh_write_code(prompt)
                    else:
                        gen = self._a1_two_step_write_code(prompt)

                    sample_time = (time.time() - reset_time) / self._samples_per_prompt

                    # === Evaluator (sandbox) =============================
                    eval_result = self._run_evaluator(
                        gen, prompt, cur_global, sample_time, **kwargs)

                    # === Dispatcher (route by result class) ==============
                    self._dispatch(
                        eval_result, prompt, gen,
                        sample_time=sample_time,
                        cur_global=cur_global,
                        **kwargs,
                    )

                    # === A3 housekeeping: periodic LT update =============
                    self._a3_maybe_update_long_term()

            except Exception as e:
                logging.exception("Sampler loop error: %s", e)
                continue

    # ==================================================================
    # Evaluator wrapper (a sample crosses this on its way to dispatch)
    # ==================================================================
    def _run_evaluator(
            self, gen: _Generation, prompt, cur_global: int,
            sample_time: float, **kwargs) -> 'evaluator.EvalResult':
        chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
        return chosen_evaluator.analyse(
            gen.code,
            prompt.island_id,
            prompt.version_generated,
            thought=gen.thought,
            **kwargs,
            global_sample_nums=cur_global,
            sample_time=sample_time,
        )

    # ==================================================================
    # Dispatcher  —  the ONLY place that fans an EvalResult out to A2/A3/L2.
    # ==================================================================
    def _dispatch(
            self,
            eval_result: 'evaluator.EvalResult',
            prompt,
            gen: _Generation,
            *,
            sample_time: float,
            cur_global: int,
            **kwargs,
    ):
        """Classify the result and route it. Mutually exclusive branches.

        Branches:
            * ``valid``         — sample registered by evaluator. Update
              best score; tell A3 the hint produced a scored child.
            * ``runtime_crash`` — sandbox raised. Record into L2; if
              A2 is enabled, ask it for a one-shot fix; tell A3 the
              hint led to a crash (with whatever score the fix produced,
              or ``None`` if A2 is absent / failed).
            * ``format_fail``   — A1 already burned its single
              format-fix retry inside ``_a1_eoh_write_code``. We only
              count it here and DO NOT bother A3 (the hint isn't to
              blame for an empty/garbled function body).

        Pre-branch: bookkeeping that EVERY result needs (counters).
        Post-branch: if a legacy V2 Reflector was injected via the
        ``reflector=`` constructor arg AND it exposes ``triage``, we
        run the legacy triage path. The legacy path is fully isolated;
        it never touches ErrorMemory or A3.
        """
        operator = gen.operator
        err = eval_result.error_trace or ''
        is_format_fail = err.startswith('InvalidSampleError')
        is_runtime_crash = bool(err) and not is_format_fail

        # Per-operator success counter is part of the run summary.
        if eval_result.is_valid and operator in self._operator_valid_counts:
            self._operator_valid_counts[operator] += 1

        best_before = self._best_score
        final_child_score: Optional[float] = eval_result.reduced_score

        if is_format_fail:
            self._on_format_fail(eval_result, gen, cur_global)
            final_child_score = None
        elif is_runtime_crash:
            final_child_score = self._on_runtime_crash(
                eval_result, prompt, gen,
                sample_time=sample_time, cur_global=cur_global, **kwargs)
        else:
            self._on_valid_child(eval_result, gen)

        # Best-score bookkeeping is centralised — every branch eventually
        # produces (or fails to produce) something we want to compare
        # against ``self._best_score``.
        self._update_best_score(eval_result)
        self._a4_observe_event(
            gen=gen,
            cur_global=cur_global,
            child_score=final_child_score,
            is_valid=bool(final_child_score is not None),
            error_trace=err,
            best_before=best_before,
            best_after=self._best_score,
        )

        # Legacy V2 path (kept only for backward compat with old
        # configs). Runs after the new dispatch so the new memories
        # are already updated; skipped entirely when not configured.
        if self._is_legacy_v2_reflector():
            self._legacy_v2_triage(
                eval_result, prompt, gen,
                sample_time=sample_time, cur_global=cur_global, **kwargs)

    # ─── valid branch ─────────────────────────────────────────────────
    def _on_valid_child(self, eval_result, gen: _Generation):
        """A scored child was registered. Notify A3 of the outcome."""
        self._a3_observe_outcome(
            gen, child_score=eval_result.reduced_score)

    # ─── runtime-crash branch ─────────────────────────────────────────
    def _on_runtime_crash(
            self, eval_result, prompt, gen: _Generation,
            *, sample_time, cur_global, **kwargs):
        """Crash routing: L2 record → A2 fix → A3 observe (final score)."""
        self._runtime_error_count += 1
        logging.info(
            'Runtime-failed sample #%d op=%s (cum=%d): %s',
            cur_global, gen.operator, self._runtime_error_count,
            (eval_result.error_trace or '')[:160])

        # L2 records the crash so A1's NEXT prompt sees the avoidance
        # block. Strictly cross-sample; never reaches A2 or A3.
        if self._error_memory is not None:
            self._error_memory.record(
                gen.operator, eval_result.error_trace or '')
            logging.info(
                '[error_memory] op=%s recorded (buf=%d, lifetime=%d)',
                gen.operator,
                len(self._error_memory),
                self._error_memory.total_recorded)

        # A2 attempts a one-shot fix when configured. Otherwise the
        # final outcome attributed to this hint is "no valid score".
        final_score: Optional[float] = None
        if self._a2_bug_fixer is not None and not self._is_legacy_v2_reflector():
            fix_result = self._a2_fix_runtime_bug(
                gen, eval_result, prompt,
                sample_time=sample_time, cur_global=cur_global, **kwargs)
            if fix_result is not None and fix_result.is_valid:
                final_score = fix_result.reduced_score
                self._update_best_score(fix_result)

        self._a3_observe_outcome(gen, child_score=final_score)
        return final_score

    # ─── format-fail branch ───────────────────────────────────────────
    def _on_format_fail(self, eval_result, gen: _Generation, cur_global: int):
        """A1's single format-fix retry already failed inside A1. Only count."""
        self._invalid_sample_count += 1
        logging.info(
            'Rejected invalid sample #%d op=%s (cum=%d): %s',
            cur_global, gen.operator, self._invalid_sample_count,
            (eval_result.error_trace or '')[:160])
        # Deliberate omission: A3 is NOT notified. Empty/garbled body
        # is a coding-level glitch; not a strategic mis-step that A3
        # should learn from.

    # ==================================================================
    # A3 housekeeping (periodic LT update + observe wrapper)
    # ==================================================================
    def _a3_observe_outcome(
            self, gen: _Generation, *, child_score: Optional[float]):
        """Tell A3 how its hint played out. Empty hint ⇒ no-op upstream."""
        if self._a3_algo_guide is None:
            return
        try:
            self._a3_algo_guide.observe_outcome(
                hint=gen.hint,
                operator=gen.operator,
                code=gen.code or '',
                child_score=child_score,
                parent_score=gen.parent_score,
            )
        except Exception as e:
            logging.warning('[a3] observe_outcome raised: %s', e)

    def _a3_maybe_update_long_term(self):
        """Trigger the periodic LT distill. No-op when disabled or empty."""
        if self._a3_algo_guide is None:
            return
        self._a3_samples_since_lt += 1
        if self._a3_samples_since_lt < self._a3_lt_update_period:
            return
        self._a3_samples_since_lt = 0
        try:
            self._a3_algo_guide.update_long_term()
            stats = self._a3_algo_guide.stats
            logging.info(
                '[a3] LT update fired (st_calls=%d, lt_calls=%d, '
                'st_buf=%d, has_lt=%s, past=%d)',
                stats['st_calls'], stats['lt_calls'],
                stats['st_buffer'], stats['has_long_term'],
                stats.get('past_outcomes', 0))
        except Exception as e:
            logging.warning('[a3] LT update raised: %s', e)

    # ==================================================================
    # A4 housekeeping (search policy + trajectory memory)
    # ==================================================================
    def _a4_current_policy(self, *, active: bool):
        if not active or getattr(self, '_a4_search_controller', None) is None:
            return None
        a3_summary = ''
        if self._a3_algo_guide is not None:
            try:
                a3_summary = self._a3_algo_guide.render_trajectory_summary()
            except Exception:
                a3_summary = ''
        return self._a4_search_controller.maybe_refresh_policy(
            a3_summary=a3_summary)

    def _a4_observe_event(
            self,
            *,
            gen: _Generation,
            cur_global: int,
            child_score: Optional[float],
            is_valid: bool,
            error_trace: str,
            best_before: float,
            best_after: float,
    ) -> None:
        if getattr(self, '_a4_search_controller', None) is None:
            return
        try:
            self._a4_search_controller.observe(
                sample_id=cur_global,
                operator=gen.operator,
                thought=gen.thought,
                a3_hint=gen.hint,
                code=gen.code or '',
                child_score=child_score,
                parent_score=gen.parent_score,
                valid=is_valid,
                error_trace=error_trace,
                best_before=best_before,
                best_after=best_after,
            )
        except Exception as e:
            logging.warning('[a4] observe_event raised: %s', e)

    # ==================================================================
    # Legacy V2 reflector compatibility (kept isolated)
    # ==================================================================
    def _is_legacy_v2_reflector(self) -> bool:
        return (self._a2_bug_fixer is not None
                and hasattr(self._a2_bug_fixer, 'triage')
                and not getattr(self, '_use_eoh_operators', True))

    def _legacy_v2_triage(self, eval_result, prompt, gen: _Generation,
                          *, sample_time, cur_global, **kwargs):
        """Mirror of the original ``_do_triage`` flow.

        Only invoked when an old-style V2 ``Reflector`` (with ``triage``)
        was injected via ``reflector=``. This path runs the LLM-based
        symptom diagnosis / RAG-guided rewrite. It is intentionally
        isolated — neither A3 nor L2 are referenced here.
        """
        triage_result = self._a2_bug_fixer.triage(eval_result, self._best_score)
        branch = triage_result.branch
        if branch == 'new_sota':
            self._handle_sota(eval_result, gen.thought, **kwargs)
        elif branch == 'local_bug':
            self._handle_local_fix(
                gen.thought, eval_result, prompt,
                sample_time, cur_global, **kwargs)
        elif branch == 'global_drawback':
            self._handle_global_diagnosis(
                eval_result, gen.thought, prompt,
                sample_time, cur_global, **kwargs)

    # ==================================================================
    # A1: Coder Agent
    # ==================================================================
    # Two write modes: EoH multi-operator (default) and the legacy
    # two-step "thought then code" path.  Each returns a ``_Generation``
    # bag so the dispatcher gets the SAME shape regardless of mode and
    # can hand ``hint`` / ``parent_score`` straight to A3.

    def _a1_two_step_write_code(self, prompt) -> _Generation:
        """Original v2 path: separate thought call + code call."""
        parent_thoughts = self._extract_parent_thoughts(prompt)
        thought_context = self._build_thought_context(parent_thoughts)

        thought = self._llm.generate_thought(
            thought_context,
            interface_spec=self.FUNCTION_INTERFACE_SPEC,
        )
        logging.info("Generated thought: %s", thought[:80])

        code = self._llm.generate_code_from_thought(
            thought, prompt.function_header,
            interface_spec=self.FUNCTION_INTERFACE_SPEC,
        )
        return _Generation(thought=thought, code=code, operator='two_step')

    def _a1_eoh_write_code(self, prompt) -> _Generation:
        """EoH path: pick one of {i1,e1,e2,m1,m2,m3}, single LLM call.

        If we are still inside the EoH-style init phase
        (``init_population_size``), we ignore the operator weights entirely
        and force i1, so the initial population is built from N diverse
        from-scratch heuristics rather than parents-of-the-seed.

        A1 is also responsible for ONE format-fix retry (when the parse
        of the first response is empty). Subsequent failures are NOT
        retried here — they fall through to the dispatcher's
        ``format_fail`` branch which only counts.
        """
        parents = list(prompt.parent_implementations)
        cur_global = self._get_global_sample_nums()

        operator, parent_dicts = self._a1_pick_operator(parents, cur_global)

        # === A3 hint (verbal gradient) ============================
        # Crossover ops (e1/e2): override the random parent pick with
        # the explicit (worst, best) pair from the island so that the
        # short-term reflection refers to the SAME two snippets the
        # LLM is shown. Mutation ops (m1/m2/m3): use the LT summary.
        hint = ''
        parent_score: Optional[float] = None
        if self._a3_algo_guide is not None and operator != 'i1':
            if operator in ('e1', 'e2') and len(parents) >= 2:
                worse_fn, better_fn = parents[0], parents[-1]
                if worse_fn is not better_fn:
                    parent_dicts = [
                        eoh_operators.function_to_parent_dict(worse_fn),
                        eoh_operators.function_to_parent_dict(better_fn),
                    ]
                    parent_score = _safe_float(
                        getattr(better_fn, 'score', None))
                    hint = self._a3_algo_guide.short_term_reflect(
                        better_code=parent_dicts[1].get('code', ''),
                        worse_code=parent_dicts[0].get('code', ''),
                    )
                    if hint:
                        logging.info(
                            '[a3] op=%s ST hint (%d chars): %s',
                            operator, len(hint), hint[:120])
            elif operator in ('m1', 'm2', 'm3'):
                hint = self._a3_algo_guide.get_long_term()
                if parents:
                    parent_score = _safe_float(
                        getattr(parents[-1], 'score', None))
                if hint:
                    logging.info(
                        '[a3] op=%s using LT hint (%d chars)',
                        operator, len(hint))

        # === L2 avoidance block (cross-sample crash prevention) ===
        # ``ErrorMemory.render_for_prompt`` returns '' when empty so we
        # can pass it unconditionally.
        avoidance = (self._error_memory.render_for_prompt()
                     if self._error_memory is not None else '')

        # === Build prompt + first generation call ================
        spec = dataclasses.replace(
            self._eoh_task_spec,
            interface_spec=self.FUNCTION_INTERFACE_SPEC,
        )
        prompt_text = eoh_operators.build_prompt(
            operator, parent_dicts, spec,
            error_avoidance=avoidance,
            reflection=hint,
            init_diversity=self._a1_init_diversity_instruction(
                operator, cur_global),
        )
        raw = self._llm._call_api(prompt_text)
        thought, code = eoh_operators.parse_response(
            raw, function_name=spec.function_name)

        # === A1's single format-fix retry ========================
        if not (code or '').strip():
            thought, code = self._a1_format_fix_retry(
                operator=operator, raw_prev=raw, spec=spec,
                fallback_thought=thought)

        # Per-operator counters / log line.
        self._operator_counts[operator] = (
            self._operator_counts.get(operator, 0) + 1)
        logging.info(
            'EoH op=%s thought=%s', operator, (thought or '')[:80])

        return _Generation(
            thought=thought, code=code, operator=operator,
            hint=hint, parent_score=parent_score,
        )

    def _a1_pick_operator(self, parents, cur_global: int):
        """Operator selection + parent dict assembly. Pure A1 concern."""
        if (self._init_population_size > 0
                and cur_global <= self._init_population_size):
            logging.info(
                'EoH init phase: forcing i1 for sample #%d/%d',
                cur_global, self._init_population_size)
            return 'i1', []
        weights = self._eoh_operator_weights
        if getattr(self, '_a4_search_controller', None) is not None:
            policy = self._a4_search_controller.current_policy
            weights = policy.apply_operator_bias(self._eoh_operator_weights)
        operator = eoh_operators.pick_operator(weights)
        parent_dicts = eoh_operators._select_parents_for_operator(
            operator, parents)
        if not parent_dicts and operator != 'i1':
            if parents:
                fallback = np.random.choice(['m1', 'm2', 'm3'])
                parent_dicts = eoh_operators._select_parents_for_operator(
                    fallback, parents)
                if parent_dicts:
                    logging.info(
                        'EoH falling back from %s to %s '
                        '(only %d parents available)',
                        operator, fallback, len(parents))
                    return fallback, parent_dicts
            logging.info(
                'EoH falling back from %s to i1 (only %d parents available)',
                operator, len(parents))
            return 'i1', []
        return operator, parent_dicts

    def _init_target_island(self, next_global: int) -> Optional[int]:
        """Round-robin init samples across islands so founders diversify."""
        if (self._init_population_size <= 0
                or next_global > self._init_population_size):
            return None
        num_islands = getattr(self._database, 'num_islands', 0) or 0
        if num_islands <= 0:
            return None
        island_id = (next_global - 1) % num_islands
        logging.info(
            'EoH init phase: routing sample #%d/%d to island %d',
            next_global, self._init_population_size, island_id)
        return island_id

    def _a1_init_diversity_instruction(
            self, operator: str, cur_global: int) -> str:
        """Optional i1-only extra instruction.

        Kept as a hook, but intentionally disabled.  A1's role is code
        generation, not search-policy control; bounded exploration should come
        from A3's reflection/trajectory summary after evaluation feedback.
        Returning an empty string keeps i1 close to EoH's vanilla
        initialization prompt.
        """
        return ''

    def _a1_format_fix_retry(
            self, *, operator: str, raw_prev: str, spec, fallback_thought: str):
        """ONE retry to recover an empty/garbled function body.

        Pure A1 logic — no L2 / A2 / A3 interaction. Failure here makes
        the sample fall through as ``InvalidSampleError`` to the
        dispatcher's ``format_fail`` branch.
        """
        self._format_fix_attempts += 1
        fix_prompt = eoh_operators.build_format_fix_prompt(raw_prev, spec)
        try:
            raw2 = self._llm._call_api(fix_prompt)
        except Exception as e:
            logging.warning('Format-fix call failed: %s', e)
            raw2 = ''
        thought2, code2 = eoh_operators.parse_response(
            raw2, function_name=spec.function_name)
        if (code2 or '').strip():
            self._format_fix_successes += 1
            logging.info(
                'EoH op=%s format-fix recovered (cum=%d/%d)',
                operator, self._format_fix_successes,
                self._format_fix_attempts)
            return (thought2 or fallback_thought), code2
        logging.info(
            'EoH op=%s format-fix still failed '
            '(raw0=%d chars, raw1=%d chars)',
            operator, len(raw_prev or ''), len(raw2 or ''))
        return fallback_thought, ''

    # ==================================================================
    # A2: Bug-Fixer Agent
    # ==================================================================
    # Owns NO cross-sample memory. State is scoped to a single sample
    # and dies when the function returns. A2 does not look at L2
    # ErrorMemory and does not talk to A3. The only hand-off is back
    # through the dispatcher (return value), which then notifies A3
    # with the FINAL score (or ``None`` if A2 still couldn't get the
    # code past the sandbox).

    def _a2_fix_runtime_bug(
            self, gen: _Generation, eval_result, prompt,
            *, sample_time, cur_global, **kwargs) -> 'evaluator.EvalResult | None':
        """Repair a runtime crash in the A2 lane.

        A2 first checks its own BugFixMemory for deterministic repair
        recipes.  If no known recipe applies (or the recipe fails), it
        falls back to a small number of LLM repair attempts.  A2 memory
        stores code-level repair outcomes only; it never receives A3's
        algorithm-quality state.
        """
        max_attempts = 2
        if self._a2_config and hasattr(self._a2_config, 'max_fix_attempts'):
            max_attempts = self._a2_config.max_fix_attempts

        signature = self._a2_bug_memory.classify(eval_result.error_trace)
        patched_code, recipe = self._a2_bug_memory.deterministic_patch(
            gen.code, eval_result.error_trace)
        if patched_code:
            new_result = self._a2_evaluate_fixed_code(
                patched_code, gen, prompt,
                sample_time=sample_time, cur_global=cur_global, **kwargs)
            success = bool(new_result is not None and new_result.is_valid)
            self._a2_bug_memory.record(
                signature, recipe, success, eval_result.error_trace)
            if success:
                logging.info(
                    '[a2] op=%s deterministic fix succeeded '
                    '(signature=%s, recipe=%s, score=%.3f)',
                    gen.operator, signature, recipe,
                    new_result.reduced_score
                    if new_result.reduced_score is not None else float('nan'))
                return new_result
            logging.info(
                '[a2] op=%s deterministic fix failed '
                '(signature=%s, recipe=%s)',
                gen.operator, signature, recipe)

        for attempt in range(max_attempts):
            extra = (
                f"Your previous code produced an error:\n"
                f"{eval_result.error_trace}\n"
                f"Original function body:\n{gen.code}\n\n"
                f"Please fix the bug while keeping the same strategy."
            )
            fixed_code = self._llm.generate_code_from_thought(
                gen.thought, prompt.function_header, extra_context=extra,
                interface_spec=self.FUNCTION_INTERFACE_SPEC,
            )
            new_result = self._a2_evaluate_fixed_code(
                fixed_code, gen, prompt,
                sample_time=sample_time, cur_global=cur_global, **kwargs)
            if new_result.is_valid:
                self._a2_bug_memory.record(
                    signature, 'llm_fallback', True, eval_result.error_trace)
                logging.info(
                    '[a2] op=%s fix succeeded on attempt %d (score=%.3f)',
                    gen.operator, attempt + 1,
                    new_result.reduced_score
                    if new_result.reduced_score is not None else float('nan'))
                return new_result
        self._a2_bug_memory.record(
            signature, 'llm_fallback', False, eval_result.error_trace)
        logging.info(
            '[a2] op=%s fix failed after %d attempts',
            gen.operator, max_attempts)
        return None

    def _a2_evaluate_fixed_code(
            self, fixed_code: str, gen: _Generation, prompt,
            *, sample_time, cur_global, **kwargs) -> 'evaluator.EvalResult':
        """Evaluate an A2 repair candidate without touching other memories."""
        chosen_evaluator = np.random.choice(self._evaluators)
        return chosen_evaluator.analyse(
            fixed_code,
            prompt.island_id,
            prompt.version_generated,
            thought=gen.thought,
            **kwargs,
            global_sample_nums=cur_global,
            sample_time=sample_time,
        )

    # ------------------------------------------------------------------
    # Legacy V2 reflector branches (only used by ``_legacy_v2_triage``)
    # ------------------------------------------------------------------
    def _handle_local_fix(self, thought, eval_result, prompt,
                          sample_time, cur_global, **kwargs):
        max_attempts = 2
        if self._a2_config:
            max_attempts = self._a2_config.max_fix_attempts

        for attempt in range(max_attempts):
            extra = (
                f"Your previous code produced an error:\n"
                f"{eval_result.error_trace}\n"
                f"Please fix the bug while keeping the same strategy."
            )
            fixed_code = self._llm.generate_code_from_thought(
                thought, prompt.function_header, extra_context=extra,
                interface_spec=self.FUNCTION_INTERFACE_SPEC,
            )
            chosen_evaluator = np.random.choice(self._evaluators)
            new_result = chosen_evaluator.analyse(
                fixed_code,
                prompt.island_id,
                prompt.version_generated,
                thought=thought,
                **kwargs,
                global_sample_nums=cur_global,
                sample_time=sample_time,
            )
            if new_result.is_valid:
                self._update_best_score(new_result)
                logging.info("Local fix succeeded on attempt %d", attempt + 1)
                return
        logging.info("Local fix failed after %d attempts", max_attempts)

    def _handle_global_diagnosis(self, eval_result, old_thought, prompt,
                                 sample_time, cur_global, **kwargs):
        symptom = self._a2_bug_fixer.diagnose_symptom(
            code=eval_result.program,
            score=eval_result.reduced_score,
            thought=old_thought,
        )
        logging.info("Diagnosed symptom: %s", symptom[:80] if symptom else "None")

        knowledge = None
        if self._knowledge_base and symptom:
            knowledge = self._knowledge_base.search(symptom, self._domain_id)

        guided_context = self._build_guided_context(old_thought, symptom, knowledge)
        new_thought = self._llm.generate_thought(
            guided_context,
            interface_spec=self.FUNCTION_INTERFACE_SPEC,
        )
        new_code = self._llm.generate_code_from_thought(
            new_thought, prompt.function_header,
            interface_spec=self.FUNCTION_INTERFACE_SPEC,
        )

        chosen_evaluator = np.random.choice(self._evaluators)
        new_result = chosen_evaluator.analyse(
            new_code,
            prompt.island_id,
            prompt.version_generated,
            thought=new_thought,
            **kwargs,
            global_sample_nums=cur_global,
            sample_time=sample_time,
        )
        self._update_best_score(new_result)

    # ------------------------------------------------------------------
    # Branch C: SOTA knowledge extraction
    # ------------------------------------------------------------------
    def _handle_sota(self, eval_result, thought, **kwargs):
        self._update_best_score(eval_result)
        if self._extractor and self._knowledge_base:
            try:
                tactic = self._extractor.extract_tactic(
                    code=eval_result.program,
                    thought=thought,
                    score=eval_result.reduced_score,
                    domain_id=self._domain_id,
                )
                if tactic:
                    self._knowledge_base.add_tactic(tactic)
                    logging.info("Extracted new L4 tactic: %s", tactic.get('name', ''))
            except Exception as e:
                logging.warning("Knowledge extraction failed: %s", e)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _update_best_score(self, eval_result):
        if eval_result.reduced_score is not None:
            if eval_result.reduced_score > self._best_score:
                self._best_score = eval_result.reduced_score

    def _extract_parent_thoughts(self, prompt) -> list[str]:
        """Extract thought strings from parent functions in the prompt."""
        thoughts = []
        if hasattr(prompt, 'thoughts') and prompt.thoughts:
            thoughts = prompt.thoughts
        return thoughts

    def _build_thought_context(self, parent_thoughts: list[str]) -> str:
        """Build the context string for thought generation in normal evolution."""
        if not parent_thoughts:
            return (
                "You are designing a heuristic for an online bin-packing problem.\n"
                "The goal is to minimize the number of bins used. Items arrive one "
                "by one; for each item you assign a priority score to every candidate "
                "bin and the item goes into the highest-scored bin.\n"
                "The current baseline uses `priorities = -np.log(item / bins)` "
                "(i.e. `np.log(bins / item)`) which achieves score ≈ -212.75.\n"
                "Propose a better strategy using only `item` (float) and `bins` "
                "(1-D remaining-capacity array)."
            )
        lines = []
        for i, t in enumerate(parent_thoughts):
            lines.append(f"Strategy v{i}: {t}")
        lines.append(
            "\nThe baseline score is ≈ -212.75 (using log(bins/item))."
            "\nPropose an improved strategy that addresses weaknesses of the above "
            "and can beat the baseline. Remember you can only use `item` (float) "
            "and `bins` (1-D remaining-capacity array)."
        )
        return '\n'.join(lines)

    def _build_guided_context(self, old_thought, symptom, knowledge) -> str:
        """Build the context string for RAG-guided thought generation (Branch B)."""
        parts = [f"Previous strategy: {old_thought}"]
        if symptom:
            parts.append(f"Diagnosed core symptom: {symptom}")
        if knowledge:
            name = knowledge.get('name', '')
            mechanism = knowledge.get('mechanism', knowledge.get('core_philosophy', ''))
            hint = knowledge.get('generator_prompt_hint', '')
            tactic_desc = knowledge.get('tactic_description', '')
            parts.append("Reference knowledge from cross-domain patterns:")
            if name:
                parts.append(f"  Name: {name}")
            if mechanism:
                parts.append(f"  Mechanism: {mechanism}")
            if hint:
                parts.append(f"  Application hint: {hint}")
            if tactic_desc:
                parts.append(f"  Tactic: {tactic_desc}")
        parts.append(
            "\nPropose a completely new strategy (2-3 sentences) that "
            "incorporates the above knowledge to fix the diagnosed symptom. "
            "Remember you can only use `item` (float scalar) and `bins` "
            "(1-D remaining-capacity array) — no 2-D operations, no history."
        )
        return '\n'.join(parts)

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1
