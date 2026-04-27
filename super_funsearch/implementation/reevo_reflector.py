"""ReEvo-style verbal reflection ("verbal gradient") for EoH operators.

This module implements **A3 Algorithm-Guide Agent** in v3.5 architecture
(see ``doc/architecture_v3.5.md``).  It is the *only* module that owns
algorithm-level reflection memory.  It does not, and must not, see
runtime errors / format failures — those belong to A2 (Bug Fixer) and
``ErrorMemory`` (L2) respectively.

Why this module exists
----------------------
EoH's i1/e1/e2/m1/m2/m3 prompts make the LLM design heuristics from
scratch every time. There is no signal that tells it *why* one parent
beats another. ReEvo's contribution is exactly that: ask a separate
"reflector" LLM to compare a worse vs. better parent and emit a
short hint, then splice that hint into the next code-generation prompt.
The reflector is doing for natural language what gradient descent does
for weights — hence "verbal gradient".

We keep three memories, all owned by *this* class (so A3's mind never
mixes with A1/A2's):

* **Short-term (ST) reflection buffer** — pairwise compare
  ``(worse, better)`` and produce ``< 20 words``. Triggered before
  crossover-flavoured operators (e1/e2). Each ST output is pushed into
  ``self._st_buffer``.
* **Long-term (LT) reflection** — periodically distil the ST buffer
  (plus the previous LT string) into ``< 50 words`` of accumulated
  insight. Used as the [Reflection] block for mutation operators
  (m1/m2/m3) which only have a single elite parent and therefore can't
  do a pairwise compare.
* **Past-outcomes self-feedback** — a FIFO of ``(hint → child code,
  pattern, score Δ)`` records.  Rendered into every reflector prompt as
  ``## Your past suggestions and their measured impact``.  This closes
  the loop on A3: it now sees *which of its own past hints actually
  worked* and can refine / abandon them.  This is the v3.5 equivalent
  of "verbal gradient with reward signal" — without any model training.

Design choices kept faithful to ReEvo
-------------------------------------
1. **Prompts are loaded verbatim from disk** (``prompts_reevo/``) — the
   v2 lesson was "don't redesign reflection prompts from scratch".
2. **System / user prompts are concatenated into a single string**
   because our ``LLM._call_api`` only accepts a single text payload
   (the upstream OpenAI-compatible adapter wraps it as the user
   message). ReEvo splits them across roles which is functionally
   equivalent.
3. **ST length budget is enforced softly**: we trust the prompt's
   ``less than 20 words`` instruction and only truncate as a safety
   net, identical to ReEvo's behaviour.
4. **LT updates are sample-driven**, not generation-driven, because
   our search has no explicit "generation" boundary like ReEvo's
   `evolve()` loop. ``update_long_term`` is invoked by the sampler
   every ``lt_update_period`` samples; if the ST buffer is empty the
   call is a no-op so it's safe to invoke unconditionally.
5. **Past-outcomes records store *abstracted* code, not full bodies**.
   We deliberately render at most ``EXCERPT_MAX_LINES`` non-trivial
   lines plus a rule-based pattern label; never the full function. The
   reflector can therefore reason about *strategy*, but cannot copy
   solutions verbatim into its hint (that would amount to A3 doing
   A1's job, which is the very mixing the user explicitly forbade).
"""
from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type

from implementation import sampler
from implementation import structure_analysis


# Hard cap as a safety net only; the prompt itself asks for < 20 words.
_ST_HARD_CHAR_CAP = 400
_LT_HARD_CHAR_CAP = 1000

# Past-outcomes block budgets.  Kept tight on purpose so the prompt
# does not blow up after many samples.
_PAST_OUTCOMES_DEFAULT_SIZE = 6
_PAST_EXCERPT_MAX_LINES = 6
_PAST_EXCERPT_MAX_CHARS = 240
_PAST_PATTERN_MAX_CHARS = 120
_PAST_HINT_DISPLAY_MAX_CHARS = 140
_TRAJECTORY_MAX_GROUPS = 6


@dataclass(frozen=True)
class HintOutcome:
    """One ``(my hint → coder's child)`` self-feedback record.

    Stored in ``ReevoReflector._past_outcomes``.  Frozen so callers
    can't mutate it after the fact (we'd lose attribution invariants).
    """

    hint_text: str                 # the ST/LT hint A3 produced
    operator: str                  # 'e1' / 'e2' / 'm1' / 'm2' / 'm3' / 'i1'
    code_excerpt: str              # 4-6 trimmed lines, NOT the full body
    pattern_summary: str           # rule-based pattern label
    structure_tag: str             # coarse strategy family
    bf_equivalent: bool            # whether it behaves like Best Fit
    diagnostics: tuple[str, ...]   # deterministic behavior-risk labels
    child_score: Optional[float]   # None ⇔ runtime crash / no score
    parent_score: Optional[float]
    delta: Optional[float]         # child - parent (None if either side missing)


class ReevoReflector:
    """Pairwise short-term + accumulated long-term verbal reflector.

    Args:
        llm_class: a subclass of ``sampler.LLM`` used for reflection
            calls. The same provider as the generator is fine but a
            lower-temperature class is cleaner.
        problem_meta: dict with the keys ReEvo's prompt templates
            interpolate — ``problem_desc``, ``func_desc``, ``func_name``.
        prompt_dir: directory containing
            ``system_reflector.txt``,
            ``user_reflector_st.txt``,
            ``user_reflector_lt.txt`` — copied verbatim from the ReEvo
            repo. Defaults to ``<repo_root>/prompts_reevo``.
        lt_buffer_size: how many ST reflections to keep around for the
            next LT update. Older entries fall off automatically.
    """

    DEFAULT_LT_BUFFER_SIZE = 20
    DEFAULT_PAST_OUTCOMES_SIZE = _PAST_OUTCOMES_DEFAULT_SIZE

    def __init__(
            self,
            llm_class: Type[sampler.LLM],
            problem_meta: dict,
            prompt_dir: str | Path | None = None,
            lt_buffer_size: int = DEFAULT_LT_BUFFER_SIZE,
            past_outcomes_size: int = DEFAULT_PAST_OUTCOMES_SIZE,
    ):
        for k in ('problem_desc', 'func_desc', 'func_name'):
            if not problem_meta.get(k):
                raise ValueError(
                    f'problem_meta missing required key: {k!r}')
        self._meta = dict(problem_meta)

        self._llm = llm_class(samples_per_prompt=1)

        if prompt_dir is None:
            prompt_dir = Path(__file__).resolve().parent.parent / 'prompts_reevo'
        self._prompt_dir = Path(prompt_dir)
        self._sys_prompt = self._read('system_reflector.txt')
        self._st_template = self._read('user_reflector_st.txt')
        self._lt_template = self._read('user_reflector_lt.txt')

        self._long_term_str: str = ''
        self._st_buffer: deque[str] = deque(maxlen=lt_buffer_size)
        # Self-feedback FIFO: hint -> resulting child code/score.
        # Owned exclusively by A3.  Never receives runtime errors or
        # format-fail events (those belong to L2 ErrorMemory).
        self._past_outcomes: deque[HintOutcome] = deque(
            maxlen=max(1, int(past_outcomes_size)))
        # Bookkeeping (visible to log/UI but not part of the public API)
        self.st_call_count: int = 0
        self.lt_call_count: int = 0
        self.st_failure_count: int = 0
        self.lt_failure_count: int = 0
        self.observe_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def short_term_reflect(self, better_code: str, worse_code: str) -> str:
        """One pairwise (worse, better) comparison → < 20-word hint.

        Both ``better_code`` and ``worse_code`` should be the rendered
        function bodies (the same strings later shown in the
        crossover prompt). Returns ``''`` on LLM failure.

        The prompt sent to the LLM is:

            <system_reflector.txt>

            <past_outcomes block>          ← self-feedback (may be empty)

            <user_reflector_st.txt formatted with the parent pair>
        """
        if not better_code.strip() or not worse_code.strip():
            return ''
        user = self._st_template.format(
            func_name=self._meta['func_name'],
            func_desc=self._meta['func_desc'],
            problem_desc=self._meta['problem_desc'],
            worse_code=worse_code.strip(),
            better_code=better_code.strip(),
        )
        prompt = self._compose_prompt(user)

        self.st_call_count += 1
        try:
            raw = self._llm._call_api(prompt)
        except Exception as e:
            self.st_failure_count += 1
            logging.warning('ReevoReflector ST call failed: %s', e)
            return ''

        text = (raw or '').strip()
        if not text:
            self.st_failure_count += 1
            return ''
        if len(text) > _ST_HARD_CHAR_CAP:
            text = text[:_ST_HARD_CHAR_CAP - 1] + '…'

        self._st_buffer.append(text)
        return text

    def update_long_term(self) -> None:
        """Drain the ST buffer into a refreshed long-term reflection.

        No-op when the buffer is empty (e.g. a periodic poll fired but
        no new ST happened in the window).
        """
        if not self._st_buffer:
            return
        new_refs = '\n'.join(f'- {r}' for r in self._st_buffer)
        prior = self._long_term_str.strip() or '(no prior reflection yet)'

        user = self._lt_template.format(
            problem_desc=self._meta['problem_desc'],
            prior_reflection=prior,
            new_reflection=new_refs,
        )
        prompt = self._compose_prompt(user)

        self.lt_call_count += 1
        try:
            raw = self._llm._call_api(prompt)
        except Exception as e:
            self.lt_failure_count += 1
            logging.warning('ReevoReflector LT call failed: %s', e)
            return

        text = (raw or '').strip()
        if not text:
            self.lt_failure_count += 1
            return
        if len(text) > _LT_HARD_CHAR_CAP:
            text = text[:_LT_HARD_CHAR_CAP - 1] + '…'

        self._long_term_str = text
        # ST entries that fed this LT have been "consumed". Drop them so
        # the next LT update sees only fresh observations.
        self._st_buffer.clear()

    def get_long_term(self) -> str:
        """The current long-term hint string (may be empty before first LT)."""
        return self._long_term_str

    # ------------------------------------------------------------------
    # Self-feedback memory  (PastOutcomes)
    # ------------------------------------------------------------------
    def observe_outcome(
            self,
            hint: str,
            operator: str,
            code: str,
            child_score: Optional[float],
            parent_score: Optional[float] = None,
    ) -> None:
        """Record ``(hint → coder's child)`` for the next reflector call.

        Called by the Sampler **after evaluation** of the child this
        hint produced.  Empty hints are skipped (nothing to attribute).

        Args:
            hint: the ST or LT reflection text that fed the child's
                generation prompt.  ``''`` ⇒ no-op.
            operator: which EoH operator the child was generated under
                (``i1`` / ``e1`` / ``e2`` / ``m1`` / ``m2`` / ``m3``).
            code: the child function body (can be the full body — we
                trim it to a 4-6 line excerpt internally).
            child_score: scalar score the child obtained.  ``None`` is
                rendered as ``FAILED (runtime crash)`` in the next
                reflector prompt.
            parent_score: score of the better parent (used to compute
                the delta the reflector sees).  ``None`` is fine —
                we'll just omit the Δ.
        """
        if not (hint or '').strip():
            return
        excerpt = self._trim_to_excerpt(code or '')
        pattern = self._extract_pattern(code or '')
        structure = structure_analysis.analyze(code or '')
        if (child_score is not None and parent_score is not None):
            delta: Optional[float] = float(child_score) - float(parent_score)
        else:
            delta = None
        self._past_outcomes.append(HintOutcome(
            hint_text=hint.strip(),
            operator=str(operator or '?'),
            code_excerpt=excerpt,
            pattern_summary=pattern,
            structure_tag=structure.structure_tag,
            bf_equivalent=structure.bf_equivalent,
            diagnostics=structure.diagnostics,
            child_score=(None if child_score is None else float(child_score)),
            parent_score=(None if parent_score is None else float(parent_score)),
            delta=delta,
        ))
        self.observe_count += 1

    @property
    def past_outcomes_size(self) -> int:
        return len(self._past_outcomes)

    # ------------------------------------------------------------------
    # Inspection helpers (used by tests + logging)
    # ------------------------------------------------------------------
    @property
    def st_buffer_size(self) -> int:
        return len(self._st_buffer)

    @property
    def stats(self) -> dict:
        return {
            'st_calls': self.st_call_count,
            'st_failures': self.st_failure_count,
            'lt_calls': self.lt_call_count,
            'lt_failures': self.lt_failure_count,
            'st_buffer': len(self._st_buffer),
            'has_long_term': bool(self._long_term_str),
            'past_outcomes': len(self._past_outcomes),
            'observes': self.observe_count,
        }

    def render_past_outcomes(self) -> str:
        """Public rendering hook (mainly for tests / debug logging)."""
        return self._render_past_outcomes()

    def render_trajectory_summary(self) -> str:
        """Public trajectory summary hook (mainly for tests / debug logging)."""
        return self._render_trajectory_summary()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _read(self, fname: str) -> str:
        path = self._prompt_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f'ReevoReflector prompt file not found: {path}. '
                f'Did you copy ReEvo\'s prompts/common/*.txt into '
                f'{self._prompt_dir}?')
        return path.read_text(encoding='utf-8')

    def _compose_prompt(self, user_part: str) -> str:
        """Glue ``system + past_outcomes (if any) + user`` into one string.

        The past-outcomes block sits *between* system and user so the
        reflector reads (a) its job description, (b) what it has said
        before and how it played out, and (c) the new comparison —
        in that order.  When the buffer is empty we degrade to the
        original 2-block layout, which keeps the existing tests valid.
        """
        trajectory = self._render_trajectory_summary()
        past = self._render_past_outcomes()
        sys_part = self._sys_prompt.strip()
        blocks = [sys_part]
        if trajectory:
            blocks.append(trajectory)
        if past:
            blocks.append(past)
        if len(blocks) > 1:
            return '\n\n'.join(blocks + [user_part])
        return f'{sys_part}\n\n{user_part}'

    # ------------------- past-outcomes formatting -------------------
    def _render_trajectory_summary(self) -> str:
        """Summarize recent A3-attributed outcomes by structure family.

        This is an algorithm-level trajectory view, not a runtime-error
        view.  It intentionally uses only score/structure/pattern signals
        already owned by A3.  Runtime tracebacks remain in A2/L2.
        """
        if not self._past_outcomes:
            return ''

        groups: dict[tuple[str, bool, tuple[str, ...]], list[HintOutcome]] = {}
        for h in self._past_outcomes:
            groups.setdefault(
                (h.structure_tag, h.bf_equivalent, h.diagnostics), []).append(h)

        def _group_sort_key(item):
            _, records = item
            scored = [r.child_score for r in records if r.child_score is not None]
            best = max(scored) if scored else -float('inf')
            return (len(records), best)

        lines = ['## Recent trajectory summary by structure family']
        lines.append(
            '> This is A3-only algorithm feedback. Use it to choose a '
            'bounded search direction; do not copy code. Runtime errors '
            'belong to A2/L2, not this summary.')
        lines.append('')

        for (tag, bf_equiv, diagnostics), records in sorted(
                groups.items(), key=_group_sort_key, reverse=True
        )[:_TRAJECTORY_MAX_GROUPS]:
            scored = [r.child_score for r in records if r.child_score is not None]
            failed = len(records) - len(scored)
            avg_s = sum(scored) / len(scored) if scored else None
            best_s = max(scored) if scored else None
            deltas = [r.delta for r in records if r.delta is not None]
            avg_delta = sum(deltas) / len(deltas) if deltas else None
            diagnosis = self._diagnose_trajectory_group(
                tag, bf_equiv, diagnostics, scored, failed, avg_delta)

            line = (
                f'- {tag}; BF-equivalent={bf_equiv}: '
                f'{len(records)} sample(s)')
            if diagnostics:
                line += f', diagnostics={",".join(diagnostics)}'
            if avg_s is not None:
                line += f', avg={avg_s:.2f}, best={best_s:.2f}'
            if avg_delta is not None:
                line += f', avg_delta={avg_delta:+.2f}'
            if failed:
                line += f', failed={failed}'
            line += f'. Diagnosis: {diagnosis}'
            lines.append(line)

        lines.append('')
        lines.append(
            'Search-policy reminder: keep compact/tight-fit behaviour as '
            'the primary signal; prefer one bounded item-dependent secondary '
            'term or tie-breaker over global balancing, target Gaussian '
            'rules, or heavy Python loops.')
        return '\n'.join(lines).rstrip() + '\n'

    @staticmethod
    def _diagnose_trajectory_group(
            tag: str,
            bf_equiv: bool,
            diagnostics: tuple[str, ...],
            scored: list[float],
            failed: int,
            avg_delta: Optional[float],
    ) -> str:
        diag = set(diagnostics)
        if 'positive_residual_primary' in diag:
            return 'wrong score direction; higher score likely prefers loose bins.'
        if 'uses_random' in diag:
            return 'stochastic perturbation; avoid random tie-breakers.'
        if 'global_balance_or_simulation' in diag:
            return 'global-balance/simulation family; high regression or timeout risk.'
        if 'returns_single_index' in diag:
            return 'interface violation; priority must be a score array.'
        if 'inf_assignment_may_need_float' in diag:
            return 'code-level dtype risk; use float score arrays before -inf.'
        if 'pure_monotone_residual' in diag and bf_equiv:
            return 'pure monotone residual transform; likely BF-saturated.'
        if failed and not scored:
            return 'no valid score; archive as failure evidence only.'
        if 'loop' in tag:
            return 'loop-heavy/risky under evaluator budget.'
        if bf_equiv:
            if avg_delta is None or avg_delta <= 0:
                return 'stable compact-packing basin but likely saturated.'
            return 'compact-packing variant improved; keep as primary signal.'
        if scored and avg_delta is not None and avg_delta >= 0:
            return 'non-BF structure with non-negative reward signal; promising.'
        if tag in {'bucket_target', 'uniform_formula'}:
            return 'diverse but may drift too far from tight-fit compactness.'
        if tag in {'piecewise_rule', 'argmin_argmax'}:
            return 'structurally relevant; refine with bounded perturbations.'
        if scored and avg_delta is not None and avg_delta < 0:
            return 'regressed; keep for diagnosis but avoid repeating verbatim.'
        return 'insufficient evidence; explore only with bounded secondary terms.'

    def _render_past_outcomes(self) -> str:
        """Render the FIFO as a markdown block, newest first.

        Returns ``''`` when empty so callers can splice unconditionally.
        """
        if not self._past_outcomes:
            return ''
        lines: list[str] = ['## Your past suggestions and their measured impact']
        lines.append(
            '> Use this self-feedback to refine or replace your next hint. '
            'If a recent suggestion FAILED or made the score worse, do NOT '
            'repeat it verbatim — either constrain it with a concrete '
            'code-level rule, or propose a structurally different '
            'direction. Do NOT copy the code excerpts into your hint; '
            'describe strategies abstractly.')
        lines.append('')
        for i, h in enumerate(reversed(list(self._past_outcomes))):
            idx = i + 1  # 1 = most recent
            lines.extend(self._render_one_outcome(idx, h))
            lines.append('')
        return '\n'.join(lines).rstrip() + '\n'

    @staticmethod
    def _render_one_outcome(idx: int, h: 'HintOutcome') -> list[str]:
        hint_disp = h.hint_text
        if len(hint_disp) > _PAST_HINT_DISPLAY_MAX_CHARS:
            hint_disp = hint_disp[:_PAST_HINT_DISPLAY_MAX_CHARS - 1] + '…'
        if h.child_score is None:
            outcome_str = 'FAILED (no valid score — runtime crash or rejected child)'
        else:
            outcome_str = f'score = {h.child_score:.2f}'
            if h.delta is not None:
                arrow = ('↑ improved' if h.delta > 0
                         else ('↓ regressed' if h.delta < 0
                               else '= unchanged'))
                outcome_str += (f' (parent={h.parent_score:.2f}, '
                                f'Δ={h.delta:+.2f} {arrow})')
        block = [f'[{idx}] You said: "{hint_disp}"']
        if h.code_excerpt:
            block.append(f'    Coder ({h.operator}) wrote:')
            for code_line in h.code_excerpt.splitlines():
                block.append(f'        {code_line}')
        block.append(f'    Pattern: {h.pattern_summary}')
        block.append(
            f'    Structure: {h.structure_tag}; '
            f'BF-equivalent: {h.bf_equivalent}')
        if h.diagnostics:
            block.append(f'    Diagnostics: {", ".join(h.diagnostics)}')
        block.append(f'    Outcome: {outcome_str}')
        return block

    # ------------------- code excerpt + pattern -------------------
    @staticmethod
    def _trim_to_excerpt(code: str) -> str:
        """Pick at most ``_PAST_EXCERPT_MAX_LINES`` non-trivial lines.

        Drops blank lines, comments, ``def``/signature lines, docstrings.
        Keeps assignments / arithmetic / returns — the *strategy* lines.
        """
        if not code:
            return ''
        kept: list[str] = []
        in_docstring = False
        docstring_quote = ''
        for raw_line in code.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            # docstring tracking (handles both """ and ''')
            for q in ('"""', "'''"):
                if stripped.startswith(q):
                    if in_docstring and docstring_quote == q:
                        in_docstring = False
                        docstring_quote = ''
                    elif not in_docstring:
                        # Opening — could be one-liner like """foo"""
                        rest = stripped[3:]
                        if q in rest:
                            pass  # one-liner: do not toggle, just skip
                        else:
                            in_docstring = True
                            docstring_quote = q
                    break
            if in_docstring:
                continue
            if stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            if stripped.startswith('#'):
                continue
            if stripped.startswith('def ') or stripped.startswith('@'):
                continue
            kept.append(raw_line.rstrip())
            if len(kept) >= _PAST_EXCERPT_MAX_LINES:
                break
        excerpt = '\n'.join(kept)
        if len(excerpt) > _PAST_EXCERPT_MAX_CHARS:
            excerpt = excerpt[:_PAST_EXCERPT_MAX_CHARS - 1] + '…'
        return excerpt

    @staticmethod
    def _extract_pattern(code: str) -> str:
        """Rule-based 1-line pattern fingerprint.

        Picks ~10 binary features from the code and renders them as a
        short label so A3 can spot 'structural lock-in' at a glance.
        Deliberately rule-based (no LLM call) — fast and deterministic.
        """
        if not code:
            return '(empty)'
        tags: list[str] = []

        has_argmin = bool(re.search(r'np\.argmin\b', code))
        has_argmax = bool(re.search(r'np\.argmax\b', code))
        if has_argmin or has_argmax:
            tags.append('argmin/argmax')
            if re.search(
                    r'\[[^\]]+\]\s*=\s*-?\s*'
                    r'(?:1e\d+|np\.inf|float\([\'"]inf|np\.NINF|np\.PINF)',
                    code):
                tags.append('score[idx]=large_value')
            elif re.search(r'\[[^\]]+\]\s*\*=', code):
                tags.append('score[idx]*=')
            elif re.search(r'\[[^\]]+\]\s*\+=', code):
                tags.append('score[idx]+=')
            else:
                tags.append('score[idx]=...')

        if (re.search(r'bins\s*%\s*item\b', code)
                or re.search(r'%\s*item\b', code)):
            tags.append('modular(bins%item)')

        if (re.search(r'\*\*\s*[3-9]\b', code)
                or re.search(r'np\.power\([^,]+,\s*[3-9]', code)):
            tags.append('high-order power')

        transcend = [fn for fn in ('cos', 'sin', 'exp', 'log', 'tanh')
                     if re.search(rf'np\.{fn}\b', code)]
        if transcend:
            tags.append(f'transcendental({"/".join(transcend)})')

        if re.search(r'np\.where\b', code) or re.search(r'np\.select\b', code):
            tags.append('piecewise(where/select)')

        if (re.search(r'np\.full_like\([^,]+,\s*-?np\.inf', code)
                or re.search(r'np\.full_like\([^,]+,\s*-?\s*1e\d', code)):
            tags.append('masked-init')

        return_match = re.search(r'return\s+([^\n]+)', code)
        if return_match:
            rexpr = return_match.group(1)
            mul_count = rexpr.count('*') - 2 * rexpr.count('**')
            add_count = rexpr.count('+')
            if mul_count > add_count and mul_count > 0:
                tags.append('mul-dominant')
            elif add_count > 0 and mul_count <= 0:
                tags.append('additive')

        if not tags:
            if re.search(r'bins\s*-\s*item', code):
                tags.append('uniform(bins-item formula)')
            else:
                tags.append('uniform formula')

        label = ' + '.join(tags)
        if len(label) > _PAST_PATTERN_MAX_CHARS:
            label = label[:_PAST_PATTERN_MAX_CHARS - 1] + '…'
        return label


BIN_PACKING_PROBLEM_META = {
    'problem_desc': (
        'designing a heuristic priority function for the 1D online bin '
        'packing problem. Items arrive one by one and must be placed in '
        'the bin with the highest priority score; the goal is to '
        'minimize the total number of bins used.'),
    'func_desc': (
        'The function takes the current item size (float scalar) and a '
        '1-D numpy array of remaining bin capacities (already filtered '
        'so every entry is >= item) and must return a 1-D numpy array '
        'of priority scores with the same shape as the bins array.'),
    'func_name': 'priority',
}
