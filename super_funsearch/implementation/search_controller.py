"""A4 Search-Controller: LLM-guided search scheduling.

A4 is intentionally a controller, not a coder:

* It reads compact trajectory events: operator, A3 hint, A1 thought,
  structure tag, BF-equivalence, score, and error class.
* It outputs a constrained JSON policy: exploration-vs-mutation bias,
  parent-source preferences, and a short horizon.
* It never writes formulas or code. The executable policy is only a set of
  bounded multipliers used by Sampler and ProgramsDatabase.
"""
from __future__ import annotations

import json
import logging
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional, Type

import numpy as np

from implementation import structure_analysis


_MIN_OPERATOR_BIAS = 0.25
_MAX_OPERATOR_BIAS = 2.00
_MIN_PARENT_BIAS = 0.05
_MAX_PARENT_BIAS = 2.00
_ALLOWED_PHASES = {
    'default_balanced_search',
    'exploit_near_frontier_non_bf',
    'recover_from_broad_regression',
    'escape_bf_saturation',
    'avoid_runtime_risk',
    'probe_bounded_exploration',
}
_ALLOWED_OPERATORS = ('e1', 'e2', 'm1', 'm2', 'm3')


def _clamp_float(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(f):
        return default
    return max(lo, min(hi, f))


def _short(text: str | None, limit: int = 180) -> str:
    text = re.sub(r'\s+', ' ', str(text or '')).strip()
    if len(text) <= limit:
        return text
    return text[:limit - 3] + '...'


@dataclass(frozen=True)
class SearchEvent:
    """One evaluated child summarized for A4's trajectory memory."""

    sample_id: int
    operator: str
    operator_group: str
    thought: str
    a3_hint: str
    structure_tag: str
    bf_equivalent: bool
    diagnostics: tuple[str, ...]
    child_score: Optional[float]
    parent_score: Optional[float]
    delta: Optional[float]
    valid: bool
    error_type: Optional[str]
    best_before: float
    best_after: float

    @property
    def result_label(self) -> str:
        if not self.valid:
            return f'failed:{self.error_type or "unknown"}'
        if self.best_after > self.best_before:
            return 'new_best'
        if self.delta is None:
            return 'scored'
        if self.delta > 0:
            return 'improved_parent'
        if self.delta < 0:
            return 'regressed_parent'
        return 'unchanged_parent'


@dataclass
class SearchPolicy:
    """Constrained policy consumed by Sampler and ProgramsDatabase."""

    phase: str = 'default_balanced_search'
    operator_bias: dict[str, float] = field(default_factory=lambda: {
        'broad_explore': 1.0,
        'mutation': 1.0,
    })
    operator_name_bias: dict[str, float] = field(default_factory=dict)
    parent_bias: dict[str, Any] = field(default_factory=dict)
    horizon_samples: int = 15
    reason: str = 'default balanced policy'

    @classmethod
    def from_jsonable(cls, raw: dict[str, Any]) -> 'SearchPolicy':
        op_raw = raw.get('operator_bias') or raw.get('operator_policy') or {}
        op_name_raw = (
            raw.get('operator_bias_by_name') or raw.get('operator_name_bias') or {})
        parent_raw = raw.get('parent_bias') or {}
        if not parent_raw:
            # Backward-compatible rendering if the LLM returns prose fields.
            parent_raw = {
                'preferred_parent_source': raw.get('preferred_parent_source', ''),
                'avoid_parent_source': raw.get('avoid_parent_source', ''),
            }

        phase = _short(raw.get('phase') or 'default_balanced_search', 80)
        if phase not in _ALLOWED_PHASES:
            phase = 'default_balanced_search'

        operator_name_bias: dict[str, float] = {}
        if isinstance(op_name_raw, dict):
            for op in _ALLOWED_OPERATORS:
                if op in op_name_raw:
                    operator_name_bias[op] = _clamp_float(
                        op_name_raw.get(op), _MIN_OPERATOR_BIAS,
                        _MAX_OPERATOR_BIAS, 1.0)

        return cls(
            phase=phase,
            operator_bias={
                'broad_explore': _clamp_float(
                    op_raw.get('broad_explore'), _MIN_OPERATOR_BIAS,
                    _MAX_OPERATOR_BIAS, 1.0),
                'mutation': _clamp_float(
                    op_raw.get('mutation'), _MIN_OPERATOR_BIAS,
                    _MAX_OPERATOR_BIAS, 1.0),
            },
            operator_name_bias=operator_name_bias,
            parent_bias=dict(parent_raw),
            horizon_samples=int(_clamp_float(
                raw.get('horizon_samples'), 5, 50, 15)),
            reason=_short(raw.get('reason') or '', 240),
        )

    def apply_operator_bias(self, base_weights: dict[str, float]) -> dict[str, float]:
        """Return EoH operator weights after group/name-level A4 bias."""
        broad = _clamp_float(
            self.operator_bias.get('broad_explore'), _MIN_OPERATOR_BIAS,
            _MAX_OPERATOR_BIAS, 1.0)
        mutation = _clamp_float(
            self.operator_bias.get('mutation'), _MIN_OPERATOR_BIAS,
            _MAX_OPERATOR_BIAS, 1.0)
        adjusted: dict[str, float] = {}
        for op, weight in base_weights.items():
            group_bias = broad if op in ('e1', 'e2') else mutation
            name_bias = _clamp_float(
                self.operator_name_bias.get(op), _MIN_OPERATOR_BIAS,
                _MAX_OPERATOR_BIAS, 1.0)
            adjusted[op] = max(1e-9, float(weight) * group_bias * name_bias)
        return adjusted

    def parent_multiplier(
            self,
            *,
            structure_tag: str,
            bf_equivalent: bool,
            score: float,
            best_score: float,
    ) -> float:
        """Extra parent-sampling multiplier requested by A4.

        This is deliberately soft: A4 can tilt probabilities, not hard-filter
        clusters. Unknown/free-form LLM fields are ignored unless they match
        the small whitelist below.
        """
        if not np.isfinite(score) or not np.isfinite(best_score):
            return 1.0

        bias = self.parent_bias or {}
        multiplier = 1.0
        scale = max(10.0, abs(best_score))
        close_gap = max(10.0, 0.05 * scale)
        catastrophic_gap = max(100.0, 0.25 * scale)
        is_near = score >= best_score - close_gap
        is_catastrophic = score < best_score - catastrophic_gap

        prefer_tags = set(bias.get('prefer_structure_tags') or [])
        avoid_tags = set(bias.get('avoid_structure_tags') or [])
        prefer_text = str(bias.get('preferred_parent_source', '')).lower()
        avoid_text = str(bias.get('avoid_parent_source', '')).lower()

        if prefer_tags and structure_tag in prefer_tags:
            multiplier *= _clamp_float(
                bias.get('prefer_structure_multiplier'), _MIN_PARENT_BIAS,
                _MAX_PARENT_BIAS, 1.25)
        if avoid_tags and structure_tag in avoid_tags:
            multiplier *= _clamp_float(
                bias.get('avoid_structure_multiplier'), _MIN_PARENT_BIAS,
                _MAX_PARENT_BIAS, 0.5)

        if ('near-frontier' in prefer_text or
                bool(bias.get('prefer_near_frontier_non_bf'))):
            if is_near and not bf_equivalent:
                multiplier *= _clamp_float(
                    bias.get('prefer_near_frontier_non_bf_multiplier'),
                    _MIN_PARENT_BIAS, _MAX_PARENT_BIAS, 1.35)

        if ('bf' in avoid_text and 'saturat' in avoid_text) or bool(
                bias.get('avoid_bf_saturated')):
            if is_near and bf_equivalent:
                multiplier *= _clamp_float(
                    bias.get('avoid_bf_saturated_multiplier'),
                    _MIN_PARENT_BIAS, _MAX_PARENT_BIAS, 0.75)

        if ('catastrophic' in avoid_text or
                bool(bias.get('avoid_catastrophic_non_bf'))):
            if is_catastrophic and not bf_equivalent:
                multiplier *= _clamp_float(
                    bias.get('avoid_catastrophic_non_bf_multiplier'),
                    _MIN_PARENT_BIAS, _MAX_PARENT_BIAS, 0.2)

        if bool(bias.get('avoid_direction_inversion')) and is_catastrophic:
            multiplier *= _clamp_float(
                bias.get('avoid_direction_inversion_multiplier'),
                _MIN_PARENT_BIAS, _MAX_PARENT_BIAS, 0.3)

        if ('loop' in avoid_text or structure_tag in avoid_tags):
            if 'loop' in structure_tag:
                multiplier *= _clamp_float(
                    bias.get('avoid_loop_multiplier'), _MIN_PARENT_BIAS,
                    _MAX_PARENT_BIAS, 0.3)

        return _clamp_float(multiplier, _MIN_PARENT_BIAS, _MAX_PARENT_BIAS, 1.0)


class SearchController:
    """LLM-backed A4 controller with bounded executable outputs."""

    def __init__(
            self,
            llm_class: Type,
            *,
            event_capacity: int = 30,
            min_events_before_policy: int = 6,
            default_horizon: int = 15,
    ):
        self._llm = llm_class(samples_per_prompt=1)
        self._events: deque[SearchEvent] = deque(maxlen=event_capacity)
        self._min_events = max(1, int(min_events_before_policy))
        self._default_horizon = max(5, int(default_horizon))
        self._current_policy = SearchPolicy(horizon_samples=self._default_horizon)
        self._samples_until_refresh = self._default_horizon
        self._policy_history: deque[tuple[SearchPolicy, str]] = deque(maxlen=8)

    @property
    def current_policy(self) -> SearchPolicy:
        return self._current_policy

    @property
    def num_events(self) -> int:
        return len(self._events)

    def observe(
            self,
            *,
            sample_id: int,
            operator: str,
            thought: str,
            a3_hint: str,
            code: str,
            child_score: Optional[float],
            parent_score: Optional[float],
            valid: bool,
            error_trace: str,
            best_before: float,
            best_after: float,
    ) -> None:
        structure = structure_analysis.analyze(code or '')
        delta = None
        if child_score is not None and parent_score is not None:
            delta = float(child_score) - float(parent_score)
        event = SearchEvent(
            sample_id=int(sample_id),
            operator=str(operator or '?'),
            operator_group=self._operator_group(operator),
            thought=_short(thought, 220),
            a3_hint=_short(a3_hint, 220),
            structure_tag=structure.structure_tag,
            bf_equivalent=structure.bf_equivalent,
            diagnostics=structure.diagnostics,
            child_score=(None if child_score is None else float(child_score)),
            parent_score=(None if parent_score is None else float(parent_score)),
            delta=delta,
            valid=bool(valid),
            error_type=self._classify_error(error_trace),
            best_before=float(best_before),
            best_after=float(best_after),
        )
        self._events.append(event)
        self._samples_until_refresh -= 1

    def maybe_refresh_policy(self, *, a3_summary: str = '') -> SearchPolicy:
        if len(self._events) < self._min_events:
            return self._current_policy
        if self._samples_until_refresh > 0:
            return self._current_policy

        prompt = self._build_prompt(a3_summary=a3_summary)
        try:
            raw = self._llm._call_api(prompt)
            policy = self._parse_policy(raw)
            self._current_policy = policy
            self._samples_until_refresh = policy.horizon_samples
            self._policy_history.append((policy, raw[:1000]))
            logging.info(
                '[a4] policy phase=%s broad=%.2f mutation=%.2f op_bias=%s '
                'horizon=%d reason=%s',
                policy.phase,
                policy.operator_bias.get('broad_explore', 1.0),
                policy.operator_bias.get('mutation', 1.0),
                {k: round(v, 2) for k, v in policy.operator_name_bias.items()},
                policy.horizon_samples,
                policy.reason[:160],
            )
        except Exception as e:
            self._samples_until_refresh = self._default_horizon
            logging.warning('[a4] policy refresh failed; keeping current policy: %s', e)
        return self._current_policy

    def _build_prompt(self, *, a3_summary: str) -> str:
        return (
            "You are A4 Search-Controller for an evolutionary code-search system.\n"
            "Your job is scheduling only. Do NOT write code, formulas, or heuristic snippets.\n"
            "You may look at concrete recent events, but your output must be JSON only.\n"
            "Choose group-level operator bias and parent-source bias for the next short horizon.\n\n"
            "Phase MUST be exactly one of:\n"
            "- default_balanced_search: no clear directional evidence yet.\n"
            "- exploit_near_frontier_non_bf: near-frontier non-BF variants appeared; refine them.\n"
            "- recover_from_broad_regression: broad exploration caused large regressions; reduce it.\n"
            "- escape_bf_saturation: many BF-equivalent samples and no improvement; briefly explore.\n"
            "- avoid_runtime_risk: runtime/timeout/loop-heavy risk is high; avoid risky parents.\n"
            "- probe_bounded_exploration: cautiously test bounded alternatives around tight-fit.\n\n"
            "Operator groups:\n"
            "- broad_explore: e1/e2, used to explore structurally different ideas.\n"
            "- mutation: m1/m2/m3, used to refine existing parents.\n\n"
            "Optional short-horizon operator-name bias may tilt within a group:\n"
            "- Lower m3 when simplify/generalize produces BF clones or timeout.\n"
            "- Raise m1/m2 when near-frontier non-BF parents need local refinement.\n"
            "- Do not permanently declare any operator useless; these are temporary multipliers.\n\n"
            "Allowed JSON schema:\n"
            "{\n"
            '  "phase": "one_of_the_allowed_phase_names_above",\n'
            '  "operator_bias": {"broad_explore": 0.25-2.0, "mutation": 0.25-2.0},\n'
            '  "operator_bias_by_name": {"e1": 0.25-2.0, "e2": 0.25-2.0, "m1": 0.25-2.0, "m2": 0.25-2.0, "m3": 0.25-2.0},\n'
            '  "parent_bias": {\n'
            '    "prefer_near_frontier_non_bf": true/false,\n'
            '    "prefer_structure_tags": ["piecewise_rule", "residual_formula"],\n'
            '    "avoid_structure_tags": ["loop_heavy"],\n'
            '    "avoid_catastrophic_non_bf": true/false,\n'
            '    "avoid_bf_saturated": true/false,\n'
            '    "avoid_direction_inversion": true/false\n'
            "  },\n"
            '  "horizon_samples": 5-50,\n'
            '  "reason": "one sentence explaining scheduling only"\n'
            "}\n\n"
            "Important risk pattern for bin packing:\n"
            "- Best Fit prefers SMALL post-placement residual r = bins - item.\n"
            "- If candidates use positive scores such as r, r**2, or remaining**p without a negative sign, they prefer LARGE residuals and often collapse toward Worst/Far Fit.\n"
            "- Treat those events as direction-inversion failures. Do not tell A1 a formula; schedule away from parent sources that produced this risk.\n\n"
            "Decision hints:\n"
            "- If broad exploration recently caused large regressions, lower broad_explore and raise mutation.\n"
            "- If many samples are BF-equivalent with no best-score improvement, briefly raise broad_explore.\n"
            "- If near-frontier non-BF variants appear, prefer those parents and raise mutation.\n"
            "- If m3 simplification repeatedly returns BF clones or timeout, lower m3 and prefer m1/m2.\n"
            "- If catastrophic regressions look like direction inversion, set avoid_direction_inversion=true and prefer safer tight-fit-neighborhood parents.\n"
            "- If loop-heavy/timeout patterns appear, avoid loop-heavy parents; operator groups may stay unchanged.\n"
            "- Do not conclude a specific individual operator is useless.\n\n"
            "Diagnostic labels you may see:\n"
            "- positive_residual_primary: score direction likely inverted; avoid those parent sources.\n"
            "- pure_monotone_residual: BF-equivalent residual transform; useful baseline but often saturated.\n"
            "- uses_random: stochastic perturbation; reduce parent/operator sources that caused it.\n"
            "- global_balance_or_simulation: global balancing or simulation-like code; often regresses or times out.\n"
            "- returns_single_index / inf_assignment_may_need_float: implementation-risk evidence for scheduling only.\n\n"
            f"Current A3 trajectory summary:\n{a3_summary.strip() or '(none)'}\n\n"
            f"Recent structure-family stats:\n{self._render_family_stats()}\n\n"
            f"Recent trajectory events:\n{self._render_events()}\n\n"
            "Return JSON only."
        )

    def _render_family_stats(self) -> str:
        rows: dict[tuple[str, bool, tuple[str, ...]], dict[str, Any]] = {}
        for ev in self._events:
            key = (ev.structure_tag, ev.bf_equivalent, ev.diagnostics)
            row = rows.setdefault(key, {
                'n': 0,
                'valid': 0,
                'timeouts': 0,
                'best': None,
                'delta_sum': 0.0,
                'delta_n': 0,
            })
            row['n'] += 1
            if ev.valid:
                row['valid'] += 1
            if ev.error_type == 'timeout':
                row['timeouts'] += 1
            if ev.child_score is not None:
                row['best'] = (
                    ev.child_score if row['best'] is None
                    else max(row['best'], ev.child_score))
            if ev.delta is not None:
                row['delta_sum'] += ev.delta
                row['delta_n'] += 1

        lines: list[str] = []
        for (tag, bf, diagnostics), row in sorted(
                rows.items(), key=lambda kv: (-kv[1]['n'], kv[0][0]))[:8]:
            avg_delta = (
                'n/a' if row['delta_n'] == 0
                else f"{row['delta_sum'] / row['delta_n']:+.2f}")
            best = 'None' if row['best'] is None else f"{row['best']:.2f}"
            diag = ','.join(diagnostics) if diagnostics else 'none'
            lines.append(
                f'- {tag}, BF={bf}, diagnostics={diag}: '
                f'n={row["n"]}, valid={row["valid"]}, '
                f'timeout={row["timeouts"]}, best={best}, avg_delta={avg_delta}')
        return '\n'.join(lines) if lines else '(none)'

    def _render_events(self) -> str:
        lines: list[str] = []
        for ev in list(self._events)[-12:]:
            score = 'None' if ev.child_score is None else f'{ev.child_score:.2f}'
            parent = 'None' if ev.parent_score is None else f'{ev.parent_score:.2f}'
            delta = 'None' if ev.delta is None else f'{ev.delta:+.2f}'
            lines.append(
                f'#{ev.sample_id} op={ev.operator} group={ev.operator_group} '
                f'structure={ev.structure_tag} BF={ev.bf_equivalent} '
                f'diagnostics={",".join(ev.diagnostics) or "none"} '
                f'score={score} parent={parent} delta={delta} '
                f'result={ev.result_label}')
            if ev.a3_hint:
                lines.append(f'  A3 hint: {ev.a3_hint}')
            if ev.thought:
                lines.append(f'  A1 thought: {ev.thought}')
        return '\n'.join(lines) if lines else '(none)'

    @staticmethod
    def _parse_policy(raw: str) -> SearchPolicy:
        text = str(raw or '').strip()
        match = re.search(r'\{.*\}', text, flags=re.DOTALL)
        if not match:
            raise ValueError('A4 response did not contain a JSON object')
        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError('A4 JSON root must be an object')
        return SearchPolicy.from_jsonable(data)

    @staticmethod
    def _operator_group(operator: str) -> str:
        if operator in ('e1', 'e2'):
            return 'broad_explore'
        if operator in ('m1', 'm2', 'm3'):
            return 'mutation'
        if operator == 'i1':
            return 'init'
        return 'other'

    @staticmethod
    def _classify_error(error_trace: str) -> Optional[str]:
        err = (error_trace or '').lower()
        if not err:
            return None
        if err.startswith('invalidsampleerror'.lower()):
            return 'format'
        if 'timeout' in err or 'timed out' in err:
            return 'timeout'
        return 'runtime'
