"""Unit tests for the ReEvo-style verbal reflector + EoH prompt injection.

These tests run with NO real LLM calls — we substitute a tiny fake
``LLM`` class that returns scripted strings. They verify:

  * Pairwise short-term reflection feeds the buffer and returns the LLM
    output verbatim (modulo the safety hard-cap).
  * Long-term update consumes the ST buffer, calls the LLM with the
    expected interpolated prompt, and exposes the returned string via
    ``get_long_term``.
  * ``update_long_term`` is a no-op when the buffer is empty.
  * LLM exceptions never propagate — failures are counted in stats.
  * ``eoh_operators.build_prompt`` splices the reflection block into all
    operators (i1/e1/e2/m1/m2/m3) under the documented header, and is
    a true no-op when reflection text is empty.

Run from the super_funsearch directory:

    python -m unittest test_reevo_reflector -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from implementation import eoh_operators
from implementation import reevo_reflector as reevo_reflector_lib


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
class _ScriptedLLM:
    """Deterministic stand-in for sampler.LLM.

    Returns the next response from ``responses`` on each ``_call_api``
    call, recording the prompt for assertions. If the sentinel value
    ``RAISE`` is hit, ``_call_api`` raises so we can test the failure
    path.
    """
    RAISE = object()
    instances: list['_ScriptedLLM'] = []

    @classmethod
    def with_responses(cls, *responses):
        """Return a *factory* that hands out a fresh _ScriptedLLM every
        time it's called like a class — matches what ``ReevoReflector``
        does internally with ``llm_class(samples_per_prompt=1)``."""
        def factory(samples_per_prompt: int = 1):
            inst = cls(list(responses))
            cls.instances.append(inst)
            return inst
        return factory

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[str] = []

    def _call_api(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self._responses:
            return ''
        nxt = self._responses.pop(0)
        if nxt is self.RAISE:
            raise RuntimeError('scripted LLM failure')
        return nxt


def _make_reflector(*responses):
    """Helper: clean per-test reflector with scripted LLM."""
    _ScriptedLLM.instances.clear()
    factory = _ScriptedLLM.with_responses(*responses)
    refl = reevo_reflector_lib.ReevoReflector(
        llm_class=factory,
        problem_meta=reevo_reflector_lib.BIN_PACKING_PROBLEM_META,
    )
    return refl, _ScriptedLLM.instances[-1]


# ---------------------------------------------------------------------------
# 1. Short-term reflection
# ---------------------------------------------------------------------------
class ShortTermReflectionTests(unittest.TestCase):

    def test_st_returns_llm_output_and_grows_buffer(self):
        refl, llm = _make_reflector('Use waste-aware ratios, not raw remaining.')
        out = refl.short_term_reflect(
            better_code='def priority(item, bins): return -np.abs(bins - item)',
            worse_code='def priority(item, bins): return -bins',
        )
        self.assertEqual(out, 'Use waste-aware ratios, not raw remaining.')
        self.assertEqual(refl.st_buffer_size, 1)
        self.assertEqual(refl.stats['st_calls'], 1)
        self.assertEqual(refl.stats['st_failures'], 0)
        # The prompt sent must mention BOTH parents and the problem name.
        sent = llm.calls[0]
        self.assertIn('priority', sent)
        self.assertIn('-np.abs(bins - item)', sent)
        self.assertIn('-bins', sent)
        self.assertIn('Worse code', sent)
        self.assertIn('Better code', sent)

    def test_st_with_empty_code_short_circuits(self):
        refl, llm = _make_reflector('Should never be returned.')
        self.assertEqual(refl.short_term_reflect('', 'def f(): pass'), '')
        self.assertEqual(refl.short_term_reflect('def f(): pass', '   '), '')
        self.assertEqual(llm.calls, [])
        self.assertEqual(refl.stats['st_calls'], 0)

    def test_st_handles_llm_exception(self):
        refl, llm = _make_reflector(_ScriptedLLM.RAISE)
        out = refl.short_term_reflect('def a(): pass', 'def b(): pass')
        self.assertEqual(out, '')
        self.assertEqual(refl.stats['st_calls'], 1)
        self.assertEqual(refl.stats['st_failures'], 1)
        self.assertEqual(refl.st_buffer_size, 0)

    def test_st_hard_caps_oversize_response(self):
        big = 'x' * 10_000
        refl, _ = _make_reflector(big)
        out = refl.short_term_reflect('def a(): pass', 'def b(): pass')
        self.assertLess(len(out), 500)
        self.assertTrue(out.endswith('…'))


# ---------------------------------------------------------------------------
# 2. Long-term reflection
# ---------------------------------------------------------------------------
class LongTermReflectionTests(unittest.TestCase):

    def test_lt_no_op_when_buffer_empty(self):
        refl, llm = _make_reflector('Should not be called.')
        refl.update_long_term()
        self.assertEqual(llm.calls, [])
        self.assertEqual(refl.stats['lt_calls'], 0)
        self.assertEqual(refl.get_long_term(), '')

    def test_lt_drains_buffer_and_stores_summary(self):
        refl, llm = _make_reflector(
            'Hint A',                             # ST 1
            'Hint B',                             # ST 2
            'Combined: prefer waste minimisation',# LT
        )
        refl.short_term_reflect('def a(): pass', 'def b(): pass')
        refl.short_term_reflect('def c(): pass', 'def d(): pass')
        self.assertEqual(refl.st_buffer_size, 2)

        refl.update_long_term()
        # Buffer drained, LT populated.
        self.assertEqual(refl.st_buffer_size, 0)
        self.assertEqual(
            refl.get_long_term(),
            'Combined: prefer waste minimisation')
        # LT prompt must include the prior reflections in bullet form.
        lt_prompt = llm.calls[-1]
        self.assertIn('- Hint A', lt_prompt)
        self.assertIn('- Hint B', lt_prompt)
        self.assertIn('(no prior reflection yet)', lt_prompt)
        self.assertEqual(refl.stats['lt_calls'], 1)
        self.assertEqual(refl.stats['lt_failures'], 0)

    def test_lt_uses_prior_summary_on_subsequent_call(self):
        refl, llm = _make_reflector(
            'Hint X',                             # ST
            'First LT summary',                   # LT 1
            'Hint Y',                             # ST
            'Refined LT summary',                 # LT 2
        )
        refl.short_term_reflect('def a(): pass', 'def b(): pass')
        refl.update_long_term()
        refl.short_term_reflect('def c(): pass', 'def d(): pass')
        refl.update_long_term()

        self.assertEqual(refl.get_long_term(), 'Refined LT summary')
        # Second LT prompt must echo the first LT as `prior_reflection`.
        last_lt_prompt = llm.calls[-1]
        self.assertIn('First LT summary', last_lt_prompt)
        self.assertIn('- Hint Y', last_lt_prompt)
        self.assertNotIn('(no prior reflection yet)', last_lt_prompt)

    def test_lt_handles_llm_exception(self):
        refl, _ = _make_reflector('Hint', _ScriptedLLM.RAISE)
        refl.short_term_reflect('def a(): pass', 'def b(): pass')
        refl.update_long_term()
        self.assertEqual(refl.stats['lt_calls'], 1)
        self.assertEqual(refl.stats['lt_failures'], 1)
        self.assertEqual(refl.get_long_term(), '')


# ---------------------------------------------------------------------------
# 3. Prompt injection — eoh_operators.build_prompt(reflection=...)
# ---------------------------------------------------------------------------
_PARENT_A = {
    'algorithm': 'Best-fit-style',
    'code': 'def priority(item, bins):\n    return -np.abs(bins - item)',
}
_PARENT_B = {
    'algorithm': 'Worst-fit-style',
    'code': 'def priority(item, bins):\n    return bins',
}
_REFL = 'Prefer minimising waste; penalise leftover space sharply.'


class ReflectionInjectionTests(unittest.TestCase):

    def test_empty_reflection_does_not_change_prompt(self):
        with_refl = eoh_operators.build_prompt(
            'i1', [], eoh_operators.BIN_PACKING_TASK, reflection='')
        without_refl = eoh_operators.build_prompt(
            'i1', [], eoh_operators.BIN_PACKING_TASK)
        self.assertEqual(with_refl, without_refl)
        self.assertNotIn('[Reflection from prior comparisons]', with_refl)

    def test_reflection_is_spliced_for_head_operators(self):
        for op in ('i1', 'e1', 'e2', 'm1', 'm2'):
            parents = {
                'i1': [],
                'e1': [_PARENT_A, _PARENT_B],
                'e2': [_PARENT_A, _PARENT_B],
                'm1': [_PARENT_A],
                'm2': [_PARENT_A],
            }[op]
            prompt = eoh_operators.build_prompt(
                op, parents, eoh_operators.BIN_PACKING_TASK,
                reflection=_REFL)
            self.assertIn('[Reflection from prior comparisons]', prompt,
                          msg=f'operator={op}')
            self.assertIn(_REFL, prompt, msg=f'operator={op}')

    def test_reflection_is_spliced_for_m3_path(self):
        prompt = eoh_operators.build_prompt(
            'm3', [_PARENT_A], eoh_operators.BIN_PACKING_TASK,
            reflection=_REFL)
        self.assertIn('[Reflection from prior comparisons]', prompt)
        self.assertIn(_REFL, prompt)

    def test_reflection_and_avoidance_coexist(self):
        prompt = eoh_operators.build_prompt(
            'e2', [_PARENT_A, _PARENT_B], eoh_operators.BIN_PACKING_TASK,
            error_avoidance='## Recent failures to avoid:\n1. (e2) ValueError: axis 1',
            reflection=_REFL)
        self.assertIn('Recent failures to avoid', prompt)
        self.assertIn('[Reflection from prior comparisons]', prompt)
        self.assertIn(_REFL, prompt)
        # Avoidance should appear before reflection (so the LLM sees
        # don't-do-this before strategic hints).
        self.assertLess(
            prompt.index('Recent failures to avoid'),
            prompt.index('[Reflection from prior comparisons]'))


# ---------------------------------------------------------------------------
# 4. PastOutcomes self-feedback memory (A3 closed-loop)
# ---------------------------------------------------------------------------
class PastOutcomesTests(unittest.TestCase):
    """The A3 self-feedback FIFO is the v3.5 'verbal gradient with reward'
    addition.  These tests pin down four invariants:
      * empty hint never gets recorded (nothing to attribute);
      * trim_to_excerpt drops blank/comment/signature lines and obeys
        the line/char budget;
      * extract_pattern tags asymmetric scoring, modular periodicity,
        high-order powers, transcendentals etc. independently;
      * render_past_outcomes injects the block into both ST and LT
        prompts when the buffer is non-empty, and stays a no-op when
        empty (so existing prompt assertions remain valid).
    """

    def test_observe_outcome_skips_empty_hint(self):
        refl, llm = _make_reflector('unused')
        refl.observe_outcome(
            hint='   ', operator='e1',
            code='priorities = bins - item\nreturn priorities',
            child_score=-209.0, parent_score=-210.0)
        self.assertEqual(refl.past_outcomes_size, 0)
        self.assertEqual(refl.observe_count, 0)

    def test_observe_outcome_records_delta_and_pattern(self):
        refl, _ = _make_reflector('unused')
        body = (
            'priorities = bins * (bins - item)**2\n'
            'idx = np.argmin(bins - item)\n'
            'priorities[idx] *= item\n'
            'return priorities')
        refl.observe_outcome(
            hint='try multiplicative + asymmetric',
            operator='e2',
            code=body,
            child_score=-209.55,
            parent_score=-211.0,
        )
        self.assertEqual(refl.past_outcomes_size, 1)
        self.assertEqual(refl.observe_count, 1)
        block = refl.render_past_outcomes()
        self.assertIn('Your past suggestions and their measured impact', block)
        self.assertIn('try multiplicative + asymmetric', block)
        self.assertIn('-209.55', block)
        self.assertIn('-211.00', block)
        self.assertIn('Δ=+1.45', block)
        # Pattern tags should reflect asymmetric scoring AND multiplicative usage.
        self.assertIn('argmin/argmax', block)
        self.assertIn('score[idx]*=', block)

    def test_observe_outcome_renders_diagnostics(self):
        refl, _ = _make_reflector('unused')
        refl.observe_outcome(
            hint='avoid loose-bin direction',
            operator='e1',
            code='r = bins - item\nreturn r',
            child_score=-5000.0,
            parent_score=-2067.0,
        )
        block = refl.render_past_outcomes()
        self.assertIn('Diagnostics:', block)
        self.assertIn('positive_residual_primary', block)

    def test_observe_outcome_runtime_crash_renders_failed(self):
        refl, _ = _make_reflector('unused')
        refl.observe_outcome(
            hint='use cosine similarity',
            operator='m1',
            code='priorities = np.cos(bins / item)\nreturn priorities',
            child_score=None, parent_score=-210.0)
        block = refl.render_past_outcomes()
        self.assertIn('FAILED', block)
        # No spurious score number when child failed.
        self.assertNotIn('score = -', block)

    def test_render_block_is_empty_when_no_outcomes(self):
        refl, _ = _make_reflector('unused')
        self.assertEqual(refl.render_past_outcomes(), '')

    def test_fifo_evicts_oldest_beyond_capacity(self):
        refl = reevo_reflector_lib.ReevoReflector(
            llm_class=_ScriptedLLM.with_responses(),
            problem_meta=reevo_reflector_lib.BIN_PACKING_PROBLEM_META,
            past_outcomes_size=2,
        )
        for i in range(4):
            refl.observe_outcome(
                hint=f'hint #{i}', operator='e1',
                code='return bins - item',
                child_score=-200 - i, parent_score=-210.0)
        self.assertEqual(refl.past_outcomes_size, 2)
        block = refl.render_past_outcomes()
        # Newest two survived: #3 (most recent) and #2.
        self.assertIn('hint #3', block)
        self.assertIn('hint #2', block)
        # Oldest two were evicted.
        self.assertNotIn('hint #0', block)
        self.assertNotIn('hint #1', block)

    def test_st_prompt_includes_past_outcomes_when_present(self):
        refl, llm = _make_reflector(
            'observed: continuous formulas in both — try asymmetric.')
        refl.observe_outcome(
            hint='earlier hint about argmin',
            operator='e1',
            code='return bins - item',
            child_score=-211.5, parent_score=-211.0)
        refl.short_term_reflect(
            better_code='def priority(item, bins): return -np.abs(bins - item)',
            worse_code='def priority(item, bins): return -bins')
        # Latest LLM call's prompt must contain the past-outcomes block.
        self.assertEqual(len(llm.calls), 1)
        sent = llm.calls[0]
        self.assertIn(
            'Your past suggestions and their measured impact', sent)
        self.assertIn('earlier hint about argmin', sent)
        # And the canonical user-template parts must still be present.
        self.assertIn('Worse code', sent)
        self.assertIn('Better code', sent)

    def test_lt_prompt_includes_past_outcomes_when_present(self):
        refl, llm = _make_reflector(
            'st-1', 'lt summary')
        refl.observe_outcome(
            hint='earlier LT-relevant hint',
            operator='m2',
            code='return bins ** 4',
            child_score=-209.6, parent_score=-210.5)
        refl.short_term_reflect('def a(): pass', 'def b(): pass')
        refl.update_long_term()
        # Two LLM calls in this test: one ST, one LT. Last must be LT.
        self.assertEqual(len(llm.calls), 2)
        lt_prompt = llm.calls[-1]
        self.assertIn(
            'Your past suggestions and their measured impact', lt_prompt)
        self.assertIn('earlier LT-relevant hint', lt_prompt)
        self.assertIn('(no prior reflection yet)', lt_prompt)

    def test_trajectory_summary_groups_recent_structures(self):
        refl, llm = _make_reflector(
            'bounded search direction')
        refl.observe_outcome(
            hint='tight fit baseline',
            operator='e1',
            code='return -(bins - item)',
            child_score=-2067.0,
            parent_score=-2067.0)
        refl.observe_outcome(
            hint='try target residual',
            operator='e2',
            code=(
                'r = bins - item\n'
                'target = 0.5 * np.max(r)\n'
                'return -np.abs(r - target)'),
            child_score=-2300.0,
            parent_score=-2067.0)

        summary = refl.render_trajectory_summary()
        self.assertIn('Recent trajectory summary by structure family', summary)
        self.assertIn('residual_formula; BF-equivalent=True', summary)
        self.assertIn('bucket_target; BF-equivalent=False', summary)
        self.assertIn('pure_monotone_residual', summary)
        self.assertIn('likely BF-saturated', summary)
        self.assertIn('bounded search direction', summary)

        refl.short_term_reflect(
            better_code='def priority(item, bins): return -(bins - item)',
            worse_code='def priority(item, bins): return bins')
        sent = llm.calls[0]
        self.assertIn('Recent trajectory summary by structure family', sent)
        self.assertIn('Search-policy reminder', sent)

    def test_pattern_tagger_recognises_modular_and_powers(self):
        Refl = reevo_reflector_lib.ReevoReflector
        self.assertIn(
            'modular(bins%item)',
            Refl._extract_pattern('return (bins % item) ** 2'))
        self.assertIn(
            'high-order power',
            Refl._extract_pattern('return bins ** 4'))
        self.assertIn(
            'transcendental',
            Refl._extract_pattern('return np.cos(bins / item)'))
        self.assertIn(
            'piecewise(where/select)',
            Refl._extract_pattern(
                'return np.where(bins == item, 1.0, bins - item)'))
        # Vanilla continuous formula gets the uniform-fallback tag.
        self.assertIn(
            'uniform',
            Refl._extract_pattern('return bins - item'))

    def test_trim_excerpt_drops_signature_blank_and_comments(self):
        Refl = reevo_reflector_lib.ReevoReflector
        body = (
            'def priority(item, bins):\n'
            '    """docstring should be skipped."""\n'
            '    \n'
            '    # this comment line should drop\n'
            '    a = bins - item\n'
            '    b = a ** 2\n'
            '    c = b + item\n'
            '    d = c * a\n'
            '    e = d + 1\n'
            '    f = e + 2\n'
            '    g = f + 3\n'
            '    return g')
        excerpt = Refl._trim_to_excerpt(body)
        self.assertNotIn('def priority', excerpt)
        self.assertNotIn('docstring', excerpt)
        self.assertNotIn('# this comment', excerpt)
        self.assertLessEqual(
            len([l for l in excerpt.splitlines() if l.strip()]), 6)
        # The first kept line must be the first real strategy line.
        self.assertIn('a = bins - item', excerpt)


# ---------------------------------------------------------------------------
# 5. Stats surface for new fields (regression guard)
# ---------------------------------------------------------------------------
class StatsSurfaceTests(unittest.TestCase):
    def test_stats_exposes_past_and_observes(self):
        refl, _ = _make_reflector('unused')
        s = refl.stats
        self.assertIn('past_outcomes', s)
        self.assertIn('observes', s)
        self.assertEqual(s['past_outcomes'], 0)
        self.assertEqual(s['observes'], 0)
        refl.observe_outcome(
            hint='h', operator='e1',
            code='return bins - item',
            child_score=-209.0, parent_score=-210.0)
        s = refl.stats
        self.assertEqual(s['past_outcomes'], 1)
        self.assertEqual(s['observes'], 1)


if __name__ == '__main__':
    unittest.main()
