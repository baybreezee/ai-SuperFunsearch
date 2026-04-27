import unittest

from implementation import search_controller


class _FakeLLM:
    def __init__(self, samples_per_prompt):
        self.samples_per_prompt = samples_per_prompt

    def _call_api(self, prompt):
        return """{
          "phase": "exploit_near_frontier_non_bf",
          "operator_bias": {"broad_explore": 0.5, "mutation": 1.5},
          "operator_bias_by_name": {"m2": 1.4, "m3": 0.5},
          "parent_bias": {
            "prefer_near_frontier_non_bf": true,
            "avoid_structure_tags": ["loop_heavy"],
            "avoid_catastrophic_non_bf": true
          },
          "horizon_samples": 12,
          "reason": "Prefer local refinement around promising non-BF parents."
        }"""


class SearchControllerTests(unittest.TestCase):

    def test_policy_applies_group_operator_bias(self):
        policy = search_controller.SearchPolicy.from_jsonable({
            'operator_bias': {'broad_explore': 0.5, 'mutation': 1.5},
        })

        weights = policy.apply_operator_bias({
            'e1': 1.0, 'e2': 1.0, 'm1': 1.0, 'm2': 1.0, 'm3': 1.0})

        self.assertEqual(weights['e1'], 0.5)
        self.assertEqual(weights['e2'], 0.5)
        self.assertEqual(weights['m1'], 1.5)
        self.assertEqual(weights['m2'], 1.5)
        self.assertEqual(weights['m3'], 1.5)

    def test_policy_applies_operator_name_bias(self):
        policy = search_controller.SearchPolicy.from_jsonable({
            'operator_bias': {'broad_explore': 1.0, 'mutation': 1.5},
            'operator_bias_by_name': {'m2': 2.0, 'm3': 0.5},
        })

        weights = policy.apply_operator_bias({
            'e1': 1.0, 'e2': 1.0, 'm1': 1.0, 'm2': 1.0, 'm3': 1.0})

        self.assertEqual(weights['m1'], 1.5)
        self.assertEqual(weights['m2'], 3.0)
        self.assertEqual(weights['m3'], 0.75)

    def test_parent_multiplier_prefers_near_frontier_non_bf(self):
        policy = search_controller.SearchPolicy.from_jsonable({
            'parent_bias': {'prefer_near_frontier_non_bf': True},
        })

        multiplier = policy.parent_multiplier(
            structure_tag='piecewise_rule',
            bf_equivalent=False,
            score=-2068.0,
            best_score=-2067.0,
        )

        self.assertGreater(multiplier, 1.0)

    def test_invalid_phase_falls_back_to_default(self):
        policy = search_controller.SearchPolicy.from_jsonable({
            'phase': 'short_name',
            'operator_bias': {'broad_explore': 0.5, 'mutation': 1.5},
        })

        self.assertEqual(policy.phase, 'default_balanced_search')

    def test_direction_inversion_bias_deprioritizes_catastrophic_parent(self):
        policy = search_controller.SearchPolicy.from_jsonable({
            'parent_bias': {'avoid_direction_inversion': True},
        })

        multiplier = policy.parent_multiplier(
            structure_tag='power_formula',
            bf_equivalent=False,
            score=-4995.0,
            best_score=-2067.0,
        )

        self.assertLess(multiplier, 1.0)

    def test_controller_refreshes_from_llm_json(self):
        controller = search_controller.SearchController(
            _FakeLLM, min_events_before_policy=1, default_horizon=1)
        for sample_id in range(1, 6):
            controller.observe(
                sample_id=sample_id,
                operator='m2',
                thought='tiered slack-aware scoring',
                a3_hint='keep compactness primary',
                code='return -(bins - item)',
                child_score=-2060.2,
                parent_score=-2067.0,
                valid=True,
                error_trace='',
                best_before=-2067.0,
                best_after=-2060.2,
            )

        policy = controller.maybe_refresh_policy(a3_summary='test summary')

        self.assertEqual(policy.phase, 'exploit_near_frontier_non_bf')
        self.assertEqual(policy.horizon_samples, 12)
        self.assertLess(policy.operator_bias['broad_explore'], 1.0)
        self.assertGreater(policy.operator_bias['mutation'], 1.0)
        self.assertLess(policy.operator_name_bias['m3'], 1.0)
        self.assertGreater(policy.operator_name_bias['m2'], 1.0)

    def test_controller_renders_diagnostics_for_a4_prompt(self):
        controller = search_controller.SearchController(
            _FakeLLM, min_events_before_policy=1, default_horizon=1)
        controller.observe(
            sample_id=1,
            operator='e2',
            thought='try residual directly',
            a3_hint='avoid repeating failed direction',
            code='r = bins - item\nreturn r',
            child_score=-5000.0,
            parent_score=-2067.0,
            valid=True,
            error_trace='',
            best_before=-2067.0,
            best_after=-2067.0,
        )

        events = controller._render_events()
        stats = controller._render_family_stats()

        self.assertIn('positive_residual_primary', events)
        self.assertIn('positive_residual_primary', stats)


if __name__ == '__main__':
    unittest.main()
