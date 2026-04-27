"""Tests for structure-aware population diversity helpers."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from implementation import code_manipulation
from implementation import programs_database
from implementation import structure_analysis


def _function(body: str):
    return code_manipulation.text_to_function(
        'def priority(item, bins):\n' + body.strip('\n') + '\n'
    )


class StructureAnalysisTests(unittest.TestCase):

    def test_residual_formula_is_best_fit_equivalent(self):
        info = structure_analysis.analyze(
            '    return -(bins - item)\n'
        )
        self.assertEqual(info.structure_tag, 'residual_formula')
        self.assertTrue(info.bf_equivalent)
        self.assertIn('pure_monotone_residual', info.diagnostics)

    def test_positive_residual_primary_is_diagnosed(self):
        info = structure_analysis.analyze(
            '    r = bins - item\n'
            '    return r\n'
        )
        self.assertIn('positive_residual_primary', info.diagnostics)

    def test_random_and_global_balance_are_diagnosed(self):
        info = structure_analysis.analyze(
            '    residual = bins - item\n'
            '    noise = np.random.random(bins.shape) * 0.01\n'
            '    return -np.abs(residual - residual.mean()) + noise\n'
        )
        self.assertIn('uses_random', info.diagnostics)
        self.assertIn('global_balance_or_simulation', info.diagnostics)

    def test_bucket_target_is_not_static_best_fit_clone(self):
        info = structure_analysis.analyze(
            '    r = bins - item\n'
            '    target = 0.5 * np.max(r)\n'
            '    return -np.abs(r - target)\n'
        )
        self.assertEqual(info.structure_tag, 'bucket_target')
        self.assertFalse(info.bf_equivalent)


class StructureAwareIslandTests(unittest.TestCase):

    def test_same_score_different_structure_makes_multiple_parents(self):
        template = code_manipulation.text_to_program(
            'import numpy as np\n\n'
            'def priority(item, bins):\n'
            '    return -(bins - item)\n'
        )
        island = programs_database.Island(
            template=template,
            function_to_evolve='priority',
            functions_per_prompt=2,
            cluster_sampling_temperature_init=0.1,
            cluster_sampling_temperature_period=30000,
        )
        scores = {'a': -10.0, 'b': -11.0}
        island.register_program(
            _function('    return -(bins - item)\n'), scores)
        island.register_program(
            _function(
                '    r = bins - item\n'
                '    target = 0.5 * np.max(r)\n'
                '    return -np.abs(r - target)\n'
            ),
            scores,
        )

        _, _, _, _, parents = island.get_prompt()

        self.assertEqual(len(parents), 2)
        self.assertEqual(
            {getattr(p, 'structure_tag', '') for p in parents},
            {'residual_formula', 'bucket_target'},
        )

    def test_softmax_underflow_does_not_crash_prompt_sampling(self):
        template = code_manipulation.text_to_program(
            'import numpy as np\n\n'
            'def priority(item, bins):\n'
            '    return -(bins - item)\n'
        )
        island = programs_database.Island(
            template=template,
            function_to_evolve='priority',
            functions_per_prompt=2,
            cluster_sampling_temperature_init=0.1,
            cluster_sampling_temperature_period=30000,
        )
        island.register_program(
            _function('    return -(bins - item)\n'), {'a': -1.0, 'b': -1.0})
        island.register_program(
            _function(
                '    r = bins - item\n'
                '    target = 0.5 * np.max(r)\n'
                '    return -np.abs(r - target)\n'
            ),
            {'a': -1000000.0, 'b': -1000000.0},
        )

        _, _, _, _, parents = island.get_prompt()

        self.assertEqual(len(parents), 2)

    def test_parent_sampling_multiplier_prefers_near_frontier_non_bf(self):
        bf_cluster = programs_database.Cluster(
            score=-2067.0,
            implementation=_function('    return -(bins - item)\n'),
            structure_info=structure_analysis.analyze(
                '    return -(bins - item)\n'),
        )
        near_non_bf = programs_database.Cluster(
            score=-2070.0,
            implementation=_function(
                '    r = bins - item\n'
                '    return -np.abs(r - 0.5 * item)\n'),
            structure_info=structure_analysis.analyze(
                '    r = bins - item\n'
                '    return -np.abs(r - 0.5 * item)\n'),
        )
        catastrophic = programs_database.Cluster(
            score=-5000.0,
            implementation=_function(
                '    r = bins - item\n'
                '    target = 0.5 * np.max(r)\n'
                '    return -np.abs(r - target)\n'),
            structure_info=structure_analysis.analyze(
                '    r = bins - item\n'
                '    target = 0.5 * np.max(r)\n'
                '    return -np.abs(r - target)\n'),
        )

        self.assertLess(
            programs_database._sampling_multiplier(bf_cluster, -2067.0),
            1.0)
        self.assertGreater(
            programs_database._sampling_multiplier(near_non_bf, -2067.0),
            1.0)
        self.assertLess(
            programs_database._sampling_multiplier(catastrophic, -2067.0),
            programs_database._sampling_multiplier(bf_cluster, -2067.0))


if __name__ == '__main__':
    unittest.main()
