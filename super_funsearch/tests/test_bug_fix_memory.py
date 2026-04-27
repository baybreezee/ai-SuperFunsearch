"""Unit tests for A2 BugFixMemory.

Run from the super_funsearch directory:

    python -m unittest test_bug_fix_memory -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from implementation.bug_fix_memory import BugFixMemory


class BugFixMemoryTests(unittest.TestCase):

    def test_classifies_float_inf_into_int_array(self):
        mem = BugFixMemory()
        sig = mem.classify('[Weibull 5k] cannot convert float infinity to integer')
        self.assertEqual(sig, BugFixMemory.DTYPE_SCORE_SIGNATURE)

    def test_classifies_inplace_float_to_int_cast(self):
        mem = BugFixMemory()
        sig = mem.classify(
            "Cannot cast ufunc 'subtract' output from dtype('float64') "
            "to dtype('int64') with casting rule 'same_kind'")
        self.assertEqual(sig, BugFixMemory.DTYPE_SCORE_SIGNATURE)

    def test_unknown_error_has_no_patch(self):
        mem = BugFixMemory()
        patched, recipe = mem.deterministic_patch(
            '    return bins\n',
            'ValueError: some unrelated bug',
        )
        self.assertIsNone(patched)
        self.assertIsNone(recipe)

    def test_dtype_patch_casts_bins_to_float_once(self):
        mem = BugFixMemory()
        body = (
            '    rem = bins - item\n'
            '    priorities = -rem\n'
            '    priorities[bins == bins.max()] = -np.inf\n'
            '    return priorities\n'
        )
        patched, recipe = mem.deterministic_patch(
            body, 'cannot convert float infinity to integer')
        self.assertEqual(recipe, BugFixMemory.DTYPE_SCORE_RECIPE)
        self.assertIn('bins = np.asarray(bins, dtype=float)', patched)
        self.assertEqual(
            patched.count('bins = np.asarray(bins, dtype=float)'), 1)
        self.assertIn('    return priorities', patched)

        patched2, recipe2 = mem.deterministic_patch(
            patched, 'cannot convert float infinity to integer')
        self.assertIsNone(patched2)
        self.assertIsNone(recipe2)

    def test_records_success_and_failure_stats(self):
        mem = BugFixMemory(capacity=2)
        sig = BugFixMemory.DTYPE_SCORE_SIGNATURE
        mem.record(sig, 'recipe-a', True, 'err one')
        mem.record(sig, 'recipe-a', False, 'err two')
        mem.record(sig, 'recipe-b', True, 'err three')

        self.assertEqual(len(mem), 2)
        stats = mem.stats()[sig]
        self.assertEqual(stats['successes'], 2)
        self.assertEqual(stats['failures'], 1)
        recent = mem.recent()
        self.assertEqual([r.error_excerpt for r in recent], ['err two', 'err three'])


if __name__ == '__main__':
    unittest.main()
