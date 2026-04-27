import json
import os
import tempfile
import unittest

from implementation import funsearch


class WarmStartTests(unittest.TestCase):

    def test_load_warm_start_records_keeps_top_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_sample(tmp, 'sample_000001.json', -2067.0)
            self._write_sample(tmp, 'sample_000002.json', -2143.0)
            self._write_sample(tmp, 'sample_000003.json', -2050.0)

            records = funsearch._load_warm_start_records(tmp, top_k=2)

        self.assertEqual([r['score'] for r in records], [-2050.0, -2067.0])

    def test_body_from_record_function_extracts_priority_body(self):
        body = funsearch._body_from_record_function(
            'def priority(item: float, bins: np.ndarray) -> np.ndarray:\n'
            '    x = bins - item\n'
            '    return -x\n'
        )

        self.assertIn('x = bins - item', body)
        self.assertIn('return -x', body)
        self.assertNotIn('def priority', body)

    @staticmethod
    def _write_sample(directory: str, name: str, score: float) -> None:
        with open(os.path.join(directory, name), 'w', encoding='utf-8') as f:
            json.dump({
                'function': (
                    'def priority(item: float, bins: np.ndarray) -> np.ndarray:\n'
                    '    return -(bins - item)\n'
                ),
                'score': score,
                'thought': None,
            }, f)


if __name__ == '__main__':
    unittest.main()
