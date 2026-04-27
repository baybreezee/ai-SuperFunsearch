"""A2 Bug-Fixer memory and deterministic repair recipes.

This module belongs to A2 only.  It records code-level runtime failures
and whether a concrete repair recipe worked.  It deliberately does not
store algorithm-quality signals; those belong to A3/ReEvo.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import textwrap


@dataclass(frozen=True)
class BugFixRecord:
    """One A2 repair attempt."""

    signature: str
    recipe: str
    success: bool
    error_excerpt: str


class BugFixMemory:
    """Small A2-owned memory of known runtime bug patterns.

    The first supported recipe fixes the current Weibull/no-numba failure
    class: numpy arrays derived from integer `bins` are later used as float
    score arrays (`-np.inf`, in-place float subtraction, etc.).
    """

    DEFAULT_CAPACITY = 20
    DTYPE_SCORE_SIGNATURE = 'numpy_int_score_array_needs_float'
    DTYPE_SCORE_RECIPE = 'cast_bins_to_float_at_function_start'

    def __init__(self, capacity: int = DEFAULT_CAPACITY):
        if capacity <= 0:
            raise ValueError(f'capacity must be > 0, got {capacity}')
        self._records: deque[BugFixRecord] = deque(maxlen=capacity)
        self._successes: dict[str, int] = defaultdict(int)
        self._failures: dict[str, int] = defaultdict(int)

    def classify(self, error_trace: str | None) -> str | None:
        """Map a runtime error trace to a known repair signature."""
        low = (error_trace or '').lower()
        if not low:
            return None
        if 'cannot convert float infinity to integer' in low:
            return self.DTYPE_SCORE_SIGNATURE
        if ("cannot cast ufunc" in low
                and "dtype('float64')" in low
                and "dtype('int64')" in low):
            return self.DTYPE_SCORE_SIGNATURE
        return None

    def deterministic_patch(
            self,
            code_body: str,
            error_trace: str | None,
    ) -> tuple[str | None, str | None]:
        """Return `(patched_body, recipe)` when a known recipe applies."""
        signature = self.classify(error_trace)
        if signature != self.DTYPE_SCORE_SIGNATURE:
            return None, None
        patched = self._ensure_bins_float(code_body)
        if not patched or patched == code_body:
            return None, None
        return patched, self.DTYPE_SCORE_RECIPE

    def record(
            self,
            signature: str | None,
            recipe: str | None,
            success: bool,
            error_trace: str | None,
    ) -> None:
        """Store the outcome of one A2 repair attempt."""
        if not signature or not recipe:
            return
        excerpt = self._first_line(error_trace)
        self._records.append(BugFixRecord(
            signature=signature,
            recipe=recipe,
            success=bool(success),
            error_excerpt=excerpt,
        ))
        if success:
            self._successes[signature] += 1
        else:
            self._failures[signature] += 1

    def stats(self) -> dict[str, dict[str, int]]:
        signatures = set(self._successes) | set(self._failures)
        return {
            sig: {
                'successes': self._successes.get(sig, 0),
                'failures': self._failures.get(sig, 0),
            }
            for sig in sorted(signatures)
        }

    def recent(self) -> list[BugFixRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    @staticmethod
    def _first_line(error_trace: str | None, max_chars: int = 160) -> str:
        line = next(
            (ln.strip() for ln in str(error_trace or '').splitlines()
             if ln.strip()),
            '',
        )
        if len(line) > max_chars:
            line = line[:max_chars - 1] + '…'
        return line

    @staticmethod
    def _ensure_bins_float(code_body: str) -> str:
        """Shadow `bins` with a float ndarray at the top of the body.

        The evaluator supplies `np` in the surrounding program, so no import is
        needed.  The generated function bodies are expected to be indented by
        four spaces; this function preserves that shape.
        """
        if not (code_body or '').strip():
            return ''
        dedented = textwrap.dedent(code_body).strip('\n')
        if not dedented.strip():
            return ''

        if 'np.asarray(bins, dtype=float)' in dedented:
            return code_body
        if 'bins.astype(float' in dedented:
            return code_body

        lines = dedented.splitlines()
        insert_at = 0

        # Keep an initial docstring at the top, if one exists.  EoH parsing
        # usually strips docstrings, but the legacy path can still surface one.
        first = lines[0].lstrip() if lines else ''
        if first.startswith(('"""', "'''")):
            quote = first[:3]
            if first.count(quote) >= 2 and len(first) > 3:
                insert_at = 1
            else:
                for i in range(1, len(lines)):
                    if quote in lines[i]:
                        insert_at = i + 1
                        break

        lines.insert(insert_at, 'bins = np.asarray(bins, dtype=float)')
        return '\n'.join(
            ('    ' + line if line.strip() else '')
            for line in lines
        )
