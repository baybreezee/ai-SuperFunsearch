"""Structure labels for candidate bin-packing priority functions.

This module is deliberately lightweight and deterministic.  It does not
change evaluation scores; it only gives the evolutionary database and A3
reflector a compact view of whether two same-score programs are actually
different search directions or just Best-Fit variants.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StructureInfo:
    structure_tag: str
    bf_equivalent: bool
    bf_match_rate: float | None = None
    diagnostics: tuple[str, ...] = ()

    def summary(self) -> str:
        if self.bf_match_rate is None:
            rate = 'n/a'
        else:
            rate = f'{self.bf_match_rate:.2f}'
        diag = ','.join(self.diagnostics) if self.diagnostics else 'none'
        return (
            f'structure={self.structure_tag}; '
            f'bf_equivalent={self.bf_equivalent}; bf_match_rate={rate}; '
            f'diagnostics={diag}'
        )


BF_EQUIVALENCE_THRESHOLD = 0.95


def analyze(code_or_function: Any) -> StructureInfo:
    """Return a compact structure label and Best-Fit equivalence flag."""
    code = _to_code(code_or_function)
    tag = classify_structure(code)
    rate = best_fit_match_rate(code, tag)
    bf_equiv = bool(rate is not None and rate >= BF_EQUIVALENCE_THRESHOLD)
    if rate is None and _static_best_fit_like(code, tag):
        bf_equiv = True
        rate = 1.0
    diagnostics = diagnose_code(code, tag, bf_equiv)
    return StructureInfo(tag, bf_equiv, rate, diagnostics)


def diagnose_code(
        code_or_function: Any,
        structure_tag: str | None = None,
        bf_equivalent: bool | None = None,
) -> tuple[str, ...]:
    """Deterministic behavior-risk labels for critic memories.

    These labels do not affect scoring. They give A3/A4 a compact diagnosis of
    why a generated heuristic may have regressed: wrong score direction,
    stochastic perturbations, pure Best-Fit-preserving transforms, etc.
    """
    code = _to_code(code_or_function)
    low = code.lower()
    compact = re.sub(r'\s+', '', low)
    tag = structure_tag or classify_structure(code)
    diagnostics: list[str] = []

    if re.search(r'\bnp\.random\b|\brandom\.', low):
        diagnostics.append('uses_random')
    if tag == 'loop_heavy' or any(
            token in low for token in (
                'simulate', 'fragmentation', 'variance', 'mean()', '.mean(')):
        diagnostics.append('global_balance_or_simulation')
    if 'returnmin(' in compact or 'returnmax(' in compact:
        diagnostics.append('returns_single_index')
    if '-np.inf' in compact and not re.search(
            r'(astype\(float\)|dtype\s*=\s*float|np\.asarray\([^)]*dtype\s*=\s*float|np\.full[^)]*dtype\s*=\s*float)',
            compact):
        diagnostics.append('inf_assignment_may_need_float')

    if _positive_residual_primary(compact):
        diagnostics.append('positive_residual_primary')

    if bf_equivalent is None:
        # Use only static BF-like detection here to avoid recursive execution.
        bf_equivalent = _static_best_fit_like(code, tag)
    if bool(bf_equivalent) and _pure_monotone_residual_like(compact):
        diagnostics.append('pure_monotone_residual')

    return tuple(dict.fromkeys(diagnostics))


def classify_structure(code_or_function: Any) -> str:
    """Classify the dominant strategy shape from the function body."""
    code = _to_code(code_or_function)
    low = code.lower()
    compact = re.sub(r'\s+', '', low)

    if re.search(r'\b(for|while)\b', low):
        return 'loop_heavy'
    if 'argmin' in low or 'argmax' in low:
        return 'argmin_argmax'
    if ('bucket' in low or 'target' in low or 'np.arange' in low
            or re.search(r'\[[^\]]*0\.25', low)):
        return 'bucket_target'
    if 'np.where' in low or 'np.select' in low:
        return 'piecewise_rule'
    if '==max_cap' in compact or '==bins.max' in compact or '-np.inf' in compact:
        if _residual_like(compact):
            return 'mask_max_bin'
        return 'masked_rule'
    if ('**2' in compact or 'pow(' in low or 'np.power' in low
            or re.search(r'\*\*\s*[0-9.]+', low)):
        if _residual_like(compact):
            return 'quadratic_residual'
        return 'power_formula'
    if any(f'np.{fn}' in low for fn in ('exp', 'log', 'sin', 'cos', 'tanh')):
        if _residual_like(compact):
            return 'transformed_residual'
        return 'transcendental_formula'
    if _residual_like(compact):
        return 'residual_formula'
    return 'uniform_formula'


def best_fit_match_rate(
        code_or_function: Any,
        structure_tag: str | None = None,
        *,
        num_cases: int = 32,
) -> float | None:
    """Compare candidate argmax decisions against Best Fit on toy states.

    To avoid hanging on generated Python loops, loop-heavy candidates are not
    executed here.  They are still labelled structurally and evaluated by the
    real sandbox.
    """
    code = _to_code(code_or_function)
    tag = structure_tag or classify_structure(code)
    if tag == 'loop_heavy':
        return None

    fn = _compile_priority(code)
    if fn is None:
        return None

    rng = np.random.default_rng(17)
    matches = 0
    total = 0
    for _ in range(num_cases):
        item = float(rng.uniform(0.05, 0.9))
        bins = rng.uniform(item + 0.01, 1.0, size=int(rng.integers(3, 10)))
        try:
            priorities = np.asarray(fn(item, bins.copy()), dtype=float)
        except Exception:
            return None
        if priorities.shape != bins.shape or not np.any(np.isfinite(priorities)):
            return None
        cand_idx = int(np.nanargmax(priorities))
        best_fit_idx = int(np.argmin(bins - item))
        matches += int(cand_idx == best_fit_idx)
        total += 1
    if total == 0:
        return None
    return matches / total


def _to_code(code_or_function: Any) -> str:
    if hasattr(code_or_function, 'body'):
        return str(getattr(code_or_function, 'body') or '')
    return str(code_or_function or '')


def _residual_like(compact_code: str) -> bool:
    return ('bins-item' in compact_code or 'item-bins' in compact_code
            or 'rem=' in compact_code or 'resid' in compact_code
            or 'delta=' in compact_code or 'post=' in compact_code)


def _static_best_fit_like(code: str, tag: str) -> bool:
    compact = re.sub(r'\s+', '', code.lower())
    if tag not in {
        'residual_formula',
        'quadratic_residual',
        'transformed_residual',
        'mask_max_bin',
    }:
        return False
    if any(tok in compact for tok in ('target', 'bucket', 'np.where', 'argmin', 'argmax')):
        return False
    return _residual_like(compact)


def _positive_residual_primary(compact_code: str) -> bool:
    """Heuristic detector for wrong score direction.

    In this evaluator, higher priority wins. A direct positive residual return
    usually prefers looser bins, which is a common catastrophic A1 mistake.
    """
    if re.search(r'return(?:bins-item|r|rem|residuals?|post|remaining)', compact_code):
        if not re.search(r'return-', compact_code):
            return True
    if re.search(r'(scores|priorities)=\(?r(?:\.astype\(float\))?', compact_code):
        return True
    if re.search(r'(scores|priorities)=\(?rem(?:\.astype\(float\))?', compact_code):
        return True
    if re.search(r'(scores|priorities)=\(?residuals?(?:\.astype\(float\))?', compact_code):
        return True
    if re.search(r'(scores|priorities)=\(?post(?:\.astype\(float\))?', compact_code):
        return True
    return False


def _pure_monotone_residual_like(compact_code: str) -> bool:
    risky_tokens = (
        'np.where', 'np.select', 'argsort', 'argmin', 'argmax', 'target',
        'mean', 'var', 'random', 'np.arange')
    if any(tok in compact_code for tok in risky_tokens):
        return False
    return _residual_like(compact_code)


def _compile_priority(code_body: str):
    body = code_body.strip('\n')
    if not body.strip():
        return None
    if body.lstrip().startswith('def '):
        source = body
    else:
        source = 'def priority(item, bins):\n' + '\n'.join(
            line if line.startswith((' ', '\t')) else '    ' + line
            for line in body.splitlines()
        )
    safe_builtins = {
        'abs': abs,
        'float': float,
        'int': int,
        'len': len,
        'max': max,
        'min': min,
        'pow': pow,
        'range': range,
        'sum': sum,
    }
    namespace = {'np': np, '__builtins__': safe_builtins}
    try:
        exec(source, namespace, namespace)
    except Exception:
        return None
    return namespace.get('priority')
