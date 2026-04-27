"""Benchmark a heuristic on the OR3 / Weibull bin-packing datasets.

Outputs the same per-instance statistics that the project's baseline
table uses:
    Avg Bins | Std Dev | Min | Max | Avg L1 Bound | Avg Gap

Usage examples
--------------
# Bench all 5 classical baselines on OR3 (no API needed):
    python bench_heuristic.py --all-baselines

# Bench a heuristic stored in a samples_*.json:
    python bench_heuristic.py --from-sample \\
        ../logs/super_funsearch/samples/samples_78.json

# Bench a heuristic from any Python file containing
# `def priority(item, bins): ...`:
    python bench_heuristic.py --code my_heuristic.py --label "Ours v3"

# Combine — typical "report-ready" command:
    python bench_heuristic.py --all-baselines --from-sample <...> --md

The script is deterministic, dependency-light (numpy + bin_packing_utils
already in the project), and is also imported by `summarize_run.py` so
that every run report automatically gets the bench table appended.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import textwrap
from pathlib import Path
from typing import Callable

import numpy as np

# Make sibling imports work no matter where this is invoked from.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import bin_packing_utils  # noqa: E402  (after sys.path mutation)


# ---------------------------------------------------------------------------
# Core simulator (priority-based interface, matches run_super_funsearch.py)
# ---------------------------------------------------------------------------

def _simulate_priority_fn(
        priority_fn: Callable,
        items: list,
        capacity: float,
        num_items: int,
) -> int:
    """Pre-allocate `num_items` empty bins, then place every item into the
    feasible bin with the highest priority. Returns the count of bins
    that ended up holding at least one item.

    This is a copy of `online_binpack` from run_super_funsearch.py — kept
    inline so this script has no dependency on the bigger experiment file.
    """
    bins = np.array([capacity for _ in range(num_items)], dtype=float)
    for item in items:
        valid_idx = np.nonzero((bins - item) >= 0)[0]
        if valid_idx.size == 0:
            # Should never happen in OR3 (each item < capacity); guard anyway.
            raise RuntimeError(f'Item {item} cannot fit any bin')
        priorities = priority_fn(item, bins[valid_idx])
        chosen = valid_idx[int(np.argmax(priorities))]
        bins[chosen] -= item
    return int((bins != capacity).sum())


# ---------------------------------------------------------------------------
# Classical baselines (implemented directly — easier than forcing FF/NF
# through the priority-on-array interface)
# ---------------------------------------------------------------------------

def _first_fit(items, capacity, _num_items=None):
    bins = []
    for item in items:
        for i in range(len(bins)):
            if bins[i] >= item:
                bins[i] -= item
                break
        else:
            bins.append(capacity - item)
    return len(bins)


def _best_fit(items, capacity, _num_items=None):
    bins = []
    for item in items:
        best_i, best_left = -1, math.inf
        for i, b in enumerate(bins):
            if b >= item and (b - item) < best_left:
                best_left = b - item
                best_i = i
        if best_i == -1:
            bins.append(capacity - item)
        else:
            bins[best_i] -= item
    return len(bins)


def _worst_fit(items, capacity, _num_items=None):
    bins = []
    for item in items:
        worst_i, worst_left = -1, -1.0
        for i, b in enumerate(bins):
            if b >= item and (b - item) > worst_left:
                worst_left = b - item
                worst_i = i
        if worst_i == -1:
            bins.append(capacity - item)
        else:
            bins[worst_i] -= item
    return len(bins)


def _next_fit(items, capacity, _num_items=None):
    bins = []
    cur = -1
    for item in items:
        if cur >= 0 and bins[cur] >= item:
            bins[cur] -= item
        else:
            bins.append(capacity - item)
            cur = len(bins) - 1
    return len(bins)


def _ffd(items, capacity, _num_items=None):
    return _first_fit(sorted(items, reverse=True), capacity)


CLASSICAL_BASELINES: dict[str, Callable] = {
    'First Fit':            _first_fit,
    'Best Fit':             _best_fit,
    'Worst Fit':            _worst_fit,
    'Next Fit':             _next_fit,
    'First Fit Decreasing': _ffd,
}


def _is_offline(name: str) -> bool:
    """Return True for algorithms that need the full item list upfront
    (i.e. NOT a fair online comparison)."""
    return name == 'First Fit Decreasing'


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _stats_from_runs(name: str, dataset: str,
                     bins_used: list[int],
                     l1_bounds: list[int]) -> dict:
    n = len(bins_used)
    mean_b = float(np.mean(bins_used))
    mean_l1 = float(np.mean(l1_bounds))
    return {
        'name': name,
        'dataset': dataset,
        'num_instances': n,
        'avg': mean_b,
        # ddof=1 for sample std (matches what most papers report); falls
        # back to 0.0 when n<=1 to avoid div-by-zero.
        'std': float(np.std(bins_used, ddof=1)) if n > 1 else 0.0,
        'min': int(min(bins_used)),
        'max': int(max(bins_used)),
        'avg_l1': mean_l1,
        'avg_gap': mean_b - mean_l1,
        'per_instance': bins_used,
        'l1_per_instance': l1_bounds,
        'is_offline': _is_offline(name),
    }


def bench_classical(name: str, dataset_name: str = 'OR3') -> dict:
    """Run a classical baseline and return the standard stats dict."""
    fn = CLASSICAL_BASELINES[name]
    dataset = bin_packing_utils.datasets[dataset_name]
    bins_used: list[int] = []
    l1_bounds: list[int] = []
    for inst in dataset.values():
        capacity = inst['capacity']
        items = inst['items']
        bins_used.append(fn(items, capacity, inst['num_items']))
        l1_bounds.append(int(bin_packing_utils.l1_bound(items, capacity)))
    return _stats_from_runs(name, dataset_name, bins_used, l1_bounds)


def bench_priority_fn(priority_fn: Callable,
                      dataset_name: str = 'OR3',
                      label: str = 'custom heuristic') -> dict:
    """Run an LLM-style priority(item, bins) function on every instance."""
    dataset = bin_packing_utils.datasets[dataset_name]
    bins_used: list[int] = []
    l1_bounds: list[int] = []
    for inst in dataset.values():
        capacity = inst['capacity']
        items = inst['items']
        n = _simulate_priority_fn(
            priority_fn, items, capacity, inst['num_items'])
        bins_used.append(n)
        l1_bounds.append(int(bin_packing_utils.l1_bound(items, capacity)))
    return _stats_from_runs(label, dataset_name, bins_used, l1_bounds)


# ---------------------------------------------------------------------------
# Heuristic loaders
# ---------------------------------------------------------------------------

def load_priority_from_code_str(code: str,
                                function_name: str = 'priority') -> Callable:
    """Compile a string containing `def priority(item, bins): ...` and
    return the callable. The LLM sandboxes always produce code that uses
    `np` (numpy) — we expose that name in the exec namespace.
    """
    namespace = {
        'np': np,
        'numpy': np,
        '__builtins__': __builtins__,
    }
    try:
        exec(code, namespace)
    except Exception as e:
        raise ValueError(f'failed to compile heuristic: {e!r}') from e
    fn = namespace.get(function_name)
    if fn is None or not callable(fn):
        raise ValueError(
            f'compiled code does not define a callable `{function_name}`')
    return fn


def load_priority_from_sample_json(json_path: Path) -> tuple[Callable, dict]:
    """Load `samples_NN.json` and return (priority_fn, raw_sample_dict)."""
    with open(json_path, encoding='utf-8') as f:
        sample = json.load(f)
    code = sample.get('function', '')
    if not code.strip():
        raise ValueError(f'no `function` field in {json_path}')
    return load_priority_from_code_str(code), sample


def load_priority_from_py_file(py_path: Path,
                               function_name: str = 'priority') -> Callable:
    """Import a .py file and pull out its `priority` function."""
    spec = importlib.util.spec_from_file_location('heuristic_module', py_path)
    if spec is None or spec.loader is None:
        raise ValueError(f'cannot import {py_path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, function_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f'{py_path} does not define `{function_name}`')
    return fn


def find_best_sample_in_dir(samples_dir: Path) -> Path | None:
    """Pick the JSON sample with the highest valid score in `samples_dir`.

    Accepts both file-naming conventions:
      * legacy `samples_<sample_order>.json` (overwriting was possible)
      * current `sample_<write_seq>.json`    (one file per disk write,
        impossible to overwrite — see `profile.py`).

    Both shapes are returned by a single ``glob('sample*.json')`` so we
    don't need to know the run vintage at the call site.
    """
    best: tuple[float, Path] | None = None
    for fp in samples_dir.glob('sample*.json'):
        try:
            with open(fp, encoding='utf-8') as f:
                d = json.load(f)
        except Exception:
            continue
        sc = d.get('score')
        if not isinstance(sc, (int, float)) or math.isnan(sc):
            continue
        # Score is `-avg_bins`; we want the largest (least negative).
        if best is None or sc > best[0]:
            best = (sc, fp)
    return best[1] if best else None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

REPORT_HEADERS = (
    'Method', 'Avg Bins', 'Std Dev', 'Min', 'Max', 'Avg L1 Bound', 'Avg Gap',
)


def format_table_md(rows: list[dict]) -> str:
    """Render rows as a Markdown table matching the project's baseline
    table format. Columns mirror REPORT_HEADERS."""
    out = ['| ' + ' | '.join(REPORT_HEADERS) + ' |',
           '|' + '|'.join(['---'] * len(REPORT_HEADERS)) + '|']
    for r in rows:
        name = r['name']
        if r.get('is_offline'):
            name = f'{name} *(offline)*'
        out.append(
            f"| {name} "
            f"| {r['avg']:.2f} "
            f"| {r['std']:.2f} "
            f"| {r['min']:d} "
            f"| {r['max']:d} "
            f"| {r['avg_l1']:.2f} "
            f"| {r['avg_gap']:.2f} |"
        )
    return '\n'.join(out)


def format_table_text(rows: list[dict]) -> str:
    """Render rows as a fixed-width text table for terminal display."""
    widths = [max(28, max(len(r['name']) + (12 if r.get('is_offline') else 0)
                          for r in rows)),
              9, 9, 6, 6, 14, 9]
    fmt_h = (f"{{:<{widths[0]}}} | {{:>{widths[1]}}} | "
             f"{{:>{widths[2]}}} | {{:>{widths[3]}}} | "
             f"{{:>{widths[4]}}} | {{:>{widths[5]}}} | "
             f"{{:>{widths[6]}}}")
    out = [fmt_h.format(*REPORT_HEADERS),
           '-' * (sum(widths) + 3 * (len(widths) - 1))]
    for r in rows:
        name = r['name'] + (' (offline)' if r.get('is_offline') else '')
        out.append(
            fmt_h.format(
                name, f"{r['avg']:.2f}", f"{r['std']:.2f}",
                str(r['min']), str(r['max']),
                f"{r['avg_l1']:.2f}", f"{r['avg_gap']:.2f}"
            )
        )
    return '\n'.join(out)


# ---------------------------------------------------------------------------
# Top-level orchestration (importable by summarize_run.py)
# ---------------------------------------------------------------------------

def bench_all_classicals(dataset_name: str = 'OR3') -> list[dict]:
    """Run every classical baseline and return one stats dict per row."""
    return [bench_classical(name, dataset_name)
            for name in CLASSICAL_BASELINES]


def bench_best_sample_with_baselines(
        samples_dir: Path,
        dataset_name: str = 'OR3',
        ours_label: str = 'Ours (best heuristic)',
) -> tuple[list[dict], Path | None]:
    """Find the best sample in `samples_dir`, bench it, and return rows
    sorted [classical baselines..., ours] for direct table rendering."""
    rows = bench_all_classicals(dataset_name)
    best_path = find_best_sample_in_dir(samples_dir)
    if best_path is not None:
        try:
            fn, sample = load_priority_from_sample_json(best_path)
            order = sample.get('sample_order', '?')
            score = sample.get('score', '?')
            label = (
                f"{ours_label} "
                f"[sample #{order}, search-score {score}]"
            )
            rows.append(bench_priority_fn(fn, dataset_name, label=label))
        except Exception as e:
            rows.append({
                'name': f'{ours_label} (LOAD FAILED: {e!r})',
                'dataset': dataset_name,
                'num_instances': 0,
                'avg': float('nan'), 'std': float('nan'),
                'min': 0, 'max': 0,
                'avg_l1': float('nan'), 'avg_gap': float('nan'),
                'per_instance': [], 'l1_per_instance': [],
                'is_offline': False,
            })
    return rows, best_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument('--all-baselines', action='store_true',
                    help='Bench all 5 classical heuristics.')
    ap.add_argument('--baseline', choices=list(CLASSICAL_BASELINES.keys()),
                    help='Bench a single classical heuristic.')
    ap.add_argument('--from-sample', type=Path,
                    help='Bench heuristic stored in a samples_*.json file.')
    ap.add_argument('--code', type=Path,
                    help='Bench heuristic from a .py file '
                         'defining `def priority(item, bins)`.')
    ap.add_argument('--label', type=str, default='Custom heuristic',
                    help='Display name for the --code/--from-sample row.')
    ap.add_argument('--dataset', default='OR3',
                    choices=list(bin_packing_utils.datasets.keys()))
    ap.add_argument('--md', action='store_true',
                    help='Output Markdown instead of plain text.')
    args = ap.parse_args()

    rows: list[dict] = []
    if args.all_baselines:
        rows.extend(bench_all_classicals(args.dataset))
    elif args.baseline:
        rows.append(bench_classical(args.baseline, args.dataset))

    if args.code:
        fn = load_priority_from_py_file(args.code)
        rows.append(bench_priority_fn(fn, args.dataset, label=args.label))
    elif args.from_sample:
        fn, sample = load_priority_from_sample_json(args.from_sample)
        order = sample.get('sample_order', '?')
        score = sample.get('score', '?')
        rows.append(bench_priority_fn(
            fn, args.dataset,
            label=f'{args.label} [sample #{order}, score {score}]'))

    if not rows:
        ap.error('Nothing to bench. Pass --all-baselines / --baseline / '
                 '--from-sample / --code.')

    if args.md:
        print(format_table_md(rows))
    else:
        print(format_table_text(rows))


if __name__ == '__main__':
    main()
