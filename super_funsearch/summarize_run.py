"""Post-mortem analyser for a Super-FunSearch run.

Usage:
    python summarize_run.py                          # auto-detect latest run
    python summarize_run.py --samples-dir <path>     # specific samples dir
    python summarize_run.py --log-file   <path>      # parse operator stats
    python summarize_run.py --baseline   <path>      # compare against another
                                                     # samples dir
    python summarize_run.py --topk 10                # show 10 best samples

Produces a markdown report on stdout summarising:
  * total / valid / invalid / runtime-error counts
  * score statistics (min, p25, median, p75, max, mean, std)
  * top-K best samples (score + thought + code snippet)
  * operator usage distribution (parsed from the log file)
  * per-operator validity rate and best score
  * comparison with a baseline run (if --baseline is provided)

The script is read-only and dependency-free (stdlib only) so it can be
re-run any number of times against the same logs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

# Local: benchmark utilities (Avg/Std/Min/Max/L1/Gap on OR3 etc.)
try:
    import bench_heuristic
except Exception as _bench_import_err:  # pragma: no cover - defensive
    bench_heuristic = None  # type: ignore
    _BENCH_IMPORT_ERR = _bench_import_err
else:
    _BENCH_IMPORT_ERR = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SAMPLES_DIR = SCRIPT_DIR.parent / 'logs' / 'super_funsearch' / 'samples'
DEFAULT_LOG_FILE = SCRIPT_DIR.parent / 'logs' / 'eoh_run.log'


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_samples(samples_dir: Path) -> list[dict]:
    """Load every sample JSON from `samples_dir`.

    Supports both naming schemes:
      * legacy `samples_<sample_order>.json` (sorted by sample_order)
      * current `sample_<write_seq>.json`    (sorted by write_seq)

    The two are kept apart by their ``glob`` pattern so we sort each
    group with the right key, then concatenate (legacy first, current
    second) which matches chronological write order in mixed dirs.
    """
    out: list[dict] = []
    if not samples_dir.exists():
        return out

    def _key_legacy(p: Path) -> int:
        m = re.search(r'samples_(\d+)', p.stem)
        return int(m.group(1)) if m else -1

    def _key_current(p: Path) -> int:
        m = re.search(r'sample_(\d+)', p.stem)
        return int(m.group(1)) if m else -1

    files = (
        sorted(samples_dir.glob('samples_*.json'), key=_key_legacy)
        + sorted(samples_dir.glob('sample_[0-9]*.json'), key=_key_current)
    )
    seen: set[Path] = set()
    for fp in files:
        if fp in seen:
            continue
        seen.add(fp)
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                out.append(json.load(f))
        except Exception as e:
            print(f"# WARN: failed to load {fp.name}: {e}")
    return out


def _open_log(path: Path):
    """Open a log file, sniffing for UTF-16 BOM (PowerShell `Tee-Object`
    writes UTF-16 LE by default on Windows). Falls back to UTF-8."""
    with open(path, 'rb') as f:
        head = f.read(2)
    if head == b'\xff\xfe':
        return open(path, 'r', encoding='utf-16', errors='replace')
    if head == b'\xfe\xff':
        return open(path, 'r', encoding='utf-16-be', errors='replace')
    return open(path, 'r', encoding='utf-8', errors='replace')


def parse_log(log_file: Path) -> dict:
    """Pull operator usage and rejection counters from the run log."""
    info = {
        'operator_counts':       Counter(),    # op -> total times invoked
        'operator_invalid':      Counter(),    # op -> rejected by InvalidSampleError
        'operator_runtime_err':  Counter(),    # op -> sandbox runtime error
        'fallback_to_i1':        0,
        'api_failures':          0,
        'launched_with':         '',
    }
    if not log_file.exists():
        return info

    op_re = re.compile(r'EoH op=(\w+)')
    rej_re = re.compile(r'Rejected invalid sample.*op=(\w+)')
    runerr_re = re.compile(r'Runtime-failed sample.*op=(\w+)')
    fallback_re = re.compile(r'EoH falling back from (\w+) to i1')
    apierr_re = re.compile(r'gave up after \d+ attempts')
    launched_re = re.compile(r'Launching Super-FunSearch.*$')

    with _open_log(log_file) as f:
        for line in f:
            m = launched_re.search(line)
            if m and not info['launched_with']:
                info['launched_with'] = m.group(0)
            m = op_re.search(line)
            if m:
                info['operator_counts'][m.group(1)] += 1
            m = rej_re.search(line)
            if m:
                info['operator_invalid'][m.group(1)] += 1
            m = runerr_re.search(line)
            if m:
                info['operator_runtime_err'][m.group(1)] += 1
            if fallback_re.search(line):
                info['fallback_to_i1'] += 1
            if apierr_re.search(line):
                info['api_failures'] += 1
    return info


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def score_stats(samples: list[dict]) -> dict:
    valid_scores = [s['score'] for s in samples
                    if isinstance(s.get('score'), (int, float))
                    and not math.isnan(s['score'])]
    n_total = len(samples)
    n_valid = len(valid_scores)
    n_invalid = n_total - n_valid
    if not valid_scores:
        return {'n_total': n_total, 'n_valid': 0, 'n_invalid': n_invalid}

    sorted_s = sorted(valid_scores)
    return {
        'n_total': n_total,
        'n_valid': n_valid,
        'n_invalid': n_invalid,
        'min':    sorted_s[0],
        'max':    sorted_s[-1],
        'mean':   statistics.fmean(valid_scores),
        'median': statistics.median(valid_scores),
        'stdev':  statistics.pstdev(valid_scores) if len(valid_scores) > 1 else 0.0,
        'p25':    _percentile(sorted_s, 0.25),
        'p75':    _percentile(sorted_s, 0.75),
        'p90':    _percentile(sorted_s, 0.90),
    }


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float('nan')
    idx = int(round(q * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def per_operator_stats(
        samples: list[dict],
        log_info: dict) -> dict[str, dict]:
    """Combine in-log operator counts with score stats (sample order based).

    The log records 'EoH op=X' BEFORE evaluation, so we associate the
    n-th 'op=X' line with the n-th sample in sample-order. This reconstructs
    a reasonably accurate per-operator score table without modifying the
    sample json schema.
    """
    op_seq: list[str] = []
    log_file = log_info.get('_log_file_path')
    if log_file and Path(log_file).exists():
        op_re = re.compile(r'EoH op=(\w+)')
        with _open_log(Path(log_file)) as f:
            for line in f:
                m = op_re.search(line)
                if m:
                    op_seq.append(m.group(1))

    per_op: dict[str, list[float]] = defaultdict(list)
    invalid_per_op: dict[str, int] = defaultdict(int)
    for i, s in enumerate(samples):
        if i >= len(op_seq):
            break
        op = op_seq[i]
        sc = s.get('score')
        if isinstance(sc, (int, float)) and not math.isnan(sc):
            per_op[op].append(sc)
        else:
            invalid_per_op[op] += 1

    out = {}
    for op in sorted(set(list(per_op.keys()) + list(invalid_per_op.keys()))):
        scores = per_op[op]
        out[op] = {
            'n_total':   len(scores) + invalid_per_op[op],
            'n_valid':   len(scores),
            'n_invalid': invalid_per_op[op],
            'best':      max(scores) if scores else None,
            'mean':      statistics.fmean(scores) if scores else None,
        }
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_benchmark_section(
        samples_dir: Path,
        dataset_name: str = 'OR3',
        topn: int = 1,
) -> str:
    """Re-execute the top-N samples on every dataset instance and
    render a Markdown table comparing Avg/Std/Min/Max/L1/Gap against
    the 5 classical baselines.

    Returns a (possibly empty) Markdown string ready to splice into
    the report. Soft-fails if anything goes wrong so summarize_run.py
    still produces the rest of the report.
    """
    if bench_heuristic is None:
        return (f'\n## 6. Per-instance benchmark on {dataset_name}\n'
                f'_bench_heuristic import failed: {_BENCH_IMPORT_ERR!r}_\n')

    out: list[str] = []
    out.append(f'\n## 6. Per-instance benchmark on {dataset_name}\n')
    out.append(
        '_All algorithms run on every dataset instance; statistics are '
        'averaged across the instances. "Avg L1 Bound" is the L1 lower '
        'bound on OPT; "Avg Gap" = Avg Bins − Avg L1._\n\n')

    try:
        rows = bench_heuristic.bench_all_classicals(dataset_name)
    except Exception as e:
        out.append(f'_classical baselines failed: {e!r}_\n')
        return ''.join(out)

    # Take the top-N valid samples and bench them.
    valid = [s for s in load_samples(samples_dir)
             if isinstance(s.get('score'), (int, float))
             and not math.isnan(s['score'])]
    valid.sort(key=lambda s: s['score'], reverse=True)
    benched_any = False
    for i, s in enumerate(valid[:topn], 1):
        code = (s.get('function') or '').strip()
        if not code:
            continue
        try:
            fn = bench_heuristic.load_priority_from_code_str(code)
            label = (
                f"Ours #{i} (sample {s.get('sample_order','?')}, "
                f"search-score {s['score']:.3f})")
            rows.append(bench_heuristic.bench_priority_fn(
                fn, dataset_name, label=label))
            benched_any = True
        except Exception as e:
            rows.append({
                'name': f"Ours #{i} (LOAD FAILED: {e!r})",
                'dataset': dataset_name, 'num_instances': 0,
                'avg': float('nan'), 'std': float('nan'),
                'min': 0, 'max': 0,
                'avg_l1': float('nan'), 'avg_gap': float('nan'),
                'per_instance': [], 'l1_per_instance': [],
                'is_offline': False,
            })

    if not benched_any:
        out.append('_no valid samples available to bench_\n')

    out.append(bench_heuristic.format_table_md(rows))
    out.append('\n')
    return ''.join(out)


def render_report(samples: list[dict], log_info: dict,
                  topk: int, baseline: dict | None) -> str:
    out: list[str] = []
    add = out.append

    add('# Super-FunSearch Run Report\n')
    if log_info.get('launched_with'):
        add(f"_Launch: {log_info['launched_with']}_\n")

    # ---- score stats ----
    stats = score_stats(samples)
    add('\n## 1. Score statistics\n')
    if stats['n_valid'] == 0:
        add(f'- Total samples: {stats["n_total"]}\n')
        add(f'- Valid (scored): 0\n')
        add(f'- Invalid (no score): {stats["n_invalid"]}\n')
        add('- **No valid samples — nothing to summarise.**\n')
    else:
        add(f"| field | value |\n|---|---|\n")
        add(f"| total samples | {stats['n_total']} |\n")
        add(f"| valid (scored) | {stats['n_valid']} ({100*stats['n_valid']/stats['n_total']:.1f}%) |\n")
        add(f"| invalid (no score) | {stats['n_invalid']} |\n")
        add(f"| best (max) | **{stats['max']:.3f}** |\n")
        add(f"| mean | {stats['mean']:.3f} |\n")
        add(f"| median | {stats['median']:.3f} |\n")
        add(f"| std | {stats['stdev']:.3f} |\n")
        add(f"| p25 / p75 / p90 | {stats['p25']:.3f} / {stats['p75']:.3f} / {stats['p90']:.3f} |\n")
        add(f"| worst (min) | {stats['min']:.3f} |\n")

    # ---- baseline diff ----
    if baseline:
        add('\n## 2. Baseline comparison\n')
        b = baseline
        if b['n_valid'] > 0 and stats['n_valid'] > 0:
            d_best = stats['max'] - b['max']
            d_mean = stats['mean'] - b['mean']
            arrow_best = '[UP]' if d_best > 0 else ('[DOWN]' if d_best < 0 else '[==]')
            arrow_mean = '[UP]' if d_mean > 0 else ('[DOWN]' if d_mean < 0 else '[==]')
            add(f'| metric | this run | baseline | delta |\n|---|---|---|---|\n')
            add(f"| best | {stats['max']:.3f} | {b['max']:.3f} | {arrow_best} {d_best:+.3f} |\n")
            add(f"| mean | {stats['mean']:.3f} | {b['mean']:.3f} | {arrow_mean} {d_mean:+.3f} |\n")
            add(f"| valid rate | {100*stats['n_valid']/stats['n_total']:.1f}% | {100*b['n_valid']/b['n_total']:.1f}% | "
                f"{(100*stats['n_valid']/stats['n_total'])-(100*b['n_valid']/b['n_total']):+.1f}pp |\n")
        else:
            add('_baseline or current run has no valid samples_\n')

    # ---- operator usage from log ----
    if log_info['operator_counts']:
        add('\n## 3. Operator usage (from log)\n')
        add('| operator | invocations | rejected (Invalid) | runtime err |\n|---|---|---|---|\n')
        for op in ['i1', 'e1', 'e2', 'm1', 'm2', 'm3']:
            c = log_info['operator_counts'].get(op, 0)
            if c == 0:
                continue
            inv = log_info['operator_invalid'].get(op, 0)
            rerr = log_info['operator_runtime_err'].get(op, 0)
            add(f"| {op} | {c} | {inv} | {rerr} |\n")
        if log_info['fallback_to_i1']:
            add(f"\n_Fallbacks to i1 (parents shortage): {log_info['fallback_to_i1']}_\n")
        if log_info['api_failures']:
            add(f"_LLM API give-ups (after retry budget): {log_info['api_failures']}_\n")

        # ---- score per operator (reconstructed) ----
        per_op = per_operator_stats(samples, log_info)
        if per_op:
            add('\n## 4. Per-operator scores (reconstructed)\n')
            add('| operator | n_total | n_valid | best | mean |\n|---|---|---|---|---|\n')
            for op, s in per_op.items():
                best = f"{s['best']:.3f}" if s['best'] is not None else '—'
                mean = f"{s['mean']:.3f}" if s['mean'] is not None else '—'
                add(f"| {op} | {s['n_total']} | {s['n_valid']} | {best} | {mean} |\n")

    # ---- top-K samples ----
    add(f'\n## 5. Top-{topk} samples\n')
    valid = [s for s in samples
             if isinstance(s.get('score'), (int, float))
             and not math.isnan(s['score'])]
    valid.sort(key=lambda s: s['score'], reverse=True)
    for i, s in enumerate(valid[:topk], 1):
        add(f"\n### #{i}  score = {s['score']:.3f}  (sample_order={s.get('sample_order','?')})\n")
        thought = (s.get('thought') or '').strip()
        if thought:
            add(f"> **Thought:** {thought[:400]}\n")
        code = s.get('function', '').strip()
        # Trim docstring + show only body up to first 12 non-doc lines.
        body_lines = _extract_body_snippet(code, max_lines=12)
        add('\n```python\n' + '\n'.join(body_lines) + '\n```\n')

    return ''.join(out)


def _extract_body_snippet(function_text: str, max_lines: int) -> list[str]:
    """Strip the def line + docstring, return up to `max_lines` body lines."""
    lines = function_text.split('\n')
    # find first non-def, non-docstring line
    out: list[str] = []
    in_doc = False
    seen_def = False
    for line in lines:
        if not seen_def and re.match(r'^\s*def\s+\w+', line):
            seen_def = True
            continue
        stripped = line.strip()
        if not seen_def:
            continue
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                # single-line docstring — skip
                continue
            in_doc = not in_doc
            continue
        if in_doc:
            continue
        if not stripped and not out:
            continue
        out.append(line.rstrip())
        if len(out) >= max_lines:
            out.append('    # ... (truncated)')
            break
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--samples-dir', type=Path, default=DEFAULT_SAMPLES_DIR)
    ap.add_argument('--log-file', type=Path, default=DEFAULT_LOG_FILE)
    ap.add_argument('--baseline', type=Path, default=None,
                    help='Path to a baseline samples_dir for diff.')
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--out', type=Path, default=None,
                    help='If set, also write the report to this path.')
    ap.add_argument('--bench-topn', type=int, default=1,
                    help='Re-run top-N samples on the dataset to compute '
                         'Avg/Std/Min/Max/L1/Gap (0 disables).')
    ap.add_argument('--bench-dataset', type=str, default='OR3',
                    help='Dataset name to benchmark on (must exist in '
                         'bin_packing_utils.datasets).')
    args = ap.parse_args()

    samples = load_samples(args.samples_dir)
    if not samples:
        print(f'No samples_*.json found under {args.samples_dir}')
        return

    log_info = parse_log(args.log_file)
    log_info['_log_file_path'] = str(args.log_file)

    baseline_stats = None
    if args.baseline:
        b_samples = load_samples(args.baseline)
        if b_samples:
            baseline_stats = score_stats(b_samples)

    report = render_report(samples, log_info, args.topk, baseline_stats)

    if args.bench_topn > 0:
        report += render_benchmark_section(
            args.samples_dir,
            dataset_name=args.bench_dataset,
            topn=args.bench_topn,
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding='utf-8')
        print(f'Report written to {args.out}')
    else:
        try:
            print(report)
        except UnicodeEncodeError:
            import sys
            sys.stdout.buffer.write(report.encode('utf-8', errors='replace'))


if __name__ == '__main__':
    main()
