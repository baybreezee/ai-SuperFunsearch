# Super-FunSearch Colab Submission

This folder contains the code needed to reproduce the main results in the project report for one-dimensional online bin packing.

## What Is Included

- `Super_FunSearch_Reproduction.ipynb`: the Google Colab notebook for reproducing the report table.
- `super_funsearch/bench_heuristic.py`: deterministic evaluator for baselines and saved heuristics.
- `super_funsearch/bin_packing_utils.py`: OR3 and Weibull 5k datasets plus L1 lower-bound utilities.
- `super_funsearch/run_super_funsearch.py`: optional full LLM evolutionary-search entry point.
- `super_funsearch/implementation/`: A1/A2/A3/A4, database, evaluator, and support modules.
- `super_funsearch/prompts_reevo/`: ReEvo-style reflector prompts.
- `super_funsearch/saved_samples/sample_000127_weibull_best.json`: saved best Weibull candidate used in the report.
- `super_funsearch/tests/`: focused unit tests for the core architecture.

No API keys are included. Do not commit `.env` files or notebook cells containing secrets.

## Colab Usage

1. Upload this folder to GitHub, or upload it as a zip to Colab.
2. Open `Super_FunSearch_Reproduction.ipynb` in Google Colab.
3. If using GitHub, set `REPO_URL` in the setup cell to your repository URL.
4. Run all deterministic cells to reproduce:
   - First Fit
   - Best Fit
   - Worst Fit
   - Next Fit
   - First Fit Decreasing
   - `Ours (v3 sample 122)`

The main result table does not require an LLM API key.

## Local Deterministic Reproduction

From this folder:

```bash
python -m venv venv
# Windows PowerShell:
venv\Scripts\Activate.ps1
# macOS/Linux:
# source venv/bin/activate
pip install -r requirements-minimal.txt
cd super_funsearch
python bench_heuristic.py --dataset "Weibull 5k" --all-baselines --from-sample saved_samples/sample_000127_weibull_best.json --label "Ours (v3 sample 122)" --md
```

Expected key result:

```text
Ours (v3 sample 122): Avg Bins 2031.60, Std 13.65, Min 2022, Max 2055, Avg L1 Bound 1987.80, Avg Gap 43.80
```

## Optional Full LLM Search

The full Super-FunSearch pipeline can be launched from `run_super_funsearch.py`, but it requires an OpenAI-compatible API key and may take a long time. The result is stochastic and may not rediscover the same saved sample.

Install the full dependency set only if you want to run the LLM search:

```bash
pip install -r requirements.txt
```

Example:

```bash
cd super_funsearch
python run_super_funsearch.py --provider openai --dataset weibull --max-samples 30 --num-islands 4 --samples-per-prompt 1 --reset-every-n-samples 50 --no-numba --search-controller --search-controller-horizon 10 --search-controller-min-events 6
```

## Architecture Summary

- A1 Coder: generates thought and code with EoH-style operators.
- A2 Bug-Fixer: repairs runtime crashes with deterministic recipes and bug-fix memory.
- A3 Algorithm-Guide: produces ReEvo-style short-term and long-term reflections.
- A4 Search-Controller: optionally schedules exploration, mutation, and parent-source bias.
- ProgramsDatabase: stores scored programs using islands and structure-aware clusters.
- L2 ErrorMemory: records recent runtime failures to prevent repeated code mistakes.

## Notes

The internal search score is negative average bin count. For example, score `-2031.6` corresponds to `Avg. Bins = 2031.60` in the report table.
