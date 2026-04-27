"""Entry point for Super-FunSearch on Bin Packing."""
import os
import ssl
import sys
import json
import http.client
import time
import logging
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

# --- Path setup ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# --- HuggingFace endpoint hardening (China retail networks) ---
# `SentenceTransformer('all-MiniLM-L6-v2')` triggers a HEAD request to
# huggingface.co which routinely fails with `[SSL: UNEXPECTED_EOF_WHILE_READING]`
# on Chinese ISPs. Two lines fix that completely:
#   1. Route through the well-known community mirror by default; user can
#      override by setting HF_ENDPOINT before launching.
#   2. After the very first successful download the model is cached under
#      ~/.cache/huggingface/. On subsequent runs we don't even need network,
#      so we also enable HF_HUB_OFFLINE=1 if the cache appears populated.
# Either env var is respected by `huggingface_hub` AND `sentence_transformers`
# automatically — no source-level change needed downstream.
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')


def _maybe_enable_hf_offline() -> None:
    """If the embedding model is already cached locally, force offline mode
    so HF doesn't even try a network HEAD request."""
    if os.environ.get('HF_HUB_OFFLINE'):
        return
    cache_root = os.path.expanduser('~/.cache/huggingface/hub')
    expected = os.path.join(
        cache_root, 'models--sentence-transformers--all-MiniLM-L6-v2')
    if os.path.isdir(expected):
        os.environ['HF_HUB_OFFLINE'] = '1'


_maybe_enable_hf_offline()

from implementation import sampler
from implementation import evaluator
from implementation import evaluator_accelerate
from implementation import code_manipulation
from implementation import funsearch
from implementation import config
import bin_packing_utils
import multiprocessing
from typing import Collection, Any


# =====================================================================
# Shared SSL context — tolerant of misbehaving servers (common with
# Chinese endpoints when client-side OpenSSL is 3.x on Windows).
# OP_IGNORE_UNEXPECTED_EOF (Python 3.11+) prevents the
# "UNEXPECTED_EOF_WHILE_READING" exception when servers close the
# connection without sending a TLS close_notify alert.
# =====================================================================
def _make_tolerant_ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    op_ignore = getattr(ssl, 'OP_IGNORE_UNEXPECTED_EOF', 0x40000000)
    try:
        ctx.options |= op_ignore
    except Exception:
        pass
    return ctx


_SSL_CTX = _make_tolerant_ssl_context()


# Bounded retry — historical code used `while True: time.sleep(3); continue`
# which would hang the whole experiment indefinitely on prolonged network
# outages. Bound the retries and let the caller see None so the sampler can
# move on to the next prompt.
_MAX_API_RETRIES = 6
_API_BACKOFF_SECONDS = (2, 4, 8, 16, 30, 60)


# =====================================================================
# 1. LLM Implementations
#    Providers via --provider:
#      - 'qwen'     -> DashScope OpenAI-compatible endpoint
#      - 'deepseek' -> api.deepseek.com (OpenAI-compatible)
#      - 'openai'   -> any OpenAI-compatible HTTPS API (official or third-party)
#    Each provider exposes a "coder" class (used by Sampler) and a
#    "general" class (used by Reflector / KnowledgeExtractor).
# =====================================================================
class QwenCoderLLM(sampler.LLM):
    """Qwen3-Coder-Plus LLM for thought + code generation.

    `qwen3-coder-plus` is the current code-tuned flagship on DashScope
    (released 2025-Q4), substantially better at structured code edits
    than the legacy `qwen-coder-plus` and on par with GPT-4o-mini for
    short-function generation tasks like our BPP `priority()`.
    """

    def __init__(self, samples_per_prompt: int):
        super().__init__(samples_per_prompt)
        self._api_key = os.getenv('QWEN_API_KEY', '')
        self._host = 'dashscope.aliyuncs.com'
        self._path = '/compatible-mode/v1/chat/completions'
        self._model = 'qwen3-coder-plus'

    def _call_api(self, prompt: str) -> str:
        return _call_openai_compat(self._host, self._path, self._api_key,
                                   self._model, prompt, temperature=0.8,
                                   tag='QwenCoder')


class QwenLLM(sampler.LLM):
    """Qwen-Max LLM (general reasoning) used for ReEvo / Reflector calls.

    Reflector benefits from a smarter, more deterministic model than the
    coder. We mirror DeepSeekReflectorLLM's design choice (`temp=0.3`)
    so that long-term reflections are stable across runs.
    """

    def __init__(self, samples_per_prompt: int):
        super().__init__(samples_per_prompt)
        self._api_key = os.getenv('QWEN_API_KEY', '')
        self._host = 'dashscope.aliyuncs.com'
        self._path = '/compatible-mode/v1/chat/completions'
        self._model = 'qwen-max'

    def _call_api(self, prompt: str) -> str:
        return _call_openai_compat(self._host, self._path, self._api_key,
                                   self._model, prompt, temperature=0.3,
                                   tag='QwenRef')


class DeepSeekChatLLM(sampler.LLM):
    """DeepSeek-Chat (V3) – used for code generation in Sampler.

    DeepSeek folded `deepseek-coder` into `deepseek-chat` in 2024-09, so a
    single chat model now serves both code and reasoning. Endpoint is OpenAI
    chat-completions compatible.
    """

    def __init__(self, samples_per_prompt: int):
        super().__init__(samples_per_prompt)
        self._api_key = os.getenv('DEEPSEEK_API_KEY', '')
        self._host = 'api.deepseek.com'
        self._path = '/v1/chat/completions'
        self._model = 'deepseek-chat'

    def _call_api(self, prompt: str) -> str:
        return _call_openai_compat(self._host, self._path, self._api_key,
                                   self._model, prompt, temperature=0.8,
                                   tag='DeepSeekChat')


class DeepSeekReflectorLLM(sampler.LLM):
    """Same provider as DeepSeekChatLLM, lower temperature, used for
    reflection / knowledge extraction where determinism helps."""

    def __init__(self, samples_per_prompt: int):
        super().__init__(samples_per_prompt)
        self._api_key = os.getenv('DEEPSEEK_API_KEY', '')
        self._host = 'api.deepseek.com'
        self._path = '/v1/chat/completions'
        self._model = 'deepseek-chat'

    def _call_api(self, prompt: str) -> str:
        return _call_openai_compat(self._host, self._path, self._api_key,
                                   self._model, prompt, temperature=0.3,
                                   tag='DeepSeekRef')


def _openai_base_to_host_path(api_base: str) -> tuple[str, str]:
    """Resolves env OPENAI_API_BASE to (host, path) for *HTTPS* POST.

    Examples:
    - https://api.openai.com  ->  api.openai.com, /v1/chat/completions
    - https://api.bltcy.ai/v1 ->  api.bltcy.ai, /v1/chat/completions
    """
    base = (api_base or 'https://api.openai.com').strip().rstrip('/')
    if not base.lower().startswith(('http://', 'https://')):
        base = 'https://' + base
    u = urlparse(base)
    host = u.netloc or 'api.openai.com'
    pfx = (u.path or '').rstrip('/')
    if not pfx:
        pfx = '/v1'
    if not pfx.startswith('/'):
        pfx = '/' + pfx
    path = f'{pfx}/chat/completions'
    return host, path


class OpenAICoderLLM(sampler.LLM):
    """OpenAI-compatible coder: official api.openai.com or any relay (course key).

    Env:
      OPENAI_API_KEY       — required
      OPENAI_API_BASE      — default https://api.openai.com; set to e.g.
                             https://api.bltcy.ai  for a third-party gateway
      OPENAI_CODER_MODEL / OPENAI_MODEL — default gpt-4o-mini; course notes use gpt-5-nano
    """

    def __init__(self, samples_per_prompt: int):
        super().__init__(samples_per_prompt)
        self._api_key = os.getenv('OPENAI_API_KEY', '')
        base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com')
        self._host, self._path = _openai_base_to_host_path(base)
        self._model = (
            os.getenv('OPENAI_CODER_MODEL') or os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        )

    def _call_api(self, prompt: str) -> str:
        return _call_openai_compat(
            self._host, self._path, self._api_key, self._model, prompt,
            temperature=0.8, tag='OpenAICoder')


class OpenAIReflectorLLM(sampler.LLM):
    """Reflector: same base URL, lower temperature, optionally different model."""

    def __init__(self, samples_per_prompt: int):
        super().__init__(samples_per_prompt)
        self._api_key = os.getenv('OPENAI_API_KEY', '')
        base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com')
        self._host, self._path = _openai_base_to_host_path(base)
        self._model = os.getenv('OPENAI_REFLECTOR_MODEL', '')
        if not self._model:
            self._model = (
                os.getenv('OPENAI_CODER_MODEL') or os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
            )

    def _call_api(self, prompt: str) -> str:
        return _call_openai_compat(
            self._host, self._path, self._api_key, self._model, prompt,
            temperature=0.3, tag='OpenAIRef')


PROVIDERS = {
    'qwen':     (QwenCoderLLM,     QwenLLM),
    'deepseek': (DeepSeekChatLLM,  DeepSeekReflectorLLM),
    'openai':   (OpenAICoderLLM,  OpenAIReflectorLLM),
}


# =====================================================================
# Shared OpenAI-compatible HTTP caller with bounded retries + tolerant
# SSL. Returns '' on permanent failure so the sampler can keep going
# instead of hanging the whole experiment.
#
# Per-(host, model) capability cache. Some OpenAI-compatible relays expose
# reasoning models (gpt-5-nano, o1-mini, ...) that reject `temperature`
# and require `max_completion_tokens` instead of `max_tokens`. We detect
# the rejection from the first 400 response and remember it so subsequent
# calls don't waste a round trip.
# =====================================================================
_PARAM_CAPS: dict[tuple[str, str], dict[str, bool]] = {}


def _call_openai_compat(host: str, path: str, api_key: str, model: str,
                        prompt: str, temperature: float, tag: str) -> str:
    cap_key = (host, model)
    cap = _PARAM_CAPS.setdefault(
        cap_key, {'skip_temperature': False, 'use_max_completion_tokens': False})

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    # Output-token budget. Has to leave room for both the visible answer
    # AND the hidden reasoning tokens that gpt-5-nano / o1-* / o3-* emit
    # internally before any user-visible content. With the previous 1024
    # cap, gpt-5-nano routinely burned ~600 reasoning + ~600 content on
    # an EoH prompt, blew through the budget, got finish_reason='length'
    # and returned content='' — which presented as `raw0=0 chars`.
    # 8192 is safely larger than any priority() function we ever expect
    # and still well below the 16k hard limit of all current providers.
    output_token_budget = 8192
    last_err = None
    extra_attempts_for_param_fix = 0
    for attempt in range(_MAX_API_RETRIES):
        body: dict = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
        }
        if cap['use_max_completion_tokens']:
            body['max_completion_tokens'] = output_token_budget
        else:
            body['max_tokens'] = output_token_budget
        if not cap['skip_temperature']:
            body['temperature'] = temperature
        payload = json.dumps(body)

        try:
            conn = http.client.HTTPSConnection(
                host, timeout=60, context=_SSL_CTX)
            conn.request('POST', path, payload, headers)
            res = conn.getresponse()
            data = res.read().decode('utf-8', errors='replace')
            try:
                conn.close()
            except Exception:
                pass

            if res.status == 200:
                parsed = json.loads(data)
                if 'choices' in parsed:
                    return parsed['choices'][0]['message']['content']
                last_err = f'no-choices: {data[:200]}'
                logging.warning("%s unexpected response (attempt %d/%d): %s",
                                tag, attempt + 1, _MAX_API_RETRIES, data[:200])
            else:
                last_err = f'HTTP {res.status}: {data[:200]}'
                low = data.lower()
                # Detect "temperature unsupported" / "max_tokens unsupported"
                # (typical messages from gpt-5-nano / o1-* reasoning models)
                changed = False
                if (res.status in (400, 422)
                        and not cap['skip_temperature']
                        and 'temperature' in low
                        and ('unsupported' in low or 'not support' in low
                             or "does not support" in low)):
                    cap['skip_temperature'] = True
                    changed = True
                    logging.info(
                        "%s: model '%s' rejects temperature; retrying without it.",
                        tag, model)
                if (res.status in (400, 422)
                        and not cap['use_max_completion_tokens']
                        and 'max_tokens' in low
                        and ('unsupported' in low or 'not support' in low
                             or "does not support" in low
                             or 'max_completion_tokens' in low)):
                    cap['use_max_completion_tokens'] = True
                    changed = True
                    logging.info(
                        "%s: model '%s' wants max_completion_tokens; switching.",
                        tag, model)
                if changed and extra_attempts_for_param_fix < 2:
                    # Don't burn a real retry slot on a parameter mismatch:
                    # immediately re-issue with corrected payload, no backoff.
                    extra_attempts_for_param_fix += 1
                    continue
                logging.warning("%s API error %d (attempt %d/%d): %s",
                                tag, res.status, attempt + 1,
                                _MAX_API_RETRIES, data[:200])
        except Exception as e:
            last_err = f'{type(e).__name__}: {e}'
            logging.warning("%s request failed (attempt %d/%d): %s",
                            tag, attempt + 1, _MAX_API_RETRIES, last_err)

        if attempt < _MAX_API_RETRIES - 1:
            time.sleep(_API_BACKOFF_SECONDS[
                min(attempt, len(_API_BACKOFF_SECONDS) - 1)])

    logging.error("%s gave up after %d attempts. Last error: %s",
                  tag, _MAX_API_RETRIES, last_err)
    return ''


# =====================================================================
# 3. Sandbox Implementation (reused from original notebook)
# =====================================================================
class Sandbox(evaluator.Sandbox):
    """Sandbox that runs generated code in a subprocess."""

    DEFAULT_NUMBA_ACCELERATE = True

    def __init__(self, verbose=False, numba_accelerate=None):
        self._verbose = verbose
        if numba_accelerate is None:
            numba_accelerate = self.DEFAULT_NUMBA_ACCELERATE
        self._numba_accelerate = bool(numba_accelerate)

    def run(self, program, function_to_run, function_to_evolve,
            inputs, test_input, timeout_seconds, **kwargs):
        dataset = inputs[test_input]
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self._compile_and_run_function,
            args=(program, function_to_run, function_to_evolve,
                  dataset, self._numba_accelerate, result_queue)
        )
        process.start()
        process.join(timeout=timeout_seconds)

        if process.is_alive():
            process.terminate()
            process.join()
            return None, False, 'Execution timed out'

        if result_queue.qsize() != 0:
            item = result_queue.get_nowait()
            if len(item) == 3:
                return item
            return item[0], item[1], None
        return None, False, 'No result produced'

    def _compile_and_run_function(self, program, function_to_run,
                                  function_to_evolve, dataset,
                                  numba_accelerate, result_queue):
        try:
            if numba_accelerate:
                program = evaluator_accelerate.add_numba_decorator(
                    program=program, function_to_evolve=function_to_evolve)
            all_globals_namespace = {}
            exec(program, all_globals_namespace)
            function_to_run = all_globals_namespace[function_to_run]
            results = function_to_run(dataset)
            if not isinstance(results, (int, float)):
                result_queue.put((None, False, f'Non-numeric result: {type(results)}'))
                return
            result_queue.put((results, True, None))
        except Exception as e:
            result_queue.put((None, False, str(e)))


# =====================================================================
# 4. Specification (same as original)
# =====================================================================
specification = r'''
import numpy as np


def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns indices of bins in which item can fit."""
    return np.nonzero((bins - item) >= 0)[0]


def online_binpack(
        items: tuple[float, ...], bins: np.ndarray
) -> tuple[list[list[float, ...], ...], np.ndarray]:
    """Performs online binpacking of `items` into `bins`."""
    packing = [[] for _ in bins]
    for item in items:
        valid_bin_indices = get_valid_bin_indices(item, bins)
        priorities = priority(item, bins[valid_bin_indices])
        best_bin = valid_bin_indices[np.argmax(priorities)]
        bins[best_bin] -= item
        packing[best_bin].append(item)
    packing = [bin_items for bin_items in packing if bin_items]
    return packing, bins


@funsearch.run
def evaluate(instances: dict) -> float:
    """Evaluate heuristic function on a set of online binpacking instances."""
    num_bins = []
    for name in instances:
        instance = instances[name]
        capacity = instance['capacity']
        items = instance['items']
        bins = np.array([capacity for _ in range(instance['num_items'])])
        _, bins_packed = online_binpack(items, bins)
        num_bins.append((bins_packed != capacity).sum())
    return -np.mean(num_bins)


@funsearch.evolve
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.

    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.

    Return:
        Array of same size as bins with priority score of each bin.
    """
    # Best Fit baseline: prefer the bin with the smallest remaining
    # capacity that still fits. This is a *trivial textbook* heuristic
    # (used as a baseline in both FunSearch and EoH papers); we use it
    # as the registered sample_0 only because the framework needs a
    # working `priority` body for signature extraction. The real
    # "initial population" is the N i1-generated samples that follow,
    # mirroring vanilla EoH's get_prompt_i1 init phase.
    return -(bins - item)
'''


# =====================================================================
# 5. End-of-run benchmark (defined BEFORE the __main__ block so it is
#    visible inside the `finally:` clause below)
# =====================================================================

def _emit_benchmark_table(log_dir: str, dataset_name: str = 'OR3') -> None:
    """Locate the most recent samples directory and append the standard
    Avg/Std/Min/Max/L1/Gap table for the best heuristic + 5 classical
    baselines. Imported lazily so failures here don't break startup.

    `dataset_name` controls which key in `bin_packing_utils.datasets` is
    benchmarked against — defaults to 'OR3' for backward compatibility but
    can be set to e.g. 'Weibull 5k' so we get apples-to-apples numbers
    comparable with the EoH and FunSearch papers.
    """
    import bench_heuristic
    samples_dir = Path(log_dir) / 'samples'
    # Accept both file-naming conventions (legacy `samples_<order>.json` and
    # current `sample_<write_seq>.json`); a single `sample*.json` glob covers
    # both because the legacy name also starts with `sample`.
    if not samples_dir.exists() or not any(samples_dir.glob('sample*.json')):
        print('\n[bench] No samples to benchmark - skipping final table.')
        return

    rows = bench_heuristic.bench_all_classicals(dataset_name)
    best_path = bench_heuristic.find_best_sample_in_dir(samples_dir)
    if best_path is not None:
        try:
            fn, sample = bench_heuristic.load_priority_from_sample_json(
                best_path)
            label = (
                f"Ours (best heuristic) "
                f"[sample #{sample.get('sample_order','?')}, "
                f"score {sample.get('score','?')}]")
            rows.append(bench_heuristic.bench_priority_fn(
                fn, dataset_name, label=label))
        except Exception as e:
            print(f'\n[bench] failed to load best sample {best_path.name}: {e!r}')

    num_instances = len(bin_packing_utils.datasets[dataset_name])
    print('\n' + '=' * 80)
    print(f'Per-instance benchmark on {dataset_name} ({num_instances} instances)')
    print('=' * 80)
    print(bench_heuristic.format_table_text(rows))
    print('=' * 80)
    print('Markdown version (paste into report):')
    print(bench_heuristic.format_table_md(rows))
    print('=' * 80)


# =====================================================================
# 6. Main
# =====================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Super-FunSearch on 1D online Bin Packing.')
    # Pulled from `bin_packing_utils.datasets` keys at parse time so any
    # dataset added there becomes selectable on the CLI without code edits.
    available_datasets = sorted(bin_packing_utils.datasets.keys())

    def _parse_dataset_name(raw: str) -> str:
        """Resolve user input → canonical key in `bin_packing_utils.datasets`.

        Accepts case-insensitive aliases like `weibull`, `weibull5k`,
        `or3`, `or-3` so PowerShell users don't have to quote the
        space-bearing canonical name `'Weibull 5k'`.
        """
        s = (raw or '').strip()
        # exact match wins
        if s in bin_packing_utils.datasets:
            return s
        norm = s.lower().replace('-', '').replace('_', '').replace(' ', '')
        alias_map = {
            'or3': 'OR3',
            'orlib': 'OR3',
            'orlibrary': 'OR3',
            'weibull': 'Weibull 5k',
            'weibull5k': 'Weibull 5k',
            'weibull5000': 'Weibull 5k',
        }
        canonical = alias_map.get(norm)
        if canonical and canonical in bin_packing_utils.datasets:
            return canonical
        valid = ', '.join(repr(k) for k in available_datasets) \
            + ' (aliases: or3, weibull, weibull5k)'
        raise argparse.ArgumentTypeError(
            f"invalid dataset {raw!r}; expected one of: {valid}")

    parser.add_argument('--dataset', type=_parse_dataset_name, default='OR3',
                        metavar='|'.join(available_datasets + ['or3', 'weibull']),
                        help='Which benchmark instance set to evolve against. '
                             "Default 'OR3' (20 OR-Library instances, 500 "
                             "items each, capacity 150). 'Weibull 5k' = 5 "
                             "synthetic instances with 5,000 items, capacity "
                             "100 — this is the dataset the EoH and FunSearch "
                             "papers report on, so use it whenever you need "
                             "apples-to-apples numbers comparable with their "
                             "published gap-to-LB figures. Case-insensitive "
                             "aliases accepted: 'or3', 'weibull', 'weibull5k'.")
    parser.add_argument('--max-samples', type=int, default=400,
                        help='Total LLM samples to draw (default: 400).')
    parser.add_argument('--num-islands', type=int, default=4,
                        help='Number of islands; with a small budget, '
                             'too many islands gives <1 sample/island and '
                             'evolution never gets a chance (default: 4).')
    parser.add_argument('--samples-per-prompt', type=int, default=2,
                        help='Samples per LLM call (default: 2).')
    parser.add_argument('--reset-period', type=int, default=600,
                        help='Seconds between island resets. Default 600s '
                             '(10min) so a few-hundred-sample run sees 2-3 '
                             'resets and the diversity mechanism actually '
                             'kicks in. Original code shipped with 4*60*60 '
                             '(4h) which never fires for short experiments. '
                             'IGNORED if --reset-every-n-samples > 0.')
    parser.add_argument('--reset-every-n-samples', type=int, default=0,
                        help='Reset the weakest half of islands every N '
                             'successfully-registered samples. Recommended '
                             'for short runs (e.g. 50 with a 500-sample '
                             'budget = 10 resets, deterministic w.r.t. the '
                             'budget). Set to 0 to fall back to the '
                             'time-based --reset-period.')
    parser.add_argument('--init-population', type=int, default=20,
                        help='EoH-style init phase: force the first N LLM '
                             'samples to use the i1 (initialisation) prompt, '
                             'mirroring EoH\'s vanilla flow that builds an '
                             'initial population of N from-scratch '
                             'heuristics before any crossover/mutation. '
                             'Default 20 = EoH paper\'s pop_size. Set to 0 '
                             'to disable (then evolution starts immediately '
                             'using the seed in `specification`).')
    parser.add_argument('--warm-start-samples-dir', default='',
                        help='Optional path to a previous run\'s samples '
                             'directory. Valid saved functions are re-'
                             'evaluated and registered into the database '
                             'before sampling starts. Use with '
                             '--init-population 0 to skip repeated i1 init.')
    parser.add_argument('--warm-start-top-k', type=int, default=0,
                        help='When warm-starting, keep only the top K saved '
                             'valid samples by score. 0 means all valid '
                             'samples in the directory.')
    parser.add_argument('--no-warm-start-round-robin', action='store_true',
                        help='By default warm-start samples are registered '
                             'round-robin across islands. Pass this flag to '
                             'register each warm-start sample into all '
                             'islands instead.')
    parser.add_argument('--no-reflection', action='store_true',
                        help='Disable Reflector / KB / Extractor for an '
                             'apples-to-apples original-FunSearch baseline.')
    parser.add_argument('--no-eoh-operators', action='store_true',
                        help='Disable the EoH-style multi-operator generator '
                             'and fall back to the v2 two-step thought->code '
                             'pipeline.')
    parser.add_argument('--no-error-memory', action='store_true',
                        help='Disable LLAMEA-style error reinjection. By '
                             'default, the most recent runtime errors are '
                             'rendered as a "## Recent failures to avoid" '
                             'block at the top of every EoH operator '
                             'prompt so the LLM stops repeating the same '
                             'shape/axis/dtype mistakes. Pass this flag '
                             'for ablation runs.')
    parser.add_argument('--no-numba', action='store_true',
                        help='Disable numba acceleration in the sandbox. '
                             'Useful for fair EoH/FunSearch-aligned runs '
                             'where prompt-side numba constraints should '
                             'not affect which LLM-generated heuristics are '
                             'considered valid.')
    parser.add_argument('--error-memory-capacity', type=int, default=5,
                        help='How many recent runtime errors to keep in '
                             'the avoidance block (default: 5). Larger = '
                             'broader memory but more prompt bloat.')
    parser.add_argument('--no-reevo-reflector', action='store_true',
                        help='Disable the ReEvo-style verbal reflector. '
                             'When enabled (default), short-term pairwise '
                             'reflections are spliced into e1/e2 prompts '
                             'and an accumulated long-term reflection '
                             'into m1/m2/m3 prompts, mimicking '
                             'verbal-gradient evolution. Pass this flag '
                             'for ablation runs.')
    parser.add_argument('--reevo-lt-update-period', type=int, default=10,
                        help='How many registered samples between long-'
                             'term reflection updates (default: 10). '
                             'Drives the cadence at which short-term '
                             'reflections are distilled into the LT hint '
                             'used by mutation operators.')
    parser.add_argument('--search-controller', action='store_true',
                        help='Enable A4 LLM Search-Controller. A4 reads '
                             'recent trajectory events and emits bounded '
                             'JSON scheduling policy for operator groups '
                             'and parent-source sampling; it never writes '
                             'code or formulas.')
    parser.add_argument('--search-controller-horizon', type=int, default=15,
                        help='Default A4 policy horizon in samples '
                             '(default: 15).')
    parser.add_argument('--search-controller-min-events', type=int, default=6,
                        help='Minimum trajectory events before A4 may call '
                             'the LLM for a policy (default: 6).')
    parser.add_argument('--provider', choices=tuple(PROVIDERS.keys()),
                        default='deepseek',
                        help='LLM provider: qwen (DashScope), deepseek (default), '
                             'openai (OPENAI_API_KEY + OPENAI_API_BASE, OpenAI-compatible).')
    parser.add_argument(
        '--openai-api-base', default=None,
        help='Only for --provider openai: sets OPENAI_API_BASE for this run '
             '(e.g. https://api.openai.com or https://api.bltcy.ai). '
             'Overrides .env.')
    parser.add_argument(
        '--openai-coder-model', default=None,
        help='Only for --provider openai: overrides OPENAI_CODER_MODEL / OPENAI_MODEL.')
    parser.add_argument(
        '--openai-reflector-model', default=None,
        help='Only for --provider openai: overrides OPENAI_REFLECTOR_MODEL.')
    parser.add_argument('--smoke', action='store_true',
                        help='Tiny 8-sample run for end-to-end sanity check.')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    if args.smoke:
        args.max_samples = 8
        args.num_islands = 2
        args.samples_per_prompt = 2

    if args.openai_api_base:
        os.environ['OPENAI_API_BASE'] = args.openai_api_base
    if args.openai_coder_model:
        os.environ['OPENAI_CODER_MODEL'] = args.openai_coder_model
    if args.openai_reflector_model:
        os.environ['OPENAI_REFLECTOR_MODEL'] = args.openai_reflector_model

    Sandbox.DEFAULT_NUMBA_ACCELERATE = not args.no_numba

    bin_packing_inputs = {
        args.dataset: bin_packing_utils.datasets[args.dataset]
    }

    coder_cls, reflector_cls = PROVIDERS[args.provider]
    class_config = config.ClassConfig(
        llm_class=coder_cls,
        sandbox_class=Sandbox,
        reflector_llm_class=reflector_cls,
    )

    experiment_config = config.Config(
        programs_database=config.ProgramsDatabaseConfig(
            num_islands=args.num_islands,
            reset_period=args.reset_period,
            reset_period_samples=args.reset_every_n_samples,
        ),
        samples_per_prompt=args.samples_per_prompt,
        num_samplers=1,
        num_evaluators=1,
        evaluate_timeout_seconds=30,
    )

    reflector_config = config.ReflectorConfig(
        enable_reflection=not args.no_reflection,
        max_fix_attempts=2,
    )

    kb_config = config.KnowledgeBaseConfig(
        seed_path=os.path.join(os.path.dirname(__file__), 'seed_knowledge.json'),
        persist_path=os.path.join(os.path.dirname(__file__), 'knowledge_runtime.json'),
        embedding_model='all-MiniLM-L6-v2',
        similarity_threshold=0.6,
    )

    reset_desc = (f'every {args.reset_every_n_samples} samples'
                  if args.reset_every_n_samples > 0
                  else f'every {args.reset_period}s')
    error_memory_desc = (f'cap={args.error_memory_capacity}'
                         if not args.no_error_memory else 'off')
    reevo_desc = (f'lt_period={args.reevo_lt_update_period}'
                  if not args.no_reevo_reflector else 'off')
    a4_desc = (f'horizon={args.search_controller_horizon},'
               f'min_events={args.search_controller_min_events}'
               if args.search_controller else 'off')
    logging.info(
        'Launching Super-FunSearch | provider=%s dataset=%s samples=%d '
        'islands=%d spp=%d reset=%s init_pop=%d warm_start=%s reflection=%s '
        'eoh_operators=%s err_mem=%s reevo=%s a4=%s numba=%s',
        args.provider, args.dataset, args.max_samples, args.num_islands,
        args.samples_per_prompt, reset_desc, args.init_population,
        (args.warm_start_samples_dir or 'off'),
        not args.no_reflection, not args.no_eoh_operators,
        error_memory_desc, reevo_desc, a4_desc,
        Sandbox.DEFAULT_NUMBA_ACCELERATE,
    )

    # Each invocation writes to its own timestamped subdirectory so that
    # `samples/samples_*.json` and the auto-bench at the end of the run
    # only see the heuristics produced by *this* run. Without this guard
    # `find_best_sample_in_dir()` was happily reading samples from older
    # 150-sample runs and reporting them as the "best heuristic" of a
    # fresh 30-sample smoke test, which made A/B comparisons impossible.
    base_log_dir = os.path.join(
        os.path.dirname(__file__), '..', 'logs', 'super_funsearch')
    run_id = time.strftime('run_%Y%m%d_%H%M%S')
    log_dir = os.path.join(base_log_dir, run_id)
    os.makedirs(log_dir, exist_ok=True)
    logging.info('Run output directory: %s', log_dir)
    try:
        funsearch.main(
            specification=specification,
            inputs=bin_packing_inputs,
            config=experiment_config,
            max_sample_nums=args.max_samples,
            class_config=class_config,
            log_dir=log_dir,
            reflector_config=reflector_config,
            kb_config=kb_config,
            domain_id='PROB_BIN_PACKING_1D',
            use_eoh_operators=not args.no_eoh_operators,
            init_population_size=args.init_population,
            warm_start_samples_dir=args.warm_start_samples_dir,
            warm_start_top_k=args.warm_start_top_k,
            warm_start_round_robin=not args.no_warm_start_round_robin,
            enable_error_memory=not args.no_error_memory,
            error_memory_capacity=args.error_memory_capacity,
            enable_reevo_reflector=not args.no_reevo_reflector,
            reevo_lt_update_period=args.reevo_lt_update_period,
            enable_search_controller=args.search_controller,
            search_controller_horizon=args.search_controller_horizon,
            search_controller_min_events=args.search_controller_min_events,
        )
    finally:
        # Always print the standard benchmark table at the end of a run,
        # even if the search was interrupted (Ctrl+C, API quota, ...). The
        # report goes to stdout so it shows up in the tee'd log.
        try:
            _emit_benchmark_table(log_dir, dataset_name=args.dataset)
        except Exception as e:
            logging.warning('Auto-bench at end of run failed: %r', e)
