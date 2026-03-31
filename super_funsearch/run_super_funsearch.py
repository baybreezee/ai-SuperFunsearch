"""Entry point for Super-FunSearch on Bin Packing."""
import os
import sys
import json
import http.client
import time
import logging

from dotenv import load_dotenv

# --- Path setup ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

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
# 1. LLM Implementation: Qwen-Coder-Plus for code generation
# =====================================================================
class QwenCoderLLM(sampler.LLM):
    """Qwen-Coder-Plus LLM for thought generation and code generation."""

    def __init__(self, samples_per_prompt: int):
        super().__init__(samples_per_prompt)
        self._api_key = os.getenv('QWEN_API_KEY', '')
        self._host = 'dashscope.aliyuncs.com'
        self._path = '/compatible-mode/v1/chat/completions'
        self._model = 'qwen-coder-plus'

    def _call_api(self, prompt: str) -> str:
        """Single LLM API call to Qwen-Coder-Plus via DashScope."""
        while True:
            try:
                conn = http.client.HTTPSConnection(self._host)
                payload = json.dumps({
                    'max_tokens': 1024,
                    'model': self._model,
                    'temperature': 0.8,
                    'messages': [{'role': 'user', 'content': prompt}],
                })
                headers = {
                    'Authorization': f'Bearer {self._api_key}',
                    'Content-Type': 'application/json',
                }
                conn.request('POST', self._path, payload, headers)
                res = conn.getresponse()
                data = res.read().decode('utf-8')

                if res.status != 200:
                    logging.warning("QwenCoder API error %d: %s", res.status, data[:200])
                    time.sleep(3)
                    continue

                parsed = json.loads(data)
                if 'choices' not in parsed:
                    logging.warning("QwenCoder unexpected response: %s", data[:200])
                    time.sleep(3)
                    continue

                return parsed['choices'][0]['message']['content']
            except Exception as e:
                logging.warning("QwenCoder request failed: %s", e)
                time.sleep(3)
                continue


# =====================================================================
# 2. LLM Implementation: Qwen for reflection / extraction
# =====================================================================
class QwenLLM(sampler.LLM):
    """Qwen-plus LLM via DashScope-compatible OpenAI endpoint."""

    def __init__(self, samples_per_prompt: int):
        super().__init__(samples_per_prompt)
        self._api_key = os.getenv('QWEN_API_KEY', '')
        self._host = 'dashscope.aliyuncs.com'
        self._path = '/compatible-mode/v1/chat/completions'
        self._model = 'qwen-plus'

    def _call_api(self, prompt: str) -> str:
        """Single LLM API call to Qwen via DashScope."""
        while True:
            try:
                conn = http.client.HTTPSConnection(self._host)
                payload = json.dumps({
                    'model': self._model,
                    'max_tokens': 1024,
                    'temperature': 0.7,
                    'messages': [{'role': 'user', 'content': prompt}],
                })
                headers = {
                    'Authorization': f'Bearer {self._api_key}',
                    'Content-Type': 'application/json',
                }
                conn.request('POST', self._path, payload, headers)
                res = conn.getresponse()
                data = res.read().decode('utf-8')

                if res.status != 200:
                    logging.warning("Qwen API error %d: %s", res.status, data[:200])
                    time.sleep(3)
                    continue

                parsed = json.loads(data)
                if 'choices' not in parsed:
                    logging.warning("Qwen unexpected response: %s", data[:200])
                    time.sleep(3)
                    continue

                return parsed['choices'][0]['message']['content']
            except Exception as e:
                logging.warning("Qwen request failed: %s", e)
                time.sleep(3)
                continue


# =====================================================================
# 3. Sandbox Implementation (reused from original notebook)
# =====================================================================
class Sandbox(evaluator.Sandbox):
    """Sandbox that runs generated code in a subprocess."""

    def __init__(self, verbose=False, numba_accelerate=True):
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate

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
    ratios = item / bins
    log_ratios = np.log(ratios)
    priorities = -log_ratios
    return priorities
'''


# =====================================================================
# 5. Main
# =====================================================================
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    bin_packing_or3 = {'OR3': bin_packing_utils.datasets['OR3']}

    class_config = config.ClassConfig(
        llm_class=QwenCoderLLM,
        sandbox_class=Sandbox,
        reflector_llm_class=QwenLLM,
    )

    experiment_config = config.Config(
        samples_per_prompt=4,
        num_samplers=1,
        num_evaluators=1,
        evaluate_timeout_seconds=30,
    )

    reflector_config = config.ReflectorConfig(
        enable_reflection=True,
        max_fix_attempts=2,
    )

    kb_config = config.KnowledgeBaseConfig(
        seed_path=os.path.join(os.path.dirname(__file__), 'seed_knowledge.json'),
        persist_path=os.path.join(os.path.dirname(__file__), 'knowledge_runtime.json'),
        embedding_model='all-MiniLM-L6-v2',
        similarity_threshold=0.6,
    )

    funsearch.main(
        specification=specification,
        inputs=bin_packing_or3,
        config=experiment_config,
        max_sample_nums=20,
        class_config=class_config,
        log_dir=os.path.join(os.path.dirname(__file__), '..', 'logs', 'super_funsearch'),
        reflector_config=reflector_config,
        kb_config=kb_config,
        domain_id='PROB_BIN_PACKING_1D',
    )
