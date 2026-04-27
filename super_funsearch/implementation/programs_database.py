# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A programs database that implements the evolutionary algorithm."""
from __future__ import annotations

import profile
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import time
from typing import Any, Tuple, Mapping

from absl import logging
import numpy as np
import scipy

from implementation import code_manipulation
from implementation import config as config_lib
from implementation import structure_analysis

# RZ: I change the original code "tuple[float, ...]" to "Tuple[float, ...]"
Signature = Tuple[float, ...]
ClusterKey = tuple[Signature, str, bool]
_PARENT_WEIGHT_CATASTROPHIC = 0.05
_PARENT_WEIGHT_RISKY = 0.20
_PARENT_WEIGHT_BF_SATURATED = 0.60
_PARENT_WEIGHT_PROMISING_NON_BF = 1.25

# RZ: the code is also incorrect
# We should use typing.Mapping rather than abc.Mapping
ScoresPerTest = Mapping[Any, float]


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Returns the tempered softmax of 1D finite `logits`."""
    if not np.all(np.isfinite(logits)):
        non_finites = set(logits[~np.isfinite(logits)])
        raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)

    result = scipy.special.softmax(logits / temperature, axis=-1)
    # Ensure that probabilities sum to 1 to prevent error in `np.random.choice`.
    index = np.argmax(result)
    result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1:])
    return result


def _reduce_score(scores_per_test: ScoresPerTest) -> float:
    """Reduces per-test scores into a single score.
    """
    # TODO RZ: change the code to average the score of each test.
    # return scores_per_test[list(scores_per_test.keys())[-1]]
    test_scores = [scores_per_test[k] for k in scores_per_test.keys()]
    return sum(test_scores) / len(test_scores)


def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
    """Represents test scores as a canonical signature."""
    return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))


def _sampling_multiplier(
        cluster: 'Cluster',
        best_score: float,
        search_policy: Any | None = None,
) -> float:
    """Parent-sampling multiplier for bounded diversity.

    Bad structures are still stored in the island and remain visible to A3 via
    logs/outcomes, but they should not reproduce as often as near-frontier
    non-BF variants.  This multiplier is applied on top of score softmax by
    adding ``temperature * log(multiplier)`` to the logits.
    """
    score = cluster.score
    if not np.isfinite(score) or not np.isfinite(best_score):
        return 1.0

    scale = max(10.0, abs(best_score))
    catastrophic_gap = max(100.0, 0.25 * scale)
    close_gap = max(10.0, 0.05 * scale)

    multiplier = 1.0
    if score < best_score - catastrophic_gap:
        multiplier *= _PARENT_WEIGHT_CATASTROPHIC
    if 'loop' in cluster.structure_tag:
        multiplier *= _PARENT_WEIGHT_RISKY
    if cluster.bf_equivalent and score >= best_score - close_gap:
        multiplier *= _PARENT_WEIGHT_BF_SATURATED
    if (not cluster.bf_equivalent) and score >= best_score - close_gap:
        multiplier *= _PARENT_WEIGHT_PROMISING_NON_BF
    if search_policy is not None and hasattr(search_policy, 'parent_multiplier'):
        try:
            multiplier *= float(search_policy.parent_multiplier(
                structure_tag=cluster.structure_tag,
                bf_equivalent=cluster.bf_equivalent,
                score=score,
                best_score=best_score,
            ))
        except Exception as e:
            logging.warning('[a4] parent multiplier ignored: %s', e)
    return float(max(1e-12, multiplier))


@dataclasses.dataclass(frozen=True)
class Prompt:
    """A prompt produced by the ProgramsDatabase, to be sent to Samplers.

    Attributes:
      code: The prompt, ending with the header of the function to be completed.
      version_generated: The function to be completed is `_v{version_generated}`.
      island_id: Identifier of the island that produced the implementations
         included in the prompt. Used to direct the newly generated implementation
         into the same island.
      thoughts: List of thought strings from the selected parent implementations.
      function_header: The function signature that the LLM should implement.
    """
    code: str
    version_generated: int
    island_id: int
    thoughts: list[str] = dataclasses.field(default_factory=list)
    function_header: str = ''
    # Sorted (worst -> best) parent functions, including their .body and
    # .thought attributes. Needed by EoH-style multi-operator samplers that
    # build prompts from paired (algorithm, code) tuples instead of from the
    # already-assembled `code` string. Empty for the very first prompt.
    parent_implementations: list = dataclasses.field(default_factory=list)


class ProgramsDatabase:
    """A collection of programs, organized as islands."""

    def __init__(
            self,
            config: config_lib.ProgramsDatabaseConfig,
            template: code_manipulation.Program,
            function_to_evolve: str,
    ) -> None:
        self._config: config_lib.ProgramsDatabaseConfig = config
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve

        # Initialize empty islands.
        self._islands: list[Island] = []
        for _ in range(config.num_islands):
            self._islands.append(
                Island(template, function_to_evolve, config.functions_per_prompt,
                       config.cluster_sampling_temperature_init,
                       config.cluster_sampling_temperature_period))
        self._best_score_per_island: list[float] = (
                [-float('inf')] * config.num_islands)
        self._best_program_per_island: list[code_manipulation.Function | None] = (
                [None] * config.num_islands)
        self._best_scores_per_test_per_island: list[ScoresPerTest | None] = (
                [None] * config.num_islands)

        self._last_reset_time: float = time.time()
        # NEW: counts successfully-registered samples since the last reset.
        # Used when config.reset_period_samples > 0 (sample-based resets).
        self._samples_since_last_reset: int = 0

    @property
    def num_islands(self) -> int:
        return len(self._islands)

    def get_prompt(
            self,
            island_id: int | None = None,
            search_policy: Any | None = None,
    ) -> Prompt:
        """Returns a prompt containing implementations from one chosen island."""
        if island_id is None:
            island_id = np.random.randint(len(self._islands))
        else:
            island_id = int(island_id) % len(self._islands)
        code, version_generated, thoughts, function_header, parent_impls = (
            self._islands[island_id].get_prompt(search_policy=search_policy)
        )
        return Prompt(code, version_generated, island_id, thoughts,
                      function_header, parent_impls)

    def _register_program_in_island(
            self,
            program: code_manipulation.Function,
            island_id: int,
            scores_per_test: ScoresPerTest,
            **kwargs  # RZ: add this for profiling
    ) -> None:
        """Registers `program` in the specified island."""
        self._islands[island_id].register_program(program, scores_per_test)
        score = _reduce_score(scores_per_test)
        if score > self._best_score_per_island[island_id]:
            self._best_program_per_island[island_id] = program
            self._best_scores_per_test_per_island[island_id] = scores_per_test
            self._best_score_per_island[island_id] = score
            logging.info('Best score of island %d increased to %s', island_id, score)

        # ======== RZ: profiling ========
        profiler: profile.Profiler = kwargs.get('profiler', None)
        if profiler:
            global_sample_nums = kwargs.get('global_sample_nums', None)
            sample_time = kwargs.get('sample_time', None)
            evaluate_time = kwargs.get('evaluate_time', None)
            program.score = score
            program.global_sample_nums = global_sample_nums
            program.sample_time = sample_time
            program.evaluate_time = evaluate_time
            profiler.register_function(program)

    def register_program(
            self,
            program: code_manipulation.Function,
            island_id: int | None,
            scores_per_test: ScoresPerTest,
            **kwargs  # RZ: add this for profiling
    ) -> None:
        """Registers `program` in the database."""
        # In an asynchronous implementation we should consider the possibility of
        # registering a program on an island that had been reset after the prompt
        # was generated. Leaving that out here for simplicity.
        if island_id is None:
            # This is a program added at the beginning, so adding it to all islands.
            for island_id in range(len(self._islands)):
                self._register_program_in_island(program, island_id, scores_per_test, **kwargs)
        else:
            self._register_program_in_island(program, island_id, scores_per_test, **kwargs)

        # Check whether it is time to reset an island. We support two modes:
        #   1) sample-based (preferred for short runs): fires every
        #      `reset_period_samples` registered programs.
        #   2) time-based (original behaviour): fires every `reset_period`
        #      wall-clock seconds. Only used when reset_period_samples == 0.
        # Note: sample_0 (the seed registered with island_id=None at startup)
        # registers across N islands and counts as N samples here, which is
        # fine — the very first reset still fires after roughly N real LLM
        # samples, not in the middle of init.
        self._samples_since_last_reset += 1
        period_samples = getattr(self._config, 'reset_period_samples', 0) or 0
        if period_samples > 0:
            if self._samples_since_last_reset >= period_samples:
                self._samples_since_last_reset = 0
                self._last_reset_time = time.time()
                logging.info(
                    'Sample-based reset firing (every %d samples).',
                    period_samples)
                self.reset_islands()
        else:
            if time.time() - self._last_reset_time > self._config.reset_period:
                self._last_reset_time = time.time()
                self._samples_since_last_reset = 0
                self.reset_islands()

    def reset_islands(self) -> None:
        """Resets the weaker half of islands."""
        # We sort best scores after adding minor noise to break ties.
        indices_sorted_by_score: np.ndarray = np.argsort(
            self._best_score_per_island +
            np.random.randn(len(self._best_score_per_island)) * 1e-6)
        num_islands_to_reset = self._config.num_islands // 2
        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
        for island_id in reset_islands_ids:
            self._islands[island_id] = Island(
                self._template,
                self._function_to_evolve,
                self._config.functions_per_prompt,
                self._config.cluster_sampling_temperature_init,
                self._config.cluster_sampling_temperature_period)
            self._best_score_per_island[island_id] = -float('inf')
            founder_island_id = np.random.choice(keep_islands_ids)
            founder = self._best_program_per_island[founder_island_id]
            founder_scores = self._best_scores_per_test_per_island[founder_island_id]
            self._register_program_in_island(founder, island_id, founder_scores)


class Island:
    """A sub-population of the programs database."""

    def __init__(
            self,
            template: code_manipulation.Program,
            function_to_evolve: str,
            functions_per_prompt: int,
            cluster_sampling_temperature_init: float,
            cluster_sampling_temperature_period: int,
    ) -> None:
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve
        self._functions_per_prompt: int = functions_per_prompt
        self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self._cluster_sampling_temperature_period = (
            cluster_sampling_temperature_period)

        self._clusters: dict[ClusterKey, Cluster] = {}
        self._num_programs: int = 0

    def register_program(
            self,
            program: code_manipulation.Function,
            scores_per_test: ScoresPerTest,
    ) -> None:
        """Stores a program on this island, in its appropriate cluster."""
        signature = _get_signature(scores_per_test)
        info = structure_analysis.analyze(program)
        setattr(program, 'structure_tag', info.structure_tag)
        setattr(program, 'bf_equivalent', info.bf_equivalent)
        setattr(program, 'structure_summary', info.summary())
        cluster_key: ClusterKey = (
            signature, info.structure_tag, info.bf_equivalent)
        if cluster_key not in self._clusters:
            score = _reduce_score(scores_per_test)
            self._clusters[cluster_key] = Cluster(score, program, info)
        else:
            self._clusters[cluster_key].register_program(program)
        self._num_programs += 1

    def get_prompt(
            self,
            search_policy: Any | None = None,
    ) -> tuple[str, int, list[str], str,
               list[code_manipulation.Function]]:
        """Constructs a prompt containing functions from this island.

        Returns:
            (code, version_generated, thoughts, function_header,
             parent_implementations)
        """
        signatures = list(self._clusters.keys())
        cluster_scores = np.array(
            [self._clusters[signature].score for signature in signatures])

        period = self._cluster_sampling_temperature_period
        temperature = self._cluster_sampling_temperature_init * (
                1 - (self._num_programs % period) / period)
        best_score = float(np.max(cluster_scores))
        multipliers = np.array([
            _sampling_multiplier(
                self._clusters[signature], best_score, search_policy)
            for signature in signatures
        ], dtype=float)
        adjusted_scores = (
            cluster_scores + temperature * np.log(np.maximum(multipliers, 1e-12)))
        probabilities = _softmax(adjusted_scores, temperature)

        functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)

        # If all programs collapse into one structure-aware cluster, still expose
        # multiple implementations from that cluster.  This prevents EoH e1/e2
        # from falling back to i1 solely because equal-score programs were
        # stored together.
        if len(signatures) == 1:
            cluster = self._clusters[signatures[0]]
            implementations = cluster.sample_programs(self._functions_per_prompt)
            scores = [cluster.score] * len(implementations)
            indices = np.argsort(scores)
            sorted_implementations = [implementations[i] for i in indices]
            version_generated = len(sorted_implementations) + 1
            code, function_header = self._generate_prompt(sorted_implementations)
            thoughts = [
                impl.thought for impl in sorted_implementations
                if impl.thought is not None
            ]
            return code, version_generated, thoughts, function_header, list(sorted_implementations)

        nonzero_probs = int(np.count_nonzero(probabilities))
        if nonzero_probs < functions_per_prompt:
            # Very low temperatures can underflow all but the best cluster to
            # exactly zero.  Sampling without replacement then crashes even
            # though enough clusters exist.  Blend in a tiny uniform component
            # so diversity sampling remains possible.
            probabilities = probabilities + 1e-12
            probabilities = probabilities / probabilities.sum()

        idx = np.random.choice(
            len(signatures), size=functions_per_prompt, p=probabilities,
            replace=len(signatures) < functions_per_prompt)
        chosen_signatures = [signatures[i] for i in idx]
        implementations = []
        scores = []
        for signature in chosen_signatures:
            cluster = self._clusters[signature]
            implementations.append(cluster.sample_program())
            scores.append(cluster.score)

        indices = np.argsort(scores)
        sorted_implementations = [implementations[i] for i in indices]
        version_generated = len(sorted_implementations) + 1

        code, function_header = self._generate_prompt(sorted_implementations)
        thoughts = [
            impl.thought for impl in sorted_implementations
            if impl.thought is not None
        ]
        # Pass back deep-copied originals (with their .body and .thought intact).
        # _generate_prompt rewrites names in-place via copy.deepcopy already, so
        # the originals here are still untouched.
        parent_impls = list(sorted_implementations)
        return code, version_generated, thoughts, function_header, parent_impls

    def _generate_prompt(
            self,
            implementations: Sequence[code_manipulation.Function],
    ) -> tuple[str, str]:
        """Creates a prompt containing a sequence of function `implementations`.

        Returns:
            (full_prompt_code, function_header_string)
        """
        implementations = copy.deepcopy(implementations)

        versioned_functions: list[code_manipulation.Function] = []
        for i, implementation in enumerate(implementations):
            new_function_name = f'{self._function_to_evolve}_v{i}'
            implementation.name = new_function_name
            if i >= 1:
                implementation.docstring = (
                    f'Improved version of `{self._function_to_evolve}_v{i - 1}`.')
            implementation = code_manipulation.rename_function_calls(
                str(implementation), self._function_to_evolve, new_function_name)
            versioned_functions.append(
                code_manipulation.text_to_function(implementation))

        next_version = len(implementations)
        new_function_name = f'{self._function_to_evolve}_v{next_version}'
        header = dataclasses.replace(
            implementations[-1],
            name=new_function_name,
            body='',
            docstring=('Improved version of '
                       f'`{self._function_to_evolve}_v{next_version - 1}`.'),
        )
        versioned_functions.append(header)

        prompt = dataclasses.replace(self._template, functions=versioned_functions)
        function_header = str(header).strip()
        return str(prompt), function_header


class Cluster:
    """A cluster of programs on the same island and with the same Signature."""

    def __init__(
            self,
            score: float,
            implementation: code_manipulation.Function,
            structure_info: structure_analysis.StructureInfo | None = None):
        self._score = score
        self._programs: list[code_manipulation.Function] = [implementation]
        self._lengths: list[int] = [len(str(implementation))]
        self._structure_info = structure_info

    @property
    def score(self) -> float:
        """Reduced score of the signature that this cluster represents."""
        return self._score

    @property
    def structure_tag(self) -> str:
        if self._structure_info is None:
            return 'unknown'
        return self._structure_info.structure_tag

    @property
    def bf_equivalent(self) -> bool:
        return bool(self._structure_info and self._structure_info.bf_equivalent)

    @property
    def num_programs(self) -> int:
        return len(self._programs)

    def register_program(self, program: code_manipulation.Function) -> None:
        """Adds `program` to the cluster."""
        self._programs.append(program)
        self._lengths.append(len(str(program)))

    def sample_program(self) -> code_manipulation.Function:
        """Samples a program, giving higher probability to shorther programs."""
        return self.sample_programs(1)[0]

    def sample_programs(self, k: int) -> list[code_manipulation.Function]:
        """Samples up to k programs, favouring shorter implementations."""
        k = max(1, min(int(k), len(self._programs)))
        normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
                max(self._lengths) + 1e-6)
        probabilities = _softmax(-normalized_lengths, temperature=1.0)
        chosen = np.random.choice(
            self._programs, size=k, p=probabilities,
            replace=len(self._programs) < k)
        return list(chosen)
