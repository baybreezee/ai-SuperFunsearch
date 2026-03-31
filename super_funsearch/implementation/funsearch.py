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

"""Super-FunSearch: FunSearch with Reflector, RAG, and Thought-Guided Evolution."""
from __future__ import annotations

from typing import Any, Tuple, Sequence

from implementation import code_manipulation
from implementation import config as config_lib
from implementation import evaluator
from implementation import programs_database
from implementation import sampler
from implementation import profile
from implementation import reflector as reflector_lib
from implementation import knowledge_base as kb_lib
from implementation import knowledge_extractor as ke_lib


def _extract_function_names(specification: str) -> Tuple[str, str]:
    """Returns the name of the function to evolve and of the function to run."""
    run_functions = list(code_manipulation.yield_decorated(
        specification, 'funsearch', 'run'))
    if len(run_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@funsearch.run`.')
    evolve_functions = list(code_manipulation.yield_decorated(
        specification, 'funsearch', 'evolve'))
    if len(evolve_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@funsearch.evolve`.')
    return evolve_functions[0], run_functions[0]


def main(
        specification: str,
        inputs: Sequence[Any],
        config: config_lib.Config,
        max_sample_nums: int | None,
        class_config: config_lib.ClassConfig,
        **kwargs
):
    """Launches a Super-FunSearch experiment.

    Args:
        specification: the boilerplate code for the problem.
        inputs       : the data instances for the problem.
        config       : experiment config.
        max_sample_nums: maximum LLM samples. None = no stop.
        class_config : class types (LLM, Sandbox, optional Reflector LLM).
        **kwargs:
            log_dir: str — directory for profiling logs.
            reflector_config: ReflectorConfig — reflection settings.
            kb_config: KnowledgeBaseConfig — knowledge base settings.
            domain_id: str — problem domain id for knowledge base.
    """
    function_to_evolve, function_to_run = _extract_function_names(specification)
    template = code_manipulation.text_to_program(specification)
    database = programs_database.ProgramsDatabase(
        config.programs_database, template, function_to_evolve)

    # Profiler
    log_dir = kwargs.get('log_dir', None)
    profiler = profile.Profiler(log_dir) if log_dir else None

    # Evaluators
    evaluators = []
    for _ in range(config.num_evaluators):
        evaluators.append(evaluator.Evaluator(
            database, template, function_to_evolve, function_to_run,
            inputs, timeout_seconds=config.evaluate_timeout_seconds,
            sandbox_class=class_config.sandbox_class
        ))

    # Initial evaluation of the seed implementation
    initial = template.get_function(function_to_evolve).body
    evaluators[0].analyse(
        initial, island_id=None, version_generated=None, profiler=profiler)

    # --- New components ---
    reflector_config = kwargs.get(
        'reflector_config', config_lib.ReflectorConfig())
    kb_config = kwargs.get('kb_config', config_lib.KnowledgeBaseConfig())
    domain_id = kwargs.get('domain_id', 'PROB_BIN_PACKING_1D')

    # Reflector (uses a separate LLM class if provided)
    reflector_obj = None
    if reflector_config.enable_reflection:
        reflector_llm = class_config.reflector_llm_class or class_config.llm_class
        reflector_obj = reflector_lib.Reflector(
            llm_class=reflector_llm, config=reflector_config)

    # Knowledge Base
    knowledge_base = kb_lib.KnowledgeBase(config=kb_config)

    # Knowledge Extractor
    extractor_llm = class_config.reflector_llm_class or class_config.llm_class
    extractor_obj = ke_lib.KnowledgeExtractor(llm_class=extractor_llm)

    # Samplers
    samplers = [
        sampler.Sampler(
            database, evaluators, config.samples_per_prompt,
            max_sample_nums=max_sample_nums,
            llm_class=class_config.llm_class,
            reflector=reflector_obj,
            knowledge_base=knowledge_base,
            extractor=extractor_obj,
            reflector_config=reflector_config,
            domain_id=domain_id,
        )
        for _ in range(config.num_samplers)
    ]

    for s in samplers:
        s.sample(profiler=profiler)
