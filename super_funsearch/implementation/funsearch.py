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

import json
import logging
import os
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
from implementation import error_memory as error_memory_lib
from implementation import reevo_reflector as reevo_reflector_lib
from implementation import search_controller as search_controller_lib


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


def _load_warm_start_records(
        samples_dir: str,
        *,
        top_k: int = 0,
) -> list[dict]:
    """Load valid sampled functions from a previous run's samples directory.

    The records are sorted by score descending (higher is better because scores
    are negative bin counts).  ``top_k <= 0`` means keep all valid records.
    """
    if not samples_dir:
        return []
    if not os.path.isdir(samples_dir):
        raise FileNotFoundError(f'warm-start samples dir not found: {samples_dir}')

    records: list[dict] = []
    for fname in sorted(os.listdir(samples_dir)):
        if not (fname.endswith('.json') and fname.startswith('sample_')):
            continue
        path = os.path.join(samples_dir, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                rec = json.load(f)
        except Exception as e:
            logging.warning('Skipping warm-start file %s: %s', path, e)
            continue
        if rec.get('score') is None or not rec.get('function'):
            continue
        try:
            rec['_score_float'] = float(rec['score'])
        except (TypeError, ValueError):
            continue
        rec['_source_file'] = path
        records.append(rec)

    records.sort(key=lambda r: r['_score_float'], reverse=True)
    if top_k and top_k > 0:
        records = records[:top_k]
    return records


def _body_from_record_function(function_text: str) -> str:
    """Extract the function body from a saved full ``def priority`` string."""
    fn = code_manipulation.text_to_function(function_text)
    return fn.body


def _register_warm_start_population(
        *,
        warm_start_samples_dir: str,
        warm_start_top_k: int,
        warm_start_round_robin: bool,
        evaluators: Sequence[evaluator.Evaluator],
        num_islands: int,
        profiler: profile.Profiler,
) -> int:
    """Re-evaluate saved samples and register valid ones into the database."""
    records = _load_warm_start_records(
        warm_start_samples_dir, top_k=warm_start_top_k)
    if not records:
        logging.warning(
            'Warm-start requested but no valid samples were loaded from %s',
            warm_start_samples_dir)
        return 0

    registered = 0
    for i, rec in enumerate(records):
        try:
            body = _body_from_record_function(rec['function'])
        except Exception as e:
            logging.warning(
                'Skipping warm-start sample %s: cannot parse function: %s',
                rec.get('_source_file'), e)
            continue
        island_id = (i % num_islands) if warm_start_round_robin else None
        result = evaluators[0].analyse(
            body,
            island_id=island_id,
            version_generated=None,
            thought=rec.get('thought'),
            profiler=profiler,
            global_sample_nums=-(i + 1),
            sample_time=0.0,
        )
        if result.is_valid and result.registered:
            registered += 1
            logging.info(
                'Warm-start registered #%d score=%s island=%s source=%s',
                registered, result.reduced_score, island_id,
                rec.get('_source_file'))
        else:
            logging.info(
                'Warm-start rejected source=%s error=%s',
                rec.get('_source_file'), (result.error_trace or '')[:160])
    logging.info(
        'Warm-start population loaded: %d/%d valid samples from %s',
        registered, len(records), warm_start_samples_dir)
    return registered


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

    # EoH config (passed through from runner)
    use_eoh_operators = kwargs.get('use_eoh_operators', True)
    eoh_task_spec = kwargs.get('eoh_task_spec', None)
    eoh_operator_weights = kwargs.get('eoh_operator_weights', None)
    init_population_size = kwargs.get('init_population_size', 0)
    warm_start_samples_dir = kwargs.get('warm_start_samples_dir', '') or ''
    warm_start_top_k = int(kwargs.get('warm_start_top_k', 0) or 0)
    warm_start_round_robin = bool(kwargs.get('warm_start_round_robin', True))

    if warm_start_samples_dir:
        _register_warm_start_population(
            warm_start_samples_dir=warm_start_samples_dir,
            warm_start_top_k=warm_start_top_k,
            warm_start_round_robin=warm_start_round_robin,
            evaluators=evaluators,
            num_islands=config.programs_database.num_islands,
            profiler=profiler,
        )

    # LLAMEA-style error memory: shared by all samplers so failures from
    # any of them help every other one. Pass capacity via runner kwargs;
    # capacity=0 (or enable_error_memory=False) disables for ablation.
    enable_error_memory = kwargs.get('enable_error_memory', True)
    error_memory_capacity = int(kwargs.get('error_memory_capacity', 5))
    error_memory_obj = (
        error_memory_lib.ErrorMemory(capacity=error_memory_capacity)
        if enable_error_memory and error_memory_capacity > 0 else None
    )

    # ReEvo verbal-gradient reflector. Single instance shared across
    # samplers so the long-term reflection accumulates evidence from
    # *all* parallel search trajectories, not just one. Uses the
    # reflector_llm class when available (lower temperature, separate
    # endpoint), falls back to the generator's LLM otherwise.
    enable_reevo = kwargs.get('enable_reevo_reflector', True)
    reevo_lt_period = int(kwargs.get('reevo_lt_update_period', 10))
    reevo_obj = None
    if enable_reevo:
        reevo_llm_class = (
            class_config.reflector_llm_class or class_config.llm_class)
        try:
            reevo_obj = reevo_reflector_lib.ReevoReflector(
                llm_class=reevo_llm_class,
                problem_meta=reevo_reflector_lib.BIN_PACKING_PROBLEM_META,
            )
        except Exception as e:
            # Don't crash the whole run if prompts are missing — log and
            # continue with reflector disabled. The CLI flag exists
            # precisely to skip this; this branch only fires for unusual
            # filesystem layouts.
            import logging
            logging.warning(
                'ReevoReflector init failed (%s); running without it.', e)
            reevo_obj = None

    # A4 Search-Controller. It uses the reflector LLM endpoint when available
    # and only emits bounded scheduling JSON (operator group + parent bias).
    enable_a4 = bool(kwargs.get('enable_search_controller', False))
    a4_obj = None
    if enable_a4:
        a4_llm_class = class_config.reflector_llm_class or class_config.llm_class
        try:
            a4_obj = search_controller_lib.SearchController(
                llm_class=a4_llm_class,
                event_capacity=int(kwargs.get('search_controller_events', 30)),
                min_events_before_policy=int(
                    kwargs.get('search_controller_min_events', 6)),
                default_horizon=int(
                    kwargs.get('search_controller_horizon', 15)),
            )
        except Exception as e:
            logging.warning(
                'SearchController init failed (%s); running without A4.', e)
            a4_obj = None

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
            use_eoh_operators=use_eoh_operators,
            eoh_task_spec=eoh_task_spec,
            eoh_operator_weights=eoh_operator_weights,
            init_population_size=init_population_size,
            error_memory=error_memory_obj,
            reevo_reflector=reevo_obj,
            reevo_lt_update_period=reevo_lt_period,
            search_controller=a4_obj,
        )
        for _ in range(config.num_samplers)
    ]

    for s in samplers:
        s.sample(profiler=profiler)
