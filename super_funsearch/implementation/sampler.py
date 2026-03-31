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

"""Class for sampling new programs."""
from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Collection, Sequence, Type
import numpy as np
import time
import logging

from implementation import evaluator
from implementation import programs_database


class LLM(ABC):
    """Language model that predicts continuation of provided source code.

    Subclasses must implement _call_api() as the single point of LLM invocation.
    Three generation modes are provided:
      - draw_samples(): original FunSearch code continuation (backward compatible)
      - generate_thought(): produce a natural-language heuristic strategy
      - generate_code_from_thought(): translate a thought into pure Python code
    """

    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    # ------------------------------------------------------------------
    # Core API – subclasses MUST override
    # ------------------------------------------------------------------
    @abstractmethod
    def _call_api(self, prompt: str) -> str:
        """Send a single prompt to the LLM and return the raw response text."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Original FunSearch interface (kept for backward compatibility)
    # ------------------------------------------------------------------
    def _draw_sample(self, prompt: str) -> str:
        """Returns a predicted continuation of `prompt`."""
        return self._call_api(prompt)

    def draw_samples(self, prompt: str) -> Collection[str]:
        """Returns multiple predicted continuations of `prompt`."""
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]

    # ------------------------------------------------------------------
    # Two-step thought-guided generation
    # ------------------------------------------------------------------
    def generate_thought(self, context: str) -> str:
        """Step 1: evolve / create a heuristic strategy in natural language.

        Args:
            context: assembled string containing parent thoughts, diagnosis
                     symptoms, RAG knowledge, etc.
        Returns:
            A 2-3 sentence natural-language strategy description.
        """
        prompt = (
            f"{context}\n\n"
            "Based on the information above, propose an improved heuristic strategy "
            "in 2-3 sentences. Describe the core idea, what signals to use, "
            "and how to combine them. Do NOT output any code."
        )
        response = self._call_api(prompt)
        return response.strip()

    def generate_code_from_thought(
            self,
            thought: str,
            function_header: str,
            extra_context: str = '',
    ) -> str:
        """Step 2: translate a thought into pure Python code.

        Args:
            thought: the natural-language strategy.
            function_header: the function signature the LLM must implement.
            extra_context: optional extra info (e.g. error trace for local fix).
        Returns:
            Raw Python code string (will be trimmed by evaluator).
        """
        parts = [
            f"Heuristic strategy to implement:\n{thought}",
        ]
        if extra_context:
            parts.append(extra_context)
        parts.append(
            f"Implement the strategy above as a Python function. "
            f"Only output the Python code body (with proper indentation), "
            f"no explanations.\n\n{function_header}"
        )
        prompt = '\n\n'.join(parts)
        response = self._call_api(prompt)
        return response


class Sampler:
    """Node that samples program continuations and sends them for analysis."""

    _global_samples_nums: int = 1

    def __init__(
            self,
            database: programs_database.ProgramsDatabase,
            evaluators: Sequence[evaluator.Evaluator],
            samples_per_prompt: int,
            max_sample_nums: int | None = None,
            llm_class: Type[LLM] = LLM,
            # --- new components (optional, for backward compat) ---
            reflector=None,
            knowledge_base=None,
            extractor=None,
            reflector_config=None,
            domain_id: str = 'PROB_BIN_PACKING_1D',
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums
        self._reflector = reflector
        self._knowledge_base = knowledge_base
        self._extractor = extractor
        self._reflector_config = reflector_config
        self._domain_id = domain_id
        self._best_score: float = -float('inf')

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def sample(self, **kwargs):
        """Continuously gets prompts, samples programs, sends them for analysis."""
        while True:
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                break
            try:
                prompt = self._database.get_prompt()
                reset_time = time.time()

                parent_thoughts = self._extract_parent_thoughts(prompt)
                thought_context = self._build_thought_context(parent_thoughts)

                for _ in range(self._samples_per_prompt):
                    self._global_sample_nums_plus_one()
                    cur_global = self._get_global_sample_nums()

                    # --- Step 1: generate thought ---
                    thought = self._llm.generate_thought(thought_context)
                    logging.info("Generated thought: %s", thought[:80])

                    # --- Step 2: generate code from thought ---
                    code = self._llm.generate_code_from_thought(
                        thought, prompt.function_header
                    )

                    sample_time = (time.time() - reset_time) / self._samples_per_prompt

                    # --- Evaluate ---
                    chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                    eval_result: evaluator.EvalResult = chosen_evaluator.analyse(
                        code,
                        prompt.island_id,
                        prompt.version_generated,
                        **kwargs,
                        global_sample_nums=cur_global,
                        sample_time=sample_time,
                    )

                    # Attach thought to function
                    if eval_result.function:
                        eval_result.function.thought = thought

                    # --- Triage & branch ---
                    if self._reflector is not None:
                        self._do_triage(eval_result, thought, prompt, sample_time,
                                        cur_global, **kwargs)
                    else:
                        self._update_best_score(eval_result)

            except Exception as e:
                logging.warning("Sampler loop error: %s", e)
                continue

    # ------------------------------------------------------------------
    # Triage dispatcher
    # ------------------------------------------------------------------
    def _do_triage(self, eval_result, thought, prompt, sample_time,
                   cur_global, **kwargs):
        """Route to the appropriate branch based on Reflector diagnosis."""
        triage_result = self._reflector.triage(eval_result, self._best_score)
        branch = triage_result.branch

        if branch == 'new_sota':
            self._handle_sota(eval_result, thought, **kwargs)
        elif branch == 'local_bug':
            self._handle_local_fix(
                thought, eval_result, prompt, sample_time, cur_global, **kwargs)
        elif branch == 'global_drawback':
            self._handle_global_diagnosis(
                eval_result, thought, prompt, sample_time, cur_global, **kwargs)

        self._update_best_score(eval_result)

    # ------------------------------------------------------------------
    # Branch A: local fix (keep thought, retry code)
    # ------------------------------------------------------------------
    def _handle_local_fix(self, thought, eval_result, prompt,
                          sample_time, cur_global, **kwargs):
        max_attempts = 2
        if self._reflector_config:
            max_attempts = self._reflector_config.max_fix_attempts

        for attempt in range(max_attempts):
            extra = (
                f"Your previous code produced an error:\n"
                f"{eval_result.error_trace}\n"
                f"Please fix the bug while keeping the same strategy."
            )
            fixed_code = self._llm.generate_code_from_thought(
                thought, prompt.function_header, extra_context=extra
            )
            chosen_evaluator = np.random.choice(self._evaluators)
            new_result = chosen_evaluator.analyse(
                fixed_code,
                prompt.island_id,
                prompt.version_generated,
                **kwargs,
                global_sample_nums=cur_global,
                sample_time=sample_time,
            )
            if new_result.function:
                new_result.function.thought = thought
            if new_result.is_valid:
                self._update_best_score(new_result)
                logging.info("Local fix succeeded on attempt %d", attempt + 1)
                return
        logging.info("Local fix failed after %d attempts", max_attempts)

    # ------------------------------------------------------------------
    # Branch B: global diagnosis + RAG
    # ------------------------------------------------------------------
    def _handle_global_diagnosis(self, eval_result, old_thought, prompt,
                                 sample_time, cur_global, **kwargs):
        symptom = self._reflector.diagnose_symptom(
            code=eval_result.program,
            score=eval_result.reduced_score,
            thought=old_thought,
        )
        logging.info("Diagnosed symptom: %s", symptom[:80] if symptom else "None")

        knowledge = None
        if self._knowledge_base and symptom:
            knowledge = self._knowledge_base.search(symptom, self._domain_id)

        guided_context = self._build_guided_context(old_thought, symptom, knowledge)
        new_thought = self._llm.generate_thought(guided_context)
        new_code = self._llm.generate_code_from_thought(
            new_thought, prompt.function_header
        )

        chosen_evaluator = np.random.choice(self._evaluators)
        new_result = chosen_evaluator.analyse(
            new_code,
            prompt.island_id,
            prompt.version_generated,
            **kwargs,
            global_sample_nums=cur_global,
            sample_time=sample_time,
        )
        if new_result.function:
            new_result.function.thought = new_thought
        self._update_best_score(new_result)

    # ------------------------------------------------------------------
    # Branch C: SOTA knowledge extraction
    # ------------------------------------------------------------------
    def _handle_sota(self, eval_result, thought, **kwargs):
        self._update_best_score(eval_result)
        if self._extractor and self._knowledge_base:
            try:
                tactic = self._extractor.extract_tactic(
                    code=eval_result.program,
                    thought=thought,
                    score=eval_result.reduced_score,
                    domain_id=self._domain_id,
                )
                if tactic:
                    self._knowledge_base.add_tactic(tactic)
                    logging.info("Extracted new L4 tactic: %s", tactic.get('name', ''))
            except Exception as e:
                logging.warning("Knowledge extraction failed: %s", e)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _update_best_score(self, eval_result):
        if eval_result.reduced_score is not None:
            if eval_result.reduced_score > self._best_score:
                self._best_score = eval_result.reduced_score

    def _extract_parent_thoughts(self, prompt) -> list[str]:
        """Extract thought strings from parent functions in the prompt."""
        thoughts = []
        if hasattr(prompt, 'thoughts') and prompt.thoughts:
            thoughts = prompt.thoughts
        return thoughts

    def _build_thought_context(self, parent_thoughts: list[str]) -> str:
        """Build the context string for thought generation in normal evolution."""
        if not parent_thoughts:
            return (
                "You are designing a heuristic for an online bin-packing problem. "
                "The function 'priority(item, bins)' returns a score for each bin. "
                "Propose an effective strategy."
            )
        lines = []
        for i, t in enumerate(parent_thoughts):
            lines.append(f"Strategy v{i}: {t}")
        lines.append(
            "\nPropose an improved strategy that addresses weaknesses of the above."
        )
        return '\n'.join(lines)

    def _build_guided_context(self, old_thought, symptom, knowledge) -> str:
        """Build the context string for RAG-guided thought generation (Branch B)."""
        parts = [f"Previous strategy: {old_thought}"]
        if symptom:
            parts.append(f"Diagnosed core symptom: {symptom}")
        if knowledge:
            name = knowledge.get('name', '')
            mechanism = knowledge.get('mechanism', knowledge.get('core_philosophy', ''))
            hint = knowledge.get('generator_prompt_hint', '')
            tactic_desc = knowledge.get('tactic_description', '')
            parts.append("Reference knowledge from cross-domain patterns:")
            if name:
                parts.append(f"  Name: {name}")
            if mechanism:
                parts.append(f"  Mechanism: {mechanism}")
            if hint:
                parts.append(f"  Application hint: {hint}")
            if tactic_desc:
                parts.append(f"  Tactic: {tactic_desc}")
        parts.append(
            "\nPropose a completely new strategy (2-3 sentences) that "
            "incorporates the above knowledge to fix the diagnosed symptom."
        )
        return '\n'.join(parts)

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1
