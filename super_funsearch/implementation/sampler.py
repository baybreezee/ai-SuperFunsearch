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
import re
import textwrap
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
    def generate_thought(self, context: str, interface_spec: str = '') -> str:
        """Step 1: evolve / create a heuristic strategy in natural language.

        Args:
            context: assembled string containing parent thoughts, diagnosis
                     symptoms, RAG knowledge, etc.
            interface_spec: function interface specification to ground the LLM.
        Returns:
            A 2-3 sentence natural-language strategy description.
        """
        prompt = (
            f"{interface_spec}\n\n"
            f"{context}\n\n"
            "Based on the information above, propose an improved heuristic strategy "
            "in 2-3 sentences. Describe the core idea, what mathematical signals "
            "derived ONLY from `item` (float) and `bins` (1-D remaining-capacity "
            "array) to use, and how to combine them into a score for each bin. "
            "Do NOT output any code."
        )
        response = self._call_api(prompt)
        return response.strip()

    def generate_code_from_thought(
            self,
            thought: str,
            function_header: str,
            extra_context: str = '',
            interface_spec: str = '',
    ) -> str:
        """Step 2: translate a thought into pure Python code.

        Args:
            thought: the natural-language strategy.
            function_header: the function signature the LLM must implement.
            extra_context: optional extra info (e.g. error trace for local fix).
            interface_spec: function interface specification to ground the LLM.
        Returns:
            Raw Python code string (will be trimmed by evaluator).
        """
        baseline = (
            "### Current baseline (score ≈ -212.75):\n"
            "```python\n"
            "def priority(item: float, bins: np.ndarray) -> np.ndarray:\n"
            "    ratios = item / bins\n"
            "    log_ratios = np.log(ratios)\n"
            "    priorities = -log_ratios\n"
            "    return priorities\n"
            "```\n"
            "Your code MUST beat this baseline. Lower (more negative) score is "
            "worse; higher is better."
        )
        parts = [interface_spec] if interface_spec else []
        parts.append(baseline)
        parts.append(f"Heuristic strategy to implement:\n{thought}")
        if extra_context:
            parts.append(extra_context)
        parts.append(
            "Now write the COMPLETE Python function implementing the strategy. "
            "Requirements:\n"
            "1. Start with `def priority(item: float, bins: np.ndarray) -> np.ndarray:`\n"
            "2. Include a `return` statement that returns a 1-D np.ndarray "
            "of the same shape as `bins`.\n"
            "3. Do NOT include `import` statements — `numpy` is already "
            "available as `np`.\n"
            "4. Output ONLY the function definition, no explanation."
        )
        prompt = '\n\n'.join(parts)
        response = self._call_api(prompt)
        return self._extract_function_body(response)

    @staticmethod
    def _extract_function_body(raw_code: str) -> str:
        """Post-process LLM output: strip markdown fences, imports, def line,
        and docstring — return only the indented function body."""
        code = raw_code.strip()

        # 1. Strip markdown code fences
        code = re.sub(r'^```[a-zA-Z]*\n?', '', code)
        code = re.sub(r'\n?```$', '', code)
        code = code.strip()

        # 2. Remove top-level import lines
        lines = code.split('\n')
        lines = [l for l in lines if not re.match(
            r'^(import |from \S+ import )', l.strip())]
        code = '\n'.join(lines).strip()

        # 3. If there's a def line, extract only the body after it
        def_match = re.search(r'^def \w+\(.*?\).*?:\s*$', code, re.MULTILINE)
        if def_match:
            after_def = code[def_match.end():]
            body_lines = after_def.split('\n')

            # Skip docstring if present
            stripped = after_def.lstrip('\n')
            if stripped.lstrip().startswith('"""') or stripped.lstrip().startswith("'''"):
                quote = '"""' if '"""' in stripped else "'''"
                first_quote = stripped.find(quote)
                second_quote = stripped.find(quote, first_quote + 3)
                if second_quote != -1:
                    after_docstring = stripped[second_quote + 3:]
                    body_lines = after_docstring.split('\n')

            code = '\n'.join(body_lines)

        # 4. Ensure body has proper indentation (at least 4 spaces)
        if code.strip():
            dedented = textwrap.dedent(code)
            indented_lines = []
            for line in dedented.split('\n'):
                if line.strip():
                    indented_lines.append('    ' + line)
                else:
                    indented_lines.append('')
            code = '\n'.join(indented_lines)

        # 5. Safety net: if no return statement, add one
        if code.strip() and not re.search(r'^\s*return\s', code, re.MULTILINE):
            last_var = None
            for line in reversed(code.strip().split('\n')):
                m = re.match(r'\s*(\w+)\s*=', line)
                if m:
                    last_var = m.group(1)
                    break
            if last_var:
                code = code.rstrip() + f'\n    return {last_var}'
            else:
                code = code.rstrip() + '\n    return bins - item'

        return code


class Sampler:
    """Node that samples program continuations and sends them for analysis."""

    _global_samples_nums: int = 1

    FUNCTION_INTERFACE_SPEC = (
        "## Function interface (MUST follow strictly)\n"
        "```\n"
        "def priority(item: float, bins: np.ndarray) -> np.ndarray\n"
        "```\n"
        "- `item`: a **float scalar** — the size of the item to pack.\n"
        "- `bins`: a **1-D np.ndarray** — each element is the **remaining capacity** "
        "of that bin. The array has already been filtered to only include bins where "
        "the item fits (i.e. every element >= item). There is NO other information "
        "about the bins (no 2-D structure, no history of packed items).\n"
        "- **Return**: a **1-D np.ndarray of the same shape as `bins`**. "
        "Each element is the priority score for the corresponding bin. "
        "The bin with the **highest** score will be chosen.\n\n"
        "### Critical constraints\n"
        "- Do NOT use `axis=1`, `.sum(axis=1)`, `.shape[1]`, or treat `bins` as 2-D.\n"
        "- Do NOT return indices or sorted arrays — return a **score array**.\n"
        "- Do NOT filter or mask bins (they are already valid).\n"
        "- Only use `item` (scalar) and `bins` (1-D array) as inputs.\n"
        "- You may use numpy operations.\n"
    )

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
                    thought = self._llm.generate_thought(
                        thought_context,
                        interface_spec=self.FUNCTION_INTERFACE_SPEC,
                    )
                    logging.info("Generated thought: %s", thought[:80])

                    # --- Step 2: generate code from thought ---
                    code = self._llm.generate_code_from_thought(
                        thought, prompt.function_header,
                        interface_spec=self.FUNCTION_INTERFACE_SPEC,
                    )

                    sample_time = (time.time() - reset_time) / self._samples_per_prompt

                    # --- Evaluate ---
                    chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                    eval_result: evaluator.EvalResult = chosen_evaluator.analyse(
                        code,
                        prompt.island_id,
                        prompt.version_generated,
                        thought=thought,
                        **kwargs,
                        global_sample_nums=cur_global,
                        sample_time=sample_time,
                    )

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
                thought, prompt.function_header, extra_context=extra,
                interface_spec=self.FUNCTION_INTERFACE_SPEC,
            )
            chosen_evaluator = np.random.choice(self._evaluators)
            new_result = chosen_evaluator.analyse(
                fixed_code,
                prompt.island_id,
                prompt.version_generated,
                thought=thought,
                **kwargs,
                global_sample_nums=cur_global,
                sample_time=sample_time,
            )
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
        new_thought = self._llm.generate_thought(
            guided_context,
            interface_spec=self.FUNCTION_INTERFACE_SPEC,
        )
        new_code = self._llm.generate_code_from_thought(
            new_thought, prompt.function_header,
            interface_spec=self.FUNCTION_INTERFACE_SPEC,
        )

        chosen_evaluator = np.random.choice(self._evaluators)
        new_result = chosen_evaluator.analyse(
            new_code,
            prompt.island_id,
            prompt.version_generated,
            thought=new_thought,
            **kwargs,
            global_sample_nums=cur_global,
            sample_time=sample_time,
        )
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
                "You are designing a heuristic for an online bin-packing problem.\n"
                "The goal is to minimize the number of bins used. Items arrive one "
                "by one; for each item you assign a priority score to every candidate "
                "bin and the item goes into the highest-scored bin.\n"
                "The current baseline uses `priorities = -np.log(item / bins)` "
                "(i.e. `np.log(bins / item)`) which achieves score ≈ -212.75.\n"
                "Propose a better strategy using only `item` (float) and `bins` "
                "(1-D remaining-capacity array)."
            )
        lines = []
        for i, t in enumerate(parent_thoughts):
            lines.append(f"Strategy v{i}: {t}")
        lines.append(
            "\nThe baseline score is ≈ -212.75 (using log(bins/item))."
            "\nPropose an improved strategy that addresses weaknesses of the above "
            "and can beat the baseline. Remember you can only use `item` (float) "
            "and `bins` (1-D remaining-capacity array)."
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
            "incorporates the above knowledge to fix the diagnosed symptom. "
            "Remember you can only use `item` (float scalar) and `bins` "
            "(1-D remaining-capacity array) — no 2-D operations, no history."
        )
        return '\n'.join(parts)

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1
