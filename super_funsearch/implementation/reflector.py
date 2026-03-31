"""Reflector module: triage evaluation results and diagnose symptoms."""
from __future__ import annotations

import dataclasses
import logging
from typing import Type

from implementation import evaluator
from implementation import sampler
from implementation import config as config_lib


@dataclasses.dataclass
class TriageResult:
    """Result of the triage step."""
    branch: str  # "local_bug" | "global_drawback" | "new_sota" | "normal"
    error_trace: str | None = None
    symptom: str | None = None


class Reflector:
    """Analyzes evaluation feedback and routes to the appropriate branch.

    - triage(): pure rule-based, no LLM call, fast.
    - diagnose_symptom(): calls LLM to extract abstract symptom (Branch B only).
    """

    def __init__(
            self,
            llm_class: Type[sampler.LLM],
            config: config_lib.ReflectorConfig | None = None,
    ):
        self._llm = llm_class(samples_per_prompt=1)
        self._config = config or config_lib.ReflectorConfig()

    def triage(
            self,
            eval_result: evaluator.EvalResult,
            best_score: float,
    ) -> TriageResult:
        """Rule-based triage. No LLM call.

        Decision logic:
          1. error_trace is not empty AND no valid score → local_bug
          2. valid score that exceeds best_score       → new_sota
          3. valid score but not SOTA                  → normal (no special action)
          4. no valid score (but no error either)      → global_drawback
          5. valid score but worse than parent          → global_drawback
        """
        has_error = bool(eval_result.error_trace)
        has_score = eval_result.is_valid and eval_result.reduced_score is not None

        if has_error and not has_score:
            return TriageResult(
                branch='local_bug',
                error_trace=eval_result.error_trace,
            )

        if has_score:
            if eval_result.reduced_score > best_score:
                return TriageResult(branch='new_sota')
            return TriageResult(branch='normal')

        return TriageResult(branch='global_drawback')

    def diagnose_symptom(
            self,
            code: str,
            score: float | None,
            thought: str | None,
    ) -> str:
        """Call LLM to extract an abstract 1-2 sentence symptom description.

        Only triggered when triage returns 'global_drawback'.
        """
        score_str = f"{score:.4f}" if score is not None else "N/A (execution failed)"
        thought_str = thought or "No explicit strategy recorded."

        prompt = (
            f"The following Python heuristic function scored {score_str} "
            f"(lower/negative is worse) on an online bin-packing benchmark.\n\n"
            f"Strategy intent: {thought_str}\n\n"
            f"Code:\n```python\n{code}\n```\n\n"
            f"Do NOT modify the code. Analyze the algorithm's fundamental "
            f"weakness at the STRATEGY level (not implementation bugs). "
            f"Summarize the core symptom in 1-2 concise sentences."
        )
        try:
            response = self._llm._call_api(prompt)
            return response.strip()
        except Exception as e:
            logging.warning("diagnose_symptom failed: %s", e)
            return "Unable to diagnose symptom."
