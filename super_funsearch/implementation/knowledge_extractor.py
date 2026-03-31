"""Knowledge Extractor: distill SOTA code into reusable L4 tactics."""
from __future__ import annotations

import json
import logging
from typing import Type

from implementation import sampler


class KnowledgeExtractor:
    """Analyzes high-scoring code to extract reusable L4 tactics.

    Triggered only when a new SOTA (best score) is achieved.
    Uses the LLM to produce a structured JSON tactic entry.
    """

    def __init__(self, llm_class: Type[sampler.LLM]):
        self._llm = llm_class(samples_per_prompt=1)

    def extract_tactic(
            self,
            code: str,
            thought: str | None,
            score: float | None,
            domain_id: str,
    ) -> dict | None:
        """Extract a L4 tactic from SOTA code.

        Returns a dict conforming to the L4_Specific_Tactics schema,
        or None if extraction fails.
        """
        thought_str = thought or "No explicit strategy recorded."
        score_str = f"{score:.4f}" if score is not None else "N/A"

        prompt = (
            f"The following Python heuristic function achieved a new best score "
            f"of {score_str} on an online bin-packing benchmark.\n\n"
            f"Strategy intent: {thought_str}\n\n"
            f"Code:\n```python\n{code}\n```\n\n"
            f"Analyze why this code performs well. Then output a JSON object "
            f"(and ONLY the JSON, no markdown fences) with this exact structure:\n"
            f'{{\n'
            f'  "name": "short tactic name in English",\n'
            f'  "applicable_symptoms": [\n'
            f'    "problem symptom this tactic solves (1)",\n'
            f'    "problem symptom this tactic solves (2)"\n'
            f'  ],\n'
            f'  "tactic_description": "2-3 sentence description of the tactic",\n'
            f'  "linked_domain_id": "{domain_id}",\n'
            f'  "linked_pattern_ids": []\n'
            f'}}'
        )

        try:
            response = self._llm._call_api(prompt)
            tactic = self._parse_json(response)
            if tactic is None:
                return None

            tactic.setdefault('linked_domain_id', domain_id)
            tactic.setdefault('linked_pattern_ids', [])
            tactic['provenance'] = {
                'score_improvement': score_str,
                'author': 'Auto_Extracted',
            }
            return tactic
        except Exception as e:
            logging.warning("KnowledgeExtractor.extract_tactic failed: %s", e)
            return None

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Try to extract a JSON object from LLM output."""
        text = text.strip()
        # Strip markdown code fences if present
        if text.startswith('```'):
            lines = text.split('\n')
            lines = [l for l in lines if not l.strip().startswith('```')]
            text = '\n'.join(lines)

        # Find the first { ... } block
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or end <= start:
            logging.warning("No JSON object found in LLM response")
            return None

        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError as e:
            logging.warning("JSON parse error: %s", e)
            return None
