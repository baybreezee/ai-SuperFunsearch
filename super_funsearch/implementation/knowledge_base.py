"""Four-layer RAG knowledge base with semantic retrieval."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np

from implementation import config as config_lib


class KnowledgeBase:
    """Manages a 4-layer knowledge hierarchy (L1-L4) with embedding-based search.

    Layers:
      L1_Meta_Thoughts        – high-level philosophical heuristics
      L2_Cross_Domain_Patterns – cross-domain algorithmic patterns
      L3_Problem_Domains       – problem domain descriptions (not searched)
      L4_Specific_Tactics      – domain-specific tactics (auto-growing)

    Search strategy:
      1. L4 with hard domain_id filter → cosine similarity on applicable_symptoms
      2. If no L4 match above threshold → fall back to L2 → L1
    """

    def __init__(self, config: config_lib.KnowledgeBaseConfig | None = None):
        self._config = config or config_lib.KnowledgeBaseConfig()
        self._embedding_model = None
        self._data: dict[str, list] = {
            'L1_Meta_Thoughts': [],
            'L2_Cross_Domain_Patterns': [],
            'L3_Problem_Domains': [],
            'L4_Specific_Tactics': [],
        }
        self._embeddings_cache: dict[str, np.ndarray] = {}

        self._load_seed()
        self._init_embedding_model()
        self._build_index()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _load_seed(self):
        seed_path = self._config.seed_path
        if os.path.exists(self._config.persist_path):
            seed_path = self._config.persist_path
            logging.info("Loading persisted knowledge from %s", seed_path)
        elif os.path.exists(self._config.seed_path):
            logging.info("Loading seed knowledge from %s", seed_path)
        else:
            logging.warning("No knowledge file found at %s or %s",
                            self._config.seed_path, self._config.persist_path)
            return

        with open(seed_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        for key in self._data:
            if key in raw:
                self._data[key] = raw[key]

    def _init_embedding_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(
                self._config.embedding_model)
            logging.info("Loaded embedding model: %s",
                         self._config.embedding_model)
        except ImportError:
            logging.warning(
                "sentence-transformers not installed. "
                "KnowledgeBase will use keyword fallback.")
        except Exception as e:
            logging.warning("Failed to load embedding model: %s", e)

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns shape (N, dim)."""
        if self._embedding_model is not None:
            return self._embedding_model.encode(texts, normalize_embeddings=True)
        return np.zeros((len(texts), 1))

    def _build_index(self):
        """Pre-compute embeddings for all applicable_symptoms in L1, L2, L4."""
        for layer_key in ['L4_Specific_Tactics', 'L2_Cross_Domain_Patterns',
                          'L1_Meta_Thoughts']:
            for node in self._data.get(layer_key, []):
                symptoms = node.get('applicable_symptoms', [])
                if symptoms:
                    node_id = self._get_node_id(node, layer_key)
                    combined = ' | '.join(symptoms)
                    emb = self._embed([combined])
                    self._embeddings_cache[node_id] = emb[0]

    def _get_node_id(self, node: dict, layer_key: str) -> str:
        id_fields = {
            'L1_Meta_Thoughts': 'meta_id',
            'L2_Cross_Domain_Patterns': 'pattern_id',
            'L4_Specific_Tactics': 'tactic_id',
        }
        field = id_fields.get(layer_key, 'id')
        return node.get(field, str(id(node)))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, symptom: str, domain_id: str) -> dict[str, Any] | None:
        """Search the knowledge base for the most relevant entry.

        Returns the best matching knowledge node dict, or None.
        The returned dict has an extra 'matched_layer' key indicating source.
        """
        if not symptom:
            return None

        symptom_emb = self._embed([symptom])[0]

        # --- Phase 1: L4 with domain_id hard filter ---
        l4_match = self._search_layer(
            'L4_Specific_Tactics', symptom_emb,
            filter_fn=lambda n: n.get('linked_domain_id') == domain_id
        )
        if l4_match and l4_match[1] >= self._config.similarity_threshold:
            result = dict(l4_match[0])
            result['matched_layer'] = 'L4'
            result['similarity'] = float(l4_match[1])
            return result

        # --- Phase 2: L2 global search ---
        l2_match = self._search_layer('L2_Cross_Domain_Patterns', symptom_emb)
        if l2_match and l2_match[1] >= self._config.similarity_threshold * 0.8:
            result = dict(l2_match[0])
            result['matched_layer'] = 'L2'
            result['similarity'] = float(l2_match[1])
            return result

        # --- Phase 3: L1 global search ---
        l1_match = self._search_layer('L1_Meta_Thoughts', symptom_emb)
        if l1_match:
            result = dict(l1_match[0])
            result['matched_layer'] = 'L1'
            result['similarity'] = float(l1_match[1])
            return result

        return None

    def _search_layer(
            self,
            layer_key: str,
            query_emb: np.ndarray,
            filter_fn=None,
    ) -> tuple[dict, float] | None:
        """Find the best matching node in a given layer.

        Returns (node_dict, cosine_similarity) or None.
        """
        best_node = None
        best_sim = -1.0

        for node in self._data.get(layer_key, []):
            if filter_fn and not filter_fn(node):
                continue
            node_id = self._get_node_id(node, layer_key)
            node_emb = self._embeddings_cache.get(node_id)
            if node_emb is None:
                continue
            sim = float(np.dot(query_emb, node_emb))
            if sim > best_sim:
                best_sim = sim
                best_node = node

        if best_node is None:
            return None
        return best_node, best_sim

    # ------------------------------------------------------------------
    # Write (Branch C: SOTA extraction)
    # ------------------------------------------------------------------
    def add_tactic(self, tactic: dict) -> None:
        """Add a new L4 tactic and persist the knowledge base."""
        if 'tactic_id' not in tactic:
            existing_ids = [t.get('tactic_id', '') for t in
                            self._data['L4_Specific_Tactics']]
            next_num = len(existing_ids) + 1
            tactic['tactic_id'] = f'TAC_AUTO_{next_num:03d}'

        self._data['L4_Specific_Tactics'].append(tactic)

        symptoms = tactic.get('applicable_symptoms', [])
        if symptoms:
            combined = ' | '.join(symptoms)
            emb = self._embed([combined])
            self._embeddings_cache[tactic['tactic_id']] = emb[0]

        self._persist()
        logging.info("Added L4 tactic: %s", tactic.get('name', tactic['tactic_id']))

    def _persist(self):
        """Save current knowledge state to disk."""
        try:
            with open(self._config.persist_path, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning("Failed to persist knowledge base: %s", e)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        return {k: len(v) for k, v in self._data.items()}
