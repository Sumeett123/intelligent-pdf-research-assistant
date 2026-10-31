from __future__ import annotations

import os
from collections import OrderedDict
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        # This project currently runs with the CPU-only PyTorch build. Passing
        # the device explicitly avoids SentenceTransformer attempting to move
        # meta tensors during model initialization.
        self.device = os.getenv("EMBEDDING_DEVICE", "cpu").strip().lower() or "cpu"
        if self.device != "cpu":
            raise ValueError(
                f"Unsupported embedding device '{self.device}'. "
                "Use EMBEDDING_DEVICE=cpu for the installed CPU-only runtime."
            )
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._query_cache_size = 128

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        clean_texts = [text.strip() for text in texts if text and text.strip()]
        if not clean_texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self.model.encode(
            clean_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        cached = self._query_cache.get(query)
        if cached is not None:
            self._query_cache.move_to_end(query)
            return cached.copy()

        vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        result = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        self._query_cache[query] = result
        self._query_cache.move_to_end(query)
        if len(self._query_cache) > self._query_cache_size:
            self._query_cache.popitem(last=False)
        return result.copy()

    def embed_query_variants(self, queries: Sequence[str]) -> np.ndarray:
        """Encode multiple formulations for semantic multi-query retrieval."""
        clean_queries = list(dict.fromkeys(query.strip() for query in queries if query and query.strip()))
        if not clean_queries:
            return np.empty((0, 0), dtype=np.float32)
        vectors = [
            self.embed_query(query).reshape(-1)
            for query in clean_queries
        ]
        return np.asarray(vectors, dtype=np.float32)
