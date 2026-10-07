from __future__ import annotations

from typing import Any, Dict, List

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict[str, Any]] = []

    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        if vectors.size == 0:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results: List[Dict[str, Any]] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0 or index >= len(self.metadata):
                continue
            metadata = self.metadata[int(index)]
            results.append({
                "chunk_id": metadata.get("chunk_id"),
                "document_name": metadata.get("document_name"),
                "page_number": metadata.get("page_number"),
                "section_index": metadata.get("section_index"),
                "score": float(score),
                "text": metadata.get("text", ""),
                "metadata": metadata,
            })
        return results

    def search_many(self, query_vectors: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Search all semantic query variants and retain each chunk's best score."""
        if self.index.ntotal == 0 or query_vectors.size == 0:
            return []
        queries = np.asarray(query_vectors, dtype=np.float32)
        faiss.normalize_L2(queries)
        scores, indices = self.index.search(queries, min(top_k, self.index.ntotal))
        best_by_chunk: Dict[str, Dict[str, Any]] = {}
        for query_scores, query_indices in zip(scores, indices):
            for score, index in zip(query_scores, query_indices):
                if index < 0 or index >= len(self.metadata):
                    continue
                metadata = self.metadata[int(index)]
                chunk_id = metadata.get("chunk_id")
                if not chunk_id:
                    continue
                result = {
                    "chunk_id": chunk_id,
                    "document_name": metadata.get("document_name"),
                    "page_number": metadata.get("page_number"),
                    "section_index": metadata.get("section_index"),
                    "score": float(score),
                    "text": metadata.get("text", ""),
                    "metadata": metadata,
                }
                if chunk_id not in best_by_chunk or result["score"] > best_by_chunk[chunk_id]["score"]:
                    best_by_chunk[chunk_id] = result
        return sorted(best_by_chunk.values(), key=lambda item: item["score"], reverse=True)[:top_k]

    def clear(self) -> None:
        self.index.reset()
        self.metadata.clear()

    def __len__(self) -> int:
        return self.index.ntotal
