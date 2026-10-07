from __future__ import annotations

import re
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b[\w-]+\b", text.lower())


class BM25Retriever:
    def __init__(self):
        self.index: BM25Okapi | None = None
        self.metadata: List[Dict[str, Any]] = []

    def build(self, chunks: List[Dict[str, Any]]) -> None:
        self.metadata = chunks
        tokenized = [tokenize(chunk.get("text", "")) for chunk in chunks]
        self.index = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        if self.index is None or not self.metadata:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results: List[Dict[str, Any]] = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            metadata = self.metadata[idx]
            results.append(
                {
                    "chunk_id": metadata.get("chunk_id"),
                    "document_name": metadata.get("document_name"),
                    "page_number": metadata.get("page_number"),
                    "section_index": metadata.get("section_index"),
                    "score": score,
                    "metadata": metadata,
                    "text": metadata.get("text", ""),
                }
            )
        return results

    def clear(self) -> None:
        self.index = None
        self.metadata.clear()
