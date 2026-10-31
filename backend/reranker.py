from __future__ import annotations

import os
import re
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import CrossEncoder


def normalize_heading_query(query: str) -> str:
    return re.sub(
        r"^(?:please\s+)?(?:explain|describe|state|define)\s+(?:the|a|an)\s+",
        "",
        query.strip().lower(),
    ).strip(" .?:-")


def is_heading_match(query: str, text: str) -> bool:
    heading_query = normalize_heading_query(query)
    return bool(
        heading_query
        and re.search(
            rf"\b{re.escape(heading_query)}\b\s+(?:these|among|the|is|are)",
            text,
            flags=re.IGNORECASE,
        )
    )


class Reranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.device = os.getenv("RERANKER_DEVICE", "cpu").strip().lower() or "cpu"
        if self.device != "cpu":
            raise ValueError(
                f"Unsupported reranker device '{self.device}'. "
                "Use RERANKER_DEVICE=cpu for the installed CPU-only runtime."
            )
        self.model = CrossEncoder(self.model_name, device=self.device)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 50) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        pairs = []
        normalized_candidates: List[Dict[str, Any]] = []
        for candidate in candidates:
            text = candidate.get("text") or candidate.get("metadata", {}).get("text", "")
            if not text and "metadata" in candidate and isinstance(candidate["metadata"], dict):
                text = candidate["metadata"].get("text", "")
            pair = (query, text)
            pairs.append(pair)
            candidate_copy = dict(candidate)
            candidate_copy["text"] = text
            normalized_candidates.append(candidate_copy)

        scores = self.model.predict(pairs)
        reranked: List[Dict[str, Any]] = []
        query_terms = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
        for candidate, score in zip(normalized_candidates, scores):
            entry = dict(candidate)
            entry["rerank_score"] = float(score)
            text_terms = set(re.findall(r"[a-zA-Z0-9]+", candidate["text"].lower()))
            lexical_overlap = len(query_terms.intersection(text_terms)) / max(len(query_terms), 1)
            heading_match = is_heading_match(query, candidate["text"])
            entry["lexical_overlap"] = round(lexical_overlap, 4)
            entry["combined_score"] = float(score) + (0.5 * lexical_overlap) + (2.0 if heading_match else 0.0)
            reranked.append(entry)

        reranked.sort(key=lambda item: item["combined_score"], reverse=True)
        return reranked[:top_k]
