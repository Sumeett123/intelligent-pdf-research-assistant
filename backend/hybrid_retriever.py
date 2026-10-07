from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


def reciprocal_rank_fusion(dense_ranked: Iterable[str], bm25_ranked: Iterable[str], k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for rank, chunk_id in enumerate(dense_ranked, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    for rank, chunk_id in enumerate(bm25_ranked, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def rank_candidates(score_map: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(score_map.items(), key=lambda item: item[1], reverse=True)


def hybrid_search(dense_results: List[Dict[str, Any]], bm25_results: List[Dict[str, Any]], k: int = 60, top_k: int = 50) -> List[Dict[str, Any]]:
    dense_ranked = [item.get("chunk_id") for item in dense_results if item.get("chunk_id")]
    bm25_ranked = [item.get("chunk_id") for item in bm25_results if item.get("chunk_id")]
    scores = reciprocal_rank_fusion(dense_ranked, bm25_ranked, k=k)

    ranked = rank_candidates(scores)
    selected: List[Dict[str, Any]] = []
    chunk_lookup = {
        item.get("chunk_id"): item for item in (dense_results + bm25_results) if item.get("chunk_id")
    }
    for chunk_id, score in ranked[:top_k]:
        match = chunk_lookup.get(chunk_id)
        if match is None:
            continue
        result = dict(match)
        result["fusion_score"] = float(score)
        selected.append(result)
    return selected
