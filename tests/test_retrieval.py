from backend.hybrid_retriever import reciprocal_rank_fusion, rank_candidates
from backend.vector_store import VectorStore
from backend.reranker import is_heading_match, normalize_heading_query


def test_reciprocal_rank_fusion_combines_rankings():
    dense = ["a", "b", "c"]
    bm25 = ["c", "a", "d"]
    scores = reciprocal_rank_fusion(dense, bm25, k=60)

    assert set(scores.keys()) == {"a", "b", "c", "d"}
    assert scores["a"] > scores["b"]
    assert scores["c"] > scores["d"]


def test_rank_candidates_returns_sorted_candidates():
    candidates = {"doc_1": 0.9, "doc_2": 0.7, "doc_3": 0.3}
    ranked = rank_candidates(candidates)

    assert [item[0] for item in ranked] == ["doc_1", "doc_2", "doc_3"]


def test_heading_match_ignores_question_instruction_words():
    query = "State the Implications for Management"

    assert normalize_heading_query(query) == "implications for management"
    assert is_heading_match(query, "Implications for Management These IT impacts affect managers.") is True
    assert is_heading_match(query, "Implications for management on business strategy.") is False


def test_vector_store_search_many_keeps_best_variant_score():
    import numpy as np

    store = VectorStore(2)
    store.add(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), [
        {"chunk_id": "a", "text": "advantages"},
        {"chunk_id": "b", "text": "networks"},
    ])

    results = store.search_many(np.asarray([[0.8, 0.6], [1.0, 0.0]], dtype=np.float32), top_k=2)

    assert [result["chunk_id"] for result in results] == ["a", "b"]
