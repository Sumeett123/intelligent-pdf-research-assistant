from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.assistant import DEFAULT_TOP_K, ResearchAssistant
from backend.hybrid_retriever import hybrid_search


def build_sample_pdf(path: Path) -> None:
    document = fitz.open()
    pages = [
        "The central research question is how hybrid retrieval affects PDF question-answering quality. The system combines dense semantic signals with lexical matching to improve factual grounding.",
        "Hybrid retrieval unifies semantic vector search and BM25 keyword search. Reciprocal rank fusion combines multiple rankings into a single candidate ordering without relying on a single retrieval signal.",
        "BM25 remains useful for exact phrase matches, technical vocabulary, and rarer terms that semantic embeddings might miss. For academic PDFs, this helps when key terminology matters.",
        "Vector similarity captures semantic proximity across paraphrases. This is important when a question uses different wording than the document text but still describes the same concept.",
        "Reciprocal rank fusion computes a combined score by summing inverse rank contributions from the dense and BM25 rankings. This reduces the risk of a single weak method dominating.",
        "Cross-encoder reranking evaluates candidate passages together with the query to decide which evidence is most relevant. This step refines the final context before answer generation.",
        "The answer generation step is strictly grounded in the retrieved context. This design aims to reduce hallucinations and maintain evidence-backed outputs for research tasks.",
        "Evaluation uses Recall@K, Precision@K, MRR, and latency to compare retrieval strategies. These metrics quantify both quality and efficiency across the same document corpus.",
        "Recall@K measures whether relevant evidence appears within the first K retrieved chunks. It captures the coverage of the retrieval system for a given question.",
        "MRR measures the rank of the first truly relevant result. A lower rank indicates worse quality for users who need an answer quickly.",
        "Retrieval latency measures how quickly results are produced, which matters when a system must respond under time constraints or heavy user load.",
        "Keyword-only retrieval is limited because exact lexical overlap may fail with paraphrasing or technical synonyms. This motivates the addition of semantic retrieval.",
        "When the evidence is absent, the system responds by stating that the uploaded documents do not contain enough information to answer the question. This preserves correctness and transparency.",
        "Conversational context is maintained within the current session, allowing follow-up questions to resolve references such as pronouns or previously mentioned entities.",
        "Future work includes OCR support for scanned PDFs, longer context windows, and broader benchmark datasets for academic PDF retrieval research.",
    ]

    for page_number, text in enumerate(pages, start=1):
        page = document.new_page()
        page.insert_text((72, 72), text, fontsize=11)
        page.insert_text((72, 100), f"Research Note {page_number}", fontsize=10)

    document.save(path)
    document.close()


def evaluate_method(name: str, assistant: ResearchAssistant, question: str, relevant_pages: list[int], top_k: int = DEFAULT_TOP_K):
    start = time.perf_counter()

    if name == "BM25":
        results = assistant.bm25_retriever.search(question, top_k=top_k)
    elif name == "Vector Search":
        query_vector = assistant.embedding_service.embed_query(question)
        results = assistant.vector_store.search(query_vector, top_k)
    elif name == "Hybrid Search":
        query_vector = assistant.embedding_service.embed_query(question)
        dense = assistant.vector_store.search(query_vector, top_k)
        bm25 = assistant.bm25_retriever.search(question, top_k=top_k)
        results = hybrid_search(dense, bm25, top_k=top_k)
    else:
        query_vector = assistant.embedding_service.embed_query(question)
        dense = assistant.vector_store.search(query_vector, top_k)
        bm25 = assistant.bm25_retriever.search(question, top_k=top_k)
        candidates = hybrid_search(dense, bm25, top_k=top_k)
        results = assistant.reranker.rerank(question, candidates, top_k=top_k)

    latency = time.perf_counter() - start
    relevant_ids = {
        chunk["chunk_id"]
        for chunk in assistant.chunks
        if chunk["page_number"] in relevant_pages
    }

    retrieved_ids = [item.get("chunk_id") for item in results[:top_k]]
    hit_ids = [item for item in retrieved_ids if item in relevant_ids]
    recall = len(hit_ids) / len(relevant_ids) if relevant_ids else 0.0
    precision = len(hit_ids) / len(retrieved_ids) if retrieved_ids else 0.0
    mrr = 0.0
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            mrr = 1.0 / rank
            break

    return {
        "method": name,
        "recall@50": round(recall, 4),
        "precision@50": round(precision, 4),
        "mrr": round(mrr, 4),
        "latency_ms": round(latency * 1000, 2),
    }


def main() -> None:
    sample_path = Path("data/sample_research.pdf")
    sample_path.parent.mkdir(exist_ok=True)
    if not sample_path.exists():
        build_sample_pdf(sample_path)

    assistant = ResearchAssistant()
    assistant.add_document(sample_path, chunk_size=900, overlap=120)

    evaluation_file = Path("evaluation/test_questions.json")
    questions = json.loads(evaluation_file.read_text(encoding="utf-8"))

    methods = ["BM25", "Vector Search", "Hybrid Search", "Hybrid Search + Rerank"]
    summary = []
    for method_name in methods:
        method_scores = []
        for item in questions:
            method_scores.append(
                evaluate_method(method_name, assistant, item["question"], item["relevant_pages"], top_k=DEFAULT_TOP_K)
            )

        aggregate = {
            "method": method_name,
            "recall@50": round(sum(row["recall@50"] for row in method_scores) / len(method_scores), 4),
            "precision@50": round(sum(row["precision@50"] for row in method_scores) / len(method_scores), 4),
            "mrr": round(sum(row["mrr"] for row in method_scores) / len(method_scores), 4),
            "latency_ms": round(sum(row["latency_ms"] for row in method_scores) / len(method_scores), 2),
        }
        summary.append(aggregate)

    print("\nRetrieval evaluation results")
    print("-" * 90)
    print(f"{'Method':<24} {'Recall@50':>12} {'Precision@50':>14} {'MRR':>10} {'Latency (ms)':>16}")
    print("-" * 90)
    for row in summary:
        print(
            f"{row['method']:<24} {row['recall@50']:>12.4f} {row['precision@50']:>14.4f} {row['mrr']:>10.4f} {row['latency_ms']:>14.2f}"
        )
    print("-" * 90)


if __name__ == "__main__":
    main()
