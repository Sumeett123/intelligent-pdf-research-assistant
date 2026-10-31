from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from backend.bm25_retriever import BM25Retriever
from backend.chunker import chunk_pages, chunk_text
from backend.citation import make_citation
from backend.embeddings import EmbeddingService
from backend.hybrid_retriever import hybrid_search
from backend.llm_service import LLMService
from backend.pdf_processor import extract_pages_from_pdf
from backend.query_understanding import expand_query
from backend.reranker import Reranker
from backend.summarization import DocumentSummarizer, is_summary_request
from backend.vector_store import VectorStore

DEFAULT_TOP_K = 50
ANSWER_CONTEXT_LIMIT = 12


class ResearchAssistant:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        reranker: Reranker | None = None,
        llm_service: LLMService | None = None,
    ):
        # Dependency reuse is optional. Standard Streamlit/API operation still
        # constructs its own services; extension comparison can share the
        # heavy read-only models across isolated per-paper indexes.
        self.embedding_service = embedding_service or EmbeddingService()
        self.dimension = self.embedding_service.model.get_sentence_embedding_dimension()
        self.vector_store = VectorStore(self.dimension)
        self.bm25_retriever = BM25Retriever()
        self.reranker = reranker or Reranker()
        self.llm_service = llm_service or LLMService()
        self.summarizer = DocumentSummarizer()
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.chunks: List[Dict[str, Any]] = []

    def add_document(self, file_path: str | Path, chunk_size: int = 900, overlap: int = 120) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        pages = extract_pages_from_pdf(path)
        if not pages:
            raise ValueError("The uploaded file is invalid or contains no extractable text.")

        document_name = path.name
        self.chunks = [chunk for chunk in self.chunks if chunk["document_name"] != document_name]

        new_chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
        if not new_chunks:
            raise ValueError("The uploaded PDF produced no chunks after processing.")

        self.chunks.extend(new_chunks)
        self._rebuild_indexes()

        self.documents[document_name] = {
            "document_name": document_name,
            "page_count": len(pages),
            "chunk_count": len(new_chunks),
            "status": "Ready",
            "chunk_size": chunk_size,
            "overlap": overlap,
        }
        return self.documents[document_name]

    def add_text_document(
        self,
        document_name: str,
        text: str,
        chunk_size: int = 900,
        overlap: int = 120,
    ) -> Dict[str, Any]:
        """Index browser-extracted text through the same RAG pipeline as PDFs.

        Browser content has no reliable PDF page boundaries, so it is represented
        as one virtual page. The resulting chunks still pass through embeddings,
        BM25, hybrid retrieval, and reranking exactly like uploaded PDFs.
        """
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("The page contains no readable text.")

        self.chunks = [chunk for chunk in self.chunks if chunk["document_name"] != document_name]
        text_chunks = chunk_text(normalized_text, chunk_size=chunk_size, overlap=overlap)
        if not text_chunks:
            raise ValueError("The page contains no readable text after processing.")

        new_chunks = [
            {
                "document_name": document_name,
                "page_number": 1,
                "chunk_id": f"{document_name}-p1-c{index}",
                "section_index": 1,
                "text": chunk,
            }
            for index, chunk in enumerate(text_chunks, start=1)
        ]
        self.chunks.extend(new_chunks)
        self._rebuild_indexes()
        self.documents[document_name] = {
            "document_name": document_name,
            "page_count": 1,
            "chunk_count": len(new_chunks),
            "status": "Ready",
            "chunk_size": chunk_size,
            "overlap": overlap,
        }
        return self.documents[document_name]

    def _rebuild_indexes(self) -> None:
        if not self.chunks:
            self.vector_store.clear()
            self.bm25_retriever.clear()
            return

        texts = [chunk.get("text", "") for chunk in self.chunks]
        vectors = self.embedding_service.embed(texts)
        self.vector_store = VectorStore(vectors.shape[1])
        self.vector_store.add(vectors, self.chunks)
        self.bm25_retriever.build(self.chunks)

    def clear_documents(self) -> None:
        self.documents.clear()
        self.chunks.clear()
        self.vector_store.clear()
        self.bm25_retriever.clear()
        

    def _matches_question(self, question: str, chunk: Dict[str, Any]) -> bool:
        stop_words = {
            "what", "which", "when", "where", "why", "how", "who", "explain", "describe",
            "define", "tell", "give", "show", "about", "from", "with", "this", "that",
        }
        question_terms = {
            term for term in re.findall(r"[a-zA-Z0-9]+", expand_query(question).lower())
            if len(term) > 2 and term not in stop_words
        }
        if not question_terms:
            return True

        text_terms = set(re.findall(r"[a-zA-Z0-9]+", str(chunk.get("text", "")).lower()))
        exact_matches = question_terms.intersection(text_terms)
        if exact_matches:
            return True

        # Allow ordinary morphology such as "characteristic" / "characteristics".
        return any(
            len(term) >= 6 and any(token.startswith(term[:6]) or term.startswith(token[:6]) for token in text_terms)
            for term in question_terms
        )

    def _select_relevant_section(self, question: str, reranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not reranked:
            return []

        question_terms = {
            term for term in re.findall(r"[a-zA-Z0-9]+", expand_query(question).lower())
            if len(term) > 2 and term not in {"what", "which", "when", "where", "why", "how", "who", "explain", "describe", "define", "tell", "give", "show", "about", "from", "with", "this", "that"}
        }
        sections: Dict[tuple[str, int], List[Dict[str, Any]]] = {}
        for chunk in reranked:
            metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
            section_key = (
                str(chunk.get("document_name") or metadata.get("document_name", "")),
                int(chunk.get("section_index", metadata.get("section_index", -1))),
            )
            sections.setdefault(section_key, []).append(chunk)

        ranked_sections = []
        for section_key, section_chunks in sections.items():
            section_text = " ".join(str(chunk.get("text", "")).lower() for chunk in section_chunks)
            section_terms = set(re.findall(r"[a-zA-Z0-9]+", section_text))
            coverage = len(question_terms.intersection(section_terms))
            best_score = max(float(chunk.get("combined_score", chunk.get("rerank_score", 0.0))) for chunk in section_chunks)
            # The cross-encoder score represents semantic relevance. Lexical
            # coverage is only a tie-breaker because paraphrases may share no
            # words with the source document.
            ranked_sections.append((best_score, coverage, section_key, section_chunks))

        ranked_sections.sort(key=lambda item: (item[0], item[1]), reverse=True)
        _, _, best_key, best_chunks = ranked_sections[0]

        selected = list(best_chunks)
        for chunk in self.chunks:
            if (
                chunk.get("document_name") == best_key[0]
                and chunk.get("section_index") == best_key[1]
                and chunk.get("chunk_id") not in {item.get("chunk_id") for item in selected}
            ):
                selected.append(chunk)
        return selected

    def _select_answer_chunks(self, reranked: List[Dict[str, Any]], limit: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        if not reranked:
            return []

        selected: List[Dict[str, Any]] = []
        selected_ids = set()

        def add_chunk(chunk: Dict[str, Any]) -> None:
            chunk_id = chunk.get("chunk_id")
            if not chunk_id or chunk_id in selected_ids or len(selected) >= limit:
                return
            selected_ids.add(chunk_id)
            selected.append(chunk)

        for chunk in reranked[:DEFAULT_TOP_K]:
            add_chunk(chunk)

        # Expand only within the same question/answer section as a strong hit.
        for hit in reranked[:DEFAULT_TOP_K]:
            section_index = hit.get("section_index")
            if section_index is None:
                continue
            related = [
                chunk for chunk in self.chunks
                if chunk.get("document_name") == hit.get("document_name")
                and chunk.get("section_index") == section_index
            ]
            for chunk in related:
                add_chunk(chunk)

        return selected

    def ask(
        self,
        question: str,
        retrieval_mode: str = "Hybrid Search",
        top_k: int = DEFAULT_TOP_K,
        rerank_top_k: int = DEFAULT_TOP_K,
        chat_history: str = "",
        context_limit: int = ANSWER_CONTEXT_LIMIT,
    ) -> Dict[str, Any]:
        started_at = time.perf_counter()
        if not question or not question.strip():
            raise ValueError("Please provide a question.")
        if not self.chunks:
            raise ValueError("Please upload at least one PDF before asking a question.")

        original_question = question.strip()
        if is_summary_request(original_question):
            started_generation = time.perf_counter()
            summary_context = self.summarizer.select_context(self.chunks)
            response = self.summarizer.summarize(
                original_question,
                self.chunks,
                self.llm_service,
            )
            completed_at = time.perf_counter()
            sources = [
                make_citation(
                    chunk.get("document_name", "unknown.pdf"),
                    chunk.get("page_number", 0),
                    chunk.get("chunk_id", "unknown"),
                )
                for chunk in summary_context
            ]
            return {
                "answer": response.get("answer", "").strip(),
                "sources": response.get("sources", sources),
                "retrieval_mode": "Document Summary",
                "candidate_count": len(summary_context),
                "reranked_count": 0,
                "relevant_count": len(summary_context),
                "timings_ms": {
                    "total": round((completed_at - started_at) * 1000, 1),
                    "retrieval": 0.0,
                    "reranking": 0.0,
                    "generation": round((completed_at - started_generation) * 1000, 1),
                },
            }
        retrieval_question = expand_query(original_question)
        semantic_queries = [original_question]
        if retrieval_question != original_question:
            semantic_queries.append(retrieval_question)
        mode = retrieval_mode.lower()
        query_vectors = self.embedding_service.embed_query_variants(semantic_queries)
        retrieval_started_at = time.perf_counter()
        if mode == "vector search":
            dense_results = self.vector_store.search_many(query_vectors, top_k)
            bm25_results = []
        elif mode == "bm25 search":
            dense_results = []
            bm25_results = self.bm25_retriever.search(retrieval_question, top_k)
        else:
            with ThreadPoolExecutor(max_workers=2) as executor:
                dense_future = executor.submit(self.vector_store.search_many, query_vectors, top_k)
                bm25_future = executor.submit(self.bm25_retriever.search, retrieval_question, top_k)
                dense_results = dense_future.result()
                bm25_results = bm25_future.result()
        retrieval_finished_at = time.perf_counter()

        if mode == "vector search":
            candidates = dense_results
        elif mode == "bm25 search":
            candidates = bm25_results
        else:
            candidates = hybrid_search(dense_results, bm25_results, top_k=top_k)

        if not candidates:
            raise ValueError("No relevant chunks were found for this question.")

        # Rerank with the understood query so paraphrases such as "pros of"
        # are compared with document wording such as "advantages of".
        reranked = self.reranker.rerank(retrieval_question, candidates, top_k=rerank_top_k)
        rerank_finished_at = time.perf_counter()
        relevant_reranked = self._select_relevant_section(question, reranked)
        if not relevant_reranked:
            raise ValueError("The uploaded documents do not contain enough relevant evidence to answer this question.")
        answer_chunks = self._select_answer_chunks(relevant_reranked, limit=context_limit)

        llm_response = self.llm_service.generate(question, answer_chunks, chat_history=chat_history)
        completed_at = time.perf_counter()

        sources = []
        seen_sources = set()
        for chunk in answer_chunks:
            source_key = (chunk.get("document_name", "unknown.pdf"), chunk.get("page_number", 0))
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            sources.append(make_citation(chunk.get("document_name", "unknown.pdf"), chunk.get("page_number", 0), chunk.get("chunk_id", "unknown")))

        answer_text = llm_response.get("answer", "")
        answer_text = answer_text.strip()
        if not answer_text:
            answer_text = "The uploaded documents do not contain enough information to answer this question."

        return {
            "answer": answer_text,
            "sources": sources,
            "retrieval_mode": retrieval_mode,
            "candidate_count": len(candidates),
            "reranked_count": len(reranked),
            "relevant_count": len(relevant_reranked),
            "timings_ms": {
                "total": round((completed_at - started_at) * 1000, 1),
                "retrieval": round((retrieval_finished_at - retrieval_started_at) * 1000, 1),
                "reranking": round((rerank_finished_at - retrieval_finished_at) * 1000, 1),
                "generation": round((completed_at - rerank_finished_at) * 1000, 1),
            },
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        return [
            {
                "document_name": data["document_name"],
                "page_count": data["page_count"],
                "chunk_count": data["chunk_count"],
                "status": data["status"],
            }
            for _, data in self.documents.items()
        ]

    def get_document_stats(self) -> Dict[str, Any]:
        return {"document_count": len(self.documents), "chunk_count": len(self.chunks)}
