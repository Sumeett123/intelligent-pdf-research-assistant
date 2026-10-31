"""Extension-only paper discovery and comparison helpers.

This module deliberately owns no Streamlit state.  Every comparison gets a
fresh ResearchAssistant, so its documents and indexes cannot leak into the
main application's upload/chat session.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from backend.assistant import ResearchAssistant
from backend.hybrid_retriever import hybrid_search
from backend.llm_service import LLMGenerationError
from backend.query_understanding import expand_query

OPENALEX = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR = "https://api.semanticscholar.org/graph/v1/paper/search"
SUPPORTED_PUBLISHERS = {"IEEE", "Springer", "PLOS", "Frontiers", "MDPI"}
ASPECTS = {
    "Objectives": "State only the explicit research objective, problem, and concrete contribution. Do not include methods unless the paper presents them as a contribution.",
    "Algorithms": "List only explicitly named algorithms, models, or techniques. If none are named, state that they are not reported.",
    "Datasets": "State only named datasets, data sources, sample sizes, geographic coverage, and input variables explicitly reported.",
    "Methodology": "Describe only the explicit data preparation, training/testing, baselines, splits, and evaluation procedure.",
    "Results": "Report only measured results, metrics, comparisons, or numerical findings. Do not treat goals, claims, or future work as results.",
    "Limitations": "Report only explicit limitations, threats to validity, constraints, or future work. Do not infer limitations from generic conclusions.",
}

ASPECT_TERMS = {
    "Objectives": {"objective", "aim", "aims", "purpose", "propose", "proposed", "problem", "contribution", "study", "develop", "presents"},
    "Algorithms": {"algorithm", "algorithms", "model", "models", "classifier", "regression", "forest", "svm", "neural", "knn", "naive", "xgboost", "learning"},
    "Datasets": {"dataset", "datasets", "data", "kaggle", "source", "records", "instances", "samples", "government", "attributes", "imbalanced"},
    "Methodology": {"methodology", "method", "preprocess", "preprocessing", "training", "testing", "validation", "split", "evaluation", "workflow", "smote", "compare"},
    "Results": {"result", "results", "accuracy", "precision", "recall", "f1", "performance", "achieved", "score", "metric", "experimental"},
    "Limitations": {"limitation", "limitations", "constraint", "challenge", "future work", "future scope", "threat"},
}


def _paper_from_work(work: dict[str, Any]) -> dict[str, Any]:
    locations = [work.get("best_oa_location") or {}, *(work.get("locations") or [])]
    location = next(
        (
            item for item in locations
            if isinstance(item, dict) and (item.get("pdf_url") or item.get("landing_page_url"))
        ),
        {},
    )
    primary_source = ((work.get("primary_location") or {}).get("source") or location.get("source") or {})
    doi = str(work.get("doi") or "").replace("https://doi.org/", "")
    source_text = " ".join(
        str(value or "")
        for value in (primary_source.get("display_name"), primary_source.get("publisher"), work.get("host_venue"), (work.get("primary_location") or {}).get("landing_page_url"))
    )
    publisher = _publisher_for(doi, source_text)
    source_name = str(primary_source.get("display_name") or "Open-access source")
    return {
        "title": work.get("title") or "Untitled paper",
        "doi": doi,
        "year": work.get("publication_year"),
        "cited_by_count": work.get("cited_by_count", 0),
        "url": (work.get("primary_location") or {}).get("landing_page_url") or work.get("doi"),
        # Some OpenAlex OA records publish only an OA landing page. It is
        # useful for discovery, but direct PDF retrieval is verified later.
        "pdf_url": location.get("pdf_url") or location.get("landing_page_url"),
        "is_open_access": bool((work.get("open_access") or {}).get("is_oa")),
        "publisher": publisher,
        "source_name": source_name,
    }


def _publisher_for(doi: str, venue: str = "") -> str:
    value = f"{doi} {venue}".lower()
    if doi.lower().startswith("10.1109/") or "ieee" in value:
        return "IEEE"
    if doi.lower().startswith("10.1007/") or "springer" in value:
        return "Springer"
    if doi.lower().startswith("10.1371/") or "plos" in value:
        return "PLOS"
    if doi.lower().startswith("10.3389/") or "frontiers" in value:
        return "Frontiers"
    if doi.lower().startswith("10.3390/") or "mdpi" in value:
        return "MDPI"
    return "Unsupported publisher"


def _paper_from_semantic_scholar(work: dict[str, Any]) -> dict[str, Any] | None:
    external_ids = work.get("externalIds") or {}
    doi = str(external_ids.get("DOI") or "").replace("https://doi.org/", "")
    pdf_url = (work.get("openAccessPdf") or {}).get("url")
    if not doi or not pdf_url:
        return None
    venue = str((work.get("publicationVenue") or {}).get("name") or "")
    return {
        "title": work.get("title") or "Untitled paper",
        "doi": doi,
        "year": work.get("year"),
        "cited_by_count": work.get("citationCount", 0),
        "url": f"https://doi.org/{doi}",
        "pdf_url": pdf_url,
        "is_open_access": True,
        "publisher": _publisher_for(doi, venue),
        "source_name": venue or "Open-access source",
    }


def _has_retrievable_pdf(paper: dict[str, Any]) -> bool:
    """Verify that a claimed OA link is an actual PDF before showing it.

    Publisher landing pages often look like open access in third-party
    metadata but return HTML, a login screen, or a DOI resolver. Checking a
    small response in memory prevents a later comparison failure and never
    writes a paper to disk.
    """
    url = str(paper.get("pdf_url") or "")
    if not url.startswith("https://"):
        return False
    try:
        with httpx.stream(
            "GET",
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchAssistant/1.0)"},
            follow_redirects=True,
            timeout=6.0,
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            first_bytes = next(response.iter_bytes(chunk_size=8), b"")
            return "application/pdf" in content_type or first_bytes.startswith(b"%PDF")
    except httpx.HTTPError:
        return False


def search_papers(
    topic: str,
    limit: int = 5,
    exclude_dois: list[str] | None = None,
    publishers: list[str] | None = None,
    min_citations: int = 50,
) -> list[dict[str, Any]]:
    """Return new IEEE/Springer candidates only; full text is never fetched."""
    response = httpx.get(OPENALEX, params={"search": topic, "per-page": 200}, timeout=20.0)
    response.raise_for_status()
    excluded = {doi.lower().strip() for doi in (exclude_dois or [])}
    papers = [_paper_from_work(work) for work in response.json().get("results", [])]
    requested = set(publishers or SUPPORTED_PUBLISHERS).intersection(SUPPORTED_PUBLISHERS)
    preferred = [
        paper for paper in papers
        if paper["publisher"] in requested
        and paper["doi"]
        and paper["pdf_url"]
        and int(paper["cited_by_count"] or 0) >= min_citations
        and paper["doi"].lower() not in excluded
    ]
    # OpenAlex does not always expose a direct OA PDF for conference papers.
    # Semantic Scholar's OA field provides a second, legal discovery source;
    # it is used only when OpenAlex cannot supply enough eligible papers.
    all_candidates = [*preferred]
    if len(preferred) < limit:
        try:
            fallback = httpx.get(
                SEMANTIC_SCHOLAR,
                params={
                    "query": topic,
                    "limit": 100,
                    "fields": "title,year,citationCount,externalIds,openAccessPdf,publicationVenue",
                },
                timeout=20.0,
            )
            fallback.raise_for_status()
            for work in fallback.json().get("data", []):
                paper = _paper_from_semantic_scholar(work)
                if (
                    not paper
                    or int(paper["cited_by_count"] or 0) < min_citations
                    or paper["doi"].lower() in excluded
                ):
                    continue
                if any(existing["doi"].lower() == paper["doi"].lower() for existing in all_candidates):
                    continue
                all_candidates.append(paper)
        except httpx.HTTPError:
            pass
    # A record is selectable only after confirming its OA URL responds with a
    # PDF. This is deliberately done before the UI claims availability.
    # First favor the publications the student checked. If that selection
    # cannot fill five slots, use any reputable OA record with a real PDF.
    fallback = [
        paper for paper in papers
        if paper["doi"].lower() not in excluded
        and int(paper["cited_by_count"] or 0) >= min_citations
        and paper not in preferred
    ]
    # Bound discovery so the extension stays responsive even when publishers
    # throttle or do not respond to an OA-link check.
    candidates = [*preferred, *all_candidates[len(preferred):], *fallback]
    candidates.sort(key=lambda paper: int(paper.get("cited_by_count") or 0), reverse=True)
    candidates = candidates[:25]
    with ThreadPoolExecutor(max_workers=5) as executor:
        verified = list(executor.map(_has_retrievable_pdf, candidates))
    results = [paper for paper, is_pdf in zip(candidates, verified) if is_pdf]
    for paper in results:
        if paper["publisher"] not in SUPPORTED_PUBLISHERS:
            paper["publisher"] = paper.get("source_name") or "Open-access source"
    results.sort(key=lambda paper: int(paper.get("cited_by_count") or 0), reverse=True)
    return results[:limit]


def lookup_doi(doi: str) -> dict[str, Any]:
    clean = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi.strip(), flags=re.I)
    response = httpx.get(f"{OPENALEX}/https://doi.org/{clean}", timeout=20.0)
    response.raise_for_status()
    return _paper_from_work(response.json())


def download_open_pdf(paper: dict[str, Any], destination: Path) -> Path | None:
    """Download only an OpenAlex-provided open-access PDF; never scrape publishers."""
    pdf_url = paper.get("pdf_url")
    if not pdf_url:
        return None
    response = httpx.get(
        pdf_url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchAssistant/1.0)"},
        follow_redirects=True,
        timeout=45.0,
    )
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        return None
    destination.write_bytes(response.content)
    return destination


def compare_papers(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("Add at least two readable papers to compare.")

    # Separate assistants preserve one hybrid index per paper, preventing a
    # strong passage in one paper from hiding evidence in another.
    paper_assistants: list[tuple[str, ResearchAssistant]] = []
    # Model loading dominates latency. Each paper retains its own chunks and
    # indexes, but all reuse the same immutable embedding/reranker/LLM clients.
    first_paper = ResearchAssistant()
    first_paper.add_document(paths[0])
    paper_assistants.append((paths[0].name, first_paper))
    for path in paths[1:]:
        paper = ResearchAssistant(
            embedding_service=first_paper.embedding_service,
            reranker=first_paper.reranker,
            llm_service=first_paper.llm_service,
        )
        paper.add_document(path)
        paper_assistants.append((path.name, paper))

    def retrieve_aspect_chunks(paper: ResearchAssistant, question: str) -> list[dict[str, Any]]:
        """Hybrid retrieve aspect evidence from several relevant PDF pages.

        Unlike normal Q&A section expansion, literature comparison must not
        restrict all aspects to one winning page. This keeps one strong chunk
        per page before adding additional high-scoring evidence.
        """
        expanded = expand_query(question)
        variants = [question] if expanded == question else [question, expanded]
        vectors = paper.embedding_service.embed_query_variants(variants)
        dense = paper.vector_store.search_many(vectors, 30)
        lexical = paper.bm25_retriever.search(expanded, 30)
        candidates = hybrid_search(dense, lexical, top_k=30)
        reranked = paper.reranker.rerank(expanded, candidates, top_k=20)
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        selected_pages: set[int] = set()
        for chunk in reranked:
            chunk_id = str(chunk.get("chunk_id") or "")
            page = int(chunk.get("page_number") or 0)
            if chunk_id and page not in selected_pages:
                selected.append(chunk)
                selected_ids.add(chunk_id)
                selected_pages.add(page)
            if len(selected) >= 4:
                break
        for chunk in reranked:
            chunk_id = str(chunk.get("chunk_id") or "")
            if chunk_id and chunk_id not in selected_ids:
                selected.append(chunk)
                selected_ids.add(chunk_id)
            if len(selected) >= 6:
                break
        return selected

    def extractive_evidence(paper: ResearchAssistant, aspect: str, question: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """Keep comparison useful if an external LLM request fails.

        This follows the same extension-owned hybrid retrieval path (dense +
        BM25 + fusion + reranking), then creates a grounded extractive answer
        from the cited passages instead of returning an empty table cell.
        """
        expanded = expand_query(question)
        if not chunks:
            return {"answer": "No directly relevant evidence was found in this paper.", "sources": []}
        question_terms = set(re.findall(r"[a-zA-Z]{4,}", expanded.lower())) | ASPECT_TERMS[aspect]
        candidates: list[tuple[int, int, str, dict[str, Any]]] = []
        seen = set()
        for chunk_index, chunk in enumerate(chunks):
            page_text = str(chunk.get("text", ""))
            for sentence in re.split(r"(?<=[.!?])\s+", page_text):
                cleaned = re.sub(r"\s+", " ", sentence).strip()
                normalized = cleaned.lower()
                if (
                    len(cleaned) < 45
                    or len(cleaned) > 420
                    or normalized in seen
                    or any(noise in normalized for noise in ("issn", "copyright", "www.", "http", "references"))
                ):
                    continue
                seen.add(normalized)
                terms = set(re.findall(r"[a-zA-Z]{4,}", normalized))
                score = len(question_terms.intersection(terms))
                candidates.append((score, -chunk_index, cleaned, chunk))
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = [item for item in candidates if item[0] > 0][:2]
        if not selected:
            return {
                "answer": f"Not explicitly reported in the retrieved evidence for {aspect.lower()}.",
                "sources": [],
            }
        return {
            "answer": " ".join(sentence for _, _, sentence, _ in selected),
            "sources": paper.llm_service._unique_sources([chunk for _, _, _, chunk in selected]),
        }

    def is_substantive_answer(answer: str) -> bool:
        body = answer.split("Sources:", 1)[0].strip()
        normalized = re.sub(r"[^a-zA-Z]", "", body).lower()
        return (
            len(normalized) >= 70
            and not body.lower().startswith("the uploaded documents do not contain enough")
            and body.lower() not in {"sources", "source"}
        )

    def not_reported(aspect: str) -> dict[str, Any]:
        return {
            "answer": f"Not explicitly reported in the retrieved evidence for {aspect.lower()}.",
            "sources": [],
            "mode": "not reported",
        }

    comparison: dict[str, list[dict[str, Any]]] = {aspect: [] for aspect in ASPECTS}
    llm_errors: list[str] = []
    for name, paper in paper_assistants:
        field_chunks = {aspect: retrieve_aspect_chunks(paper, question) for aspect, question in ASPECTS.items()}
        try:
            generated = paper.llm_service.generate_comparison_fields(field_chunks)
            for aspect in ASPECTS:
                answer = generated[aspect]
                not_supported = answer.lower().startswith("not explicitly reported")
                comparison[aspect].append({
                    "document_name": name,
                    "answer": answer,
                    "sources": [] if not_supported else paper.llm_service._unique_sources(field_chunks[aspect]),
                    "mode": "not reported" if not_supported else "LLM-grounded",
                })
        except LLMGenerationError as exc:
            error_text = str(exc)
            if error_text not in llm_errors:
                llm_errors.append(error_text)
            for aspect, question in ASPECTS.items():
                result = extractive_evidence(paper, aspect, question, field_chunks[aspect])
                mode = "cited extract" if result["sources"] else "not reported"
                comparison[aspect].append({"document_name": name, **result, "mode": mode})

    contradictions = []
    review_aspects = ("Algorithms", "Datasets", "Results", "Methodology")
    evidence_text = "\n\n".join(
        f"{aspect}:\n" + "\n".join(f"- {item['document_name']}: {item['answer']}" for item in comparison[aspect])
        for aspect in review_aspects
    )
    try:
        conflict_review = paper_assistants[0][1].llm_service.analyze_comparison_conflicts(evidence_text)
        for line in conflict_review.splitlines():
            parts = [part.strip() for part in line.split("|", 2)]
            if len(parts) == 3 and parts[0] in review_aspects and parts[1].upper() == "YES":
                contradictions.append({
                    "aspect": parts[0],
                    "status": "Review required",
                    "explanation": parts[2],
                    "evidence": [item for item in comparison[parts[0]] if item["sources"]],
                })
    except LLMGenerationError:
        # Without a configured LLM it is safer to present the comparison
        # evidence and make no automated contradiction claim.
        pass

    return {
        "comparison": comparison,
        "contradictions": contradictions,
        "documents": [name for name, _ in paper_assistants],
        "llm_errors": llm_errors,
    }
