from __future__ import annotations

import os
import re
import uuid
import httpx
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.assistant import DEFAULT_TOP_K, ResearchAssistant
from backend.llm_service import LLMGenerationError
from backend.extension_research import SUPPORTED_PUBLISHERS, compare_papers, download_open_pdf, lookup_doi, search_papers

app = FastAPI(title="PDF Hybrid RAG API")
assistant = ResearchAssistant()
extension_sessions: Dict[str, ResearchAssistant] = {}

# This is a local demonstration backend bound to the user's machine. The
# extension may execute under a Chrome-extension origin or the origin of the
# page being read, so allow browser origins to reach the local API. Replace
# this with an explicit production allow-list before deploying publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    retrieval_mode: str = "Hybrid Search"
    top_k: int = DEFAULT_TOP_K
    rerank_top_k: int = DEFAULT_TOP_K


class ExtensionIngestRequest(BaseModel):
    session_id: str | None = None
    title: str
    url: str
    content: str


class ExtensionAskRequest(BaseModel):
    session_id: str
    question: str
    chat_history: str = ""


class ExtensionPaperSearchRequest(BaseModel):
    topic: str
    exclude_dois: list[str] = []
    publishers: list[str] = ["IEEE", "Springer", "PLOS", "Frontiers", "MDPI"]
    min_citations: int = 50


class ExtensionDoiRequest(BaseModel):
    doi: str
    publishers: list[str] = ["IEEE", "Springer", "PLOS", "Frontiers", "MDPI"]


def _extension_document_name(title: str, url: str) -> str:
    clean_title = re.sub(r"[^a-zA-Z0-9._ -]", "", title).strip()[:100] or "Browser page"
    return f"{clean_title} ({url[:120]})"


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/documents")
def documents() -> Dict[str, Any]:
    return {"documents": assistant.list_documents(), "stats": assistant.get_document_stats()}


@app.post("/clear")
def clear_documents() -> Dict[str, str]:
    assistant.clear_documents()
    return {"status": "documents cleared"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    temp_dir = Path("data")
    temp_dir.mkdir(exist_ok=True)
    target = temp_dir / file.filename
    contents = await file.read()
    target.write_bytes(contents)

    try:
        info = assistant.add_document(target)
        return {"status": "uploaded", "document": info}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ask")
def ask_question(request: AskRequest) -> Dict[str, Any]:
    try:
        return assistant.ask(
            question=request.question,
            retrieval_mode=request.retrieval_mode,
            top_k=request.top_k,
            rerank_top_k=request.rerank_top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMGenerationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/extension/ingest")
def ingest_extension_page(request: ExtensionIngestRequest) -> Dict[str, Any]:
    """Create or refresh an isolated RAG session for one browser page."""
    if len(request.content.strip()) < 40:
        raise HTTPException(status_code=400, detail="The page does not contain enough readable text.")
    if len(request.content) > 1_000_000:
        raise HTTPException(status_code=413, detail="The page text is too large to process.")

    session_id = request.session_id or str(uuid.uuid4())
    session_assistant = extension_sessions.setdefault(session_id, ResearchAssistant())
    document = session_assistant.add_text_document(
        _extension_document_name(request.title, request.url),
        request.content,
    )
    return {"session_id": session_id, "document": document}


@app.post("/extension/ask")
def ask_extension_page(request: ExtensionAskRequest) -> Dict[str, Any]:
    session_assistant = extension_sessions.get(request.session_id)
    if not session_assistant:
        raise HTTPException(status_code=404, detail="Page session expired. Refresh the page context and try again.")
    try:
        return session_assistant.ask(question=request.question, chat_history=request.chat_history)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMGenerationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/extension/research/search")
def search_extension_papers(request: ExtensionPaperSearchRequest) -> Dict[str, Any]:
    """Find five candidate papers for the extension, without touching app state."""
    if len(request.topic.strip()) < 3:
        raise HTTPException(status_code=400, detail="Enter a topic with at least three characters.")
    try:
        return {
            "papers": search_papers(
                request.topic.strip(),
                exclude_dois=request.exclude_dois,
                publishers=request.publishers,
                min_citations=max(50, request.min_citations),
            )
        }
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Paper search is temporarily unavailable.") from exc


@app.post("/extension/research/doi")
def validate_extension_doi(request: ExtensionDoiRequest) -> Dict[str, Any]:
    """Resolve citation metadata only; the extension never downloads the PDF."""
    try:
        paper = lookup_doi(request.doi)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=404, detail="DOI metadata could not be found.") from exc
    if paper["publisher"] not in set(request.publishers).intersection(SUPPORTED_PUBLISHERS):
        raise HTTPException(status_code=400, detail="This DOI is not from one of your selected supported publishers.")
    if not paper["pdf_url"]:
        raise HTTPException(status_code=400, detail="No open-access full-text link is available for this DOI.")
    return {"paper": paper}


@app.post("/extension/research/compare")
async def compare_extension_papers(
    request: Request,
) -> Dict[str, Any]:
    """Compare PDFs in an extension-owned temporary directory only."""
    # Parse multipart data explicitly. Chrome content scripts can append files
    # from dynamically-created inputs, which is more reliable here than a
    # typed FastAPI list binding.
    form = await request.form()
    papers_payload = str(form.get("papers", "[]"))
    try:
        requested_papers = __import__("json").loads(papers_payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Selected papers must be a JSON list.") from exc
    if not isinstance(requested_papers, list):
        raise HTTPException(status_code=400, detail="Selected papers must be a list.")

    # Keep extension uploads out of the repository's data directory and out
    # of the Streamlit application's document lifecycle.
    session_dir = Path(tempfile.gettempdir()) / "research_assistant_extension" / str(uuid.uuid4())
    session_dir.mkdir(parents=True, exist_ok=True)
    uploaded_files = [
        value for _, value in form.multi_items()
        if hasattr(value, "filename") and hasattr(value, "read") and str(getattr(value, "filename", "")).lower().endswith(".pdf")
    ]
    for upload in uploaded_files[:5]:
        target = session_dir / f"uploaded-{uuid.uuid4().hex[:8]}-{Path(upload.filename).name}"
        target.write_bytes(await upload.read())

    def fetch_selected_paper(item: Any) -> tuple[Path | None, str]:
        doi = str(item.get("doi", "")) if isinstance(item, dict) else ""
        selected_publisher = str(item.get("publisher", "")) if isinstance(item, dict) else ""
        selected_pdf_url = str(item.get("pdf_url", "")) if isinstance(item, dict) else ""
        try:
            paper = lookup_doi(doi)
            # Preferred IEEE/Springer/etc. records retain their known source.
            # Fallback records are allowed only when the search already
            # verified their direct open-access PDF link.
            known_source = paper["publisher"] in SUPPORTED_PUBLISHERS
            if (known_source and paper["publisher"] != selected_publisher) or not selected_pdf_url.startswith("https://"):
                return None, doi
            safe_name = re.sub(r"[^a-zA-Z0-9._ -]", "_", paper["title"])[:80] or "paper"
            if selected_pdf_url.startswith("https://"):
                paper["pdf_url"] = selected_pdf_url
            target = download_open_pdf(paper, session_dir / f"{uuid.uuid4().hex[:8]}-{safe_name}.pdf")
            return target, "" if target else doi
        except (httpx.HTTPError, ValueError):
            return None, doi

    # Full-text requests are independent. Fetch them concurrently rather than
    # making students wait once per selected paper.
    fetched = []
    if requested_papers:
        with ThreadPoolExecutor(max_workers=min(5, len(requested_papers))) as executor:
            fetched = list(executor.map(fetch_selected_paper, requested_papers[:5]))
    paths = list(session_dir.glob("uploaded-*.pdf")) + [path for path, _ in fetched if path]
    unavailable = [doi for _, doi in fetched if doi]
    if len(paths) < 2:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Fewer than two selected papers currently have retrievable open-access full text.")
    try:
        result = compare_papers(paths)
        result["unavailable_dois"] = unavailable
        return result
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        shutil.rmtree(session_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("API_PORT", "8000")))
