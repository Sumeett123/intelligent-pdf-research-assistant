from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.assistant import DEFAULT_TOP_K, ResearchAssistant

app = FastAPI(title="PDF Hybrid RAG API")
assistant = ResearchAssistant()


class AskRequest(BaseModel):
    question: str
    retrieval_mode: str = "Hybrid Search"
    top_k: int = DEFAULT_TOP_K
    rerank_top_k: int = DEFAULT_TOP_K


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("API_PORT", "8000")))
