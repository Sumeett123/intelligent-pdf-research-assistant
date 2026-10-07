from __future__ import annotations

from typing import Any, Dict, Iterable, List


def make_citation(document_name: str, page_number: int, chunk_id: str) -> Dict[str, Any]:
    return {
        "document_name": document_name,
        "page_number": page_number,
        "chunk_id": chunk_id,
    }


def format_sources(sources: Iterable[Dict[str, Any]]) -> str:
    grouped: Dict[str, set[int]] = {}
    for source in sources:
        document_name = source.get("document_name", "Unknown")
        page_number = int(source.get("page_number", 0))
        grouped.setdefault(document_name, set()).add(page_number)

    lines: List[str] = []
    for document_name, pages in grouped.items():
        lines.append(f"📄 {document_name}")
        for page in sorted(pages):
            lines.append(f"Page {page}")
    return "\n".join(lines) if lines else "No sources available."
