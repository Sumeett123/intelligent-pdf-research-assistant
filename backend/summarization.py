from __future__ import annotations

import re
from typing import Any, Dict, List


SUMMARY_CONTEXT_LIMIT = 24
_SUMMARY_PATTERNS = (
    r"\bsummary\b",
    r"\bsummarize\b",
    r"\bsummarise\b",
    r"\boverview\b",
    r"\bbrief(?:ly)?\b",
    r"\bkey points?\b",
    r"\bmain points?\b",
    r"\bwhole document\b",
    r"\bentire document\b",
    r"\bwhole pdf\b",
    r"\bentire pdf\b",
)


def is_summary_request(question: str) -> bool:
    """Return whether the user is asking for a document-level summary."""
    normalized = question.strip().lower()
    return bool(normalized and any(re.search(pattern, normalized) for pattern in _SUMMARY_PATTERNS))


class DocumentSummarizer:
    """Coordinates document-wide summary context without changing Q&A retrieval."""

    def select_context(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ordered = sorted(
            (chunk for chunk in chunks if str(chunk.get("text", "")).strip()),
            key=lambda chunk: (
                str(chunk.get("document_name", "")),
                int(chunk.get("page_number", 0)),
                str(chunk.get("chunk_id", "")),
            ),
        )
        return ordered[:SUMMARY_CONTEXT_LIMIT]

    def summarize(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        llm_service: Any,
    ) -> Dict[str, Any]:
        context = self.select_context(chunks)
        if not context:
            return {
                "answer": "The uploaded documents do not contain enough information to summarize.",
                "sources": [],
            }
        return llm_service.generate_summary(question, context)
