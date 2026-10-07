from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Union

import fitz


def clean_extracted_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    # PDF extraction commonly inserts a newline in the middle of hyphenated words.
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    meaningful_lines = [line for line in lines if line]
    return "\n".join(meaningful_lines).strip()


def extract_pages_from_pdf(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        return []

    try:
        document = fitz.open(str(path))
    except Exception:
        return []

    pages: List[Dict[str, Any]] = []
    for page_number in range(document.page_count):
        page = document[page_number]
        text = clean_extracted_text(page.get_text("text"))
        if not text:
            continue
        pages.append(
            {
                "document_name": path.name,
                "page_number": page_number + 1,
                "text": text,
                "image_count": len(page.get_images(full=True)),
            }
        )

    document.close()
    return pages


def extract_multiple_pdfs(file_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
    pages: List[Dict[str, Any]] = []
    for file_path in file_paths:
        pages.extend(extract_pages_from_pdf(file_path))
    return pages
