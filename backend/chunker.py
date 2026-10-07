from __future__ import annotations

import re
from typing import Any, Dict, List


QUESTION_START = re.compile(r"^\s*(?:q(?:uestion)?\s*\d+|q\d+)\s*[.):\-]?\s*", re.IGNORECASE)


def _question_sections(text: str) -> List[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    starts = [index for index, line in enumerate(lines) if QUESTION_START.match(line)]
    if len(starts) < 2:
        return ["\n".join(lines)]

    sections: List[str] = []
    if starts[0] > 0:
        sections.append("\n".join(lines[:starts[0]]))
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(lines)
        sections.append("\n".join(lines[start:end]))
    return sections


def _section_parts(text: str) -> List[tuple[bool, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    starts = [index for index, line in enumerate(lines) if QUESTION_START.match(line)]
    if not starts:
        return [(False, "\n".join(lines))]

    parts: List[tuple[bool, str]] = []
    if starts[0] > 0:
        parts.append((False, "\n".join(lines[:starts[0]])))
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(lines)
        parts.append((True, "\n".join(lines[start:end])))
    return parts


def _starts_numbered_list(text: str) -> bool:
    return bool(re.search(r"(?m)^\s*\d+\.\s+", text))


def _sentence_units(text: str) -> List[str]:
    paragraphs = re.split(r"\n\s*\n+", text.strip())
    units: List[str] = []
    for paragraph in paragraphs:
        paragraph = re.sub(r"\s*\n\s*", " ", paragraph).strip()
        if not paragraph:
            continue
        units.extend(sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", paragraph) if sentence.strip())
    return units


def _split_long_unit(unit: str, chunk_size: int) -> List[str]:
    words = unit.split()
    pieces: List[str] = []
    current: List[str] = []
    current_length = 0
    for word in words:
        next_length = current_length + len(word) + (1 if current else 0)
        if current and next_length > chunk_size:
            pieces.append(" ".join(current))
            current = [word]
            current_length = len(word)
        else:
            current.append(word)
            current_length = next_length
    if current:
        pieces.append(" ".join(current))
    return pieces or [unit]


def _overlap_words(text: str, overlap: int) -> str:
    if overlap <= 0:
        return ""
    words: List[str] = []
    length = 0
    for word in reversed(text.split()):
        next_length = length + len(word) + (1 if words else 0)
        if words and next_length > overlap:
            break
        words.append(word)
        length = next_length
    return " ".join(reversed(words))


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    cleaned = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n").replace("\r", "\n")).strip()
    if not cleaned:
        return []

    units = _sentence_units(cleaned)
    if not units:
        return [cleaned]

    chunks: List[str] = []
    current = ""
    for unit in units:
        for piece in _split_long_unit(unit, chunk_size) if len(unit) > chunk_size else [unit]:
            candidate = f"{current} {piece}".strip()
            if current and len(candidate) > chunk_size:
                chunks.append(current)
                overlap_text = _overlap_words(current, overlap)
                current = f"{overlap_text} {piece}".strip()
                while len(current) > chunk_size and overlap_text:
                    overlap_text = overlap_text.split(" ", 1)[-1] if " " in overlap_text else ""
                    current = f"{overlap_text} {piece}".strip()
            else:
                current = candidate

    if current:
        chunks.append(current)
    return chunks or [cleaned]


def chunk_pages(pages: List[Dict[str, Any]], chunk_size: int = 900, overlap: int = 120) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    active_section_index = 0
    previous_text = ""
    continued_numbered_section = False
    has_question_structure = any(
        QUESTION_START.match(line.strip())
        for page in pages
        for line in str(page.get("text", "")).splitlines()
        if line.strip()
    )
    for page in pages:
        text = page.get("text", "")
        sections = _section_parts(text) or [(False, text)]
        chunk_index = 0
        if not has_question_structure:
            continues_section = _starts_numbered_list(text) and (
                "implications for management" in previous_text.lower() or continued_numbered_section
            )
            if not continues_section:
                active_section_index += 1
            continued_numbered_section = _starts_numbered_list(text)
        for starts_new_section, section in sections:
            if has_question_structure and (starts_new_section or active_section_index == 0):
                active_section_index += 1
            split_chunks = chunk_text(section, chunk_size=chunk_size, overlap=overlap)
            for chunk in split_chunks:
                chunk_index += 1
                chunk_id = f"{page['document_name']}-p{page['page_number']}-c{chunk_index}"
                chunks.append(
                    {
                        "document_name": page["document_name"],
                        "page_number": page["page_number"],
                        "chunk_id": chunk_id,
                        "section_index": active_section_index,
                        "text": chunk,
                    }
                )
        previous_text = str(text)
    return chunks
