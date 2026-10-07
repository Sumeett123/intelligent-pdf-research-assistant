from backend.chunker import chunk_pages, chunk_text


def test_chunk_text_keeps_sentence_boundaries_and_word_safe_overlap():
    chunks = chunk_text(
        "Sentence one is useful. Sentence two carries evidence. Sentence three adds detail.",
        chunk_size=45,
        overlap=12,
    )

    assert len(chunks) == 3
    assert all(chunk == chunk.strip() for chunk in chunks)
    assert all(not chunk.startswith(("e ", "es ")) for chunk in chunks[1:])
    assert "evidence" in chunks[1]


def test_chunk_text_splits_long_sentences_without_losing_words():
    chunks = chunk_text("alpha beta gamma delta epsilon zeta", chunk_size=18, overlap=0)

    assert chunks == ["alpha beta gamma", "delta epsilon zeta"]


def test_chunk_pages_keeps_question_sections_separate():
    pages = [
        {
            "document_name": "questions.pdf",
            "page_number": 6,
            "text": "Q1. Explain MIS. Ans: MIS integrates operational data for managers.\n"
            "Q2. Explain DSS. Ans: DSS supports complex decisions.",
        }
    ]

    chunks = chunk_pages(pages, chunk_size=200, overlap=0)

    assert len(chunks) == 2
    assert chunks[0]["section_index"] != chunks[1]["section_index"]
    assert "Q2" not in chunks[0]["text"]
    assert "Q1" not in chunks[1]["text"]


def test_numbered_section_continues_across_pages():
    pages = [
        {"document_name": "report.pdf", "page_number": 1, "text": "Implications for Management\nThese impacts matter."},
        {"document_name": "report.pdf", "page_number": 2, "text": "1. Understanding the changing context of IT."},
        {"document_name": "report.pdf", "page_number": 3, "text": "2. Keeping abreast of developments."},
    ]

    chunks = chunk_pages(pages, chunk_size=200, overlap=0)

    assert {chunk["section_index"] for chunk in chunks} == {1}