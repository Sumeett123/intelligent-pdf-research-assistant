from backend.summarization import DocumentSummarizer, is_summary_request


def test_summary_request_detection():
    assert is_summary_request("Give a summary of the PDF")
    assert is_summary_request("What is the overview of this document?")
    assert not is_summary_request("What are the advantages of information systems?")


def test_summary_context_is_document_ordered_and_bounded():
    summarizer = DocumentSummarizer()
    chunks = [
        {"document_name": "b.pdf", "page_number": 2, "chunk_id": "b-p2-c1", "text": "B"},
        {"document_name": "a.pdf", "page_number": 2, "chunk_id": "a-p2-c1", "text": "A2"},
        {"document_name": "a.pdf", "page_number": 1, "chunk_id": "a-p1-c1", "text": "A1"},
    ]

    selected = summarizer.select_context(chunks)

    assert [chunk["chunk_id"] for chunk in selected] == ["a-p1-c1", "a-p2-c1", "b-p2-c1"]
