from backend.assistant import ResearchAssistant


def test_answer_context_expands_only_within_matching_section():
    assistant = ResearchAssistant.__new__(ResearchAssistant)
    assistant.chunks = [
        {"chunk_id": "doc-p6-c1", "document_name": "doc.pdf", "page_number": 6, "section_index": 1, "text": "MIS definition"},
        {"chunk_id": "doc-p6-c2", "document_name": "doc.pdf", "page_number": 6, "section_index": 1, "text": "MIS characteristics"},
        {"chunk_id": "doc-p6-c3", "document_name": "doc.pdf", "page_number": 6, "section_index": 2, "text": "DSS section"},
        {"chunk_id": "doc-p7-c1", "document_name": "doc.pdf", "page_number": 7, "section_index": 2, "text": "Unrelated page"},
    ]
    reranked = [dict(assistant.chunks[0], rerank_score=2.0)]

    selected = assistant._select_answer_chunks(reranked)

    assert [chunk["chunk_id"] for chunk in selected] == ["doc-p6-c1", "doc-p6-c2"]


def test_question_section_can_continue_across_pages():
    from backend.chunker import chunk_pages

    pages = [
        {"document_name": "doc.pdf", "page_number": 1, "text": "Q1. Explain the method. Ans: The method starts here."},
        {"document_name": "doc.pdf", "page_number": 2, "text": "The method continues with the final steps."},
        {"document_name": "doc.pdf", "page_number": 3, "text": "Q2. Explain another method. Ans: A different method."},
    ]

    chunks = chunk_pages(pages, chunk_size=200, overlap=0)

    assert chunks[0]["section_index"] == chunks[1]["section_index"]
    assert chunks[1]["section_index"] != chunks[2]["section_index"]


def test_non_question_bank_pages_do_not_share_one_section():
    from backend.chunker import chunk_pages

    pages = [
        {"document_name": "report.pdf", "page_number": 1, "text": "Introduction to the report."},
        {"document_name": "report.pdf", "page_number": 2, "text": "Methods used in the report."},
    ]

    chunks = chunk_pages(pages, chunk_size=200, overlap=0)

    assert chunks[0]["section_index"] != chunks[1]["section_index"]


def test_relevance_gate_rejects_generic_chunk_for_distinctive_query():
    assistant = ResearchAssistant.__new__(ResearchAssistant)
    generic_chunk = {"text": "An information system combines hardware and software."}

    assert assistant._matches_question("Explain the heat diagram", generic_chunk) is False
    assert assistant._matches_question("Explain information systems", generic_chunk) is True


def test_relevant_section_uses_retrieval_metadata_when_section_is_nested():
    assistant = ResearchAssistant.__new__(ResearchAssistant)
    assistant.chunks = [
        {"chunk_id": "doc-p1-c1", "document_name": "doc.pdf", "page_number": 1, "section_index": 1, "text": "Types of information systems include TPS."},
        {"chunk_id": "doc-p2-c1", "document_name": "doc.pdf", "page_number": 2, "section_index": 2, "text": "Unrelated advantages of computers."},
    ]
    reranked = [
        {
            "chunk_id": "doc-p1-c1",
            "metadata": assistant.chunks[0],
            "text": assistant.chunks[0]["text"],
            "combined_score": 1.0,
        },
        {
            "chunk_id": "doc-p2-c1",
            "metadata": assistant.chunks[1],
            "text": assistant.chunks[1]["text"],
            "combined_score": 0.9,
        },
    ]

    selected = assistant._select_relevant_section("Explain the types of information systems", reranked)

    assert [chunk["chunk_id"] for chunk in selected] == ["doc-p1-c1"]


def test_fallback_answer_does_not_repeat_question_header():
    from backend.llm_service import LLMService

    service = LLMService.__new__(LLMService)
    result = service._fallback_answer(
        "Explain the importance of the information system to society.",
        [{"document_name": "doc.pdf", "page_number": 1, "text": "Q1. Explain the importance of the information system to society. Ans: Information systems support coordination and decision-making."}],
    )

    assert "Explain the importance" not in result["answer"]
    assert "support coordination" in result["answer"]