from backend.citation import make_citation, format_sources


def test_make_citation_uses_document_and_page_number():
    citation = make_citation("research.pdf", 4, "chunk-12")
    assert citation["document_name"] == "research.pdf"
    assert citation["page_number"] == 4
    assert citation["chunk_id"] == "chunk-12"


def test_format_sources_groups_sources_by_document():
    sources = [
        make_citation("paper.pdf", 2, "c1"),
        make_citation("paper.pdf", 5, "c2"),
        make_citation("notes.pdf", 1, "c3"),
    ]
    rendered = format_sources(sources)
    assert "paper.pdf" in rendered
    assert "notes.pdf" in rendered
    assert "Page 2" in rendered
    assert "Page 5" in rendered
