import fitz

from backend.pdf_processor import extract_pages_from_pdf


def test_extract_pages_from_pdf_returns_clean_page_text(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This sample PDF explains the hybrid retrieval method.")
    doc.save(pdf_path)
    doc.close()

    pages = extract_pages_from_pdf(pdf_path)

    assert len(pages) == 1
    assert pages[0]["page_number"] == 1
    assert "hybrid retrieval" in pages[0]["text"].lower()


def test_extract_pages_from_pdf_rejects_empty_or_invalid_file(tmp_path):
    invalid_path = tmp_path / "invalid.pdf"
    invalid_path.write_bytes(b"not a pdf")

    pages = extract_pages_from_pdf(invalid_path)
    assert pages == []
