import pytest

from backend.llm_service import LLMGenerationError, LLMService


def test_answer_generation_does_not_use_local_fallback(monkeypatch):
    service = LLMService.__new__(LLMService)
    service.provider = "groq"
    service.model = "test-model"

    def fail_provider(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(service, "_provider_call", fail_provider)

    with pytest.raises(LLMGenerationError, match="could not generate an answer"):
        service.generate(
            "What is the answer?",
            [{"document_name": "doc.pdf", "page_number": 1, "text": "Evidence."}],
        )


def test_summary_generation_does_not_use_local_fallback(monkeypatch):
    service = LLMService.__new__(LLMService)
    service.provider = "groq"
    service.model = "test-model"

    def fail_provider(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(service, "_provider_call", fail_provider)

    with pytest.raises(LLMGenerationError, match="could not generate a summary"):
        service.generate_summary(
            "Summarize the document.",
            [{"document_name": "doc.pdf", "page_number": 1, "text": "Evidence."}],
        )
