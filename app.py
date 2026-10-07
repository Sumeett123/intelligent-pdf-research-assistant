from __future__ import annotations

from pathlib import Path

import streamlit as st

from backend.assistant import DEFAULT_TOP_K, ResearchAssistant

st.set_page_config(page_title="Intelligent PDF Research Assistant", layout="wide")
st.title("Intelligent PDF Research Assistant")
st.caption("Ask questions about your research documents using hybrid retrieval and grounded generation.")

INDEX_SCHEMA_VERSION = "section-retrieval-v3"
if st.session_state.get("index_schema_version") != INDEX_SCHEMA_VERSION:
    st.session_state.assistant = ResearchAssistant()
    st.session_state.index_schema_version = INDEX_SCHEMA_VERSION
    st.session_state.messages = []
    st.session_state.processed_files = set()
elif "assistant" not in st.session_state:
    st.session_state.assistant = ResearchAssistant()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

assistant = st.session_state.assistant

with st.sidebar:
    st.header("Document settings")
    chunk_size = st.number_input("Chunk size", min_value=300, max_value=2000, value=900, step=50)
    chunk_overlap = st.number_input("Chunk overlap", min_value=50, max_value=300, value=120, step=10)
    retrieval_mode = st.selectbox("Retrieval mode", ["Hybrid Search", "Vector Search", "BM25 Search"])
    provider = st.selectbox("LLM provider", ["groq", "openai"], index=0 if assistant.llm_service.provider == "groq" else 1)
    model_options = {
        "groq": ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "allam-2-7b"],
        "openai": ["gpt-4.1", "gpt-4o", "gpt-4o-mini"],
    }
    configured_model = assistant.llm_service.model
    default_model_index = model_options[provider].index(configured_model) if configured_model in model_options[provider] else 0
    llm_model = st.selectbox("LLM model", model_options[provider], index=default_model_index)
    assistant.llm_service.provider = provider
    assistant.llm_service.model = llm_model
    if not assistant.llm_service.has_credentials():
        st.warning(
            f"No {provider.upper()} API key is configured. Answers will use grounded document extraction, "
            "not generated LLM prose. Add the key to .env and restart the app for natural-language answers."
        )

    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            if file.name in st.session_state.processed_files:
                continue
            save_path = Path("data") / file.name
            save_path.parent.mkdir(exist_ok=True)
            save_path.write_bytes(file.getvalue())
            try:
                assistant.add_document(save_path, chunk_size=int(chunk_size), overlap=int(chunk_overlap))
                st.session_state.processed_files.add(file.name)
                st.success(f"Loaded {file.name}")
            except Exception as exc:  # pragma: no cover - UI error handling
                st.error(f"Failed to process {file.name}: {exc}")

    if st.button("Clear documents"):
        assistant.clear_documents()
        st.session_state.processed_files.clear()
        st.session_state.messages = []
        st.rerun()

    st.caption(f"Model configured: {assistant.llm_service.configuration_status()}")

st.subheader("Documents")
documents = assistant.list_documents()
if not documents:
    st.info("Upload one or more PDFs to begin indexing and retrieval.")
else:
    for doc in documents:
        with st.container():
            st.markdown(f"### {doc['document_name']}")
            st.caption(f"{doc['page_count']} pages • {doc['chunk_count']} chunks • {doc['status']}")

st.subheader("Chat")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            with st.expander("Sources Used", expanded=False):
                for source in message["sources"]:
                    document_name = source.get("document_name") or "Unknown document"
                    page_number = source.get("page_number") or "?"
                    st.markdown(f"- **{document_name}** — Page {page_number}")

prompt = st.chat_input("Ask about your documents")
if prompt:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    try:
        response = assistant.ask(
            question=prompt,
            retrieval_mode=retrieval_mode,
            top_k=DEFAULT_TOP_K,
            rerank_top_k=DEFAULT_TOP_K,
        )
        answer = response["answer"]
        sources = response["sources"]
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        with st.chat_message("assistant"):
            st.write(answer)
            with st.expander("Sources Used", expanded=False):
                for source in sources:
                    document_name = source.get("document_name") or "Unknown document"
                    page_number = source.get("page_number") or "?"
                    st.markdown(f"- **{document_name}** — Page {page_number}")
    except Exception as exc:  # pragma: no cover - UI error handling
        error_message = f"Error: {exc}"
        st.session_state.messages.append({"role": "assistant", "content": error_message, "sources": []})
        with st.chat_message("assistant"):
            st.error(error_message)
