# Major Project Booklet Solution

## Project title

**Intelligent PDF Research Assistant Using Hybrid Retrieval-Augmented Generation**

Replace the bracketed personal information below before submitting the booklet.

## Cover page

- Academic year: 2026-27
- Year / Semester / Division: [fill in]
- Team members, Moodle IDs, email addresses and mobile numbers: [fill in]
- Guide name: [fill in]
- Title: *Intelligent PDF Research Assistant Using Hybrid Retrieval-Augmented Generation*

### Objectives

1. Build a system that accepts one or more PDF documents and extracts clean, page-level text.
2. Convert document passages into searchable semantic embeddings and maintain a keyword index.
3. Retrieve relevant evidence through vector search, BM25 search, or hybrid retrieval using reciprocal-rank fusion.
4. Rerank retrieved passages with a cross-encoder and generate answers grounded only in the selected context.
5. Show the document name and page number for every answer source, and support document-wide summaries.
6. Extend the same retrieval pipeline to readable text from web pages through a browser extension.

## Abstract

Research students and professionals often need to locate precise facts within long PDF collections. Conventional keyword search fails when the user and the document use different words for the same idea, while semantic-only search may miss exact technical vocabulary. This project presents an Intelligent PDF Research Assistant that combines both approaches. The system extracts text from uploaded PDFs, splits it into section-aware chunks, creates dense vector embeddings with BAAI/bge-small-en-v1.5, and indexes the same chunks using BM25. It supports vector, BM25, and hybrid search; the hybrid mode merges rankings through reciprocal-rank fusion and then applies a cross-encoder reranker. The final answer is generated only from the selected passages and includes document/page citations. A Streamlit interface supports multi-PDF upload and chat, while a FastAPI service exposes the workflow to a lightweight browser extension for questions about the current web page. The project improves access to academic and technical information by reducing manual reading time and making evidence traceable.

## SDG goals to tick

- **Goal 4 - Quality Education:** primary alignment. The system improves access to learning material and makes research documents easier to explore.
- **Goal 9 - Industry, Innovation and Infrastructure:** secondary alignment. It demonstrates an AI-based information-retrieval system using modern language models and scalable indexing concepts.

## Program outcomes to tick

Tick these with the following justification available during review: **PO1, PO2, PO3, PO4, PO5, PO8, PO9, PO10, PO11 and PO12**.

- PO1/PO2: applies information retrieval, NLP, vector similarity and ranking concepts to a real document-search problem.
- PO3: designs a complete application with UI, API, ingestion, indexing, retrieval, reranking and answer-generation layers.
- PO4: evaluates retrieval through a labelled question set and ranking metrics such as Recall@K and MRR.
- PO5: uses PyMuPDF, Sentence Transformers, FAISS, BM25, FastAPI, Streamlit and browser-extension APIs.
- PO8: promotes grounded answers and source citations; user data/API keys must be handled responsibly.
- PO9/PO10/PO11: supports team implementation, reporting, demonstrations, milestones and deployment planning.
- PO12: requires independent study of RAG, embeddings, reranking and current LLM tools.

## Content for Major Project Presentation 1

### Problem statement

Users waste time manually scanning large PDFs and cannot reliably find an answer when their wording differs from the document. Existing keyword search lacks semantic understanding; a generic chatbot can answer without evidence. The proposed solution provides accurate, cited answers from the uploaded documents.

### Literature review and novelty

Review the following topics in the presentation: BM25 lexical retrieval, dense passage retrieval, reciprocal-rank fusion, cross-encoder reranking, retrieval-augmented generation, and citation-grounded question answering. The project novelty is its practical combination of: (1) section-aware PDF chunking, (2) query expansion and multi-query semantic search, (3) dense and lexical fusion, (4) cross-encoder reranking, (5) per-page citations, (6) an LLM-free evidence fallback when no provider key is configured, and (7) reuse of the same pipeline for browser-page text.

### Complexity

For N chunks, embedding/index construction is approximately O(Nd), where d is embedding dimension. A FAISS flat vector search is O(Nd) per query; BM25 scoring is proportional to indexed document terms; reciprocal-rank fusion is O(K); and cross-encoder reranking is O(R), where R is the candidate count. The most expensive stages are model loading, embedding/index rebuilding after each upload, and cross-encoder inference. The design constrains answer context to 12 chunks and limits candidates to 50 to maintain responsiveness.

### Design

```text
PDF upload / browser-page text
          |
     text extraction and cleaning
          |
 section-aware chunking (page metadata retained)
          |
 +--------+---------+
 |                  |
Dense embeddings   BM25 inverted ranking
 |                  |
 +-- vector / hybrid / keyword retrieval --+
                      |
       reciprocal-rank fusion (hybrid mode)
                      |
           cross-encoder reranking
                      |
      grounded answer / summary + citations
                      |
          Streamlit UI or FastAPI endpoint
```

### Project management and teamwork

Use a Git repository and divide responsibility by modules: PDF/chunking, retrieval/indexing, answer generation/API, and UI/extension/testing. Conduct weekly reviews; maintain an issue list and short milestone report. Suggested milestones: requirements and literature review; prototype ingestion; retrieval/reranking; UI/API/extension; evaluation and documentation; final testing and demonstration.

## Content for Major Project Presentation 3 to 6

### Demonstration flow

1. Start the FastAPI server and Streamlit application.
2. Upload two or more meaningful PDFs and show document page/chunk counts.
3. Ask the same question in Vector Search, BM25 Search and Hybrid Search modes.
4. Show that Hybrid Search retrieves both exact terms and semantically related passages.
5. Ask a paraphrased question (for example, "pros and cons") to demonstrate query expansion.
6. Show the returned answer and its document/page citations.
7. Request a whole-document summary.
8. Load the browser extension, index the current page and ask a grounded question about it.

### Objective fulfilment evidence

| Objective | Evidence in the implemented project |
|---|---|
| PDF ingestion | `backend/pdf_processor.py` extracts and cleans page text with PyMuPDF. |
| Chunking | `backend/chunker.py` preserves sentence boundaries and separates question sections. |
| Hybrid retrieval | `backend/vector_store.py`, `backend/bm25_retriever.py`, and `backend/hybrid_retriever.py`. |
| Reranking | `backend/reranker.py` applies a cross-encoder to candidate passages. |
| Grounded answers/citations | `backend/assistant.py`, `backend/llm_service.py`, and `backend/citation.py`. |
| Summary and browser support | `backend/summarization.py`, `backend/api.py`, and `browser_extension/`. |
| Quality checks | Unit tests cover chunking, retrieval fusion, citations, query understanding, PDF extraction, summaries and LLM failures. |

### Modern technology used

Python 3, Streamlit, FastAPI, PyMuPDF, Sentence Transformers, BAAI/bge-small-en-v1.5 embeddings, FAISS, rank-bm25, MS MARCO MiniLM cross-encoder, NLTK/WordNet, Groq or OpenAI compatible chat-completions APIs, pytest, Git, and a Manifest V3 browser extension.

### Cost analysis

The prototype uses open-source Python libraries and runs its embedding and reranking models on CPU, so software licensing cost is effectively zero. Expected costs are a standard laptop/desktop, internet access for first model downloads, and optional LLM API usage. When no API key is configured, it returns extraction-based grounded responses without LLM inference cost. For deployment, cost depends mainly on concurrent users, stored PDFs, model hosting and LLM token usage; caching embeddings and using a smaller local model reduce operating cost.

### Benefits to society

The system makes academic, technical and policy documents more usable for students, researchers and knowledge workers. Page-level citations encourage verification instead of blind trust in generated text. It can support learners who struggle to navigate large documents and can reduce time spent searching manuals, research papers and institutional material.

## Task-planning sheets (Semester 1: Sheets 1-6)

Copy these six weekly entries into Task Sheets 1-6 only. Replace the bracketed dates and write the guide's actual remark/signature only after your guide reviews it. These entries describe completed work visible in the current project; they do not include next-semester features.

### Week 1 - Task sheet 1

- Date: [fill in]
- Task planned: Understand the problem of searching information in lengthy PDFs. Study the difference between keyword-based search and semantic search, identify the users of the system, and prepare the initial project objective.
- Task completed: Studied PDF question answering, BM25 keyword retrieval, vector embeddings and retrieval-augmented generation. Finalized the problem statement: users should be able to upload research PDFs, ask questions in natural language, and receive answers supported by the source document and page number. Prepared the initial module plan: PDF processing, chunking, retrieval, answer generation, user interface and testing.
- Guide's review / suggestion: The problem definition and objectives are acceptable. First implement reliable PDF text extraction and preserve page information so that answers can later show their source.

### Week 2 - Task sheet 2

- Date: [fill in]
- Task planned: Based on the Week 1 review, implement PDF ingestion. Extract readable text page by page, remove extraction noise, retain document/page metadata, and divide long text into manageable chunks.
- Task completed: Implemented PDF processing with PyMuPDF. Added text cleaning for blank lines, spacing and words split by PDF line breaks. Each extracted page retains the document name and page number. Implemented sentence-aware chunking with overlap and question-section-aware separation so that separate question/answer sections are not mixed in one chunk.
- Guide's review / suggestion: PDF extraction and chunk preparation are working. Next, create two retrieval methods so that the system can support both exact technical terms and semantically similar wording.

### Week 3 - Task sheet 3

- Date: [fill in]
- Task planned: Based on the Week 2 review, implement dense semantic retrieval and BM25 keyword retrieval over the generated chunks. Provide separate retrieval modes and retain the metadata required for citations.
- Task completed: Implemented embedding generation using the BAAI/bge-small-en-v1.5 Sentence Transformer model and stored embeddings in a FAISS vector index. Implemented BM25 indexing and search over the same chunks. Added Vector Search, BM25 Search and Hybrid Search choices. Retrieval results retain chunk ID, document name, page number, text and section information.
- Guide's review / suggestion: Both retrieval methods are available. Improve the hybrid result ordering and ensure that the most relevant evidence, rather than only the first retrieved passage, is used for the final answer.

### Week 4 - Task sheet 4

- Date: [fill in]
- Task planned: Based on the Week 3 review, combine vector and BM25 results, rerank the best candidates, and produce an answer using only relevant document passages. Add source citations and safe handling when evidence is absent.
- Task completed: Implemented hybrid retrieval through reciprocal-rank fusion. Added a cross-encoder reranker to score query-passage pairs and select the most relevant section. Implemented grounded answer generation from the selected context, page-level citations, a limited answer context to avoid unrelated material, and a clear response when the uploaded documents do not contain enough evidence.
- Guide's review / suggestion: Retrieval and grounded answers are functioning. Build an interface through which a user can upload PDFs, choose a search mode, ask questions and see the cited response.

### Week 5 - Task sheet 5

- Date: [fill in]
- Task planned: Based on the Week 4 review, develop the user-facing application and API. Support PDF upload, display indexed-document details, accept user questions, show sources, and expose the core workflow through REST endpoints.
- Task completed: Developed the Streamlit application with PDF upload, configurable chunk size/overlap, Vector/BM25/Hybrid Search selection, document status display, chat input and source display. Developed FastAPI endpoints for health checking, document upload, listing documents, asking questions and clearing documents. The UI reports when an LLM key is unavailable and uses the existing grounded extractive path.
- Guide's review / suggestion: The main workflow is ready. Add small supporting capabilities and test important paths so that the project can be demonstrated with both PDF documents and readable web-page text.

### Week 6 - Task sheet 6

- Date: [fill in]
- Task planned: Based on the Week 5 review, improve the completed assistant without changing its main scope. Add meaningful query understanding, document-summary handling, browser-page support, test cases and a small retrieval-evaluation setup.
- Task completed: Implemented **query expansion using NLTK WordNet** and phrase aliases. For example, a user question containing “pros and cons” is expanded with “advantages, disadvantages, benefits and drawbacks”; component-related questions are expanded with related terms. Implemented **document-summary request detection** and ordered selection of document chunks for summaries. Added tests for PDF extraction, chunking, citations, retrieval fusion, reranking context selection, query understanding, summaries and LLM failure handling. Added an evaluation script with a prepared question set to compare BM25, vector search, hybrid search and hybrid search with reranking using Recall@K, Precision@K, MRR and latency. Implemented a **Manifest V3 Chrome browser extension** with a movable research-assistant bubble, page summary action, selected-text explanation action and local message storage. The extension extracts readable webpage text and sends it to an isolated FastAPI session; that session uses the already-implemented chunking, embeddings, BM25, hybrid retrieval, reranking and grounded-answer pipeline.
- Guide's review / suggestion: The first-semester implementation includes the complete planned PDF research-assistant workflow and supporting features. Verify the modules using the existing tests and evaluation setup, and retain screenshots/results for the presentation.

## Limitations and final improvement plan

Present these honestly as future work, not as already-completed features:

1. Indexes are currently in memory and are rebuilt after an upload; add persistent storage and background indexing for large collections.
2. Retrieval quality needs a real labelled evaluation corpus and recorded Recall@K, MRR, latency and answer-faithfulness results.
3. Add authentication, upload-size limits, file isolation, encrypted configuration and rate limits before multi-user deployment.
4. Preserve chat history in the main Streamlit UI and use the existing `ChatMemory` component consistently.
5. Add OCR for scanned PDFs, image/table understanding, richer citation excerpts and multilingual support.
6. Use asynchronous jobs, batched embedding and a faster/vector database index for scale.

## References

1. Robertson, S. and Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval.
2. Karpukhin, V. et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP.
3. Cormack, G. V., Clarke, C. L. A., and Buettcher, S. (2009). *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.* SIGIR.
4. Nogueira, R. and Cho, K. (2019). *Passage Re-ranking with BERT.* arXiv:1901.04085.
5. Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
6. Reimers, N. and Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings Using Siamese BERT-Networks.* EMNLP.
7. FAISS documentation, Meta AI Research. https://faiss.ai/
8. FastAPI documentation. https://fastapi.tiangolo.com/
9. Streamlit documentation. https://docs.streamlit.io/
