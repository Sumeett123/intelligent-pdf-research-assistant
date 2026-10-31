# Research Assistant Bubble extension

This independent Chrome extension adds a movable page-level assistant and sends page content to this project's FastAPI backend. The backend creates an isolated temporary text document for each browser page, then applies the existing chunking, embeddings, BM25 retrieval, hybrid fusion, reranking, and grounded answer generation flow.

## Paper comparison

The **Compare papers** tab is fully separate from the Streamlit workflow. Enter a topic to search the top five papers through OpenAlex and optionally retain their DOI values as reference metadata. Then upload two to five local PDFs that you already have. The extension never downloads papers; it indexes only the explicitly uploaded PDFs in a temporary extension session and returns a cited comparison of objectives, algorithms, datasets, methodology, results, and limitations.

Potential contradictions are deliberately marked **Review required** rather than treated as errors: students should inspect the cited page evidence and account for dataset or experimental-condition differences. A DOI alone is not enough for a page-cited comparison, so upload its PDF manually; the extension does not bypass publisher access controls.

## Run locally

1. Start the project backend: `uvicorn backend.api:app --host 127.0.0.1 --port 8000`.
2. In Chrome, open `chrome://extensions`, enable **Developer mode**, choose **Load unpacked**, and select this `browser_extension` folder.
3. Open a normal `http` or `https` page and click the purple bubble.

## Current scope

- Supports normal web pages, web articles, and text that browser pages expose (including many online document readers).
- Saves up to 20 recent messages for a page in Chrome local storage.
- Uses a per-page backend RAG session; it does not modify the Streamlit application's uploaded-document index.
- Chrome's built-in PDF viewer and desktop WPS Office can restrict page-text access. The extension still supports copied/selected text where the browser makes that selection available; direct desktop WPS integration requires a separate WPS add-in or desktop companion.
