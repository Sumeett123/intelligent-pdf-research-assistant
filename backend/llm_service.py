from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

import httpx
from dotenv import load_dotenv

load_dotenv()


class LLMService:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()
        configured_model = model or os.getenv("LLM_MODEL") or os.getenv("GROQ_MODEL")
        if configured_model in {"llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"}:
            configured_model = None
        self.model = configured_model or "openai/gpt-oss-20b"

    def has_credentials(self) -> bool:
        if self.provider == "openai":
            return bool(os.getenv("OPENAI_API_KEY"))
        if self.provider == "groq":
            return bool(os.getenv("GROQ_API_KEY"))
        return False

    def configuration_status(self) -> str:
        if self.has_credentials():
            return f"{self.provider} / {self.model}"
        return f"{self.provider} / {self.model} (API key missing; grounded extractive mode)"

    def _chunk_text(self, chunk: Dict[str, Any]) -> str:
        if chunk.get("text"):
            return str(chunk.get("text"))
        metadata = chunk.get("metadata")
        if isinstance(metadata, dict) and metadata.get("text"):
            return str(metadata.get("text"))
        return ""

    def generate(self, question: str, context_chunks: List[Dict[str, Any]], chat_history: str = "") -> Dict[str, Any]:
        if not context_chunks:
            return {
                "answer": "The uploaded documents do not contain enough information to answer this question.",
                "sources": [],
            }

        context_text = "\n\n".join(
            f"[Source: {chunk['document_name']} | Page {chunk['page_number']}]\n{self._chunk_text(chunk)}"
            for chunk in context_chunks
            if self._chunk_text(chunk)
        )

        system_prompt = (
            "You are a careful research assistant. Answer only from the supplied document context. "
            "Do not invent or infer information that is not present. If the answer cannot be found in the document context, "
            "say exactly: 'The uploaded documents do not contain enough information to answer this question.' "
            "Answer every part of the question completely, but do not include unrelated topics from the context. "
            "When the question asks for components, elements, parts, or what a system consists of, "
            "identify and include the complete explicit list from the relevant document section, including "
            "continuations in nearby passages; do not substitute a shorter related summary or a list of system types. "
            "Answer the user's question directly. Do not repeat the source question, other question headers, or document instructions. "
            "Cite sources by document and page number in a final 'Sources:' section."
        )
        user_prompt = (
            f"Question: {question}\n\n"
            f"Conversation history:\n{chat_history}\n\n"
            f"Document context:\n{context_text}"
        )

        try:
            return self._provider_call(system_prompt, user_prompt, question, context_chunks)
        except Exception:
            fallback = self._fallback_answer(question, context_chunks)
            if not self.has_credentials():
                fallback["answer"] = (
                    f"No {self.provider.upper()} API key is configured, so this is a grounded extractive answer "
                    f"from the retrieved document passages.\n\n{fallback['answer']}"
                )
            return fallback

    def _provider_call(self, system_prompt: str, user_prompt: str, question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if self.provider == "groq" and groq_key:
            resp = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1600,
                },
                timeout=45.0,
            )
            resp.raise_for_status()
            payload = resp.json()
            content = payload["choices"][0]["message"]["content"]
            return {"answer": self._append_sources(content, context_chunks), "sources": self._unique_sources(context_chunks)}

        if self.provider == "openai" and openai_key:
            resp = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1600,
                },
                timeout=45.0,
            )
            resp.raise_for_status()
            payload = resp.json()
            content = payload["choices"][0]["message"]["content"]
            return {"answer": self._append_sources(content, context_chunks), "sources": self._unique_sources(context_chunks)}

        raise RuntimeError("No LLM API key configured.")

    def _append_sources(self, content: str, context_chunks: List[Dict[str, Any]]) -> str:
        if "Sources:" in content:
            return content
        source_lines = ["\n\nSources:"]
        for index, source in enumerate(self._unique_sources(context_chunks), start=1):
            source_lines.append(f"{index}. {source['document_name']} - Page {source['page_number']}")
        return content + "\n" + "\n".join(source_lines)

    def _unique_sources(self, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []
        seen = set()
        for chunk in context_chunks:
            source = {
                "document_name": chunk.get("document_name", "unknown.pdf"),
                "page_number": chunk.get("page_number", 0),
                "chunk_id": chunk.get("chunk_id", "unknown"),
            }
            key = (source["document_name"], source["page_number"])
            if key in seen:
                continue
            seen.add(key)
            sources.append(source)
        return sources

    def _fallback_response(self, answer: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        sources = self._unique_sources(context_chunks)
        source_text = "\n".join(
            f"{index + 1}. {source['document_name']} - Page {source['page_number']}"
            for index, source in enumerate(sources)
        )
        return {"answer": answer.strip() + "\n\nSources:\n" + source_text, "sources": sources}

    def _fallback_answer(self, question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        stop_words = {
            "what", "which", "when", "where", "why", "how", "who", "explain", "describe",
            "define", "tell", "give", "show", "about", "from", "with", "this", "that",
        }
        question_terms = {
            term for term in re.findall(r"[a-zA-Z0-9]+", question.lower())
            if len(term) > 2 and term not in stop_words
        }
        sentences: List[str] = []
        seen = set()
        for chunk in context_chunks:
            text = self._chunk_text(chunk).strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                cleaned = re.sub(
                    r"^\s*Q\d+\s*[.):\-]?\s*.*?\bAns(?:wer)?\s*:\s*",
                    "",
                    sentence,
                    flags=re.IGNORECASE,
                ).strip()
                cleaned = re.sub(r"^\s*(?:Ans(?:wer)?\s*:\s*|Q\d*\.?\s*)", "", cleaned, flags=re.IGNORECASE).strip()
                if cleaned.lower().startswith(("explain ", "state ", "describe ", "what is ", "what are ")):
                    continue
                normalized = re.sub(r"\s+", " ", cleaned)
                if len(normalized) < 20 or normalized.lower() in seen:
                    continue
                seen.add(normalized.lower())
                sentences.append(normalized)

        scored = []
        for index, sentence in enumerate(sentences):
            sentence_terms = set(re.findall(r"[a-zA-Z0-9]+", sentence.lower()))
            score = len(question_terms.intersection(sentence_terms))
            scored.append((score, -index, sentence))

        selected = [sentence for score, _, sentence in sorted(scored, reverse=True)[:8] if score > 0]
        if not selected:
            return {
                "answer": "The uploaded documents do not contain enough relevant information to answer this question.",
                "sources": [],
            }
        return self._fallback_response(" ".join(selected), context_chunks)

    def _ordered_context_text(self, context_chunks: List[Dict[str, Any]]) -> str:
        entries = []
        seen = set()
        for chunk in context_chunks:
            text = self._chunk_text(chunk).strip()
            if not text:
                continue
            key = (chunk.get("document_name", ""), chunk.get("page_number", 0), text)
            if key in seen:
                continue
            seen.add(key)
            chunk_id = str(chunk.get("chunk_id", ""))
            match = re.search(r"-c(\d+)$", chunk_id)
            chunk_number = int(match.group(1)) if match else 0
            entries.append((int(chunk.get("page_number", 0)), chunk_number, text))
        entries.sort(key=lambda item: (item[0], item[1]))
        return " ".join(item[2] for item in entries)

    def _legacy_section_answer(self, question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any] | None:
        question_lower = question.lower()
        document_text = self._ordered_context_text(context_chunks)
        if not document_text:
            return None

        if any(term in question_lower for term in ["importance", "society"]):
            match = re.search(r"(?is)Q1\.?\s+.*?Ans\s*:\s*(.*?)(?=Q2\.?\s+Explain|$)", document_text)
            if match:
                section = match.group(1).strip()
                section = re.sub(r"\s*➢\s*", "\n- ", section)
                section = re.sub(r"\s+", " ", section)
                return self._fallback_response(section[:2200], context_chunks)

        if any(term in question_lower for term in ["type", "types", "summarize", "define"]):
            match = re.search(r"(?is)Q2\.?\s+.*?Ans\s*:\s*(.*?)(?=Q3\.?\s+Explain|$)", document_text)
            if match:
                section = match.group(1).strip()
                section = re.sub(r"\s*[•▪]\s*", "\n- ", section)
                section = re.sub(r"\s+", " ", section)
                return self._fallback_response(section[:5000], context_chunks)

        return None

    def _legacy_fallback_answer(self, question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = []
        for chunk in context_chunks:
            text = self._chunk_text(chunk).strip()
            if not text:
                continue

            answer_sections = re.findall(r"(?i)(?:Ans|Answer)\s*:\s*(.*?)(?=(?:\s*Q\d+\.|\s*Q\d+\s|\s*Q[\w-]+\.|\s*Q[\w-]+\s|\Z))", text, flags=re.S)
            if answer_sections:
                texts.extend(section.strip() for section in answer_sections if section.strip())
                continue

            for marker in ["Advantages:", "Disadvantages:"]:
                idx = text.upper().find(marker.upper())
                if idx != -1:
                    texts.append(text[idx + len(marker):].strip())
                    break
            else:
                texts.append(text)

        if not texts:
            return {
                "answer": "The uploaded documents do not contain enough information to answer this question.",
                "sources": [],
            }

        combined = " ".join(texts)
        structured_answer = ""
        combined = re.sub(
            r"(?i)first part of (?:the )?question should be written from previous Q\.?\s*4\.?",
            "",
            combined,
        )
        combined = re.sub(r"(?i)first part of (?:the )?question should be written from previous question", "", combined)

        question_lower = question.lower()
        if "advantage" in question_lower and "disadvantage" in question_lower:
            definition_sentences: List[str] = []
            advantage_sentences: List[str] = []
            disadvantage_sentences: List[str] = []
            page_texts: Dict[tuple[str, int], List[str]] = {}
            for chunk in context_chunks:
                page_key = (str(chunk.get("document_name", "")), int(chunk.get("page_number", 0)))
                chunk_text = self._chunk_text(chunk)
                if chunk_text and chunk_text not in page_texts.setdefault(page_key, []):
                    page_texts[page_key].append(chunk_text)

            for page_chunks in page_texts.values():
                chunk_text = " ".join(page_chunks)
                definition_sentences.extend(
                    sentence.strip()
                    for sentence in re.split(r"(?<=[.!?])\s+", chunk_text)
                    if "an information system is" in sentence.lower()
                )
                advantage_match = re.search(
                    r"(?is)advantages? of information system\s*:?(.*?)(?=disadvantages? of information system|$)",
                    chunk_text,
                )
                if advantage_match:
                    advantage_sentences.extend(re.split(r"(?<=[.!?])\s+", advantage_match.group(1)))
                disadvantage_match = re.search(r"(?is)disadvantages? of information system\s*:?(.*)$", chunk_text)
                if disadvantage_match:
                    disadvantage_sentences.extend(re.split(r"(?<=[.!?])\s+", disadvantage_match.group(1)))

            def unique_meaningful(sentences: List[str], limit: int) -> List[str]:
                result: List[str] = []
                seen = set()
                for sentence in sentences:
                    cleaned = re.sub(r"^\s*(?:[ivx]+|\d+)\.?\s*", "", sentence, flags=re.I).strip()
                    cleaned = re.sub(r"^\s*first part of .*?(?=\b(?:Advantages?|Disadvantages?)\b|$)", "", cleaned, flags=re.I).strip()
                    if len(cleaned) < 35 or cleaned.lower() in seen:
                        continue
                    seen.add(cleaned.lower())
                    result.append(cleaned)
                    if len(result) == limit:
                        break
                return result

            definition = unique_meaningful(definition_sentences, 1)
            advantages = unique_meaningful(advantage_sentences, 8)
            disadvantages = unique_meaningful(disadvantage_sentences, 5)
            structured_parts = definition + advantages + disadvantages
            if structured_parts:
                combined = " ".join(structured_parts)
                structured_answer = ""
                if definition:
                    structured_answer += definition[0]
                if advantages:
                    structured_answer += "\n\nAdvantages:\n" + " ".join(advantages)
                if disadvantages:
                    structured_answer += "\n\nDisadvantages:\n" + " ".join(disadvantages)

        sentences = re.split(r"(?<=[.!?])\s+", combined)
        stop_words = {
            "what", "why", "how", "when", "where", "which", "who", "is", "are", "the", "a", "an",
            "to", "of", "for", "and", "or", "explain", "tell", "state", "define", "summarize", "about",
        }
        question_terms = {
            term for term in re.findall(r"[a-zA-Z]+", question.lower())
            if term not in stop_words and len(term) > 2
        }
        scored_sentences: List[tuple[int, str]] = []
        cleaned_sentences: List[str] = []
        seen = set()
        for sentence in sentences:
            cleaned = re.sub(r"^\s*Ans\s*:\s*", "", sentence, flags=re.I)
            cleaned = re.sub(r"^\s*Q\d*\.?\s*", "", cleaned, flags=re.I)
            cleaned = re.sub(r"^\s*\d+\.?\s*", "", cleaned)
            cleaned = re.sub(r"(?i)^first part of .*?(?=\b(?:Advantages?|Disadvantages?)\b|$)", "", cleaned).strip()
            cleaned = cleaned.strip()
            if cleaned.lower().startswith("ing "):
                cleaned = cleaned[4:]
            if not cleaned:
                continue
            if cleaned.lower().startswith("what ") or cleaned.lower().startswith("explain ") or "what are its advantages and disadvantages" in cleaned.lower():
                continue
            if any(keyword in cleaned.lower() for keyword in ["question", "question bank", "department of computer engineering"]):
                continue
            normalized = re.sub(r"\s+", " ", cleaned)
            if len(normalized) < 20:
                continue
            if normalized.lower() in seen:
                continue
            seen.add(normalized.lower())
            cleaned_sentences.append(normalized)

        for sentence in cleaned_sentences:
            normalized = sentence.lower()
            score = sum(1 for term in question_terms if term and term in normalized)
            for keyword in ["information", "system", "advantage", "disadvantage", "society", "decision", "communication", "organization", "data", "network", "people", "software", "hardware"]:
                if keyword in normalized:
                    score += 1
            scored_sentences.append((score, sentence.strip()))

        ranked = sorted(enumerate(scored_sentences), key=lambda item: (item[1][0], -item[0]), reverse=True)
        selected_indexes = sorted(index for index, (score, sentence) in ranked[:8] if score > 0 and sentence)
        selection = [scored_sentences[index][1] for index in selected_indexes]
        if not selection:
            selection = sentences[:4]

        answer = structured_answer or " ".join(selection)
        if structured_answer:
            answer = answer.strip()
            if len(answer) > 1800:
                answer = answer[:1800].rsplit(" ", 1)[0] + "."
            return self._fallback_response(answer, context_chunks)
        sentence_parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", answer) if part.strip()]
        if len(sentence_parts) > 6:
            answer = " ".join(sentence_parts[:6])
        if len(answer) > 900:
            answer = " ".join(sentence_parts[:4])

        sources = []
        seen_sources = set()
        for chunk in context_chunks:
            source = {
                "document_name": chunk.get("document_name", "unknown.pdf"),
                "page_number": chunk.get("page_number", 0),
                "chunk_id": chunk.get("chunk_id", "unknown"),
            }
            source_key = (source["document_name"], source["page_number"])
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            sources.append(source)
            if len(sources) == 3:
                break

        return {
            "answer": answer + "\n\nSources:\n" + "\n".join(f"{index + 1}. {source['document_name']} - Page {source['page_number']}" for index, source in enumerate(sources)),
            "sources": sources,
        }
