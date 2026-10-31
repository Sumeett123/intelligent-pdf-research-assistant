from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

try:
    import nltk
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer
except ImportError:  # pragma: no cover - dependency is declared in requirements
    nltk = None
    wordnet = None
    WordNetLemmatizer = None


_PHRASE_ALIASES = (
    (r"\bpros?\s+and\s+cons?\b", "advantages disadvantages benefits drawbacks"),
    (r"\bpros?\s+of\b", "advantages benefits"),
    (r"\bstrengths?\s+and\s+weakness(?:es)?\b", "advantages disadvantages benefits drawbacks"),
    (r"\bpositive\s+and\s+negative\s+(?:aspects?|points?|effects?)\b", "advantages disadvantages benefits drawbacks"),
    (r"\bbenefits?\s+and\s+drawbacks?\b", "advantages disadvantages pros cons"),
    (
        r"\b(?:core|main|key|essential)\s+(?:components?|parts?|elements?)\b",
        "necessary elements required components constituent parts",
    ),
    (
        r"\b(?:components?|parts?|elements?)\s+of\b",
        "necessary elements required components constituent parts",
    ),
)

_STOP_WORDS = {
    "a", "an", "and", "are", "about", "be", "by", "can", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "that",
    "the", "their", "this", "to", "what", "when", "where", "which",
    "who", "why", "with",
}
_POS_BY_SUFFIX = (
    (("ing", "ed", "ize", "ise", "ify"), "v"),
    (("ly",), "r"),
    (("ous", "ful", "ive", "al", "ic", "less"), "a"),
)

if nltk is not None:
    project_nltk_data = Path(__file__).resolve().parent.parent / "data" / "nltk_data"
    if project_nltk_data.exists():
        nltk.data.path.insert(0, str(project_nltk_data))

_LEMMATIZER = WordNetLemmatizer() if WordNetLemmatizer is not None else None


def _wordnet_available() -> bool:
    if wordnet is None:
        return False
    try:
        wordnet.synsets("research")
    except LookupError:
        return False
    return True


def require_wordnet() -> None:
    """Fail explicitly if the configured NLP vocabulary is unavailable."""
    if nltk is None or wordnet is None:
        raise RuntimeError("NLTK and WordNet are required for query understanding.")
    if not _wordnet_available():
        raise RuntimeError(
            "WordNet data is missing. Install it with: "
            "python -m nltk.downloader wordnet omw-1.4"
        )


def _guess_pos(word: str) -> str | None:
    for suffixes, pos in _POS_BY_SUFFIX:
        if word.endswith(suffixes) and len(word) > len(suffixes[0]) + 2:
            return pos
    return None


@lru_cache(maxsize=4096)
def _synonyms(word: str) -> tuple[str, ...]:
    if not _wordnet_available():
        return ()

    normalized = _LEMMATIZER.lemmatize(word) if _LEMMATIZER is not None else word
    guessed_pos = _guess_pos(word)
    # If suffixes do not identify a part of speech, search every WordNet
    # lexical category. This makes the vocabulary work for arbitrary terms,
    # rather than only terms in a hand-maintained alias table.
    synsets = wordnet.synsets(normalized, pos=guessed_pos) if guessed_pos else wordnet.synsets(normalized)
    candidates: list[str] = []
    seen = {word, normalized}
    # WordNet returns common senses first. Taking a few senses gives broader
    # coverage while the per-term cap prevents BM25 query pollution.
    for synset in synsets[:4]:
        lemmas = sorted(synset.lemmas(), key=lambda lemma: lemma.count(), reverse=True)
        for lemma in lemmas:
            synonym = lemma.name().replace("_", " ").lower()
            if synonym in seen or not re.fullmatch(r"[a-z][a-z -]+", synonym):
                continue
            seen.add(synonym)
            candidates.append(synonym)
            if len(candidates) >= 4:
                return tuple(candidates)
    return tuple(candidates)


def _terms(query: str) -> Iterable[str]:
    for term in re.findall(r"\b[a-zA-Z][a-zA-Z-]+\b", query.lower()):
        if len(term) > 2 and term not in _STOP_WORDS:
            yield term


def expand_query(query: str) -> str:
    """Expand arbitrary query terms with a broad lexical vocabulary.

    This is not intended to enumerate all possible paraphrases. WordNet is a
    general lexical resource for any recognized term, while the
    SentenceTransformer vector search handles unseen wording and sentence
    structure. The original query is always preserved.
    """
    require_wordnet()
    expanded = query.strip()
    phrase_terms: set[str] = set()
    for pattern, aliases in _PHRASE_ALIASES:
        if re.search(pattern, expanded, flags=re.IGNORECASE):
            expanded = f"{expanded} {aliases}"
            phrase_terms.update(re.findall(r"[a-zA-Z]+", pattern.replace(r"\b", "").lower()))

    additions: list[str] = []
    seen: set[str] = set(re.findall(r"\b[a-zA-Z][a-zA-Z-]+\b", expanded.lower()))
    for term in _terms(query):
        if term in phrase_terms:
            continue
        for synonym in _synonyms(term):
            if synonym not in seen:
                seen.add(synonym)
                additions.append(synonym)
    return f"{expanded} {' '.join(additions)}".strip()
