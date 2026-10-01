"""Lightweight tokenizer shared by indexing, answering, and evaluation."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")

# Question words and function words. Domain nouns stay in the index so rare
# terms such as "lithium" or "bereavement" can outrank generic matches.
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "for",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "as",
        "if",
        "into",
        "about",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "does",
        "do",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
        "must",
        "may",
        "many",
        "much",
        "there",
        "their",
        "they",
        "them",
        "we",
        "our",
        "you",
        "your",
        "i",
        "me",
        "my",
        "not",
        "no",
        "nor",
        "than",
        "then",
        "per",
        "each",
        "every",
        "before",
        "after",
        "during",
        "within",
        "under",
        "over",
        "between",
        "only",
        "also",
        "using",
    }
)


def _normalize(token: str) -> str:
    # Keep digits so "Sev-1" and "Sev-2" stay distinct. Fold a trailing plural s
    # so "list" and "lists" match, but leave words that end in "ss" alone.
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in _TOKEN.findall(text.lower()):
        if token in STOPWORDS:
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        tokens.append(_normalize(token))
    return tokens
