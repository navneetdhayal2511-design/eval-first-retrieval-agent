"""Okapi BM25 over handbook chunks. No third-party search library."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from retrieval_agent.chunking import Chunk
from retrieval_agent.tokenize import tokenize


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("BM25Index requires at least one chunk")
        self.k1 = k1
        self.b = b
        self.chunks = list(chunks)
        self._docs: list[list[str]] = [tokenize(chunk.indexed_text) for chunk in self.chunks]
        self._doc_freq: list[Counter[str]] = [Counter(tokens) for tokens in self._docs]
        self._lengths = [len(tokens) for tokens in self._docs]
        self._avgdl = sum(self._lengths) / len(self._lengths)
        self._df: dict[str, int] = defaultdict(int)
        for freq in self._doc_freq:
            for term in freq:
                self._df[term] += 1
        self._n = len(self.chunks)

    def idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))

    def score_chunk(self, query_terms: list[str], doc_index: int) -> float:
        freq = self._doc_freq[doc_index]
        length = self._lengths[doc_index] or 1
        score = 0.0
        for term in query_terms:
            tf = freq.get(term, 0)
            if tf == 0:
                continue
            denom = tf + self.k1 * (1 - self.b + self.b * length / self._avgdl)
            score += self.idf(term) * (tf * (self.k1 + 1)) / denom
        return score

    def search(self, question: str, k: int = 4) -> list[Hit]:
        query_terms = tokenize(question)
        if not query_terms:
            return []
        ranked = [
            Hit(chunk=self.chunks[i], score=self.score_chunk(query_terms, i))
            for i in range(self._n)
        ]
        ranked = [hit for hit in ranked if hit.score > 0]
        ranked.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        return ranked[:k]


def token_coverage(question: str, text: str) -> float:
    """Share of question tokens that actually appear in the candidate text."""
    query_terms = tokenize(question)
    if not query_terms:
        return 0.0
    present = set(tokenize(text))
    return sum(1 for term in query_terms if term in present) / len(query_terms)
