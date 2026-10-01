"""Cited answers. The default reader is extractive so the project runs offline."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from retrieval_agent.bm25 import BM25Index, Hit, token_coverage
from retrieval_agent.chunking import Chunk
from retrieval_agent.tokenize import tokenize

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    refused: bool
    citations: list[dict[str, str]] = field(default_factory=list)
    retrieved: list[dict[str, float | str]] = field(default_factory=list)
    top_score: float = 0.0
    coverage: float = 0.0
    support_text: str = ""

    def citation_documents(self) -> list[str]:
        return [item["document"] for item in self.citations]


def split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE.split(text.replace("\n", " ")) if part.strip()]
    return parts or [text.strip()]


def best_sentence(question: str, text: str) -> str:
    query = set(tokenize(question))
    sentences = split_sentences(text)
    ranked = sorted(
        sentences,
        key=lambda sentence: (
            len(query & set(tokenize(sentence))),
            -len(sentence),
        ),
        reverse=True,
    )
    return ranked[0]


class ExtractiveReader:
    """Pick the sentence in the retrieved section that covers the question."""

    def read(self, question: str, chunk: Chunk) -> str:
        # Handbook sections are short. Quoting the whole section keeps the
        # supporting sentence instead of a nearby sentence that echoes the question.
        if len(chunk.text) <= 600:
            return " ".join(chunk.text.split())
        return best_sentence(question, chunk.text)


class RetrievalAgent:
    """Retrieve, abstain when support is weak, then read a cited answer.

    `min_coverage` is a fixed design choice: at least half of the question's
    content words must appear in the top section. `score_threshold` is chosen
    on the dev split only. See evaluate.choose_threshold.
    """

    def __init__(
        self,
        index: BM25Index,
        score_threshold: float,
        min_coverage: float = 0.5,
        k: int = 4,
        reader: ExtractiveReader | None = None,
    ) -> None:
        self.index = index
        self.score_threshold = score_threshold
        self.min_coverage = min_coverage
        self.k = k
        self.reader = reader or ExtractiveReader()

    def ask(self, question: str) -> Answer:
        hits = self.index.search(question, k=self.k)
        retrieved = [
            {
                "document": hit.chunk.document,
                "heading": hit.chunk.heading,
                "score": round(hit.score, 4),
            }
            for hit in hits
        ]
        if not hits:
            return Answer(
                question=question,
                text="I cannot find this in the handbook.",
                refused=True,
                retrieved=retrieved,
            )

        top: Hit = hits[0]
        coverage = token_coverage(question, top.chunk.indexed_text)
        if top.score < self.score_threshold or coverage < self.min_coverage:
            return Answer(
                question=question,
                text="I cannot find this in the handbook.",
                refused=True,
                retrieved=retrieved,
                top_score=top.score,
                coverage=coverage,
            )

        return Answer(
            question=question,
            text=self.reader.read(question, top.chunk),
            refused=False,
            citations=[{"document": top.chunk.document, "heading": top.chunk.heading}],
            retrieved=retrieved,
            top_score=top.score,
            coverage=coverage,
        )


class FirstHitBaseline:
    """Scan the corpus in file order and answer from the first weak match.

    This is the baseline the BM25 agent has to beat. It has no IDF and it
    does not abstain when only a generic word matched.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)

    def ask(self, question: str) -> Answer:
        query = set(tokenize(question))
        for chunk in self.chunks:
            if query & set(tokenize(chunk.indexed_text)):
                return Answer(
                    question=question,
                    text=best_sentence(question, chunk.text),
                    refused=False,
                    citations=[{"document": chunk.document, "heading": chunk.heading}],
                    top_score=1.0,
                    coverage=token_coverage(question, chunk.indexed_text),
                    support_text=chunk.text,
                )
        return Answer(
            question=question,
            text="I cannot find this in the handbook.",
            refused=True,
        )
