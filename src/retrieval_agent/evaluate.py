"""Choose the abstention threshold on dev questions, then score a frozen split."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from retrieval_agent.answer import Answer, FirstHitBaseline, RetrievalAgent
from retrieval_agent.bm25 import BM25Index
from retrieval_agent.chunking import Chunk, load_corpus


@dataclass(frozen=True)
class Example:
    example_id: str
    split: str
    answerable: bool
    question: str
    document: str
    gold_substring: str
    gold_keywords: tuple[str, ...]

    @classmethod
    def from_obj(cls, obj: dict) -> "Example":
        return cls(
            example_id=obj["id"],
            split=obj["split"],
            answerable=bool(obj["answerable"]),
            question=obj["question"],
            document=obj.get("document", ""),
            gold_substring=obj.get("gold_substring", ""),
            gold_keywords=tuple(obj.get("gold_keywords", [])),
        )


def load_examples(path: Path) -> list[Example]:
    examples: list[Example] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            examples.append(Example.from_obj(json.loads(line)))
    return examples


def keyword_recall(answer: Answer, example: Example) -> float:
    if not example.answerable:
        return 1.0 if answer.refused else 0.0
    if answer.refused or not example.gold_keywords:
        return 0.0
    text = answer.text.lower()
    hits = sum(1 for keyword in example.gold_keywords if keyword.lower() in text)
    return hits / len(example.gold_keywords)


def is_correct(answer: Answer, example: Example) -> bool:
    if not example.answerable:
        return answer.refused and not answer.citations
    if answer.refused:
        return False
    cited = example.document in answer.citation_documents()
    supported = example.gold_substring.lower() in answer.text.lower() or keyword_recall(answer, example) >= 0.5
    return cited and supported


def balanced_accuracy(rows: list[tuple[Example, Answer]]) -> float:
    groups: dict[bool, list[bool]] = {True: [], False: []}
    for example, answer in rows:
        groups[example.answerable].append(is_correct(answer, example))
    rates = []
    for flags in groups.values():
        if flags:
            rates.append(sum(flags) / len(flags))
    return sum(rates) / len(rates) if rates else 0.0


def choose_threshold(dev_examples: list[Example], index: BM25Index, min_coverage: float = 0.5) -> float:
    """Sweep abstention cutoffs on the dev split and keep the best one.

    The test split is not read here. Ties prefer the higher cutoff so the
    agent abstains instead of guessing.
    """
    if not dev_examples:
        raise ValueError("dev split is empty")
    scored: list[float] = []
    for example in dev_examples:
        hits = index.search(example.question, k=1)
        scored.append(hits[0].score if hits else 0.0)
    candidates = sorted(set([0.0, *scored]))
    best_threshold = 0.0
    best_score = -1.0
    for threshold in candidates:
        agent = RetrievalAgent(index, score_threshold=threshold, min_coverage=min_coverage)
        rows = [(example, agent.ask(example.question)) for example in dev_examples]
        score = balanced_accuracy(rows)
        if score > best_score or (score == best_score and threshold > best_threshold):
            best_score = score
            best_threshold = threshold
    return best_threshold


def summarize(name: str, examples: list[Example], answers: list[Answer], index: BM25Index | None) -> dict:
    pairs = list(zip(examples, answers, strict=True))
    answerable = [(ex, ans) for ex, ans in pairs if ex.answerable]
    unanswerable = [(ex, ans) for ex, ans in pairs if not ex.answerable]

    def rate(rows: list[tuple[Example, Answer]]) -> float:
        if not rows:
            return 0.0
        return sum(is_correct(ans, ex) for ex, ans in rows) / len(rows)

    retrieval_hits = 0
    for example, answer in answerable:
        if index is not None:
            hits = index.search(example.question, k=3)
            if any(example.gold_substring.lower() in hit.chunk.text.lower() for hit in hits):
                retrieval_hits += 1
        elif example.gold_substring.lower() in answer.support_text.lower():
            retrieval_hits += 1

    keyword_values = [keyword_recall(answer, example) for example, answer in answerable]
    citations = [
        example.document in answer.citation_documents()
        for example, answer in answerable
        if not answer.refused
    ]
    return {
        "system": name,
        "n": len(examples),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "balanced_accuracy": round(balanced_accuracy(pairs), 4),
        "answerable_accuracy": round(rate(answerable), 4),
        "refusal_accuracy": round(rate(unanswerable), 4),
        "retrieval_recall_at_3": round(retrieval_hits / len(answerable), 4) if answerable else 0.0,
        "keyword_recall": round(sum(keyword_values) / len(keyword_values), 4) if keyword_values else 0.0,
        "citation_accuracy": round(sum(citations) / len(citations), 4) if citations else 0.0,
        "answered": sum(1 for answer in answers if not answer.refused),
        "refused": sum(1 for answer in answers if answer.refused),
    }


def failure_taxonomy(examples: list[Example], answers: list[Answer]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for example, answer in zip(examples, answers, strict=True):
        if is_correct(answer, example):
            continue
        if example.answerable and answer.refused:
            counts["missed_answer"] += 1
        elif not example.answerable and not answer.refused:
            counts["answered_when_unsupported"] += 1
        elif example.answerable and example.document not in answer.citation_documents():
            counts["wrong_citation"] += 1
        else:
            counts["incomplete_answer"] += 1
    return dict(counts)


def build_report(root: Path, split: str = "test") -> dict:
    corpus = load_corpus(root / "data" / "corpus")
    examples = load_examples(root / "data" / "eval" / "questions.jsonl")
    dev = [example for example in examples if example.split == "dev"]
    heldout = [example for example in examples if example.split == split]
    if split != "dev" and not heldout:
        raise ValueError(f"No examples in split {split!r}")

    index = BM25Index(corpus)
    threshold = choose_threshold(dev, index)
    agent = RetrievalAgent(index, score_threshold=threshold)
    baseline = FirstHitBaseline(corpus)

    agent_answers = [agent.ask(example.question) for example in heldout]
    baseline_answers = [baseline.ask(example.question) for example in heldout]

    report = {
        "split": split,
        "score_threshold": round(threshold, 4),
        "min_coverage": agent.min_coverage,
        "threshold_fit_on": "dev",
        "systems": [
            summarize("bm25_extractive", heldout, agent_answers, index),
            summarize("first_hit_baseline", heldout, baseline_answers, None),
        ],
        "failure_taxonomy": failure_taxonomy(heldout, agent_answers),
        "predictions": [
            {
                "id": example.example_id,
                "question": example.question,
                "answerable": example.answerable,
                "correct": is_correct(answer, example),
                "refused": answer.refused,
                "answer": answer.text,
                "citations": answer.citations,
            }
            for example, answer in zip(heldout, agent_answers, strict=True)
        ],
    }
    return report


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
