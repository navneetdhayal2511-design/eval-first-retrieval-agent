import unittest
from pathlib import Path

from retrieval_agent.answer import RetrievalAgent
from retrieval_agent.bm25 import BM25Index
from retrieval_agent.chunking import load_corpus
from retrieval_agent.evaluate import (
    balanced_accuracy,
    build_report,
    choose_threshold,
    is_correct,
    load_examples,
)

ROOT = Path(__file__).resolve().parents[1]


class RetrievalAgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chunks = load_corpus(ROOT / "data" / "corpus")
        cls.index = BM25Index(cls.chunks)
        cls.examples = load_examples(ROOT / "data" / "eval" / "questions.jsonl")
        cls.dev = [example for example in cls.examples if example.split == "dev"]
        cls.threshold = choose_threshold(cls.dev, cls.index)
        cls.agent = RetrievalAgent(cls.index, score_threshold=cls.threshold)

    def test_corpus_has_one_chunk_per_section(self) -> None:
        documents = {chunk.document for chunk in self.chunks}
        self.assertGreaterEqual(len(self.chunks), 18)
        self.assertIn("01-safety.md", documents)
        self.assertIn("06-access.md", documents)

    def test_bm25_ranks_the_lithium_section_first(self) -> None:
        hits = self.index.search("Where are lithium battery packs stored?", k=3)
        self.assertEqual(hits[0].chunk.document, "01-safety.md")
        self.assertEqual(hits[0].chunk.heading, "Lithium batteries")
        self.assertIn("fire cabinet B2", hits[0].chunk.text)

    def test_threshold_is_selected_from_dev_only(self) -> None:
        dev_ids = {example.example_id for example in self.dev}
        self.assertTrue(all(example_id.startswith("dev-") for example_id in dev_ids))
        self.assertGreater(self.threshold, 0)

    def test_unsupported_salary_question_is_refused(self) -> None:
        answer = self.agent.ask("What is the chief executive salary?")
        self.assertTrue(answer.refused)
        self.assertEqual(answer.citations, [])

    def test_cited_answer_names_the_source_file(self) -> None:
        answer = self.agent.ask("What Vault path holds production retrieval secrets?")
        self.assertFalse(answer.refused)
        self.assertIn("kv/prod/retrieval", answer.text)
        self.assertEqual(answer.citations[0]["document"], "06-access.md")

    def test_held_out_report_beats_the_baseline(self) -> None:
        report = build_report(ROOT, split="test")
        systems = {item["system"]: item for item in report["systems"]}
        self.assertGreater(
            systems["bm25_extractive"]["balanced_accuracy"],
            systems["first_hit_baseline"]["balanced_accuracy"],
        )
        self.assertGreaterEqual(systems["bm25_extractive"]["retrieval_recall_at_3"], 0.9)
        self.assertEqual(report["threshold_fit_on"], "dev")

    def test_dev_predictions_are_not_required_to_label_the_test_set(self) -> None:
        rows = [(example, self.agent.ask(example.question)) for example in self.dev]
        self.assertGreaterEqual(balanced_accuracy(rows), 0.8)
        self.assertTrue(any(is_correct(answer, example) for example, answer in rows))


if __name__ == "__main__":
    unittest.main()
