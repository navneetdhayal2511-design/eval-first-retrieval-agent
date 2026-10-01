"""Command line: ask one question, or score the frozen eval split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from retrieval_agent.answer import RetrievalAgent
from retrieval_agent.bm25 import BM25Index
from retrieval_agent.chunking import load_corpus
from retrieval_agent.evaluate import build_report, choose_threshold, load_examples, project_root


def _print_answer(question: str, root: Path) -> int:
    corpus = load_corpus(root / "data" / "corpus")
    index = BM25Index(corpus)
    dev = [example for example in load_examples(root / "data" / "eval" / "questions.jsonl") if example.split == "dev"]
    threshold = choose_threshold(dev, index)
    answer = RetrievalAgent(index, score_threshold=threshold).ask(question)
    print(f"Question: {question}")
    print(f"Refused: {answer.refused}")
    print(f"Answer: {answer.text}")
    if answer.citations:
        for citation in answer.citations:
            print(f"Source: {citation['document']} :: {citation['heading']}")
    else:
        print("Source: none")
    if answer.retrieved:
        print("Retrieved:")
        for item in answer.retrieved:
            print(f"  {item['score']:7.3f}  {item['document']} :: {item['heading']}")
    return 0


def _print_eval(root: Path, split: str) -> int:
    report = build_report(root, split=split)
    out_dir = root / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{split}_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Split: {report['split']}")
    print(f"Threshold fit on dev: {report['score_threshold']}")
    print(f"Minimum token coverage: {report['min_coverage']}")
    print()
    header = f"{'system':<22} {'balanced':>10} {'answer':>10} {'refuse':>10} {'recall@3':>10} {'keywords':>10}"
    print(header)
    print("-" * len(header))
    for system in report["systems"]:
        print(
            f"{system['system']:<22} "
            f"{system['balanced_accuracy']:10.3f} "
            f"{system['answerable_accuracy']:10.3f} "
            f"{system['refusal_accuracy']:10.3f} "
            f"{system['retrieval_recall_at_3']:10.3f} "
            f"{system['keyword_recall']:10.3f}"
        )
    print()
    print(f"Failure taxonomy: {report['failure_taxonomy'] or 'none'}")
    print(f"Wrote {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Eval-first retrieval agent")
    parser.add_argument("--root", type=Path, default=None, help="Project root. Defaults to the repo directory.")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Answer one question from the handbook")
    ask.add_argument("question")

    evaluate = sub.add_parser("eval", help="Score the agent against the first-hit baseline")
    evaluate.add_argument("--split", default="test", choices=("dev", "test"))

    args = parser.parse_args(argv)
    root = args.root or project_root()
    if args.command == "ask":
        return _print_answer(args.question, root)
    return _print_eval(root, args.split)


if __name__ == "__main__":
    sys.exit(main())
