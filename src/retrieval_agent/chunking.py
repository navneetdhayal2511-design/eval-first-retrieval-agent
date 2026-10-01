"""Split handbook markdown into retrievable sections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document: str
    heading: str
    text: str

    @property
    def indexed_text(self) -> str:
        return f"{self.heading}\n{self.text}"


def chunk_markdown(text: str, document: str) -> list[Chunk]:
    """One chunk per heading. Body text before the first section is kept."""
    heading = "Overview"
    buckets: list[tuple[str, list[str]]] = [(heading, [])]
    for line in text.splitlines():
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            heading = line[3:].strip() or "Overview"
            buckets.append((heading, []))
            continue
        buckets[-1][1].append(line)

    chunks: list[Chunk] = []
    for section, lines in buckets:
        body = "\n".join(lines).strip()
        if not body:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{document}::{section}",
                document=document,
                heading=section,
                text=body,
            )
        )
    return chunks


def load_corpus(corpus_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), path.name))
    if not chunks:
        raise FileNotFoundError(f"No markdown documents in {corpus_dir}")
    return chunks
