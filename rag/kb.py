"""Knowledge-base loading and the two chunking strategies (Task 2 + Task 3)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import config


@dataclass
class Document:
    doc_id: str
    title: str
    text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    text = " ".join(text.split())
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def load_documents(kb_dir=config.KB_DIR) -> list[Document]:
    docs = []
    for path in sorted(kb_dir.glob("*.md")):
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        title = lines[0].lstrip("# ").strip()
        body = " ".join(l.strip() for l in lines[1:] if l.strip())
        docs.append(Document(doc_id=path.stem, title=title, text=body))
    return docs


def chunk_fixed(doc: Document, size: int = config.FIXED_CHUNK_SIZE,
                overlap: int = config.FIXED_CHUNK_OVERLAP) -> list[Chunk]:
    """Fixed-size character windows with overlap. Window edges are nudged back to the
    nearest space so words are not cut in half (size is an upper bound)."""
    text, chunks, start, i = doc.text, [], 0, 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start + size // 2:
                end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(f"{doc.doc_id}::fixed::{i}", doc.doc_id, piece,
                                {"doc_id": doc.doc_id, "title": doc.title, "strategy": "fixed",
                                 "start_char": start, "end_char": end, "parent_text": text}))
            i += 1
        if end >= len(text):
            break
        nxt = end - overlap
        sp = text.find(" ", nxt)
        start = sp + 1 if 0 <= sp < end else nxt
    return chunks


def chunk_sentences(doc: Document, per_chunk: int = config.SENTENCES_PER_CHUNK) -> list[Chunk]:
    """Sentence-based chunks: consecutive groups of `per_chunk` whole sentences."""
    sents = split_sentences(doc.text)
    parent = " ".join(sents)
    out = []
    for i in range(0, len(sents), per_chunk):
        out.append(Chunk(f"{doc.doc_id}::sent::{i // per_chunk}", doc.doc_id,
                         " ".join(sents[i:i + per_chunk]),
                         {"doc_id": doc.doc_id, "title": doc.title, "strategy": "sentence",
                          "first_sentence": i, "parent_text": parent}))
    return out


def chunk_all(docs: list[Document], strategy: str) -> list[Chunk]:
    fn = chunk_fixed if strategy == "fixed" else chunk_sentences
    return [c for d in docs for c in fn(d)]


def expand_to_sentences(chunk_text: str, meta: dict) -> list[str]:
    """Small-to-big: return the parent document's WHOLE sentences that overlap this chunk.
    Fixed-size chunks often start or end mid-sentence; generation and grounding checks
    work on whole sentences so answers never quote a half-sentence."""
    parent = meta.get("parent_text")
    if not parent:
        return split_sentences(chunk_text)
    start = meta.get("start_char")
    if start is None:
        idx = parent.find(chunk_text)
        if idx < 0:
            return split_sentences(chunk_text)
        start, end = idx, idx + len(chunk_text)
    else:
        end = meta.get("end_char", start + len(chunk_text))
    out, pos = [], 0
    for s in split_sentences(parent):
        s_start = parent.find(s, pos)
        s_end = s_start + len(s)
        pos = s_end
        if s_start < end and s_end > start:
            out.append(s)
    return out or split_sentences(chunk_text)
