import re
from dataclasses import dataclass

from src.data.loader import Document

MIN_WORDS = 20

HEADING = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)


@dataclass
class Chunk:
    text: str
    source: str
    department: str
    title: str
    section: str
    chunk_id: str


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split on Markdown headings. Returns (heading, body) pairs."""
    sections = []
    heading = ""
    last = 0
    for match in HEADING.finditer(text):
        body = text[last : match.start()].strip()
        if body:
            sections.append((heading, body))
        heading = match.group(1).strip()
        last = match.end()
    body = text[last:].strip()
    if body:
        sections.append((heading, body))
    return sections


def split_long(para: str, max_words: int) -> list[str]:
    if len(para.split()) <= max_words:
        return [para]
    parts = para.split("\n")
    if len(parts) == 1:
        parts = re.split(r"(?<=[.!?])\s+", para)
    out: list[str] = []
    buf: list[str] = []
    count = 0
    for part in parts:
        n = len(part.split())
        if n > max_words:  # a single huge line: cut by words
            words = part.split()
            out.extend(" ".join(words[i : i + max_words]) for i in range(0, n, max_words))
            continue
        if buf and count + n > max_words:
            out.append("\n".join(buf))
            buf, count = [], 0
        buf.append(part)
        count += n
    if buf:
        out.append("\n".join(buf))
    return out


def pack_paragraphs(body: str, max_words: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    paragraphs = [p for para in body.split("\n\n") for p in split_long(para, max_words)]
    for para in paragraphs:
        n = len(para.split())
        if current and count + n > max_words:
            chunks.append("\n\n".join(current))
            tail = " ".join(chunks[-1].split()[-overlap:]) if overlap else ""
            current = [tail] if tail else []
            count = len(tail.split())
        current.append(para)
        count += n
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_document(doc: Document, max_words: int = 300, overlap: int = 50) -> list[Chunk]:
    chunks = []
    for heading, body in split_sections(doc.text):
        for piece in pack_paragraphs(body, max_words, overlap):
            context = f"{doc.title} > {heading}" if heading else doc.title
            if len(piece.split()) < MIN_WORDS:
                continue
            chunks.append(
                Chunk(
                    text=f"{context}\n\n{piece}",
                    source=doc.source,
                    department=doc.department,
                    title=doc.title,
                    section=heading,
                    chunk_id=f"{doc.source}::{len(chunks)}",
                )
            )
    return chunks


def chunk_documents(docs: list[Document], **kwargs) -> list[Chunk]:
    return [c for doc in docs for c in chunk_document(doc, **kwargs)]
