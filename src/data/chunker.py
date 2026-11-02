import re
from dataclasses import dataclass

from src.data.loader import Document

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
        body = text[last:match.start()].strip()
        if body:
            sections.append((heading, body))
        heading = match.group(1).strip()
        last = match.end()
    body = text[last:].strip()
    if body:
        sections.append((heading, body))
    return sections


def pack_paragraphs(body: str, max_words: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for para in body.split("\n\n"):
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