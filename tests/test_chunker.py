from src.data.chunker import chunk_document, split_sections
from src.data.loader import Document


def make_doc(text):
    return Document(text=text, source="a.md", department="legal", title="Policy")


def test_split_sections_uses_headings():
    sections = split_sections("Intro text\n\n## Part A\n\nAAA\n\n## Part B\n\nBBB")
    assert sections == [("", "Intro text"), ("Part A", "AAA"), ("Part B", "BBB")]


def test_chunks_respect_max_words_and_keep_metadata():
    body = "\n\n".join(f"paragraph {i} " + "word " * 40 for i in range(10))
    chunks = chunk_document(make_doc(f"## Big\n\n{body}"), max_words=100, overlap=10)
    assert len(chunks) > 1
    assert all(c.department == "legal" for c in chunks)
    assert all(c.text.startswith("Policy > Big") for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_overlap_repeats_text_between_chunks():
    body = "\n\n".join(f"p{i} " + "w " * 50 for i in range(4))
    chunks = chunk_document(make_doc(f"## S\n\n{body}"), max_words=60, overlap=5)
    assert len(chunks) >= 2
    last_words = chunks[0].text.split()[-5:]
    assert " ".join(last_words) in chunks[1].text


def test_no_chunk_far_exceeds_max_words():
    huge = "\n".join(f"- item {i} " + "word " * 20 for i in range(200))
    chunks = chunk_document(make_doc(f"## List\n\n{huge}"), max_words=100, overlap=10)
    assert max(len(c.text.split()) for c in chunks) < 200
