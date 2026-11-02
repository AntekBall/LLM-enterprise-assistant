from functools import lru_cache

from src.models.inference import generate
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import get_client, search

POOL = 50

SYSTEM_PROMPT = """You answer questions about company handbook documents using ONLY the numbered passages provided.

Rules:
- Write a direct answer in one to three complete sentences, stating the key facts (numbers, names, durations, who/what).
- After each claim, cite the passage number in square brackets, like [1] or [2][3]. Never answer with only a citation.
- If the passages state or clearly imply the answer, answer it, including yes/no questions ("Yes, ..." or "No, ...").
- Only if the passages contain no relevant information, reply exactly: "I don't have enough information in the provided documents to answer that."
- Do not use outside knowledge. Do not say "based on the provided context" and do not add a "Therefore" summary."""


def build_messages(question: str, hits) -> list[dict]:
    context = "\n\n".join(
        f"Passage {i} ({h.payload['title']} > {h.payload['section']}):\n{h.payload['text']}"
        for i, h in enumerate(hits, start=1)
    )
    user = f"{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()


def retrieve(
    question: str,
    departments: list[str] | None = None,
    top_k: int = 5,
    rerank: bool = True,
):
    client = get_client()
    if not rerank:
        return search(client, question, departments=departments, top_k=top_k)
    candidates = search(client, question, departments=departments, top_k=POOL)
    return get_reranker().rerank(question, candidates, top_k=top_k)


def answer(
    question: str,
    departments: list[str] | None = None,
    top_k: int = 5,
    rerank: bool = True,
) -> dict:
    hits = retrieve(question, departments, top_k, rerank)
    text = generate(build_messages(question, hits))
    sources = [
        {
            "n": i,
            "source": h.payload["source"],
            "title": h.payload["title"],
            "section": h.payload["section"],
            "department": h.payload["department"],
            "score": round(h.score, 3),
        }
        for i, h in enumerate(hits, start=1)
    ]
    return {"answer": text, "sources": sources}