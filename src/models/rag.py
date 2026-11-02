from functools import lru_cache

from src.models.inference import generate
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import get_client, search

POOL = 50

SYSTEM_PROMPT = """You are an assistant that answers questions about company handbook documents.
Rules:
- Answer ONLY using the numbered context passages provided.
- Cite the passages you used with their numbers in square brackets, like [1] or [2][3].
- If the context does not contain the answer, reply exactly: "I don't have enough information in the provided documents to answer that."
- Do not use outside knowledge. Be concise."""


def build_messages(question: str, hits) -> list[dict]:
    context = "\n\n".join(
        f"[{i}] ({h.payload['title']} > {h.payload['section']})\n{h.payload['text']}"
        for i, h in enumerate(hits, start=1)
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"
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