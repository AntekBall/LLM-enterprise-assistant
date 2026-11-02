from src.models.inference import generate
from src.retrieval.vector_store import get_client, search

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


def answer(question: str, departments: list[str] | None = None, top_k: int = 5) -> dict:
    hits = search(get_client(), question, departments=departments, top_k=top_k)
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