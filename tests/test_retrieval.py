from types import SimpleNamespace

from src.models.rag import SYSTEM_PROMPT, build_messages
from src.retrieval.vector_store import get_client


def make_hit(n):
    return SimpleNamespace(
        payload={"title": f"Title {n}", "section": f"Sec {n}", "text": f"text {n}"}
    )


def test_build_messages_numbers_passages_and_includes_question():
    msgs = build_messages("What is X?", [make_hit(1), make_hit(2)])
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    user = msgs[1]["content"]
    assert "Passage 1 (Title 1 > Sec 1):\ntext 1" in user
    assert "Passage 2 (Title 2 > Sec 2):\ntext 2" in user
    assert user.endswith("Question: What is X?")


def test_system_prompt_contains_canonical_refusal():
    assert (
        "I don't have enough information in the provided documents to answer that." in SYSTEM_PROMPT
    )


def test_get_client_reads_qdrant_url_from_env(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "src.retrieval.vector_store.QdrantClient",
        lambda url: seen.setdefault("url", url),
    )
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    get_client()
    assert seen["url"] == "http://qdrant:6333"
