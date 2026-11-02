import pytest
from fastapi.testclient import TestClient

from src.api import main

FAKE = {
    "answer": "20 days. [1]",
    "sources": [{
        "n": 1, "source": "a.md", "title": "T", "section": "S",
        "department": "total-rewards", "score": 0.5,
    }],
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "load_llm", lambda: None)
    monkeypatch.setattr(main, "get_reranker", lambda: None)
    monkeypatch.setattr(main, "answer", lambda q, d, k: FAKE)
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_answer_and_sources(client):
    r = client.post("/ask", json={"question": "How many vacation days?"})
    assert r.status_code == 200
    assert r.json()["answer"] == "20 days. [1]"
    assert r.json()["sources"][0]["department"] == "total-rewards"


def test_ask_rejects_unknown_department(client):
    r = client.post("/ask", json={"question": "hello there", "departments": ["nope"]})
    assert r.status_code == 422


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "top_k": 99}).status_code == 422