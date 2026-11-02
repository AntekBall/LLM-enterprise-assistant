from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-large-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, device="cuda")
    return _model


def embed_documents(texts: list[str]) -> list[list[float]]:
    return (
        get_model()
        .encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        .tolist()
    )


def embed_query(query: str) -> list[float]:
    return get_model().encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
