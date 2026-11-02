from qdrant_client import models
from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class Reranker:
    """Cross-encoder that scores (query, chunk) pairs jointly and re-sorts hits."""

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cuda"):
        self.model = CrossEncoder(model_name, device=device, max_length=512)

    def rerank(
        self, query: str, hits: list[models.ScoredPoint], top_k: int = 10
    ) -> list[models.ScoredPoint]:
        if not hits:
            return []
        pairs = [(query, h.payload["text"]) for h in hits]
        scores = self.model.predict(pairs, batch_size=16)
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            models.ScoredPoint(id=h.id, version=0, score=float(s), payload=h.payload)
            for h, s in ranked
        ]
