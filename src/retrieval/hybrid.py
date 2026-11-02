import re

import numpy as np
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from src.retrieval.vector_store import COLLECTION, search as dense_search

TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class HybridRetriever:
    """Dense (Qdrant) + BM25 keyword search, merged with Reciprocal Rank Fusion."""

    def __init__(self, client: QdrantClient):
        self.client = client
        self.ids: list[int] = []
        self.payloads: list[dict] = []

        offset = None
        while True:
            points, offset = client.scroll(
                COLLECTION, limit=1000, offset=offset,
                with_payload=True, with_vectors=False,
            )
            for p in points:
                self.ids.append(p.id)
                self.payloads.append(p.payload)
            if offset is None:
                break

        self.pos = {pid: i for i, pid in enumerate(self.ids)}
        self.departments = np.array([p["department"] for p in self.payloads])
        self.bm25 = BM25Okapi([tokenize(p["text"]) for p in self.payloads])

    def _bm25_ranked(self, query: str, departments: list[str] | None, n: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        if departments is not None:
            scores = np.where(np.isin(self.departments, departments), scores, -np.inf)
        order = np.argsort(-scores)[:n]
        return [int(self.ids[i]) for i in order if scores[i] > 0]

    def search(
        self,
        query: str,
        departments: list[str] | None = None,
        top_k: int = 10,
        candidates: int = 50,
        rrf_k: int = 60,
    ) -> list[models.ScoredPoint]:
        dense = [p.id for p in dense_search(self.client, query, departments, candidates)]
        sparse = self._bm25_ranked(query, departments, candidates)

        fused: dict[int, float] = {}
        for ranking in (dense, sparse):
            for rank, pid in enumerate(ranking, start=1):
                fused[pid] = fused.get(pid, 0.0) + 1.0 / (rrf_k + rank)

        best = sorted(fused, key=fused.get, reverse=True)[:top_k]
        return [
            models.ScoredPoint(
                id=pid, version=0, score=fused[pid],
                payload=self.payloads[self.pos[pid]],
            )
            for pid in best
        ]