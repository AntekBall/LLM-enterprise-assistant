import json
import sys
from pathlib import Path

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import get_client, search

EVAL_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/eval_candidates.jsonl")
MODE = sys.argv[2] if len(sys.argv) > 2 else "dense"
KS = (1, 3, 5, 10)
POOL = 50


def load_questions() -> list[dict]:
    with EVAL_PATH.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return [r for r in records if r.get("keep") and r.get("source")]


def find_rank(hits, record) -> int | None:
    for i, h in enumerate(hits, start=1):
        if (
            h.payload["source"] == record["source"]
            and h.payload["department"] == record["department"]
        ):
            return i
    return None


def main() -> None:
    client = get_client()
    use_hybrid = MODE.startswith("hybrid")
    use_rerank = MODE.endswith("+rerank")
    retriever = HybridRetriever(client) if use_hybrid else None
    reranker = Reranker() if use_rerank else None
    questions = load_questions()
    ranks = []
    for r in questions:
        q = r["question"]
        top_k = POOL if use_rerank else max(KS)
        if retriever:
            hits = retriever.search(q, top_k=top_k, candidates=POOL)
        else:
            hits = search(client, q, top_k=top_k)
        if reranker:
            hits = reranker.rerank(q, hits, top_k=max(KS))
        ranks.append(find_rank(hits, r))

    n = len(questions)
    print(f"mode: {MODE}")
    print(f"questions: {n}")
    for k in KS:
        hit = sum(1 for r in ranks if r is not None and r <= k) / n
        print(f"hit@{k}: {hit:.2f}")
    mrr = sum(1 / r for r in ranks if r) / n
    print(f"MRR: {mrr:.3f}")

    print("\nmisses (not in top 10):")
    for r, rank in zip(questions, ranks):
        if rank is None:
            print("-", r["question"], "->", r["department"], r["source"])


if __name__ == "__main__":
    main()
