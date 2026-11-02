import json
import sys
from pathlib import Path

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector_store import get_client, search

EVAL_PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "data/eval/hard_questions.jsonl")
DEPTH = 100
POOLS = (10, 20, 30, 50, 100)


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
    retriever = HybridRetriever(client)
    questions = load_questions()

    dense_ranks, hybrid_ranks = [], []
    for r in questions:
        dense_ranks.append(find_rank(search(client, r["question"], top_k=DEPTH), r))
        hybrid_ranks.append(
            find_rank(retriever.search(r["question"], top_k=DEPTH, candidates=DEPTH), r)
        )

    n = len(questions)
    print(f"questions: {n}\n")
    print("recall by candidate pool size")
    print("pool   dense  hybrid")
    for p in POOLS:
        d = sum(1 for x in dense_ranks if x is not None and x <= p) / n
        h = sum(1 for x in hybrid_ranks if x is not None and x <= p) / n
        print(f"{p:>4}   {d:.2f}   {h:.2f}")

    print("\nquestions not in the top 10 of either mode (rank, or - if beyond depth)")
    for r, d, h in zip(questions, dense_ranks, hybrid_ranks):
        if (d is None or d > 10) and (h is None or h > 10):
            print(f"dense={d or '-':>4} hybrid={h or '-':>4}  {r['question']}")
            print(f"{'':24}-> {r['department']}/{r['source']}")


if __name__ == "__main__":
    main()