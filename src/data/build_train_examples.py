import json
import random
from pathlib import Path

from src.data.build_train_questions import eval_docs
from src.evaluation.evaluate_answers import has_real_citation, is_refusal
from src.models.inference import generate
from src.models.rag import POOL, build_messages, get_reranker
from src.retrieval.vector_store import get_client, search

IN_PATH = Path("data/train/questions.jsonl")
OUT_PATH = Path("data/train/candidates.jsonl")
UNANSWERABLE_FRACTION = 0.18
TOP_K = 5
REFUSAL = "I don't have enough information in the provided documents to answer that."


def retrieve_excluding(client, reranker, question, exclude):
    candidates = [
        h
        for h in search(client, question, top_k=POOL)
        if (h.payload["department"], h.payload["source"]) not in exclude
    ]
    return reranker.rerank(question, candidates, top_k=TOP_K)


def main() -> None:
    rows = [json.loads(line) for line in IN_PATH.open(encoding="utf-8") if line.strip()]
    rng = random.Random(11)
    kinds = {
        r["chunk_id"]: ("unanswerable" if rng.random() < UNANSWERABLE_FRACTION else "answerable")
        for r in rows
    }

    done = set()
    if OUT_PATH.exists():
        done = {
            json.loads(line)["chunk_id"] for line in OUT_PATH.open(encoding="utf-8") if line.strip()
        }
    todo = [r for r in rows if r["chunk_id"] not in done]
    print(f"already processed: {len(done)}, to do: {len(todo)}")

    client, reranker = get_client(), get_reranker()
    eval_exclude = eval_docs()
    stats = {"kept": 0, "source_missing": 0, "refused": 0, "no_citation": 0}

    with OUT_PATH.open("a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            kind = kinds[r["chunk_id"]]
            own = (r["department"], r["source"])
            exclude = eval_exclude | {own} if kind == "unanswerable" else eval_exclude
            hits = retrieve_excluding(client, reranker, r["question"], exclude)
            messages = build_messages(r["question"], hits)
            record = {
                "chunk_id": r["chunk_id"],
                "question": r["question"],
                "kind": kind,
                "messages": messages,
                "sources": [h.payload["source"] for h in hits],
            }
            if kind == "answerable":
                if not any((h.payload["department"], h.payload["source"]) == own for h in hits):
                    stats["source_missing"] += 1
                    record["answer"] = None
                else:
                    ans = generate(messages)
                    if is_refusal(ans):
                        stats["refused"] += 1
                        record["answer"] = None
                    elif not has_real_citation(ans):
                        stats["no_citation"] += 1
                        record["answer"] = None
                    else:
                        stats["kept"] += 1
                        record["answer"] = ans
            else:
                stats["kept"] += 1
                record["answer"] = REFUSAL
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"{i}/{len(todo)} {stats}", flush=True)

    print(stats)


if __name__ == "__main__":
    main()
