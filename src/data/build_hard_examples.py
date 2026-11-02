import json
from pathlib import Path

from src.data.build_train_questions import eval_docs
from src.evaluation.evaluate_answers import has_real_citation, is_refusal
from src.models.inference import generate
from src.models.rag import POOL, build_messages, get_reranker
from src.retrieval.vector_store import get_client, search

IN_PATH = Path("data/train/hard_questions.jsonl")
OUT_PATH = Path("data/train/hard_candidates.jsonl")
TOP_K = 5
HINT = (
    "\n\nNote: these passages do contain enough information to answer. "
    "Give a direct answer (start with Yes or No for yes/no questions) and cite the passages."
)


def main() -> None:
    rows = [json.loads(line) for line in IN_PATH.open(encoding="utf-8") if line.strip()]
    done = set()
    if OUT_PATH.exists():
        done = {
            json.loads(line)["chunk_id"] for line in OUT_PATH.open(encoding="utf-8") if line.strip()
        }
    todo = [r for r in rows if r["chunk_id"] not in done]
    print(f"to do: {len(todo)}")

    client, reranker = get_client(), get_reranker()
    exclude = eval_docs()
    stats = {"kept": 0, "rescued": 0, "source_missing": 0, "bad": 0}

    with OUT_PATH.open("a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            own = (r["department"], r["source"])
            cands = [
                h
                for h in search(client, r["question"], top_k=POOL)
                if (h.payload["department"], h.payload["source"]) not in exclude
            ]
            hits = reranker.rerank(r["question"], cands, top_k=TOP_K)
            messages = build_messages(r["question"], hits)
            answer = None
            if not any((h.payload["department"], h.payload["source"]) == own for h in hits):
                stats["source_missing"] += 1
            else:
                answer = generate(messages)
                if is_refusal(answer):
                    hinted = [
                        messages[0],
                        {**messages[1], "content": messages[1]["content"] + HINT},
                    ]
                    answer = generate(hinted)
                    rescued = True
                else:
                    rescued = False
                if is_refusal(answer) or not has_real_citation(answer):
                    stats["bad"] += 1
                    answer = None
                else:
                    stats["kept"] += 1
                    stats["rescued"] += rescued
            f.write(
                json.dumps(
                    {
                        "chunk_id": r["chunk_id"],
                        "question": r["question"],
                        "kind": "answerable",
                        "messages": messages,
                        "answer": answer,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()
            if i % 25 == 0:
                print(f"{i}/{len(todo)} {stats}", flush=True)
    print(stats)


if __name__ == "__main__":
    main()
