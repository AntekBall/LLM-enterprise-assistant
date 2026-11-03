import argparse
import json
import re
from pathlib import Path

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import get_client, search

EVAL_PATH = Path("data/eval/hard_questions.jsonl")
RESULTS_DIR = Path("data/eval/results")
RUNS = {
    "Baseline (prompt v1)": "baseline",
    "Prompt v2": "prompt_v2",
    "Fine-tuned v1": "finetuned",
    "Fine-tuned v2": "finetuned_v2",
}
KS = (1, 3, 5, 10)
POOL = 50
OUT = Path("docs/results.md")

REFUSAL = re.compile(
    r"(do not|don.{0,6}t) have enough information|cannot be answered|not (enough|sufficient) information|passages? (do|does) not",
    re.I,
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def ratio(num: int, den: int) -> str:
    return f"{num / den:.2f}" if den else "n/a"


def find_rank(hits, record) -> int | None:
    for i, h in enumerate(hits, start=1):
        if (
            h.payload["source"] == record["source"]
            and h.payload["department"] == record["department"]
        ):
            return i
    return None


def retrieval_table() -> str:
    questions = [r for r in load_jsonl(EVAL_PATH) if r.get("keep") and r.get("source")]
    client = get_client()
    hybrid = HybridRetriever(client)
    reranker = Reranker()
    rows = []
    for mode in ("dense", "dense+rerank", "hybrid", "hybrid+rerank"):
        use_hybrid = mode.startswith("hybrid")
        use_rerank = mode.endswith("+rerank")
        ranks = []
        for r in questions:
            q = r["question"]
            top_k = POOL if use_rerank else max(KS)
            if use_hybrid:
                hits = hybrid.search(q, top_k=top_k, candidates=POOL)
            else:
                hits = search(client, q, top_k=top_k)
            if use_rerank:
                hits = reranker.rerank(q, hits, top_k=max(KS))
            ranks.append(find_rank(hits, r))
        n = len(questions)
        hit = [sum(1 for x in ranks if x is not None and x <= k) / n for k in KS]
        mrr = sum(1 / x for x in ranks if x) / n
        rows.append([mode] + [f"{h:.2f}" for h in hit] + [f"{mrr:.3f}"])
        print("done:", mode)
    header = ["Retrieval"] + [f"hit@{k}" for k in KS] + ["MRR"]
    return f"Retrieval on {len(questions)} questions with a known source\n\n" + table(header, rows)


def answer_table() -> str:
    rows = []
    n_ans = n_unans = 0
    for name, stem in RUNS.items():
        path = RESULTS_DIR / f"{stem}.judged.jsonl"
        if not path.exists():
            continue
        data = load_jsonl(path)
        for r in data:
            r["refused"] = bool(REFUSAL.search(r["answer"]))
        ans = [r for r in data if r["answerable"]]
        unans = [r for r in data if not r["answerable"]]
        n_ans, n_unans = len(ans), len(unans)
        answered = [r for r in ans if not r["refused"]]
        with_src = [r for r in ans if r["retrieved"]]
        judged = [r for r in data if not r["refused"] and r.get("faithful") is not None]
        rows.append(
            [
                name,
                ratio(sum(bool(r.get("correct")) for r in ans), len(ans)),
                ratio(sum(bool(r.get("correct")) for r in with_src), len(with_src)),
                ratio(sum(bool(r.get("correct")) for r in answered), len(answered)),
                ratio(sum(r["refused"] for r in ans), len(ans)),
                ratio(sum(r["refused"] for r in unans), len(unans)),
                ratio(sum(bool(r["faithful"]) for r in judged), len(judged)),
                ratio(sum(bool(r.get("cited")) for r in answered), len(answered)),
            ]
        )
    header = [
        "Run",
        "Correct",
        "Correct (source in context)",
        "Correct (answered only)",
        "False refusals",
        "Correct refusals",
        "Faithful",
        "Cited",
    ]
    return f"Answer quality ({n_ans} answerable, {n_unans} unanswerable)\n\n" + table(header, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-retrieval", action="store_true")
    args = parser.parse_args()
    parts = [answer_table()]
    if not args.skip_retrieval:
        parts.insert(0, retrieval_table())
    report = "\n\n".join(parts)
    print("\n" + report)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(report + "\n", encoding="utf-8")
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
