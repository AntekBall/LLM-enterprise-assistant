import json
import re
import sys
from pathlib import Path

from numpy import empty

from src.models.inference import generate
from src.models.rag import build_messages, retrieve

args = [a for a in sys.argv[1:] if not a.startswith("--")]
EVAL_PATH = Path("data/eval/hard_questions.jsonl")
OUT_PATH = Path(args[0] if args else "data/eval/results/baseline.jsonl")
RERANK = "--no-rerank" not in sys.argv
TOP_K = 5

def load_questions() -> list[dict]:
    with EVAL_PATH.open(encoding="utf-8") as f:
        return [r for r in map(json.loads, filter(str.strip, f)) if r.get("keep")]


def is_refusal(text: str) -> bool:
    return bool(re.search(
        r"(do not|don.{0,6}t) have enough information|cannot be answered|not (enough|sufficient) information|passages? (do|does) not",
        text, re.I))


def has_real_citation(text: str) -> bool:
    stripped = re.sub(r"\[\d+\]", "", text).strip()
    return bool(re.search(r"\[\d+\]", text)) and len(stripped.split()) >= 4


def run_one(r: dict) -> dict:
    q = r["question"]
    answerable = r["source"] is not None
    hits = retrieve(q, top_k=TOP_K, rerank=RERANK)
    ans = generate(build_messages(q, hits))
    return {
        "question": q,
        "answerable": answerable,
        "reference": r["reference_answer"],
        "answer": ans,
        "refused": is_refusal(ans),
        "cited": has_real_citation(ans),
        "retrieved": answerable and any(
            h.payload["source"] == r["source"] and h.payload["department"] == r["department"]
            for h in hits
        ),
        "sources": [h.payload["source"] for h in hits],
        "context": "\n\n".join(f"[{i}] {h.payload['text']}" for i, h in enumerate(hits, 1)),
    }


def main() -> None:
    questions = load_questions()
    rows = []
    for i, r in enumerate(questions, 1):
        rows.append(run_one(r))
        print(f"{i}/{len(questions)}", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    answered = [r for r in ans if not r["refused"]]
    empty = sum(len(re.sub(r"\[\d+\]", "", r["answer"]).split()) < 4 and not r["refused"] for r in rows)


    print(f"empty/marker-only answers: {empty}")
    print(f"\nrerank: {RERANK}  top_k: {TOP_K}")
    print(f"answerable ({len(ans)})")
    print(f"  source in context:       {sum(r['retrieved'] for r in ans) / len(ans):.2f}")
    print(f"  false refusals:          {sum(r['refused'] for r in ans) / len(ans):.2f}")
    print(f"  has citation (answered): {sum(r['cited'] for r in answered) / max(len(answered), 1):.2f}")
    print(f"unanswerable ({len(unans)})")
    print(f"  correct refusals:        {sum(r['refused'] for r in unans) / len(unans):.2f}")
    print(f"\nsaved to {OUT_PATH}. Next: judge_answers.py")


if __name__ == "__main__":
    main()