import json
import random
from pathlib import Path

from src.data.chunker import chunk_documents
from src.data.loader import load_documents
from src.models.inference import generate

EVAL_FILES = [
    Path("data/eval/hard_questions.jsonl"),
    Path("data/processed/eval_candidates.jsonl"),
]
OUT_PATH = Path("data/train/questions.jsonl")
N_QUESTIONS = 1200
MAX_PER_DOC = 2
MIN_WORDS, MAX_WORDS = 60, 250

PROMPT = """You write questions that employees might ask a company handbook assistant.
Read the passage and write ONE question that the passage answers with a specific fact
(a number, a name, a duration, a rule, who is responsible, or a yes/no with its condition).
Rules:
- The question must make sense on its own. Never say "the passage", "the text" or "this document".
- Do not copy long phrases from the passage. Use natural, everyday wording.
- Do not mention the passage's title or section names in the question.
Reply with JSON only, in this form: {"question": "..."}"""


def eval_docs() -> set[tuple[str, str]]:
    docs = set()
    for path in EVAL_FILES:
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("source"):
                docs.add((r["department"], r["source"]))
    return docs


def parse_question(text: str) -> str | None:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        q = json.loads(text[start : end + 1]).get("question")
    except json.JSONDecodeError:
        return None
    return q.strip() if isinstance(q, str) and len(q.split()) >= 4 else None


def pick_chunks(chunks, excluded, n, seed=7):
    rng = random.Random(seed)
    usable = [
        c
        for c in chunks
        if (c.department, c.source) not in excluded
        and MIN_WORDS <= len(c.text.split()) <= MAX_WORDS
    ]
    rng.shuffle(usable)
    per_doc, picked = {}, []
    for c in usable:
        key = (c.department, c.source)
        if per_doc.get(key, 0) >= MAX_PER_DOC:
            continue
        per_doc[key] = per_doc.get(key, 0) + 1
        picked.append(c)
        if len(picked) == n:
            break
    return picked


def main() -> None:
    chunks = chunk_documents(load_documents())
    excluded = eval_docs()
    picked = pick_chunks(chunks, excluded, N_QUESTIONS)
    print(f"excluded eval documents: {len(excluded)}")
    print(f"chunks picked: {len(picked)} from {len({(c.department, c.source) for c in picked})} documents")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT_PATH.exists():
        done = {json.loads(l)["chunk_id"] for l in OUT_PATH.open(encoding="utf-8") if l.strip()}
    todo = [c for c in picked if c.chunk_id not in done]
    print(f"already done: {len(done)}, to do: {len(todo)}")

    with OUT_PATH.open("a", encoding="utf-8") as f:
        for i, c in enumerate(todo, 1):
            messages = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": c.text},
            ]
            q = parse_question(generate(messages, max_new_tokens=120))
            if q:
                record = {
                    "question": q,
                    "source": c.source,
                    "department": c.department,
                    "chunk_id": c.chunk_id,
                    "context": c.text,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            if i % 25 == 0:
                print(f"{i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()