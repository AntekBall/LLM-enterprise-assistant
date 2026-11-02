import json
import random
from pathlib import Path

from src.data.chunker import chunk_documents
from src.data.loader import load_documents
from src.models.inference import generate

OUT_PATH = Path("data/processed/eval_candidates.jsonl")
N_QUESTIONS = 60  
MIN_WORDS, MAX_WORDS = 80, 250

PROMPT = """You write test questions for a company handbook assistant.
Read the passage and write ONE question that an employee might ask and that the passage answers fully.
Rules:
- The question must make sense on its own. Never say "the passage", "the text" or "this document".
- Do not copy long phrases from the passage. Prefer natural wording.
- Then write a short reference answer using only facts from the passage.
Reply with JSON only, in this form: {"question": "...", "answer": "..."}"""


def parse_json(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not data.get("question") or not data.get("answer"):
        return None
    return data


def pick_chunks(chunks, n: int, seed: int = 42):
    """One chunk per source file, so questions cover many documents."""
    rng = random.Random(seed)
    usable = [c for c in chunks if MIN_WORDS <= len(c.text.split()) <= MAX_WORDS]
    rng.shuffle(usable)
    picked, seen = [], set()
    for c in usable:
        if c.source in seen:
            continue
        seen.add(c.source)
        picked.append(c)
        if len(picked) == n:
            break
    return picked


def main() -> None:
    chunks = chunk_documents(load_documents())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for c in pick_chunks(chunks, N_QUESTIONS):
            messages = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": c.text},
            ]
            data = parse_json(generate(messages, max_new_tokens=300))
            if data is None:
                continue
            record = {
                "question": data["question"],
                "reference_answer": data["answer"],
                "source": c.source,
                "department": c.department,
                "chunk_id": c.chunk_id,
                "context": c.text,
                "keep": True,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
    print(f"wrote {kept} candidates to {OUT_PATH}")


if __name__ == "__main__":
    main()