import json
import random
from pathlib import Path

from src.data.build_train_questions import eval_docs, parse_question
from src.data.chunker import chunk_documents
from src.data.loader import load_documents
from src.models.inference import generate

USED_PATH = Path("data/train/questions.jsonl")
OUT_PATH = Path("data/train/hard_questions.jsonl")
N_QUESTIONS = 700
MIN_WORDS, MAX_WORDS = 60, 250

STYLES = [
    'a yes/no question starting with "Can I", "May I" or "Am I allowed to"',
    'a yes/no question starting with "Do I need to", "Is it required" or "Is it okay if"',
    "a question describing a short personal situation and asking what should happen",
    "a question that paraphrases the key fact using everyday words, avoiding the passage's own terms",
]

PROMPT = """You write questions that employees might ask a company handbook assistant.
Read the passage and write ONE question, written as {style}, that the passage answers
(directly or by clear implication).
Rules:
- The question must make sense on its own. Never say "the passage", "the text" or "this document".
- Do not copy long phrases from the passage.
Reply with JSON only, in this form: {{"question": "..."}}"""


def main() -> None:
    used = {
        json.loads(line)["chunk_id"] for line in USED_PATH.open(encoding="utf-8") if line.strip()
    }
    excluded = eval_docs()
    chunks = [
        c
        for c in chunk_documents(load_documents())
        if c.chunk_id not in used
        and (c.department, c.source) not in excluded
        and MIN_WORDS <= len(c.text.split()) <= MAX_WORDS
    ]
    rng = random.Random(21)
    rng.shuffle(chunks)
    per_doc, picked = {}, []
    for c in chunks:
        key = (c.department, c.source)
        if per_doc.get(key, 0) >= 2:
            continue
        per_doc[key] = per_doc.get(key, 0) + 1
        picked.append(c)
        if len(picked) == N_QUESTIONS:
            break

    done = set()
    if OUT_PATH.exists():
        done = {
            json.loads(line)["chunk_id"] for line in OUT_PATH.open(encoding="utf-8") if line.strip()
        }
    todo = [c for c in picked if c.chunk_id not in done]
    print(f"picked {len(picked)}, to do {len(todo)}")

    with OUT_PATH.open("a", encoding="utf-8") as f:
        for i, c in enumerate(todo, 1):
            style = STYLES[i % len(STYLES)]
            messages = [
                {"role": "system", "content": PROMPT.format(style=style)},
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
