import json
import random
from pathlib import Path

REFUSAL_SHARE = 0.05


def read(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def build(old, new):
    rows = [r for r in read(old) + read(new)]
    answerable = [r for r in rows if r["kind"] == "answerable"]
    refusals = [r for r in rows if r["kind"] == "unanswerable"]
    rng = random.Random(5)
    n_ref = int(len(answerable) * REFUSAL_SHARE / (1 - REFUSAL_SHARE))
    mixed = answerable + rng.sample(refusals, min(n_ref, len(refusals)))
    rng.shuffle(mixed)
    return mixed


for name in ("train", "val"):
    mixed = build(f"data/train/{name}.jsonl", f"data/train/hard_{name}.jsonl")
    with open(f"data/train/v2_{name}.jsonl", "w", encoding="utf-8") as f:
        for r in mixed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_ref = sum(r["kind"] == "unanswerable" for r in mixed)
    print(f"v2_{name}: {len(mixed)} examples, {n_ref} refusals")
