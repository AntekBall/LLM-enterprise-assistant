import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

JUDGE_MODEL = "microsoft/phi-4"
IN_PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "data/train/candidates.jsonl")
PREFIX = sys.argv[2] if len(sys.argv) > 2 else ""
TRAIN_PATH = Path(f"data/train/{PREFIX}train.jsonl")
VAL_PATH = Path(f"data/train/{PREFIX}val.jsonl")
VAL_FRACTION = 0.05

SYSTEM = "You are a strict grader. Reply with only YES or NO."

FAITHFUL = """Context passages and question:
{context}

Answer: {answer}

Is every factual claim in the answer supported by the passages? Ignore citation markers like [1]. Reply YES or NO."""

ANSWERS = """Context passages and question:
{context}

Answer: {answer}

Does the answer directly and specifically answer the question? Reply YES or NO."""

HAS_ANSWER = """Context passages and question:
{context}

Do the passages contain the information needed to answer the question? Reply YES or NO."""

tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ),
    device_map="cuda",
)


def judge(prompt: str) -> bool | None:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=3, do_sample=False)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    text = text.strip().upper()
    return True if text.startswith("YES") else False if text.startswith("NO") else None


def main() -> None:
    rows = [json.loads(line) for line in IN_PATH.open(encoding="utf-8") if line.strip()]
    rows = [r for r in rows if r["answer"]]
    kept, dropped = [], {"answerable": 0, "unanswerable": 0}

    for i, r in enumerate(rows, 1):
        context = r["messages"][1]["content"]
        if r["kind"] == "answerable":
            ok = (
                judge(FAITHFUL.format(context=context, answer=r["answer"])) is True
                and judge(ANSWERS.format(context=context, answer=r["answer"])) is True
            )
        else:
            ok = judge(HAS_ANSWER.format(context=context)) is False
        if ok:
            kept.append(r)
        else:
            dropped[r["kind"]] += 1
        if i % 50 == 0:
            print(f"{i}/{len(rows)} kept={len(kept)} dropped={dropped}", flush=True)

    random.Random(3).shuffle(kept)
    n_val = int(len(kept) * VAL_FRACTION)
    splits = {VAL_PATH: kept[:n_val], TRAIN_PATH: kept[n_val:]}
    for path, items in splits.items():
        with path.open("w", encoding="utf-8") as f:
            for r in items:
                messages = r["messages"] + [{"role": "assistant", "content": r["answer"]}]
                f.write(
                    json.dumps({"messages": messages, "kind": r["kind"]}, ensure_ascii=False) + "\n"
                )

    for path, items in splits.items():
        n_un = sum(r["kind"] == "unanswerable" for r in items)
        print(f"{path}: {len(items)} examples ({n_un} refusals)")
    print(f"dropped: {dropped}")


if __name__ == "__main__":
    main()
