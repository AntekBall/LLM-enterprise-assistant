import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

JUDGE_MODEL = "microsoft/phi-4"
IN_PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "data/eval/results/baseline.jsonl")
OUT_PATH = IN_PATH.with_suffix(".judged.jsonl")

SYSTEM = "You are a strict grader. Reply with only YES or NO."

CORRECT_PROMPT = """Question: {question}

Reference answer: {reference}

Candidate answer: {answer}

Does the candidate answer state the key facts of the reference answer (same numbers, names, and yes/no conclusion) without contradicting it? Extra detail is fine. Reply YES or NO."""

FAITHFUL_PROMPT = """Context passages:
{context}

Answer: {answer}

Is every factual claim in the answer supported by the context passages? Ignore citation markers like [1]. Reply YES or NO."""

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
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    text = text.strip().upper()
    if text.startswith("YES"):
        return True
    if text.startswith("NO"):
        return False
    return None


def rate(rows: list[dict], key: str) -> str:
    vals = [r[key] for r in rows if r[key] is not None]
    return f"{sum(vals) / len(vals):.2f} (n={len(vals)})" if vals else "n/a"


def main() -> None:
    with IN_PATH.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    for i, r in enumerate(rows, 1):
        r["correct"] = None
        r["faithful"] = None
        if r["answerable"] and not r["refused"]:
            r["correct"] = judge(CORRECT_PROMPT.format(
                question=r["question"], reference=r["reference"], answer=r["answer"]))
        if not r["refused"]:
            r["faithful"] = judge(FAITHFUL_PROMPT.format(
                context=r["context"], answer=r["answer"]))
        print(f"{i}/{len(rows)}", flush=True)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    answered = [r for r in ans if not r["refused"]]
    retrieved = [r for r in ans if r["retrieved"]]

    print(f"\njudge: {JUDGE_MODEL}")
    print(f"answerable ({len(ans)})")
    print(f"  correct (all):           {sum(bool(r['correct']) for r in ans) / len(ans):.2f}")
    print(f"  correct (source in ctx): {sum(bool(r['correct']) for r in retrieved) / max(len(retrieved), 1):.2f}")
    print(f"  faithful (answered):     {rate(answered, 'faithful')}")
    print(f"unanswerable ({len(unans)})")
    print(f"  faithful (if answered):  {rate([r for r in unans if not r['refused']], 'faithful')}")

    print("\n-- spot check by hand")
    judged = [r for r in rows if r["correct"] is not None]
    for r in random.Random(0).sample(judged, min(8, len(judged))):
        print(f"\nQ: {r['question']}\nREF: {r['reference']}\nANS: {r['answer']}\nJUDGE correct={r['correct']}")


if __name__ == "__main__":
    main()