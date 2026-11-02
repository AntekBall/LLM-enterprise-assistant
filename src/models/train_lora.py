import json
import os

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

cfg = yaml.safe_load(open("configs/model_config.yaml", encoding="utf-8"))
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", cfg["mlflow_experiment"])


def load_split(path: str) -> Dataset:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    # Prompt/completion format: loss is computed on the assistant answer only.
    return Dataset.from_list(
        [{"prompt": r["messages"][:-1], "completion": r["messages"][-1:]} for r in rows]
    )


def main() -> None:
    t = cfg["training"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map="cuda",
        dtype=torch.bfloat16,
    )

    lora = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    )

    args = SFTConfig(
        output_dir=cfg["adapter_dir"],
        num_train_epochs=t["epochs"],
        learning_rate=t["learning_rate"],
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=t["grad_accum"],
        max_length=t["max_length"],
        warmup_steps=t["warmup_ratio"],
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="no",
        seed=t["seed"],
        report_to="mlflow",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=load_split(cfg["train_file"]),
        eval_dataset=load_split(cfg["val_file"]),
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(cfg["adapter_dir"])
    print(f"adapter saved to {cfg['adapter_dir']}")


if __name__ == "__main__":
    main()
