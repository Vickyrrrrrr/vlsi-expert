#!/usr/bin/env python3
"""
QLoRA fine-tune CODER head: writes syntactically correct, synthesizable Verilog.
Base: Qwen2.5-Coder-32B-Instruct
Data: 500+ (spec → Verilog) from VerilogEval + RTLLM
Output: models/vlsi-coder-lora (~80MB adapter)
"""

import os
import json
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer

# ── Config ────────────────────────────────────────────────────────────
BASE_MODEL = str(Path(__file__).parent.parent / "models" / "vlsi-moe-merged" / "merged")
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "vlsi-coder-lora"
DATA_PATH = Path(__file__).parent.parent / "data" / "train_pairs.jsonl"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LORA_RANK = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
MAX_SEQ_LENGTH = 4096
BATCH_SIZE = 1  # MI300X handles this
GRAD_ACCUM = 8  # effective batch = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3


def format_coder_prompt(example: dict) -> str:
    """Format training prompt for coder model."""
    spec = example["spec"]
    verilog = example["verilog"]
    return (
        "You are an expert VLSI engineer. Generate correct, synthesizable Verilog RTL "
        "for the given specification. Output ONLY the Verilog code with no explanation.\n\n"
        f"### Specification\n{spec}\n\n"
        f"### Verilog RTL\n```verilog\n{verilog}\n```"
    )


def main():
    print("=" * 60)
    print("  VLSI Expert — CODER Head Training")
    print(f"  Base model: {BASE_MODEL}")
    print(f"  LoRA rank:  {LORA_RANK}")
    print(f"  Output:     {OUTPUT_DIR}")
    print("=" * 60)
    print()

    # Load data
    print("[1/4] Loading training data...")
    pairs = []
    with open(DATA_PATH) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    # Filter to syntax-OK pairs only for coder training
    clean_pairs = [p for p in pairs if p.get("syntax_ok", True)]
    print(f"  Using {len(clean_pairs)} syntax-verified pairs out of {len(pairs)} total")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Load model + tokenizer
    print(f"[2/4] Loading merged model ({BASE_MODEL})...")
    if not Path(BASE_MODEL).exists():
        print(f"  ERROR: Merged model not found at {BASE_MODEL}")
        print("  Run: python scripts/merge_moe.py first")
        sys.exit(1)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare for QLoRA
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    # LoRA config
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create dataset
    train_ds = Dataset.from_list(clean_pairs)

    # Training args
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        optim="adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
    )

    # Trainer
    print("[3/4] Starting training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        formatting_func=format_coder_prompt,
    )
    trainer.train()

    # Save
    print("[4/4] Saving LoRA adapter...")
    trainer.model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"  Saved to: {OUTPUT_DIR}")
    print("  Done!")


if __name__ == "__main__":
    main()
