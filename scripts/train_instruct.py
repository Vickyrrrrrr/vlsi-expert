#!/usr/bin/env python3
"""
QLoRA fine-tune INSTRUCT head: fixes Verilog errors, generates SDC, analyzes timing.
Base: DeepSeek-R1-Distill-Qwen-32B (reasoning native)
Data: 200+ (error → fix) pairs from syntax failures
Output: models/vlsi-instruct-lora (~80MB adapter)
"""

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
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "vlsi-instruct-lora"
DATA_PATH = Path(__file__).parent.parent / "data" / "error_fix_pairs.jsonl"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
MAX_SEQ_LENGTH = 3072
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3

# Additional manual error→fix training examples
MANUAL_FIX_EXAMPLES = [
    {
        "instruction": "Fix this Verilog error",
        "input": "ERROR: missing 'reg' for signal assignment in always block",
        "output": "Change 'output logic [7:0] count' to 'output reg [7:0] count' when used inside 'always @(posedge clk)' block. In Verilog, signals assigned inside always blocks must be declared as 'reg' type."
    },
    {
        "instruction": "Fix timing violation",
        "input": "ERROR: setup timing violation WNS=-3.2ns on path from alu_result to output_reg at 500MHz",
        "output": "Timing violation on alu_result → output_reg (8.2ns path at 500MHz/2ns clock). Solutions: (1) Pipeline the path by inserting a register between ALU and output. (2) Relax clock to 300MHz (3.33ns). (3) Use faster cells (if available in PDK). Recommended: insert pipeline stage with retiming."
    },
    {
        "instruction": "Generate SDC constraints",
        "input": "Clock: clk at 100MHz. Input delay: data_in 0.5ns. Output delay: result 1.0ns. Clock uncertainty: 0.1ns",
        "output": "# SDC Constraints\ncreate_clock -name clk -period 10.0 [get_ports clk]\nset_input_delay -clock clk 0.5 [get_ports data_in]\nset_output_delay -clock clk 1.0 [get_ports result]\nset_clock_uncertainty 0.1 [get_clocks clk]\nset_false_path -from [get_ports rst_n]"
    },
    {
        "instruction": "Fix module port mismatch",
        "input": "ERROR: Port 'count[7:0]' of module 'counter' has width 8, but connected to 'cnt[3:0]' of width 4",
        "output": "Port width mismatch: counter expects 8-bit count, received 4-bit cnt. Fix: Change the instantiating signal to 'wire [7:0] cnt' to match the module port width. The counter module definition is correct — the instantiating code needs the signal widened."
    },
]


def format_instruct_prompt(example: dict) -> str:
    """Format training prompt for instruct/reasoning model."""
    instruction = example.get("instruction", "Fix this Verilog error")
    user_input = example.get("input", "")
    output = example.get("output", "")

    # DeepSeek-R1 CoT format
    return (
        f"### Instruction\n{instruction}\n\n"
        f"### Input\n{user_input}\n\n"
        f"### Response\n{output}"
    )


def main():
    print("=" * 60)
    print("  VLSI Expert — INSTRUCT Head Training")
    print(f"  Base model: {BASE_MODEL}")
    print(f"  LoRA rank:  {LORA_RANK}")
    print(f"  Output:     {OUTPUT_DIR}")
    print("=" * 60)
    print()

    # Load data
    print("[1/4] Loading training data...")
    pairs = []
    if DATA_PATH.exists():
        with open(DATA_PATH) as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
    # Add manual examples
    pairs.extend(MANUAL_FIX_EXAMPLES)
    print(f"  Using {len(pairs)} fix+instruct pairs")

    # 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print("[2/4] Loading base model (DeepSeek-R1-Distill-Qwen-32B)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

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

    train_ds = Dataset.from_list(pairs)

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

    print("[3/4] Starting training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        formatting_func=format_instruct_prompt,
    )
    trainer.train()

    print("[4/4] Saving LoRA adapter...")
    trainer.model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"  Saved to: {OUTPUT_DIR}")
    print("  Done!")


if __name__ == "__main__":
    main()
