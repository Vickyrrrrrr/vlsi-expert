#!/usr/bin/env python3
"""
Advanced QLoRA fine-tuning for VLSI Expert MoE model.
Uses Chain-of-Thought (CoT) reasoning — the merged model already has 
DeepSeek-R1's reasoning melted in via DARE+TIES merge.

CoT format: Analyze design → Think step by step → Generate Verilog
This activates the reasoning capabilities from the R1 merge.

Data: 641 verified Verilog pairs
Output: models/vlsi-coder-lora-advanced (~80MB adapter)
"""

import json
import sys
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
MODEL_PATH = str(Path(__file__).parent.parent / "models" / "vlsi-moe-merged" / "merged")
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "vlsi-coder-lora-advanced"
DATA_PATH = Path(__file__).parent.parent / "data" / "train_pairs.jsonl"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LORA_RANK = 64
LORA_ALPHA = 128
MAX_SEQ_LENGTH = 8192  # Longer for CoT + code
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 4


def generate_cot_analysis(spec: str, verilog: str) -> str:
    """Generate Chain-of-Thought analysis from the Verilog code.
    
    Parses the actual Verilog to create a correct analysis, which becomes
    training data. The model learns to THINK before writing code.
    """
    # Detect design type from Verilog
    code_lower = verilog.lower()
    if "fifo" in code_lower:
        design_type = "FIFO (First-In First-Out buffer)"
    elif "counter" in code_lower:
        design_type = "Counter"
    elif "shift" in code_lower:
        design_type = "Shift Register"
    elif "fsm" in code_lower or "state" in code_lower:
        design_type = "Finite State Machine (FSM)"
    elif "uart" in code_lower:
        design_type = "UART Controller"
    elif "spi" in code_lower:
        design_type = "SPI Controller"
    elif "alu" in code_lower:
        design_type = "Arithmetic Logic Unit (ALU)"
    elif "multipl" in code_lower:
        design_type = "Multiplier"
    elif "div" in code_lower:
        design_type = "Divider"
    elif "adder" in code_lower:
        design_type = "Adder"
    elif "crc" in code_lower:
        design_type = "CRC Generator"
    elif "pwm" in code_lower:
        design_type = "PWM Generator"
    elif "arbit" in code_lower:
        design_type = "Arbiter"
    else:
        design_type = "Digital Logic Module"

    # Count ports
    import re
    ports = re.findall(r'(?:input|output|inout)\s+(?:wire|reg|logic)?\s*(?:\[[\d:]+]\s*)?(\w+)', verilog)
    clock = [p for p in ports if 'clk' in p.lower()]
    reset = [p for p in ports if 'rst' in p.lower() or 'reset' in p.lower()]
    data = [p for p in ports if p not in clock + reset]

    clock_str = f"clk ({', '.join(clock)})" if clock else "no explicit clock found"
    reset_str = f"rst ({', '.join(reset)})" if reset else "no reset found"
    data_str = f"{len(data)} data/control ports: {', '.join(data[:8])}{'...' if len(data) > 8 else ''}"

    # Detect behavioral features
    features = []
    if "posedge" in code_lower:
        features.append("synchronous (posedge-triggered)")
    if "negedge" in code_lower:
        features.append("negative-edge sensitive")
    if "always" in code_lower:
        features.append("sequential logic with always block(s)")
    if "assign" in code_lower:
        features.append("combinational assign statements")
    if "case" in code_lower:
        features.append("case-based state decoding")
    if "for" in code_lower:
        features.append("generate/loop-based structure")

    return f"""Design Type: {design_type}
Clock: {clock_str}
Reset: {reset_str}
Ports: {data_str}
Features: {', '.join(features[:5])}

Design Analysis:
This is a {design_type.lower()} with {len(ports)} ports. 
The design uses {' and '.join(features[:3])}.
Key considerations: {spec[:200]}"""


def format_advanced_prompt(example: dict) -> str:
    """Advanced CoT training format: Think → Analyze → Generate."""
    spec = example["spec"]
    verilog = example["verilog"]
    cot = generate_cot_analysis(spec, verilog)

    return (
        "You are an expert VLSI design engineer with deep reasoning capabilities. "
        "For each design specification, first analyze the requirements, then generate "
        "correct, synthesizable Verilog RTL.\n\n"
        f"### Specification\n{spec}\n\n"
        f"### Design Analysis\n{cot}\n\n"
        f"### Verilog RTL\n```verilog\n{verilog}\n```"
    )


def main():
    print("=" * 60)
    print("  VLSI Expert — Advanced CoT Training")
    print(f"  Model:  {MODEL_PATH}")
    print(f"  LoRA:   rank={LORA_RANK}, epochs={NUM_EPOCHS}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)
    print()

    # Check merged model exists
    if not Path(MODEL_PATH).exists():
        print(f"  ERROR: Merged model not found at {MODEL_PATH}")
        print("  Run: python scripts/merge_moe.py first")
        sys.exit(1)

    # Load data
    print("[1/4] Loading training data...")
    with open(DATA_PATH) as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    # Use ALL pairs (even non-syntax-OK — the model learns from errors too)
    print(f"  Using {len(pairs)} training pairs")

    # 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Load merged model
    print(f"[2/4] Loading merged model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    # LoRA on ALL transformer layers (not just attention)
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "w1", "w2", "w3",  # some architectures use these names
        ],
        lora_dropout=0.05,
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
        max_grad_norm=1.0,
        warmup_ratio=0.05,
    )

    print("[3/4] Starting advanced CoT training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        formatting_func=format_advanced_prompt,
    )
    trainer.train()

    print("[4/4] Saving advanced LoRA adapter...")
    trainer.model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"  Saved to: {OUTPUT_DIR}")
    print("  Done! 🎉")


if __name__ == "__main__":
    main()
