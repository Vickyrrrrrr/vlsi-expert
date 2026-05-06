#!/usr/bin/env python3
"""
Dual-model inference: Qwen2.5-Coder-32B for Verilog + DeepSeek-R1 for reasoning.
Task router sends to the right model. Both loaded on MI300X simultaneously.

Usage: python scripts/dual_inference.py --test
"""

import argparse
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Model paths ────────────────────────────────────────────────────────
CODER_PATH = "Qwen/Qwen2.5-Coder-32B-Instruct"
REASON_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

# Router patterns (from train_router.py)
CODER_PATTERNS = [
    "generate verilog", "write rtl", "create module", "design a",
    "implement", "verilog code for", "hdl for", "rtl for",
    "module for", "build a", "make a", "create a",
]
REASON_PATTERNS = [
    "fix error", "fix the", "debug", "why does", "analyze",
    "explain why", "what is wrong", "correct this", "repair",
    "timing violation", "synthesis failed", "syntax error",
    "sdc constraint", "timing constraint", "create_clock",
    "set_input_delay", "set_output_delay", "clock period",
]


def route_to_model(text: str) -> tuple:
    """Returns: (model_name, model, tokenizer)"""
    text_lower = text.lower()
    for p in REASON_PATTERNS:
        if p in text_lower:
            return "REASON"
    return "CODER"


def load_models():
    """Load both models — fits in 192GB VRAM."""
    print("Loading CODER (Qwen2.5-Coder-32B)...")
    coder = AutoModelForCausalLM.from_pretrained(
        CODER_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    coder_tok = AutoTokenizer.from_pretrained(CODER_PATH, trust_remote_code=True)
    coder_tok.pad_token = coder_tok.eos_token

    print("Loading REASON (DeepSeek-R1-32B)...")
    reason = AutoModelForCausalLM.from_pretrained(
        REASON_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    reason_tok = AutoTokenizer.from_pretrained(REASON_PATH, trust_remote_code=True)
    reason_tok.pad_token = reason_tok.eos_token

    print("✅ Both models loaded!")
    return (coder, coder_tok), (reason, reason_tok)


def generate_chip(query: str, models, toks):
    """Route to the right model and generate chip design output."""
    route = route_to_model(query)
    print(f"\n  📍 Routed to: {route}")

    if route == "CODER":
        model, tok = models[0], toks[0]
        prompt = (
            f"Generate correct, synthesizable Verilog RTL for: {query}\n\n"
            "Output ONLY the code in a ```verilog fence:\n"
        )
    else:
        model, tok = models[1], toks[1]
        prompt = (
            f"Analyze and fix this VLSI issue: {query}\n\n"
            "Think step by step about the root cause, then provide a detailed fix:\n"
        )

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    start = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, temperature=0.2, do_sample=True, pad_token_id=tok.eos_token_id)
    elapsed = time.time() - start
    result = tok.decode(out[0], skip_special_tokens=True)
    print(f"  ⚡ Generated {len(result)} chars in {elapsed:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--query", type=str, default="")
    args = parser.parse_args()

    print("=" * 60)
    print("  VLSI Expert — Dual Model Inference")
    print("  Coder: Qwen2.5-Coder-32B")
    print("  Reason: DeepSeek-R1-Distill-Qwen-32B")
    print("=" * 60)

    (coder_m, coder_t), (reason_m, reason_t) = load_models()

    if args.test:
        tests = [
            "Generate Verilog for 8-bit up counter with synchronous reset",
            "Fix this error: synthesis failed — missing module instantiation for dual_port_ram",
            "Write RTL for a SPI master controller with configurable clock divider",
            "Analyze timing violation: WNS -3.2ns on path from ALU to output register at 500MHz",
        ]
        for t in tests:
            result = generate_chip(t, (coder_m, reason_m), (coder_t, reason_t))
            print(f"  Output preview: {result[:120]}...")
            print()
    else:
        while True:
            try:
                query = input("\n> Describe your chip or error: ")
                if not query:
                    break
                result = generate_chip(query, (coder_m, reason_m), (coder_t, reason_t))
                print(f"\n{result}")
            except (EOFError, KeyboardInterrupt):
                break


if __name__ == "__main__":
    main()
