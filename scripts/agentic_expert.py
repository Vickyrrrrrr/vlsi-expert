#!/usr/bin/env python3
"""
Use VLSI Expert model with AgentIC pipeline.
Creates a CrewAI-compatible LLM wrapper around the FFN-merged model.

Usage:
  python agentic_expert.py "8-bit up counter with synchronous reset"
"""

import sys
import argparse

def load_model():
    """Load the VLSI Expert model. Returns (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        "vxkyyy/vlsi-moe-ffn-merged",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("vxkyyy/vlsi-moe-ffn-merged")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate(desc: str, pdk: str = "sky130", freq: int = 100) -> str:
    """Generate Verilog RTL from a natural language description."""
    model, tok = load_model()

    prompt = (
        f"Generate correct, synthesizable Verilog RTL for the following specification.\n"
        f"Target: {pdk} PDK at {freq}MHz.\n\n"
        f"### Specification\n{desc}\n\n### Verilog RTL\nmodule"
    )

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=800,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tok.eos_token_id,
            use_cache=False,
        )

    verilog = tok.decode(out[0], skip_special_tokens=True)
    return verilog


def main():
    parser = argparse.ArgumentParser(description="VLSI Expert — Generate chip design")
    parser.add_argument("desc", help="Design description, e.g. '8-bit counter'")
    parser.add_argument("--pdk", default="sky130")
    parser.add_argument("--freq", type=int, default=100)
    args = parser.parse_args()

    print("=" * 60)
    print("  VLSI Expert — AI Chip Designer")
    print(f"  Spec: {args.desc[:60]}")
    print(f"  PDK: {args.pdk} at {args.freq}MHz")
    print("=" * 60)
    print()

    verilog = generate(args.desc, args.pdk, args.freq)

    print("### Generated Verilog RTL ###\n")
    print(verilog)
    print(f"\n{'='*60}")
    print(f"  Lines: {len(verilog.splitlines())}")
    print("=" * 60)


if __name__ == "__main__":
    main()
