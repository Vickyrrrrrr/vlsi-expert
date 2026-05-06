#!/usr/bin/env python3
"""
FFN-ONLY merge: Absorb DeepSeek-R1's FFN knowledge into Qwen2.5-Coder.
Keeps Coder's attention/generation layers untouched (no reshape crashes).

Method: DARE + TIES on FFN layers only (gate_proj, up_proj, down_proj).
Attention layers (q/k/v/o) kept from Coder — no compatibility issues.

Result: One model that GENERATES like Coder but REASONS with R1 knowledge.
Time: ~20 min on MI300X (reuse cached models from first merge).
"""

import os, sys, json, torch, gc
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "models" / "vlsi-moe-ffn-merged"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DARE_DROP = 0.90

# FFN layers to merge (knowledge transfer)
MERGE_LAYERS = {"gate_proj", "up_proj", "down_proj"}

# Layers to SKIP (keep from Coder untouched)
SKIP_LAYERS = {"q_proj", "k_proj", "v_proj", "o_proj", "lm_head", "embed_tokens", "rotary_emb", "norm"}


def should_merge(name: str) -> bool:
    """Check if this parameter should receive FFN knowledge from R1."""
    for skip in SKIP_LAYERS:
        if skip in name:
            return False
    for merge in MERGE_LAYERS:
        if merge in name:
            return True
    return False  # skip by default (bias, etc.)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print("  VLSI Expert — FFN-Only Knowledge Merge")
    print("  Base:  Qwen2.5-Coder-32B (keeps attention + generation)")
    print("  Donor: DeepSeek-R1-32B (FFN knowledge only)")
    print("  DARE drop: 90%")
    print("=" * 60)
    print()

    # Load base (Coder — keeps everything)
    print("[1/4] Loading CODER (base, generation engine)...")
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    # Load donor (R1 — FFN knowledge only)
    print("[2/4] Loading REASON (donor, FFN knowledge)...")
    donor = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    # Merge only FFN layers
    print("[3/4] Merging FFN layers (gate/up/down projections)...")
    merged_count = 0
    skipped_count = 0

    for (base_name, base_p), (_, donor_p) in zip(base.named_parameters(), donor.named_parameters()):
        if not should_merge(base_name):
            skipped_count += 1
            continue

        with torch.no_grad():
            delta = donor_p.data - base_p.data

            # DARE: randomly drop 90% of delta, rescale remaining 10x
            mask = torch.rand_like(delta) > DARE_DROP
            delta_dared = delta * mask * (1.0 / (1.0 - DARE_DROP))

            # TIES: resolve sign conflicts
            base_sign = torch.sign(base_p.data)
            delta_sign = torch.sign(delta_dared)
            conflict = (base_sign != delta_sign) & (base_sign != 0) & (delta_sign != 0)
            delta_dared[conflict] = 0

            # Apply surviving delta
            base_p.data = base_p.data + delta_dared * 0.3  # conservative merge

            merged_count += 1

            if merged_count % 20 == 0:
                print(f"    Merged {merged_count} layers... ({skipped_count} skipped)")

    del donor
    gc.collect()
    torch.cuda.empty_cache()

    total = merged_count + skipped_count
    print(f"\n  FFN layers merged:  {merged_count}/{total}")
    print(f"  Attention skipped:  {skipped_count}/{total}")

    # Save
    out = OUTPUT_DIR / "merged"
    print(f"\n[4/4] Saving merged model ({out})...")
    base.save_pretrained(str(out))
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct", trust_remote_code=True)
    tok.save_pretrained(str(out))

    # Quick generation test
    print("\n  🧪 Quick test — generating Verilog...")
    test = "8-bit up counter with synchronous reset and enable"
    inp = tok(test + "\n\n### Verilog\nmodule", return_tensors="pt").to(base.device)

    with torch.no_grad():
        out = base.generate(**inp, max_new_tokens=200, temperature=0.2, do_sample=True, pad_token_id=tok.eos_token_id, use_cache=False)

    result = tok.decode(out[0], skip_special_tokens=True)
    has_module = "module" in result.lower()
    has_endmodule = "endmodule" in result.lower()
    print(f"  module found: {has_module} | endmodule found: {has_endmodule}")
    print(f"  Preview: {result[200:500]}")
    print(f"\n  ✅ Merge complete! Model: {out}")


if __name__ == "__main__":
    main()
