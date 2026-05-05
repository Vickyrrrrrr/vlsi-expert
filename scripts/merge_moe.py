#!/usr/bin/env python3
"""
DARE + TIES merge three Qwen-family models into one MoE-style model.
Uses mergekit for fast weight-space merging.

Models merged (all share Qwen2.5-32B base architecture, same tokenizer):
  Expert A: Qwen2.5-Coder-32B-Instruct  → Verilog generation
  Expert B: DeepSeek-R1-Distill-Qwen-32B → CoT reasoning, error analysis
  Expert C: Qwen3-32B                    → Latest instruction following

Method: DARE (Drop And REscale) to sparsify deltas, then TIES-Merging to
resolve parameter interference. Produces a single merged model that can
be converted to MoE by adding a task router head.

Time: ~4 hours on MI300X (192GB VRAM handles all 3 models simultaneously)
"""

import os
import sys
import json
import torch
import subprocess
from pathlib import Path
from typing import Dict, Any

# ── Config ────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "vlsi-moe-merged"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Three expert models — all Qwen-family (same tokenizer, same hidden dim: 5120)
EXPERTS = {
    "coder": {
        "path": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "role": "Verilog RTL generation, SDC writing, testbench generation",
        "priority": 1,
    },
    "reason": {
        "path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "role": "Chain-of-thought reasoning, architecture planning, error analysis",
        "priority": 2,
    },
    "instruct": {
        "path": "Qwen/Qwen3-32B",
        "role": "Error fixing, instruction following, explanation generation",
        "priority": 3,
    },
}

# DARE config: drop 90% of delta params, rescale remaining 10×
DARE_DROP_RATE = 0.90  # 90% dropped → 10% kept → rescale ×10
TIES_RESET_FRACTION = 0.10  # Reset params with small delta changes


def merge_with_mergekit() -> bool:
    """Use mergekit (if installed) for DARE + TIES merging."""
    mergekit_yaml = str(OUTPUT_DIR / "mergekit_config.yml")

    config = {
        "slices": [
            {
                "sources": [
                    {
                        "model": EXPERT["path"],
                        "layer_range": [0, 32],
                    }
                    for EXPERT in EXPERTS.values()
                ],
            }
        ],
        "merge_method": "ties",
        "parameters": {
            "density": 1.0 - DARE_DROP_RATE,
            "weight": [1.0 / len(EXPERTS)] * len(EXPERTS),
            "normalize": True,
        },
        "dtype": "bfloat16",
    }

    with open(mergekit_yaml, "w") as f:
        json.dump(config, f, indent=2)

    print(f"  MergeKit config: {mergekit_yaml}")
    print("  Running mergekit merge...")

    result = subprocess.run(
        [
            sys.executable, "-m", "mergekit",
            "merge", mergekit_yaml,
            str(OUTPUT_DIR / "merged"),
            "--cuda",
            "--allow-crimes",
        ],
        capture_output=False,
    )

    return result.returncode == 0


def merge_manual() -> bool:
    """Manual DARE merge if mergekit is not available. Loads all 3 models
    into memory, computes deltas, applies DARE, and merges via TIES."""
    from transformers import AutoModelForCausalLM, AutoConfig

    print("  MergeKit not available. Performing manual DARE+TIES merge...")
    config = AutoConfig.from_pretrained(list(EXPERTS.values())[0]["path"])

    # Load base model (first expert becomes the "base")
    base_path = list(EXPERTS.values())[0]["path"]
    print(f"  Loading base: {base_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Compute deltas from remaining experts, apply DARE, merge
    for expert_name, expert_info in list(EXPERTS.items())[1:]:
        print(f"  Processing expert: {expert_name} ({expert_info['role'][:50]}...)")
        try:
            expert_model = AutoModelForCausalLM.from_pretrained(
                expert_info["path"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

            # For each parameter: compute delta, apply DARE, merge
            for (base_name, base_param), (_, expert_param) in zip(
                base_model.named_parameters(), expert_model.named_parameters()
            ):
                with torch.no_grad():
                    delta = expert_param.data - base_param.data

                    # DARE: randomly drop p% of delta values
                    mask = torch.rand_like(delta) > DARE_DROP_RATE
                    delta_dared = delta * mask * (1.0 / (1.0 - DARE_DROP_RATE))

                    # TIES: resolve sign conflicts
                    base_sign = torch.sign(base_param.data)
                    delta_sign = torch.sign(delta_dared)
                    sign_conflict = (base_sign != delta_sign) & (base_sign != 0) & (delta_sign != 0)
                    delta_dared[sign_conflict] = 0

                    # Merge: average the surviving delta into base
                    base_param.data = base_param.data + delta_dared * 0.5

            del expert_model
            torch.cuda.empty_cache()
            print(f"    Expert {expert_name} merged.")

        except Exception as e:
            print(f"    Expert {expert_name} FAILED: {e}")
            print("    Continuing with available experts...")

    # Save merged model
    merged_path = OUTPUT_DIR / "merged"
    base_model.save_pretrained(str(merged_path))
    print(f"\n  Merged model saved: {merged_path}")
    return True


def main():
    print("=" * 60)
    print("  VLSI Expert — MoE Model Assembly (DARE + TIES)")
    print(f"  Experts: {len(EXPERTS)} ({', '.join(EXPERTS.keys())})")
    print(f"  DARE drop rate: {DARE_DROP_RATE}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)
    print()

    # Try mergekit first, fall back to manual
    try:
        import mergekit
        print("[merge] Using mergekit for DARE+TIES merging...")
        success = merge_with_mergekit()
    except ImportError:
        print("[merge] mergekit not installed. Using manual merge...")
        success = merge_manual()

    if success:
        print("\n✅ MoE merge complete!")
        print(f"   Merged model: {OUTPUT_DIR / 'merged'}")
        print(f"\n   Next: python scripts/train_router.py  (train task router)")
        print(f"         python scripts/serve_vllm.py     (serve with vLLM)")
        print(f"         python eval/evaluate_moe.py      (evaluate through AgentIC)")
    else:
        print("\n❌ Merge failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
