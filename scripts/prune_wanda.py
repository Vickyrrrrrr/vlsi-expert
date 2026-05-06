#!/usr/bin/env python3
"""
Wanda-based pruning: 1:8 sparsity on the CoT-trained VLSI Expert model.
Ranks weights by |weight| × activation_norm — keeps top 12.5%.

Wanda preserves weights that are BOTH large AND fire during correct Verilog generation.
More intelligent than magnitude-only pruning.

Time: ~30 min on MI300X
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────
MODEL_PATH = str(Path(__file__).parent.parent / "models" / "vlsi-moe-merged" / "merged")
LORA_PATH = str(Path(__file__).parent.parent / "models" / "vlsi-coder-lora-advanced")
OUTPUT_PATH = str(Path(__file__).parent.parent / "models" / "vlsi-expert-sparse-1-8")
DATA_PATH = str(Path(__file__).parent.parent / "data" / "train_pairs.jsonl")
SPARSITY_RATIO = 0.875  # 1:8 → keep 12.5%

CALIBRATION_SAMPLES = 50  # Use first 50 training pairs for activation profiling


def apply_wanda_pruning(model: nn.Module, calibration_data: list, tokenizer, sparsity: float):
    """Apply Wanda pruning: |weight| × ||activation|| for importance scoring."""
    print(f"  Applying Wanda pruning (sparsity={sparsity:.1%})...")
    model.eval()

    pruned_params = 0
    total_params = 0

    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            weight = module.weight.data
            total_params += weight.numel()

            # Collect activations for this layer
            activations = []
            handles = []

            def hook(_, __, output, layer_name=name):
                activations.append(output.detach())

            handles.append(module.register_forward_hook(hook))

            # Run calibration data through the model
            for item in calibration_data[:CALIBRATION_SAMPLES]:
                text = f"### Specification\n{item['spec']}\n\n### Verilog RTL\n{item['verilog']}"

                try:
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    _ = model(**inputs)
                except Exception:
                    continue

            for h in handles:
                h.remove()

            if not activations:
                # No activations captured — fall back to magnitude pruning
                sorted_idx = torch.argsort(weight.abs().flatten())
                keep_num = int(weight.numel() * (1 - sparsity))
                mask = torch.ones_like(weight.flatten(), dtype=torch.bool)
                mask[sorted_idx[:weight.numel() - keep_num]] = False
                mask = mask.reshape(weight.shape)
                weight.data = weight.data * mask
                pruned_params += weight.numel() - keep_num
                continue

            # Compute Wanda importance: |weight| × activation_norm
            act_norm = torch.mean(torch.cat([a.norm(dim=-1, keepdim=True) for a in activations], dim=0), dim=0).squeeze()
            if act_norm.dim() == 1:
                act_norm = act_norm.unsqueeze(0)

            # Ensure act_norm shape matches weight shape
            while act_norm.dim() < weight.dim():
                act_norm = act_norm.unsqueeze(-1)

            importance = weight.abs() * act_norm

            # Keep top (1 - sparsity) fraction
            keep_num = int(weight.numel() * (1 - sparsity))
            _, topk_indices = torch.topk(importance.flatten(), keep_num)

            mask = torch.zeros_like(weight.flatten(), dtype=torch.bool)
            mask[topk_indices] = True
            mask = mask.reshape(weight.shape)

            weight.data = weight.data * mask
            pruned_params += weight.numel() - keep_num

            if total_params % 10_000_000 == 0:
                print(f"    {name}: kept {keep_num}/{weight.numel()} weights")

    print(f"\n  Pruned {pruned_params:,}/{total_params:,} weights ({pruned_params/total_params:.1%} sparse)")
    return model


def verify_sparsity(model: nn.Module) -> dict:
    """Count actual sparsity achieved."""
    total = 0
    zeros = 0
    for p in model.parameters():
        total += p.numel()
        zeros += (p == 0).sum().item()
    return {"total_params": total, "zero_params": zeros, "sparsity": zeros / total}


def main():
    print("=" * 60)
    print("  VLSI Expert — Wanda Pruning (1:8 sparsity)")
    print(f"  Model: {MODEL_PATH}")
    print(f"  LoRA:  {LORA_PATH}")
    print(f"  Ratio: {SPARSITY_RATIO} (keep {1-SPARSITY_RATIO:.1%})")
    print("=" * 60)
    print()

    # Load calibration data
    print("[1/4] Loading calibration data...")
    with open(DATA_PATH) as f:
        data = [json.loads(line) for line in f if line.strip()]

    # Load model
    print("[2/4] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Load LoRA if available
    if Path(LORA_PATH).exists():
        print("  Loading CoT LoRA adapter...")
        model = PeftModel.from_pretrained(model, LORA_PATH)
        model = model.merge_and_unload()  # Merge LoRA into base for pruning

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Apply Wanda pruning
    print("[3/4] Applying Wanda pruning...")
    model = apply_wanda_pruning(model, data, tokenizer, SPARSITY_RATIO)

    # Verify
    stats = verify_sparsity(model)
    print(f"\n[4/4] Saving pruned model...")
    print(f"  Total: {stats['total_params']:,}")
    print(f"  Zeros: {stats['zero_params']:,}")
    print(f"  Actual sparsity: {stats['sparsity']:.2%}")

    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)

    # Quick test
    print("\n  Quick test — generating Verilog...")
    test_prompt = "### Specification\n8-bit up counter with synchronous reset and enable\n\n### Design Analysis\nDesign Type:"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    has_module = "module" in result.lower()
    print(f"  Output has 'module': {has_module} ({len(result)} chars)")
    print(f"  Sample: {result[200:400]}...")

    print(f"\n  Saved: {OUTPUT_PATH}")
    print("  Done!")


if __name__ == "__main__":
    main()
