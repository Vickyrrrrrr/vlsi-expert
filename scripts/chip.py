#!/usr/bin/env python3
"""
Lightweight client — call your VLSI Expert model from your local machine.
No AgentIC, no heavy deps. Just Python + requests.

Setup (run once):
  ssh -N -L 8000:localhost:8000 -i ~/.ssh/id_ed25519 root@129.212.179.250 &
  
Then use:
  python3 chip.py "8-bit counter with reset"
  python3 chip.py "UART transmitter" --pdk sky130 --freq 100
"""

import argparse
import json
import requests

VLLM_URL = "http://localhost:8000/v1/completions"
API_KEY = "agentic-vlsi-expert-secure"


def design_chip(desc: str, pdk: str = "sky130", freq: int = 100) -> str:
    """Send design spec to VLSI Expert model via vLLM endpoint."""

    # ---- Part 1: Generate Verilog RTL ----
    rtl_prompt = (
        f"Generate correct, synthesizable Verilog RTL for: {desc}\n"
        f"Target: {pdk} PDK at {freq}MHz. Output ONLY the code.\n\n"
        f"module"
    )

    rtl_response = requests.post(
        VLLM_URL,
        json={
            "model": "vlsi-expert-merged",
            "prompt": rtl_prompt,
            "max_tokens": 800,
            "temperature": 0.2,
        },
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=120,
    )

    if rtl_response.status_code != 200:
        return f"❌ Model error: HTTP {rtl_response.status_code}\n{rtl_response.text[:300]}"

    verilog = "module" + rtl_response.json()["choices"][0]["text"]
    verilog = verilog.strip()

    # ---- Part 2: Generate SDC constraints ----
    sdc_prompt = (
        f"Generate SDC timing constraints for this Verilog module. Clock: {freq}MHz. PDK: {pdk}.\n\n"
        f"```verilog\n{verilog[:1500]}\n```\n\n"
        f"Output ONLY SDC commands:"
    )

    sdc_response = requests.post(
        VLLM_URL,
        json={
            "model": "vlsi-expert-merged",
            "prompt": sdc_prompt,
            "max_tokens": 400,
            "temperature": 0.1,
        },
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=60,
    )

    sdc = sdc_response.json()["choices"][0]["text"].strip() if sdc_response.status_code == 200 else "SDC generation skipped"

    return (
        f"{'='*60}\n"
        f"  VLSI Expert — Chip Design\n"
        f"  Design: {desc[:60]}\n"
        f"  PDK: {pdk} at {freq}MHz\n"
        f"{'='*60}\n\n"
        f"--- Verilog RTL ---\n\n"
        f"{verilog}\n\n"
        f"--- SDC Constraints ---\n\n"
        f"{sdc}\n\n"
        f"Lines: {len(verilog.splitlines())}  |  Model: FFN-merged MoE  |  GPU: MI300X"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="VLSI Expert — Design chips from your terminal")
    p.add_argument("desc", help="Design description, e.g. '8-bit counter'")
    p.add_argument("--pdk", default="sky130")
    p.add_argument("--freq", type=int, default=100, help="Target frequency in MHz")
    args = p.parse_args()

    print(design_chip(args.desc, args.pdk, args.freq))
