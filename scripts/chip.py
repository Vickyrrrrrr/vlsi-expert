#!/usr/bin/env python3
"""
VLSI Expert — Remote Chip Designer Client
Call your model running on VPS from your local machine.

Setup:
  1. Set env vars in .env:
     VLSI_EXPERT_HOST=YOUR_VPS_IP
     VLSI_EXPERT_PORT=8001
     VLSI_EXPERT_KEY=agentic-vlsi-expert-secure

  2. Or use SSH tunnel (recommended):
     ssh -N -L 8001:localhost:8001 -i ~/.ssh/id_ed25519 ubuntu@YOUR_VPS_IP
     # Then VLSI_EXPERT_HOST=localhost

Usage:
  python scripts/chip.py "8-bit counter with synchronous reset"
  python scripts/chip.py "UART transmitter" --pdk sky130 --freq 100
"""

import argparse
import os
import requests

# Config from environment
VPS_HOST = os.environ.get("VLSI_EXPERT_HOST", "localhost")
VPS_PORT = os.environ.get("VLSI_EXPERT_PORT", "8001")
API_KEY = os.environ.get("VLSI_EXPERT_KEY", "agentic-vlsi-expert-secure")
BASE_URL = f"http://{VPS_HOST}:{VPS_PORT}"


def design_chip(desc: str, pdk: str = "sky130", freq: int = 100) -> str:
    """Send design spec to VLSI Expert model via OpenAI-compatible chat API."""

    # Part 1: Generate Verilog RTL
    rtl_response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "vlsi-expert",
            "messages": [
                {"role": "system", "content": "You are a VLSI design compiler. Output ONLY raw synthesizable Verilog RTL code. No explanation, no markdown, no comments, no conversational text. Start with 'module' and end with 'endmodule'."},
                {"role": "user", "content": f"Generate correct, synthesizable Verilog RTL for: {desc}\nTarget: {pdk} PDK at {freq}MHz.\n\nmodule"},
            ],
            "max_tokens": 800,
            "temperature": 0.2,
        },
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        timeout=300,
    )

    if rtl_response.status_code != 200:
        return f"❌ Model error: HTTP {rtl_response.status_code}\n{rtl_response.text[:500]}"

    verilog = rtl_response.json()["choices"][0]["message"]["content"].strip()
    
    # Strip markdown code blocks if present
    if "```" in verilog:
        lines = verilog.splitlines()
        code_lines = []
        in_code = False
        for line in lines:
            if line.strip().startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                code_lines.append(line)
        if code_lines:
            verilog = "\n".join(code_lines)
    
    # Ensure it starts with module
    if not verilog.startswith("module"):
        # Try to find module keyword
        idx = verilog.find("module")
        if idx != -1:
            verilog = verilog[idx:]
    
    verilog = verilog.strip()

    # Part 2: Generate SDC constraints
    sdc_response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "vlsi-expert",
            "messages": [
                {"role": "system", "content": "You generate SDC timing constraints. Output ONLY raw SDC commands. No explanation, no markdown, no comments."},
                {"role": "user", "content": f"Generate SDC timing constraints for this Verilog module. Clock: {freq}MHz. PDK: {pdk}.\n\n{verilog[:1500]}\n\nOutput ONLY SDC commands:"},
            ],
            "max_tokens": 400,
            "temperature": 0.1,
        },
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        timeout=120,
    )

    sdc = sdc_response.json()["choices"][0]["message"]["content"].strip() if sdc_response.status_code == 200 else "SDC generation skipped"

    return (
        f"{'='*60}\n"
        f"  VLSI Expert — Chip Design\n"
        f"  Design: {desc[:60]}\n"
        f"  PDK: {pdk} at {freq}MHz\n"
        f"  Server: {BASE_URL}\n"
        f"{'='*60}\n\n"
        f"--- Verilog RTL ---\n\n"
        f"{verilog}\n\n"
        f"--- SDC Constraints ---\n\n"
        f"{sdc}\n\n"
        f"Lines: {len(verilog.splitlines())}  |  Model: vlsi-moe-ffn-merged-formal  |  GPU: MI300X"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="VLSI Expert — Design chips from your terminal")
    p.add_argument("desc", help="Design description, e.g. '8-bit counter'")
    p.add_argument("--pdk", default="sky130")
    p.add_argument("--freq", type=int, default=100, help="Target frequency in MHz")
    args = p.parse_args()

    print(design_chip(args.desc, args.pdk, args.freq))
