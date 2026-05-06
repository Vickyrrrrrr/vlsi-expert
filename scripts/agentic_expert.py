#!/usr/bin/env python3
"""
Use VLSI Expert model with AgentIC pipeline (Local or Remote).

This script creates a CrewAI-compatible LLM wrapper that points to
your remote VPS where the model is served via OpenAI-compatible API.

Usage:
  # Ensure .env has LLM_BASE_URL pointing to your VPS
  export LLM_BASE_URL=http://YOUR_VPS_IP:8001/v1
  export LLM_API_KEY=agentic-vlsi-expert-secure

  python scripts/agentic_expert.py "8-bit up counter with synchronous reset"
"""

import os
import sys
import argparse
import re


def _strip_markdown(text: str, keyword: str = "module") -> str:
    """Strip markdown code blocks and conversational wrappers."""
    text = text.strip()
    
    # Strip markdown code blocks
    if "```" in text:
        lines = text.splitlines()
        code_lines = []
        in_code = False
        for line in lines:
            if line.strip().startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                code_lines.append(line)
        if code_lines:
            text = "\n".join(code_lines)
    
    # Ensure it starts with expected keyword
    if not text.startswith(keyword):
        idx = text.find(keyword)
        if idx != -1:
            text = text[idx:]
    
    return text.strip()


def generate(desc: str, pdk: str = "sky130", freq: int = 100) -> str:
    """Generate Verilog RTL via OpenAI-compatible API (local or remote)."""
    import requests

    base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8001/v1")
    api_key = os.environ.get("LLM_API_KEY", "agentic-vlsi-expert-secure")
    model = os.environ.get("LLM_MODEL", "vlsi-expert")

    # Ensure base_url ends with /v1
    if not base_url.endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    response = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a VLSI design compiler. Output ONLY raw synthesizable Verilog RTL code. No explanation, no markdown, no comments, no conversational text. Start with 'module' and end with 'endmodule'."},
                {"role": "user", "content": f"Generate correct, synthesizable Verilog RTL for: {desc}\nTarget: {pdk} PDK at {freq}MHz.\n\nmodule"},
            ],
            "max_tokens": 800,
            "temperature": 0.2,
        },
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=300,
    )

    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text[:500]}")

    return _strip_markdown(response.json()["choices"][0]["message"]["content"], "module")


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
    print(f"  API: {os.environ.get('LLM_BASE_URL', 'http://localhost:8001/v1')}")
    print("=" * 60)
    print()

    try:
        verilog = generate(args.desc, args.pdk, args.freq)
        print("### Generated Verilog RTL ###\n")
        print(verilog)
        print(f"\n{'='*60}")
        print(f"  Lines: {len(verilog.splitlines())}")
        print("=" * 60)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
