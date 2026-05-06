#!/usr/bin/env python3
"""
Bridge: Use VLSI Expert via remote API with AgentIC pipeline.

Prerequisites:
  1. Your VPS is running the model server:
     python scripts/serve_fastapi.py  # or serve_vllm.py

  2. Your local .env points to the VPS:
     export LLM_BASE_URL=http://YOUR_VPS_IP:8001/v1
     export LLM_API_KEY=agentic-vlsi-expert-secure

Usage:
  python scripts/build.py "8-bit counter with reset"
  python scripts/build.py "UART transmitter" --pdk sky130 --harden
"""

import os
import sys
import argparse
from pathlib import Path


def _strip_markdown(text: str, keyword: str = "module") -> str:
    """Strip markdown code blocks and conversational wrappers."""
    text = text.strip()
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
    if not text.startswith(keyword):
        idx = text.find(keyword)
        if idx != -1:
            text = text[idx:]
    return text.strip()


def build_chip(desc: str, name: str = "expert_design", pdk: str = "sky130", skip_openlane: bool = True):
    """Run AgentIC pipeline using remote VLSI Expert model."""
    try:
        from crewai import LLM
    except ImportError:
        print("❌ CrewAI not installed. Install AgentIC first:")
        print("   pip install agentic-ic")
        sys.exit(1)

    base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8001/v1")
    api_key = os.environ.get("LLM_API_KEY", "agentic-vlsi-expert-secure")
    model = os.environ.get("LLM_MODEL", "vlsi-expert")

    # CrewAI LLM wrapper pointing to remote server
    llm = LLM(
        model=f"openai/{model}",
        base_url=base_url,
        api_key=api_key,
        max_tokens=4096,
        temperature=0.2,
        timeout=300,
        num_retries=3,
    )

    print(f"🚀 Using remote model at {base_url}")
    print(f"   This will send {desc!r} through AgentIC's pipeline...")

    # AgentIC orchestrator integration
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "AgentIC" / "src"))
        from agentic.orchestrator import BuildOrchestrator

        orch = BuildOrchestrator(
            name=name,
            desc=desc,
            llm=llm,
            pdk_profile=pdk,
            skip_openlane=skip_openlane,
            max_retries=3,
            verbose=True,
        )
        orch.run()

        if orch.state.name == "SUCCESS":
            print(f"\n✅ Chip design complete!")
        else:
            print(f"\n❌ Build failed: {orch.state.name}")
    except ImportError:
        print("\n⚠️  AgentIC not found. Falling back to direct API call...")
        import requests
        r = requests.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a VLSI design compiler. Output ONLY raw synthesizable Verilog RTL code. No explanation, no markdown, no comments, no conversational text. Start with 'module' and end with 'endmodule'."},
                    {"role": "user", "content": f"Generate Verilog RTL for: {desc}\nTarget: {pdk}\n\nmodule"},
                ],
                "max_tokens": 800,
                "temperature": 0.2,
            },
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=300,
        )
        if r.status_code == 200:
            verilog = _strip_markdown(r.json()["choices"][0]["message"]["content"], "module")
            print(verilog)
        else:
            print(f"❌ API error: {r.status_code} {r.text[:500]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build chip with VLSI Expert via remote API")
    p.add_argument("desc", help="Design description")
    p.add_argument("--name", default="expert_design")
    p.add_argument("--pdk", default="sky130")
    p.add_argument("--harden", action="store_true", help="Run full GDSII hardening")
    args = p.parse_args()

    print("=" * 60)
    print("  VLSI Expert + AgentIC Pipeline (Remote)")
    print(f"  Design: {args.desc[:60]}")
    print(f"  PDK:    {args.pdk}")
    print(f"  Model:  {os.environ.get('LLM_BASE_URL', 'http://localhost:8001/v1')}")
    print("=" * 60)
    print()

    build_chip(args.desc, args.name, args.pdk, skip_openlane=not args.harden)
