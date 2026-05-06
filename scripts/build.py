#!/usr/bin/env python3
"""
Bridge: Use VLSI Expert via vLLM endpoint with AgentIC pipeline.

1. Start vLLM:  python scripts/serve.py
2. Build chip:  python scripts/build.py "8-bit counter with reset"

AgentIC's BuildOrchestrator calls the vLLM endpoint for every LLM request.
"""

import sys
import argparse
from pathlib import Path

# Add AgentIC to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "AgentIC" / "src"))


def build_chip(desc: str, name: str = "expert_design", pdk: str = "sky130", skip_openlane: bool = True):
    """Run AgentIC pipeline using VLSI Expert model via vLLM endpoint."""
    from crewai import LLM
    from agentic.orchestrator import BuildOrchestrator

    # Point to vLLM endpoint — vLLM serves OpenAI-compatible API
    llm = LLM(
        model="vlsi-expert-merged",
        base_url="http://localhost:8000/v1",
        api_key="not-needed",  # vLLM doesn't require auth locally
        max_tokens=4096,
        temperature=0.2,
    )

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
        if not skip_openlane:
            print(f"   GDSII: {orch.artifacts.get('gds', 'N/A')}")
    else:
        print(f"\n❌ Build failed: {orch.state.name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build chip with VLSI Expert via vLLM")
    p.add_argument("desc", help="Design description")
    p.add_argument("--name", default="expert_design")
    p.add_argument("--pdk", default="sky130")
    p.add_argument("--harden", action="store_true", help="Run full GDSII hardening")
    args = p.parse_args()

    print("=" * 60)
    print("  VLSI Expert + AgentIC Pipeline")
    print(f"  Design: {args.desc[:60]}")
    print(f"  PDK:    {args.pdk}")
    print(f"  Model:  vLLM http://localhost:8000/v1")
    print("=" * 60)
    print()

    build_chip(args.desc, args.name, args.pdk, skip_openlane=not args.harden)
