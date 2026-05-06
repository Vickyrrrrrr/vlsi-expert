#!/usr/bin/env python3
"""
Serve VLSI Expert MoE via vLLM on AMD MI300X + ROCm 7.2.
Starts OpenAI-compatible API on port 8000.

AgentIC can then use: --model vxkyyy/vlsi-moe-ffn-merged --base-url http://localhost:8000/v1

Usage:
  python scripts/serve.py              # Start server
  python scripts/serve.py --test       # Quick test the endpoint
"""

import argparse
import sys
import subprocess
from pathlib import Path

MODEL_PATH = str(Path(__file__).parent.parent / "models" / "vlsi-moe-ffn-merged" / "merged")
PORT = 8000


def start():
    print("=" * 60)
    print("  VLSI Expert — vLLM Server (MI300X + ROCm 7.2)")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Port:  {PORT}")
    print(f"  Endpoint: http://localhost:{PORT}/v1/completions")
    print("=" * 60)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--dtype", "bfloat16",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.85",
        "--tensor-parallel-size", "1",
        "--port", str(PORT),
        "--host", "127.0.0.1",      # Only VPS itself can access
        "--api-key", "agentic-vlsi-expert-secure",  # Require this key in requests
    ]
    subprocess.run(cmd)


def test():
    import requests
    print(f"Testing vLLM at http://localhost:{PORT}/v1/completions...")
    payload = {
        "model": MODEL_PATH,
        "prompt": "Generate Verilog for an 8-bit counter with synchronous reset\n\nmodule",
        "max_tokens": 200,
        "temperature": 0.2,
    }
    try:
        r = requests.post(
            f"http://localhost:{PORT}/v1/completions",
            json=payload,
            headers={"Authorization": "Bearer agentic-vlsi-expert-secure"},
            timeout=30,
        )
        if r.status_code == 200:
            text = r.json()["choices"][0]["text"]
            print(f"✅ Response: module{text[:200]}...")
        else:
            print(f"❌ HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"❌ {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true")
    args = p.parse_args()
    test() if args.test else start()
