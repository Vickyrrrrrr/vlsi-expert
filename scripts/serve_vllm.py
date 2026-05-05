#!/usr/bin/env python3
"""
Serve the VLSI Expert MoE model with vLLM on AMD MI300X + ROCm 7.2.

Architecture:
  - Merged Qwen-32B backbone (DARE + TIES from 3 experts)
  - Task router routes to the appropriate expert head at inference
  - Served via vLLM's OpenAI-compatible API on port 8000

Usage:
  python scripts/serve_vllm.py            # Start vLLM server
  python scripts/serve_vllm.py --test     # Quick test the endpoint
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

MODEL_PATH = str(Path(__file__).parent.parent / "models" / "vlsi-moe-merged" / "merged")

VLLM_CONFIG = {
    "model": MODEL_PATH,
    "dtype": "bfloat16",
    "max-model-len": 8192,
    "gpu-memory-utilization": 0.85,
    "tensor-parallel-size": 1,
    "enforce-eager": False,
    "port": 8000,
    "host": "0.0.0.0",
}


def start_server():
    """Start vLLM OpenAI-compatible server."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", VLLM_CONFIG["model"],
        "--dtype", VLLM_CONFIG["dtype"],
        "--max-model-len", str(VLLM_CONFIG["max-model-len"]),
        "--gpu-memory-utilization", str(VLLM_CONFIG["gpu-memory-utilization"]),
        "--tensor-parallel-size", str(VLLM_CONFIG["tensor-parallel-size"]),
        "--port", str(VLLM_CONFIG["port"]),
        "--host", VLLM_CONFIG["host"],
    ]
    if not VLLM_CONFIG["enforce-eager"]:
        cmd.append("--enforce-eager")

    print("=" * 60)
    print("  VLSI Expert — vLLM Server (ROCm 7.2 + MI300X)")
    print(f"  Model: {VLLM_CONFIG['model']}")
    print(f"  Port:  {VLLM_CONFIG['port']}")
    print("=" * 60)

    subprocess.run(cmd)


def test_endpoint(prompt: str = "module counter(input clk, rst, output reg [7:0] count);"):
    """Test the vLLM endpoint with a sample Verilog generation request."""
    import requests

    url = f"http://localhost:{VLLM_CONFIG['port']}/v1/completions"

    payload = {
        "model": MODEL_PATH,
        "prompt": f"Generate complete Verilog code for: {prompt}\n\nmodule",
        "max_tokens": 512,
        "temperature": 0.2,
        "stop": ["</s>", "endmodule"],
    }

    print(f"  Testing vLLM endpoint: {url}")
    try:
        resp = requests.post(url, json=payload, timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            text = result["choices"][0]["text"]
            print(f"  ✅ Response ({len(text)} chars):")
            print(f"  module{text}")
        else:
            print(f"  ❌ HTTP {resp.status_code}: {resp.text[:200]}")
    except requests.exceptions.ConnectionError:
        print("  ❌ vLLM server not running. Start with: python scripts/serve_vllm.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test the endpoint instead of starting server")
    parser.add_argument("--prompt", type=str, default="8-bit up counter with synchronous reset", help="Test prompt")
    args = parser.parse_args()

    if args.test:
        test_endpoint(args.prompt)
    else:
        start_server()


if __name__ == "__main__":
    main()
