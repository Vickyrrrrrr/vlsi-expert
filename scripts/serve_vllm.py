#!/usr/bin/env python3
"""
Serve VLSI Expert via vLLM on AMD MI300X + ROCm.
OpenAI-compatible /v1/chat/completions endpoint.

This is the PRODUCTION server — fastest inference, lowest latency.
If vLLM ROCm build fails, fall back to: scripts/serve_fastapi.py

Usage:
  # Download model first
  python scripts/download_model.py

  # Start vLLM server
  python scripts/serve_vllm.py --local

  # Or load directly from HuggingFace (slower startup)
  python scripts/serve_vllm.py --model vxkyyy/vlsi-moe-ffn-merged-formal

Test:
  curl http://localhost:8000/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -H "Authorization: Bearer agentic-vlsi-expert-secure" \\
    -d '{"model":"vlsi-expert","messages":[{"role":"user","content":"Generate an 8-bit counter"}],"max_tokens":800}'
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

LOCAL_MODEL = "models/vlsi-moe-ffn-merged-formal"
HF_MODEL = "vxkyyy/vlsi-moe-ffn-merged-formal"
API_KEY = os.environ.get("VLSI_API_KEY", "agentic-vlsi-expert-secure")
PORT = int(os.environ.get("VLSI_PORT", "8000"))


def check_vllm():
    """Check if vllm is installed and ROCm-compatible."""
    try:
        import vllm
        print(f"✅ vLLM found: {vllm.__version__}")
        return True
    except ImportError:
        print("❌ vLLM not installed.")
        print("\nTo install vLLM for ROCm:")
        print("   pip install vllm")
        print("\nIf ROCm binary is not available, build from source (slow):")
        print("   git clone https://github.com/vllm-project/vllm")
        print("   cd vllm && python setup.py develop")
        print("\nOr use the FastAPI fallback instead:")
        print("   python scripts/serve_fastapi.py")
        return False


def start(model_path: str, port: int):
    if not check_vllm():
        sys.exit(1)

    print("=" * 60)
    print("  VLSI Expert — vLLM Server (MI300X + ROCm)")
    print(f"  Model: {model_path}")
    print(f"  Port:  {port}")
    print(f"  Endpoint: http://0.0.0.0:{port}/v1/chat/completions")
    print(f"  API Key:  {API_KEY}")
    print("=" * 60)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--dtype", "bfloat16",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.90",
        "--tensor-parallel-size", "1",
        "--port", str(port),
        "--host", "0.0.0.0",
        "--api-key", API_KEY,
        # Enable chat template if available in tokenizer_config.json
        "--chat-template", str(Path(model_path) / "tokenizer_config.json"),
    ]

    # Remove chat-template arg if file doesn't exist
    if not Path(cmd[-1]).exists():
        cmd = cmd[:-2]

    subprocess.run(cmd)


def test(port: int):
    import requests
    url = f"http://localhost:{port}/v1/chat/completions"
    print(f"Testing vLLM at {url}...")
    payload = {
        "model": "vlsi-expert",
        "messages": [{"role": "user", "content": "Generate Verilog for an 8-bit counter with synchronous reset"}],
        "max_tokens": 200,
        "temperature": 0.2,
    }
    try:
        r = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=120,
        )
        if r.status_code == 200:
            text = r.json()["choices"][0]["message"]["content"]
            print(f"✅ Response: {text[:300]}...")
        else:
            print(f"❌ HTTP {r.status_code}: {r.text[:500]}")
    except Exception as e:
        print(f"❌ {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Serve VLSI Expert via vLLM")
    p.add_argument("--model", default=None, help="HF model ID or local path")
    p.add_argument("--local", action="store_true", help="Use local model at ./models/...")
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--test", action="store_true", help="Test endpoint")
    args = p.parse_args()

    if args.test:
        test(args.port)
        sys.exit(0)

    if args.local:
        model_path = LOCAL_MODEL
        if not Path(model_path).exists():
            print(f"❌ Local model not found at {model_path}")
            print(f"   Run: python scripts/download_model.py")
            sys.exit(1)
    else:
        model_path = args.model or HF_MODEL

    start(model_path, args.port)
