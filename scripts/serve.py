#!/usr/bin/env python3
"""
VLSI Expert — Unified Server Launcher
Auto-detects vLLM availability and launches the best option.

Usage:
  python scripts/serve.py              # Auto-detect and start
  python scripts/serve.py --fastapi    # Force FastAPI fallback
  python scripts/serve.py --vllm       # Force vLLM
  python scripts/serve.py --test       # Test running endpoint
"""

import argparse
import sys
import subprocess


def main():
    p = argparse.ArgumentParser(description="VLSI Expert Server Launcher")
    p.add_argument("--fastapi", action="store_true", help="Force FastAPI fallback")
    p.add_argument("--vllm", action="store_true", help="Force vLLM")
    p.add_argument("--test", action="store_true", help="Test endpoint")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    if args.test:
        subprocess.run([sys.executable, "scripts/serve_vllm.py", "--test", "--port", str(args.port)])
        return

    use_vllm = args.vllm
    if not args.fastapi and not args.vllm:
        try:
            import vllm
            use_vllm = True
            print(f"✅ vLLM detected ({vllm.__version__}). Using production server.")
        except ImportError:
            print("⚠️  vLLM not found. Using FastAPI fallback.")
            use_vllm = False

    if use_vllm:
        subprocess.run([sys.executable, "scripts/serve_vllm.py", "--local", "--port", str(args.port)])
    else:
        subprocess.run([sys.executable, "scripts/serve_fastapi.py", "--local", "--port", str(args.port)])


if __name__ == "__main__":
    main()
