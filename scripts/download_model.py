#!/usr/bin/env python3
"""
Download VLSI Expert model from HuggingFace Hub to local disk.

Usage:
  python scripts/download_model.py              # Default: vxkyyy/vlsi-moe-ffn-merged-formal
  python scripts/download_model.py --model vxkyyy/vlsi-moe-ffn-merged
  python scripts/download_model.py --local-dir ./models/my-model
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download

DEFAULT_MODEL = "vxkyyy/vlsi-moe-ffn-merged-formal"
DEFAULT_LOCAL_DIR = "models/vlsi-moe-ffn-merged-formal"


def download(model_id: str, local_dir: str, token: str = None):
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_id}...")
    print(f"Target: {local_path.absolute()}")
    print(f"This is ~66 GB and may take 20-60 minutes depending on connection.")
    print("=" * 60)

    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_path),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=token or os.environ.get("HF_TOKEN"),
    )

    print("\n" + "=" * 60)
    print(f"✅ Model downloaded to: {local_path.absolute()}")
    print(f"   Size: {sum(f.stat().st_size for f in local_path.rglob('*') if f.is_file()) / 1e9:.1f} GB")
    print("\nYou can now start the server:")
    print(f"   python scripts/serve_vllm.py --model {local_path}")
    print(f"   # or")
    print(f"   python scripts/serve_fastapi.py --model {local_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download VLSI Expert model from HF Hub")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF Hub model ID")
    p.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR, help="Where to save")
    p.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var)")
    args = p.parse_args()
    download(args.model, args.local_dir, args.token)
