#!/usr/bin/env python3
"""Upload VLSI Expert merged model to HuggingFace Hub."""

import os
import sys
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "vlsi-moe-ffn-merged" / "merged"
HF_REPO = "Vickyrrrrrr/vlsi-moe-ffn-merged"

print("Uploading VLSI Expert merged model to HuggingFace Hub...")
print(f"  Local path:  {MODEL_PATH}")
print(f"  HF repo:     https://huggingface.co/{HF_REPO}")
print()

# Check if logged in
try:
    from huggingface_hub import HfApi
    api = HfApi()
    user = api.whoami()
    print(f"  ✅ Logged in as: {user['name']}")
except Exception as e:
    print(f"  ❌ Not logged in. Run: huggingface-cli login")
    print(f"     Get token at: https://huggingface.co/settings/tokens")
    sys.exit(1)

# Upload the model
print(f"\n  Uploading model files (this may take 10-15 min for 65GB)...")
from huggingface_hub import upload_folder

upload_folder(
    folder_path=str(MODEL_PATH),
    repo_id=HF_REPO,
    repo_type="model",
    commit_message="VLSI Expert: FFN-merged Qwen2.5-Coder + DeepSeek-R1 via DARE+TIES",
)

print(f"\n✅ Upload complete!")
print(f"   View at: https://huggingface.co/{HF_REPO}")
print(f"   Use with:")
print(f"     model = AutoModelForCausalLM.from_pretrained('{HF_REPO}')")
