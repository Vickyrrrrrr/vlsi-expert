#!/usr/bin/env python3
"""
VLSI Expert — FastAPI Server (Fallback)
OpenAI-compatible /v1/chat/completions endpoint using transformers.

Use this when vLLM ROCm build fails or as a zero-dependency fallback.

Usage:
  python scripts/serve_fastapi.py                    # Load from HF Hub
  python scripts/serve_fastapi.py --local            # Load from ./models/...
  python scripts/serve_fastapi.py --port 8000

Test:
  curl http://localhost:8001/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -H "Authorization: Bearer agentic-vlsi-expert-secure" \\
    -d '{"model":"vlsi-expert","messages":[{"role":"user","content":"Generate an 8-bit counter"}],"max_tokens":800}'
"""

import argparse
import os
import sys
import time
import torch
import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List, Dict, Optional

# ── Config ────────────────────────────────────────────────────────────
LOCAL_MODEL = "models/vlsi-moe-ffn-merged-formal"
HF_MODEL = "vxkyyy/vlsi-moe-ffn-merged-formal"
API_KEY = os.environ.get("VLSI_API_KEY", "agentic-vlsi-expert-secure")
PORT = int(os.environ.get("VLSI_PORT", "8000"))

# ── Global model holder ───────────────────────────────────────────────
model = None
tokenizer = None
security = HTTPBearer()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "vlsi-expert"
    messages: List[ChatMessage]
    max_tokens: int = 800
    temperature: float = 0.2
    stop: Optional[List[str]] = None
    top_p: Optional[float] = 1.0


class CompletionRequest(BaseModel):
    model: str = "vlsi-expert"
    prompt: str
    max_tokens: int = 800
    temperature: float = 0.2
    stop: Optional[List[str]] = None


def verify_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


def _format_chat_prompt(messages: List[ChatMessage]) -> str:
    """Convert OpenAI chat messages to a single prompt string."""
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"System: {msg.content}\n")
        elif msg.role == "user":
            parts.append(f"User: {msg.content}\n")
        elif msg.role == "assistant":
            parts.append(f"Assistant: {msg.content}\n")
    parts.append("Assistant:")
    return "\n".join(parts)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_local = Path(LOCAL_MODEL).exists()
    model_path = LOCAL_MODEL if use_local else HF_MODEL

    print(f"Loading VLSI Expert from {'LOCAL' if use_local else 'HF Hub'}: {model_path}...")
    print("This may take 2-5 minutes for a 66GB model...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=use_local,
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=use_local,
    )
    print(f"✅ Model loaded on {model.device}")
    yield


app = FastAPI(title="VLSI Expert API (FastAPI Fallback)", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "backend": "fastapi+transformers"}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, _: bool = Depends(verify_key)):
    start = time.time()
    prompt = _format_chat_prompt(req.messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the generated part after the prompt
    generated = full_text[len(prompt):]
    elapsed = time.time() - start

    return {
        "id": f"vlsi-chat-{int(start)}",
        "object": "chat.completion",
        "created": int(start),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": generated},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": inputs["input_ids"].shape[1],
            "completion_tokens": outputs.shape[1] - inputs["input_ids"].shape[1],
            "total_tokens": outputs.shape[1],
        },
        "_elapsed_sec": round(elapsed, 2),
    }


@app.post("/v1/completions")
async def completions(req: CompletionRequest, _: bool = Depends(verify_key)):
    start = time.time()
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    elapsed = time.time() - start

    return {
        "id": f"vlsi-comp-{int(start)}",
        "object": "text_completion",
        "created": int(start),
        "model": req.model,
        "choices": [{
            "text": text,
            "index": 0,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": inputs["input_ids"].shape[1],
            "completion_tokens": outputs.shape[1],
            "total_tokens": inputs["input_ids"].shape[1] + outputs.shape[1],
        },
        "_elapsed_sec": round(elapsed, 2),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--local", action="store_true", help="Force local model path")
    args = p.parse_args()

    if args.local and not Path(LOCAL_MODEL).exists():
        print(f"❌ Local model not found at {LOCAL_MODEL}")
        print(f"   Run: python scripts/download_model.py")
        sys.exit(1)

    print(f"Starting FastAPI server on port {args.port}")
    print(f"API Key: {API_KEY}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
