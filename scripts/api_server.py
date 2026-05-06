#!/usr/bin/env python3
"""
VLSI Expert — FastAPI Server
Uses same transformers code that already works on the VPS.
OpenAI-compatible /v1/completions endpoint.

Usage:
  python scripts/api_server.py           # Start on port 8000
  curl http://localhost:8000/v1/completions -H "Authorization: Bearer agentic-vlsi-expert-secure" -d '{"model":"vlsi-expert","prompt":"8-bit counter\n\nmodule","max_tokens":200}'
"""

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

# ── Config ────────────────────────────────────────────────────────────
MODEL_PATH = str(Path(__file__).parent.parent / "models" / "vlsi-moe-ffn-merged" / "merged")
API_KEY = "agentic-vlsi-expert-secure"
PORT = 8000

# ── Global model holder ───────────────────────────────────────────────
model = None
tokenizer = None
security = HTTPBearer()


class CompletionRequest(BaseModel):
    model: str = "vlsi-expert"
    prompt: str
    max_tokens: int = 800
    temperature: float = 0.2
    stop: list = ["</s>"]


def verify_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model, tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading VLSI Expert from {MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Model loaded! Ready on port {PORT}")
    yield


app = FastAPI(title="VLSI Expert API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


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
        "id": f"vlsi-expert-{int(start)}",
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
    uvicorn.run(app, host="0.0.0.0", port=PORT)
