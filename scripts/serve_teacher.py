#!/usr/bin/env python3
"""
VLSI Teacher Server — FastAPI + Transformers (ROCm)
OpenAI-compatible /v1/chat/completions endpoint.

Usage:
  python scripts/serve_teacher.py                                    # HF Hub
  python scripts/serve_teacher.py --local                            # Local model
  python scripts/serve_teacher.py --model /path/to/model --port 8000
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

LOCAL_MODEL = "models/vlsi-moe-ffn-merged-formal"
HF_MODEL = "vxkyyy/vlsi-moe-ffn-merged-formal"
API_KEY = os.environ.get("VLSI_API_KEY", "agentic-vlsi-expert-secure")
PORT = int(os.environ.get("VLSI_TEACHER_PORT", "8000"))

model = None
tokenizer = None
security = HTTPBearer()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "vlsi-expert-teacher"
    messages: list[ChatMessage]
    max_tokens: int = 2048
    temperature: float = 0.2
    top_p: float = 1.0
    stop: list[str] | None = None


class CompletionRequest(BaseModel):
    model: str = "vlsi-expert-teacher"
    prompt: str
    max_tokens: int = 2048
    temperature: float = 0.2
    top_p: float = 1.0
    stop: list[str] | None = None


def verify_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


def _format_chat_prompt(messages: list[ChatMessage]) -> str:
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"<|im_start|>system\n{msg.content}<|im_end|>\n")
        elif msg.role == "user":
            parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>\n")
        elif msg.role == "assistant":
            parts.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_local = Path(LOCAL_MODEL).exists()
    model_path = LOCAL_MODEL if use_local else HF_MODEL

    print(f"Loading model from {'LOCAL' if use_local else 'HF HUB'}: {model_path}")
    print("~66GB model, 2-4 min on MI300X...")

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
    model.eval()
    print(f"Model loaded on {model.device}, {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")
    yield


app = FastAPI(title="VLSI Teacher Server", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "backend": "fastapi+transformers"}


@app.get("/models")
def list_models():
    return {"data": [{"id": "vlsi-expert-teacher", "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, _: bool = Depends(verify_key)):
    if model is None:
        raise HTTPException(503, "Model not loaded yet")

    t0 = time.time()
    prompt = _format_chat_prompt(req.messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=req.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = full[len(prompt):]
    elapsed = time.time() - t0

    return {
        "id": f"chatcmpl-{int(t0)}",
        "object": "chat.completion",
        "created": int(t0),
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
    if model is None:
        raise HTTPException(503, "Model not loaded yet")

    t0 = time.time()
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=req.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    elapsed = time.time() - t0

    return {
        "id": f"cmpl-{int(t0)}",
        "object": "text_completion",
        "created": int(t0),
        "model": req.model,
        "choices": [{
            "text": text,
            "index": 0,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": inputs["input_ids"].shape[1],
            "completion_tokens": outputs.shape[1] - inputs["input_ids"].shape[1],
            "total_tokens": outputs.shape[1],
        },
        "_elapsed_sec": round(elapsed, 2),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--local", action="store_true")
    p.add_argument("--port", type=int, default=PORT)
    args = p.parse_args()

    if args.model:
        LOCAL_MODEL = args.model
    elif args.local and not Path(LOCAL_MODEL).exists():
        print(f"Local model not found at {LOCAL_MODEL}")
        sys.exit(1)

    print(f"VLSI Teacher Server starting on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
