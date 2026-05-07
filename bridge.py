#!/usr/bin/env python3
"""
AgentIC Bridge — Hot-Swap API for Qwen-VLSI-SOTA-Sprint
Provides an API endpoint that allows AgentIC to switch between models.

Feature:
  Once the 14B Ternary model finishes training, the pipeline automatically
  hot-swaps from the 66GB VRAM Teacher (33B) to the 12GB VRAM Ternary Student (14B).

Endpoints:
  GET  /health                  — Server status
  GET  /models                  — List registered models + active model
  POST /v1/swap                 — Switch active model
  POST /v1/chat/completions     — OpenAI-compatible proxy to active model
  GET  /v1/metrics              — Dashboard metrics (refactors, proof-density, tok/s)

Usage:
  python bridge.py
  python bridge.py --port 8002 --teacher http://localhost:8000/v1 --draft http://localhost:8001/v1
"""

import argparse
import json
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from config import (
    ROOT, CHECKPOINT_DIR, EXPERT_DISTILLED_DIR, DATASET_DIR,
    TEACHER_URL, DRAFT_URL, API_KEY,
    TEACHER_PORT, DRAFT_PORT, BRIDGE_PORT,
)

TRIPLETS_DATA = DATASET_DIR / "reasoning_triplets.jsonl"


@dataclass
class ModelRegistry:
    models: dict = field(default_factory=dict)
    active: str = "teacher"

    def __post_init__(self):
        self.models = {
            "teacher": {
                "name": "Teacher (33B)",
                "model_id": "vxkyyy/vlsi-moe-ffn-merged-formal",
                "url": TEACHER_URL,
                "vram_gb": 66,
                "type": "fp16",
                "speeds_up_gen": False,
            },
            "draft": {
                "name": "Draft (1.5B)",
                "model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                "url": DRAFT_URL,
                "vram_gb": 5,
                "type": "fp16",
                "speeds_up_gen": True,
            },
        }

    def register_student(self, student_path: str):
        self.models["student"] = {
            "name": "Student (14B Ternary)",
            "model_id": "qwen-vlsi-sota-14b-ternary",
            "path": student_path,
            "vram_gb": 12,
            "type": "bitnet_b158",
            "speeds_up_gen": False,
        }

    def swap(self, model_name: str) -> dict:
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.models.keys())}")
        previous = self.active
        self.active = model_name
        return {
            "previous": previous,
            "active": model_name,
            "model": self.models[model_name],
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_active_url(self) -> str:
        return self.models[self.active]["url"]


registry = ModelRegistry()

CHECKPOINT_DIRS = sorted(
    [d for d in CHECKPOINT_DIR.glob("phase*") if d.is_dir()],
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
if CHECKPOINT_DIRS:
    registry.register_student(str(CHECKPOINT_DIRS[0]))

student_gguf = EXPERT_DISTILLED_DIR / "qwen-vlsi-sota-14b-ternary.gguf"
if student_gguf.exists():
    registry.models["student"]["gguf_path"] = str(student_gguf)

if EXPERT_DISTILLED_DIR.exists() and any(EXPERT_DISTILLED_DIR.iterdir()):
    registry.register_student(str(EXPERT_DISTILLED_DIR))


class SwapRequest(BaseModel):
    model: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "vlsi-expert"
    messages: list[ChatMessage]
    max_tokens: int = 2048
    temperature: float = 0.2
    top_p: float = 1.0
    stream: bool = False


def load_metrics() -> dict:
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "active_model": registry.active,
        "active_details": registry.models[registry.active],
        "teacher_status": "unknown",
        "draft_status": "unknown",
        "successful_refactors": 0,
        "total_samples": 0,
        "proof_density": 0.0,
        "tokens_per_second": 0.0,
    }

    if TRIPLETS_DATA.exists():
        try:
            rows = []
            with open(TRIPLETS_DATA) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            metrics["total_samples"] = len(rows)
            metrics["successful_refactors"] = sum(
                1 for r in rows if r.get("verification_stage") == "all-passed"
            )
            metrics["proof_density"] = (
                metrics["successful_refactors"] / max(metrics["total_samples"], 1)
            )
            metrics["avg_refactors"] = (
                sum(r.get("num_refactors", 0) for r in rows) / max(len(rows), 1)
            )
            metrics["avg_time_sec"] = (
                sum(r.get("total_time_sec", 0) for r in rows) / max(len(rows), 1)
            )
        except Exception:
            pass

    return metrics


async def check_url(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{url}/models")
            return r.status_code == 200
    except Exception:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("  AgentIC Bridge — Model Hot-Swap API")
    print(f"  Teacher: {TEACHER_URL}")
    print(f"  Draft:   {DRAFT_URL}")
    print(f"  Port:    {BRIDGE_PORT}")
    print(f"  Active:  {registry.active}")
    print("=" * 60)

    teacher_ok = await check_url(TEACHER_URL)
    draft_ok = await check_url(DRAFT_URL)

    if not teacher_ok:
        print("  WARNING: Teacher vLLM not reachable. Start it:")
        print("    ./setup.sh --serve-teacher")
    if not draft_ok:
        print("  WARNING: Draft vLLM not reachable. Start it:")
        print("    ./setup.sh --serve-draft")
    print()

    yield


app = FastAPI(title="AgentIC Bridge — VLSI Model Hot-Swap", lifespan=lifespan)


@app.get("/health")
async def health():
    teacher_ok = await check_url(TEACHER_URL)
    draft_ok = await check_url(DRAFT_URL)
    return {
        "status": "ok",
        "active_model": registry.active,
        "teacher_reachable": teacher_ok,
        "draft_reachable": draft_ok,
        "bridge_port": BRIDGE_PORT,
    }


@app.get("/models")
async def list_models():
    return {
        "active": registry.active,
        "models": registry.models,
        "can_swap_to": [k for k in registry.models if k != registry.active],
    }


@app.post("/v1/swap")
async def swap_model(req: SwapRequest):
    if req.model not in registry.models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{req.model}' not found. Available: {list(registry.models.keys())}",
        )

    model_info = registry.models[req.model]
    if "url" not in model_info:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.model}' is registered but has no active vLLM URL. "
                    f"Serve it first: ./setup.sh --serve-teacher  # or --serve-draft",
        )

    target_url = model_info["url"]
    reachable = await check_url(target_url)
    if not reachable:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{req.model}' vLLM instance at {target_url} is not reachable. "
                    f"Ensure it is running.",
        )

    result = registry.swap(req.model)
    vram_before = registry.models[result["previous"]].get("vram_gb", "?")
    vram_after = model_info.get("vram_gb", "?")

    print(f"  >> HOT SWAP: {result['previous']} → {req.model} "
          f"({vram_before}GB → {vram_after}GB VRAM)")

    return {
        **result,
        "vram_before_gb": vram_before,
        "vram_after_gb": vram_after,
        "vram_saved_gb": vram_before - vram_after if isinstance(vram_before, (int, float)) and isinstance(vram_after, (int, float)) else None,
        "message": f"Successfully hot-swapped to {req.model}",
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    active_url = registry.get_active_url()
    payload = req.model_dump()

    async with httpx.AsyncClient(timeout=300.0) as client:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        if req.stream:
            async def stream_response():
                async with client.stream(
                    "POST",
                    f"{active_url}/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        yield f"data: {json.dumps({'error': response.status_code, 'detail': body.decode()})}\n\n"
                        return
                    async for chunk in response.aiter_bytes():
                        yield chunk

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers={"X-Active-Model": registry.active},
            )
        else:
            response = await client.post(
                f"{active_url}/chat/completions",
                json=payload,
                headers=headers,
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=response.text[:1000],
                )

            result = response.json()
            result["_active_model"] = registry.active
            return result


@app.get("/v1/metrics")
async def metrics():
    metrics = load_metrics()
    teacher_ok = await check_url(TEACHER_URL)
    draft_ok = await check_url(DRAFT_URL)
    metrics.update({
        "teacher_reachable": teacher_ok,
        "draft_reachable": draft_ok,
    })

    checkpoint_dirs = sorted(CHECKPOINT_DIR.glob("phase*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if checkpoint_dirs:
        latest = checkpoint_dirs[0]
        metrics["latest_checkpoint"] = {
            "name": latest.name,
            "age_minutes": (datetime.now().timestamp() - latest.stat().st_mtime) / 60,
            "size_gb": sum(f.stat().st_size for f in latest.rglob("*") if f.is_file()) / 1e9,
        }

    if student_gguf.exists():
        metrics["gguf_export"] = {
            "path": str(student_gguf),
            "size_gb": student_gguf.stat().st_size / 1e9,
            "created": datetime.fromtimestamp(student_gguf.stat().st_mtime).isoformat(),
        }

    return metrics


@app.get("/v1/dashboard")
async def dashboard():
    m = await metrics()
    return {
        "dashboard": {
            "title": "Qwen-VLSI-SOTA-Sprint — Training Dashboard",
            "active_model": m["active_model"],
            "teacher_reachable": m.get("teacher_reachable"),
            "draft_reachable": m.get("draft_reachable"),
            "sampling_stats": {
                "total_samples": m.get("total_samples", 0),
                "successful_refactors": m.get("successful_refactors", 0),
                "proof_density": f"{m.get('proof_density', 0) * 100:.1f}%",
                "avg_refactor_attempts": m.get("avg_refactors", 0),
                "avg_generation_time_sec": m.get("avg_time_sec", 0),
            },
            "checkpoint": m.get("latest_checkpoint"),
            "export": m.get("gguf_export"),
            "phase": "phase3_qat_complete" if m.get("gguf_export") else (
                "phase2_distilling" if m.get("latest_checkpoint") else "phase1_precomputing"
            ),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="AgentIC Bridge — Model Hot-Swap API")
    parser.add_argument("--port", type=int, default=BRIDGE_PORT)
    parser.add_argument("--teacher", default=TEACHER_URL, help="Teacher vLLM URL")
    parser.add_argument("--draft", default=DRAFT_URL, help="Draft vLLM URL")
    parser.add_argument("--auto-swap", action="store_true",
                        help="Auto-swap to student if checkpoint available")
    args = parser.parse_args()

    if args.auto_swap:
        checkpoint_dirs = sorted(
            [d for d in CHECKPOINT_DIR.glob("phase*") if d.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if checkpoint_dirs:
            print(f"Auto-swap: Student checkpoint found at {checkpoint_dirs[0]}")
            try:
                registry.swap("student")
                print("  Switched to student model.")
            except ValueError:
                print("  Student model not yet registered.")

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
