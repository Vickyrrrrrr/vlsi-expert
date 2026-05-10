# SiliconSmith AI — System Architecture

## Overview

SiliconSmith AI is built as a three-layer system:
1. **Inference layer** — AMD MI300X + ROCm + vLLM
2. **API layer** — OpenAI-compatible REST endpoint
3. **Agent layer** — Tool-using Python agent for chip design workflows

---

## Hardware: AMD Instinct MI300X

| Spec | Value |
|---|---|
| GPU architecture | CDNA 3 |
| HBM3 memory | 192 GB |
| Memory bandwidth | 5.3 TB/s |
| FP16/BF16 TFLOPS | 1307 |
| ROCm version | 6.x |

The MI300X's 192GB unified memory pool enables running large MoE models with a 262K token context window that would not fit on smaller GPUs.

---

## Software Stack

```
┌─────────────────────────────────────┐
│         Agent Application           │
│    (Python, tool use, memory)       │
├─────────────────────────────────────┤
│      OpenAI Python SDK              │
│   base_url = http://127.0.0.1:8000  │
├─────────────────────────────────────┤
│         SSH Port Tunnel             │
│    Local :8000 → Remote :8000       │
├─────────────────────────────────────┤
│         vLLM v0.17.1                │
│    /v1/chat/completions             │
│    /v1/models                       │
├─────────────────────────────────────┤
│         ROCm Runtime                │
│    (HIP, MIOpen, rocBLAS)           │
├─────────────────────────────────────┤
│    AMD Instinct MI300X              │
│    192GB HBM3                       │
└─────────────────────────────────────┘
```

---

## Model Configuration

| Parameter | Value |
|---|---|
| Model | vlsi-moe-yarn (custom long-context MoE) |
| dtype | bfloat16 |
| KV cache dtype | fp8 (memory-efficient) |
| Max context length | 262,144 tokens |
| Tensor parallel size | 1 (single MI300X) |
| Serving port | 8000 |

---

## Access Pattern

```
Developer PC
    │
    │  ssh -L 8000:127.0.0.1:8000 root@DROPLET_IP
    │
    ▼
Local 127.0.0.1:8000
    │  (SSH tunnel forwards to remote)
    │
    ▼
AMD Droplet 127.0.0.1:8000
    │
    ▼
vLLM API Server (Docker container: rocm)
    │
    ▼
Model: /app/vlsi-moe-yarn on MI300X
```

---

## Agent Architecture

The agent layer (in `factory.py`, `chip.py`, `bridge.py`) provides:

- **Task planning** — decompose chip design questions into sub-tasks
- **Tool use** — RTL analysis, constraint checking, documentation lookup
- **Memory** — retain design context across multi-turn conversations
- **Streaming** — real-time token streaming for long responses

---

## Why This Architecture Works

1. **MI300X memory** allows 262K context without chunking — entire design specs fit in one prompt
2. **fp8 KV cache** reduces memory pressure, allowing longer effective context
3. **vLLM prefix caching** speeds up repeated context (e.g., always-present system prompt with chip specs)
4. **OpenAI-compatible API** means any tool that supports custom OpenAI endpoints can plug in
