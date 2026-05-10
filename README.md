# 🔬 SiliconSmith AI — Agentic VLSI Copilot

> A chip-design AI agent powered by a custom long-context MoE model running on AMD Instinct MI300X.  
> Built for the [AMD Developer Hackathon](https://lablab.ai/ai-hackathons/amd-developer) — **AI Agents & Agentic Workflows** track.

---

## What It Does

SiliconSmith AI is an agentic assistant for **VLSI and chip design**. Ask it anything from architecture tradeoffs to RTL review — it holds up to **262,144 tokens** of design context in a single conversation.

| Capability | Example |
|---|---|
| Architecture reasoning | "Tradeoffs of mesh vs. ring NoC for a 16-core SoC?" |
| RTL analysis | "Find timing issues in this Verilog FIFO module." |
| Clock domain crossing | "How do I safely cross from 200MHz to 400MHz?" |
| Constraint guidance | "Write SDC constraints for a 500MHz DDR interface." |
| Design document Q&A | Paste an entire datasheet — it reads and answers. |
| Agentic planning | Multi-step reasoning across chip design sub-tasks. |

---

## Usage

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)

response = client.chat.completions.create(
    model="/app/vlsi-moe-yarn",
    messages=[
        {"role": "system", "content": "You are SiliconSmith, an expert VLSI and chip design assistant."},
        {"role": "user",   "content": "Explain clock-domain crossing risks in a mixed-signal SoC."}
    ],
    max_tokens=512,
)
print(response.choices[0].message.content)
```

The model is served via a **vLLM OpenAI-compatible API** — any tool or script that supports a custom OpenAI `base_url` works out of the box.

---

## Infrastructure

| Component | Details |
|---|---|
| Hardware | AMD Instinct MI300X · 192GB HBM3 |
| Software | ROCm · vLLM v0.17.1 |
| Model | vlsi-moe-yarn (custom long-context MoE) |
| Max context | 262,144 tokens |
| dtype | bfloat16 · fp8 KV cache |
| API | OpenAI-compatible `/v1/chat/completions` |

---

## Serve the Model

```bash
vllm serve /app/vlsi-moe-yarn \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --host 0.0.0.0 \
  --port 8000
```

See [`serving/launch.sh`](serving/launch.sh) for the full AMD/ROCm environment setup.

---

## Architecture

```
Your PC  ──── SSH tunnel :8000 ────►  AMD MI300X Droplet
                                           │
                                      vLLM Server
                                           │
                                    vlsi-moe-yarn model
                                    (262K token context)
```

---

## Hackathon

**AMD Developer Hackathon 2026** · [lablab.ai/ai-hackathons/amd-developer](https://lablab.ai/ai-hackathons/amd-developer)  
Track: **AI Agents & Agentic Workflows** · Prize path: Build in Public `#AMDDevHackathon`

---

## License

MIT — see [LICENSE](LICENSE)
