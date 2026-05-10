# SiliconSmith AI — VLSI Expert Model

> A chip-design AI built on a **custom Qwen-based model** with reasoning FFN layers and YaRN long-context extension.  
> Served via vLLM on AMD Instinct MI300X. Built for the [AMD Developer Hackathon 2026](https://lablab.ai/ai-hackathons/amd-developer).

---

## What It Does

SiliconSmith AI is an AI assistant for **VLSI and chip design**. It holds up to **262,144 tokens** of context — enough to load an entire SoC specification or IP datasheet in one conversation.

| Capability | Example |
|---|---|
| Architecture reasoning | "Compare mesh vs. ring NoC for a 16-core AI accelerator" |
| RTL review | "Find timing issues in this Verilog FIFO module" |
| Clock-domain crossing | "How do I safely cross from 200MHz to 400MHz?" |
| Constraint guidance | "Write SDC constraints for a 500MHz DDR interface" |
| Design document Q&A | Paste a full datasheet — it reads and answers |
| Agentic planning | Multi-step reasoning across chip design sub-tasks |

---

## Model Architecture

This is **not a standard MoE**. The architecture is:

- **Base**: Qwen encoder-decoder backbone
- **Modification**: 10% of FFN layers replaced with **reasoning-optimized feed-forward blocks** for deeper logical inference
- **Context**: YaRN (Yet another RoPE extensioN) to extend context to 262,144 tokens
- **Training**: Knowledge distillation from a larger teacher model on a VLSI-domain corpus

---

## Usage

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)

response = client.chat.completions.create(
    model="vlsi-moe-yarn",
    messages=[
        {"role": "system", "content": "You are SiliconSmith, an expert VLSI and chip design assistant."},
        {"role": "user",   "content": "Explain clock-domain crossing risks in a mixed-signal SoC."}
    ],
    max_tokens=512,
)
print(response.choices[0].message.content)
```

The model exposes an **OpenAI-compatible API** — any tool, script, or agent framework that supports a custom `base_url` works without modification.

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

See [`serving/launch.sh`](serving/launch.sh) for the full ROCm environment setup.

---

## Infrastructure

| Component | Details |
|---|---|
| Hardware | AMD Instinct MI300X · 192GB HBM3 |
| Runtime | ROCm + vLLM v0.17.1 |
| Model dtype | bfloat16 · fp8 KV cache |
| Max context | 262,144 tokens |
| API | OpenAI-compatible `/v1/chat/completions` |

---

## Repository Structure

```
vlsi-expert/
├── app.py            # Main agent application
├── bridge.py         # Model bridge / API routing
├── chip.py           # Chip design task tools
├── factory.py        # Agent factory / orchestration
├── distill.py        # Knowledge distillation pipeline
├── config.py         # Configuration
├── requirements.txt  # Python dependencies
└── serving/
    ├── launch.sh     # vLLM server launch script
    └── test_client.py
```

---

## Hackathon

**AMD Developer Hackathon 2026** · [lablab.ai/ai-hackathons/amd-developer](https://lablab.ai/ai-hackathons/amd-developer)  
Track: **AI Agents & Agentic Workflows** · `#AMDDevHackathon`

---

## License

MIT — see [LICENSE](LICENSE)
