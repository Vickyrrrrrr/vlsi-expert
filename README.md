# VLSI Expert — AI Chip Designer

**Fine-tuned 33B model for hardware chip design, running on AMD MI300X via ROCm.**

Generate synthesizable Verilog RTL from natural language. Host the model on your AMD VPS, call it from anywhere.

---

## Architecture

```
Your Laptop / WSL                    AMD MI300X VPS
┌──────────────────┐                 ┌─────────────────────────────┐
│  AgentIC Pipeline │ ──HTTP/API──► │  VLSI Expert Model Server   │
│  CrewAI Agents    │    Key        │  (vLLM or FastAPI)          │
│  chip.py client   │               │  Port 8000                  │
└──────────────────┘                 └─────────────────────────────┘
        │                                      │
        │  "Generate 8-bit counter"            │  vxkyyy/vlsi-moe-ffn-merged-formal
        │─────────────────────────────────────►│  33B params, bfloat16
        │                                      │  ~66GB VRAM
        │  module counter(...)                 │  AMD Instinct MI300X
        │◄─────────────────────────────────────│  ROCm 6.2 / PyTorch 2.5
```

**Model**: [vxkyyy/vlsi-moe-ffn-merged-formal](https://huggingface.co/vxkyyy/vlsi-moe-ffn-merged-formal)  
**Base**: Qwen2.5-Coder-32B + DeepSeek-R1-Distill-Qwen-32B merged via DARE+TIES (FFN layers)  
**Fine-tuning**: LoRA (rank 128) on 307 SystemVerilog Assertion (SVA) examples  
**Hardware**: AMD Instinct MI300X (192GB VRAM)  
**Platform**: ROCm 6.2 + PyTorch 2.5.1

---

## Quick Start

### 1. On Your AMD VPS (Model Host)

```bash
git clone https://github.com/Vickyrrrrrr/vlsi-expert.git
cd vlsi-expert
./setup.sh
```

This will:
1. Create Python venv
2. Install ROCm PyTorch + deps
3. Download ~66GB model from HuggingFace
4. Start API server on port 8000

### 2. On Your Local Machine (Client)

```bash
# Option A: Direct API call
export VLSI_EXPERT_HOST=YOUR_VPS_IP
export VLSI_EXPERT_PORT=8000
python scripts/chip.py "8-bit counter with synchronous reset"

# Option B: SSH Tunnel (more secure)
ssh -N -L 8000:localhost:8000 -i ~/.ssh/id_ed25519 ubuntu@YOUR_VPS_IP
export VLSI_EXPERT_HOST=localhost
python scripts/chip.py "8-bit counter with synchronous reset"

# Option C: AgentIC Pipeline
export LLM_BASE_URL=http://YOUR_VPS_IP:8000/v1
export LLM_API_KEY=agentic-vlsi-expert-secure
export LLM_MODEL=vlsi-expert
agentic build --name counter --desc "8-bit counter" --skip-openlane
```

---

## API Endpoints

The server exposes OpenAI-compatible endpoints:

### `POST /v1/chat/completions`
```bash
curl http://YOUR_VPS_IP:8000/v1/chat/completions \
  -H "Authorization: Bearer agentic-vlsi-expert-secure" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vlsi-expert",
    "messages": [{"role": "user", "content": "Generate an 8-bit counter"}],
    "max_tokens": 800,
    "temperature": 0.2
  }'
```

### `POST /v1/completions` (legacy)
```bash
curl http://YOUR_VPS_IP:8000/v1/completions \
  -H "Authorization: Bearer agentic-vlsi-expert-secure" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vlsi-expert",
    "prompt": "Generate an 8-bit counter\\n\\nmodule",
    "max_tokens": 800
  }'
```

---

## File Structure

```
vlsi-expert/
├── scripts/
│   ├── serve.py              # Auto-detect vLLM/FastAPI launcher
│   ├── serve_vllm.py         # Production vLLM server
│   ├── serve_fastapi.py      # Fallback FastAPI server (chat + completions)
│   ├── download_model.py     # Pull model from HF Hub
│   ├── chip.py               # Remote client (call from your laptop)
│   ├── agentic_expert.py     # AgentIC integration helper
│   ├── build.py              # Full AgentIC pipeline bridge
│   ├── collect_data.py       # Training data collector
│   ├── upload_hf.py          # Upload to HuggingFace
│   └── train_lora_safe.py    # LoRA training script
├── gradio/
│   └── app.py                # Gradio demo (runs on VPS)
├── data/
│   ├── train_pairs.jsonl     # Verilog training pairs
│   └── error_fix_pairs.jsonl # Error-fix pairs
├── models/                   # Downloaded model cache (66GB)
├── archive/                  # Old MoE experiments (historical)
├── setup.sh                  # One-command VPS setup
├── .env.example              # Environment template
├── requirements.txt          # Dependencies
├── DEPLOY.md                 # AMD Developer Cloud deployment
└── SUBMIT.md                 # Hackathon submission details
```

---

## Model Details

| Attribute | Value |
|-----------|-------|
| **Base Models** | Qwen2.5-Coder-32B-Instruct + DeepSeek-R1-Distill-Qwen-32B |
| **Merge Method** | DARE+TIES (FFN layers only) |
| **Fine-tuning** | LoRA r=128, 10 epochs, 307 SVA examples |
| **Training Loss** | 4.78 → 0.051 |
| **Trainable Params** | 1.07B (3.2% of 33B) |
| **VRAM Usage** | ~120GB during training, ~66GB inference |
| **Hardware** | AMD Instinct MI300X |
| **Software** | ROCm 6.2, PyTorch 2.5.1+rocm6.2, Transformers 4.49 |

---

## Gradio Demo

Launch the web UI on your VPS:

```bash
python gradio/app.py
# → http://YOUR_VPS_IP:7860
```

Or deploy to HuggingFace Spaces:
```bash
# Space config is in README_HF.md (gradio sdk)
```

---

## License

Apache-2.0

Built for the [AMD Developer Hackathon](https://lablab.ai/ai-hackathons/amd-developer).
