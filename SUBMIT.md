# VLSI Expert — AMD Developer Hackathon Submission

## What We Built

A **33B parameter AI chip designer** fine-tuned on AMD MI300X for hardware RTL generation with formal verification (SVA) capabilities.

**Architecture:**
- **Base**: Qwen2.5-Coder-32B-Instruct merged with DeepSeek-R1-Distill-Qwen-32B via DARE+TIES (FFN layers)
- **Fine-tuning**: LoRA (rank 128) on 307 SystemVerilog Assertion examples
- **Training Loss**: 4.78 → 0.051 over 10 epochs
- **Hardware**: AMD Instinct MI300X (192GB VRAM, ROCm 6.2)

The model generates synthesizable Verilog RTL + SystemVerilog Assertions from natural language, served via OpenAI-compatible API so you can run AgentIC from your laptop while the model lives on AMD cloud.

---

## Where It Is

| Resource | URL |
|----------|-----|
| **GitHub** (this repo) | https://github.com/Vickyrrrrrr/vlsi-expert |
| **HuggingFace Model** (66GB weights) | https://huggingface.co/vxkyyy/vlsi-moe-ffn-merged-formal |
| **HuggingFace Base Model** | https://huggingface.co/vxkyyy/vlsi-moe-ffn-merged |
| **LoRA Adapter** | https://huggingface.co/vxkyyy/vlsi-formal-lora |
| **AgentIC Pipeline** | https://github.com/Vickyrrrrrr/AgentIC |
| **Gradio Demo** | (Deploy to HF Spaces from this repo) |

---

## How It Works

### Remote Architecture

```
Your Machine (Anywhere)          AMD Developer Cloud (MI300X)
┌────────────────────┐          ┌─────────────────────────────┐
│ AgentIC + CrewAI   │ ──API──► │  VLSI Expert Model Server   │
│ chip.py client     │   Key    │  /v1/chat/completions       │
│ Local .env         │          │  Port 8000                  │
└────────────────────┘          └─────────────────────────────┘
```

### Example Flow

```bash
# On VPS
./setup.sh  # Downloads model, starts server

# On your laptop
export LLM_BASE_URL=http://VPS_IP:8000/v1
export LLM_API_KEY=agentic-vlsi-expert-secure
agentic build --name counter --desc "8-bit counter" --skip-openlane
```

The model receives the spec through CrewAI/LiteLLM → generates Verilog + SVA → AgentIC verifies with Yosys/Verilator.

---

## Hackathon Submission Checklist

### Step 1: Submit on lablab.ai

Go to https://lablab.ai/ai-hackathons/amd-developer → Submit

**Primary Track:** 🤖 Track 1 — AI Agents & Agentic Workflows  
**Secondary Track:** ⚡ Track 2 — Fine-Tuning on AMD GPUs  

**Required fields:**
- **Project Title**: VLSI Expert — AI Chip Designer (33B, AMD MI300X)
- **Short Description**: 33B model fine-tuned on AMD MI300X for Verilog RTL + SVA generation. Served via OpenAI-compatible API for remote AgentIC pipeline execution.
- **Long Description**: (See below)
- **GitHub**: https://github.com/Vickyrrrrrr/vlsi-expert
- **Demo URL**: https://huggingface.co/spaces/vxkyyy/vlsi-expert (or your VPS IP)
- **Video**: 2-3 min screen recording
- **Tags**: CrewAI, AMD Developer Cloud, ROCm, Qwen, DeepSeek, DARE, TIES, vLLM, LoRA, VLSI

### Long Description Template

```
VLSI Expert is a 33B parameter language model specialized for hardware chip design.

What makes it unique:
1. MERGED: Two 32B models (Qwen2.5-Coder + DeepSeek-R1) combined via DARE+TIES
   on FFN layers only. Coder's attention generates code; R1's FFN provides reasoning.

2. FINE-TUNED: LoRA (rank 128, 1.07B trainable params) on 307 SVA examples.
   Trained on AMD Instinct MI300X with ROCm 6.2 + PyTorch 2.5.1.
   Loss dropped from 4.78 to 0.051 over 10 epochs.

3. REMOTE: Model runs on AMD Developer Cloud VPS. You call it via OpenAI-compatible
   API from your laptop. AgentIC's CrewAI agents use it for RTL generation,
   formal verification, and error fixing — all without local GPU.

4. OPEN: Full training scripts, data collectors, and server code are open-source.
   Anyone can replicate the pipeline on their own AMD GPU instance.

Use cases:
- Generate synthesizable Verilog from English specs
- Produce SystemVerilog Assertions for formal verification
- Fix Verilog errors via agentic iteration
- Generate SDC timing constraints

Tech stack:
- AMD Instinct MI300X (192GB VRAM)
- ROCm 6.2 / PyTorch 2.5.1+rocm6.2
- Transformers + PEFT (LoRA)
- FastAPI / vLLM for serving
- CrewAI + AgentIC for agent orchestration
- Yosys + Verilator for RTL verification
```

### Step 2: Video Demo (2-3 min)

```
0:00-0:15: Show GitHub repo + HF model page
0:15-0:45: SSH into AMD VPS, run ./setup.sh, show model loading
0:45-1:15: From local terminal: python scripts/chip.py "UART transmitter"
            Show generated Verilog + SVA
1:15-1:45: Show rocm-smi output (MI300X running, VRAM usage)
            Explain: 66GB model on 192GB GPU
1:45-2:15: Show AgentIC build running with remote LLM_BASE_URL
            Terminal: syntax check → simulation → PASS
2:15-2:30: Architecture diagram. Explain DARE+TIES merge + LoRA fine-tuning.
2:30-3:00: Links, thank you, built on AMD MI300X + ROCm.
```

### Step 3: Social Media (Build in Public / Extra Challenge)

**Post 1 (Technical)**:
```
Built a 33B VLSI design model on @AIatAMD MI300X with ROCm 6.2.

- Merged Qwen2.5-Coder + DeepSeek-R1 via DARE+TIES
- Fine-tuned with LoRA on SVA examples
- Serves OpenAI-compatible API for remote AgentIC pipeline

Loss: 4.78 → 0.051 🔥

Repo: github.com/Vickyrrrrrr/vlsi-expert
#AMDDeveloperHackathon @lablabai
```

**Post 2 (ROCm Feedback)**:
```
ROCm on MI300X: honest take after 48 hours 🧵

Pros:
+ 192GB VRAM fits 33B bfloat16 + LoRA training comfortably
+ PyTorch ROCm wheels install cleanly (pip install torch --index-url rocm6.2)
+ rocm-smi gives same info as nvidia-smi

Cons:
- vLLM ROCm binary doesn't exist → had to use FastAPI fallback
- Some CUDA-specific libs need HIPIFY translation
- Smaller ecosystem than CUDA but growing fast

Overall: perfectly usable for LLM training/serving. Just avoid CUDA-only tools.
#AMDDeveloperHackathon @AIatAMD
```

Tag requirements: @lablab on X, lablab.ai on LinkedIn, @AIatAMD on X, AMD Developer on LinkedIn.

### Step 4: HuggingFace Space

1. Join AMD Developer Hackathon HF Organization (link on hackathon page)
2. Create Space under organization
3. Use this repo's `gradio/app.py`
4. Set Space to use your model from HF Hub
5. Submit Space URL on lablab.ai

---

## Quick Reference

### Start Server on VPS
```bash
cd ~/vlsi-expert
source ~/vlsi-env/bin/activate
python scripts/serve.py          # Auto-detect vLLM/FastAPI
```

### Call from Local Machine
```bash
export VLSI_EXPERT_HOST=YOUR_VPS_IP
export VLSI_EXPERT_PORT=8000
python scripts/chip.py "8-bit counter"
```

### AgentIC Integration
```bash
# .env on your local machine
LLM_BASE_URL=http://YOUR_VPS_IP:8000/v1
LLM_API_KEY=agentic-vlsi-expert-secure
LLM_MODEL=vlsi-expert

agentic build --name counter --desc "8-bit counter" --skip-openlane
```

---

## What's Next

1. **vLLM ROCm build**: Get production-grade serving working on ROCm
2. **More training data**: Scale from 307 → 10K SVA + Verilog pairs
3. **Multi-GPU**: Tensor parallelism across 2x MI300X for larger models
4. **PDK expansion**: TSMC 28nm, Samsung 14nm support
5. **X402 integration**: Agent-to-agent payments for chip design services
