# VLSI Expert — AI Chip Designer (MoE)

A **Mixture-of-Experts** model for VLSI chip design. Three expert models merged via
DARE + TIES with a learned task router. Served via vLLM on AMD Instinct MI300X.

Built for the [AMD Developer Hackathon](https://lablab.ai/ai-hackathons/amd-developer).

## Architecture (MoE)

```
Input: "Generate RTL for 5-stage RISC-V pipeline"

        ┌────────────────────────────────────────┐
        │        TASK ROUTER (MLP classifier)    │
        │  "generate/write/create" → Expert 0    │
        │  "fix/error/debug" → Expert 1          │
        │  "sdc/timing" → Expert 2               │
        └───────┬──────────────┬─────────────────┘
                │              │
    ┌───────────▼───────────┐  │  ┌────────────────────────┐
    │ Expert 0: CODER       │  │  │ Expert 1+2: REASON+INSTR│
    │ Qwen2.5-Coder-32B     │  │  │ DeepSeek-R1+Qwen3-32B  │
    │ • Write Verilog RTL   │  │  │ • Fix syntax errors     │
    │ • Generate testbenches│  │  │ • Analyze timing paths  │
    │ • Write SDC configs   │  │  │ • Suggest fixes         │
    └───────────────────────┘  └──┴─────────────────────────┘

Three experts merged via DARE + TIES into one model.
Router (MLP on embeddings) selects expert per task.
Served with vLLM 0.17.1 + ROCm 7.2 on AMD MI300X (192 GB VRAM).

Effective compute sparsity: 1:3 (only one expert active at a time)
Model size: 32B backbone + 3 expert FFN layers (~64GB total, fits in VRAM)
Inference: 15-25 tok/s via vLLM continuous batching
```

## Quick Start

### Via vLLM (AMD GPU)
```bash
# Start the server
python scripts/serve_vllm.py

# In another terminal, query it
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"vlsi-moe","prompt":"Generate Verilog for an 8-bit counter","max_tokens":512}'
```

### Via HuggingFace (CPU/GPU)
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct")
model = PeftModel.from_pretrained(base, "Vickyrrrrrr/vlsi-coder-lora")
```

## Training Pipeline

```bash
# 1. Collect public Verilog data
python scripts/collect_data.py

# 2. Merge 3 experts into one MoE model
python scripts/merge_moe.py

# 3. Train task router
python scripts/train_router.py

# 4. Evaluate through AgentIC pipeline
python eval/evaluate_moe.py

# 5. Serve with vLLM
python scripts/serve_vllm.py
```

## Evaluation

Tested through [AgentIC](https://github.com/Vickyrrrrrr/AgentIC) — 27-stage RTL-to-GDSII pipeline on SkyWater 130nm.

| Model | Architecture | Synthesis Pass |
|-------|-------------|---------------|
| Qwen2.5-Coder-32B (baseline) | Dense 32B | _pending_ |
| VLSI Expert (QLoRA) | 32B + LoRA adapter | _pending_ |
| **VLSI Expert MoE** | **32B + 3 experts + router** | **_pending_** |

## Files

```
vlsi-expert/
├── scripts/
│   ├── collect_data.py        # Public Verilog datasets
│   ├── merge_moe.py           # DARE + TIES merge (3 experts → 1)
│   ├── train_router.py        # MLP router training
│   ├── train_coder.py         # QLoRA coder fine-tuning
│   ├── train_instruct.py      # QLoRA instruct fine-tuning
│   └── serve_vllm.py          # vLLM serving (ROCm 7.2)
├── eval/
│   ├── evaluate_moe.py        # MoE pipeline evaluation
│   └── evaluate.py            # Single-model evaluation
├── gradio/
│   └── app.py                 # HuggingFace Spaces demo
├── DEPLOY.md                  # AMD GPU deployment guide
└── README.md
```

## Tech Stack

| Component | Version |
|-----------|---------|
| vLLM | 0.17.1 |
| ROCm | 7.2.0 |
| GPU | AMD Instinct MI300X (192 GB VRAM) |
| Base models | Qwen2.5-Coder-32B, DeepSeek-R1-32B, Qwen3-32B |
| Merge method | DARE (90% delta dropout) + TIES-Merging |
| Router | MLP classifier on embeddings |
| Serving | vLLM OpenAI-compatible API |

## License

Apache 2.0 — same as all base models. Training data from VerilogEval v2 (MIT) and RTLLM (Apache 2.0).
