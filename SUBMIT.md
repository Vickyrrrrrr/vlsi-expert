# VLSI Expert — AMD Developer Hackathon Submission

## What We Built

A **FFN-merged Mixture-of-Experts model** for hardware chip design. Two models combined via DARE+TIES:

| Expert | Model | Role |
|--------|-------|------|
| Coder | Qwen2.5-Coder-32B-Instruct | Verilog RTL generation, SDC writing, testbench |
| Reason | DeepSeek-R1-Distill-Qwen-32B | CoT reasoning, error analysis, architecture planning |

**Merge method:** FFN-only DARE+TIES (192 FFN layers merged, 579 attention layers kept from Coder). No generation crashes — Coder's attention engine generates code while R1's FFN knowledge provides reasoning.

## Where It Is

| Resource | URL |
|----------|-----|
| **GitHub** (code + scripts) | https://github.com/Vickyrrrrrr/vlsi-expert |
| **HuggingFace Space** (live demo) | https://huggingface.co/spaces/vxkyyy/vlsi-expert |
| **HuggingFace Model** (65GB weights) | https://huggingface.co/vxkyyy/vlsi-moe-ffn-merged |
| **AgentIC Pipeline** | https://github.com/Vickyrrrrrr/AgentIC |

## How It Works

```
User: "8-bit counter with synchronous reset"

  ┌──────────────────────────────────────────────────────┐
  │  Qwen2.5-Coder-32B (Attention: 40 heads, 579 layers) │  ← Generates code
  │                        +                             │
  │  DeepSeek-R1-32B (FFN knowledge: 192 layers)         │  ← Reasons, analyzes
  └──────────────────────────────────────────────────────┘
                           │
                    Verilog RTL + SDC + Analysis
```

Both models' FFN knowledge lives in ONE model. Attention stays from Coder — no generation crashes.

## Hackathon Submission

### Step 1: Submit on lablab.ai

Go to https://lablab.ai/ai-hackathons/amd-developer → Submit

**Track:** Track 1 — AI Agents & Agentic Workflows

**Required fields:**
- Project Title: VLSI Expert — AI Chip Designer (MoE)
- GitHub: https://github.com/Vickyrrrrrr/vlsi-expert
- Demo URL: https://huggingface.co/spaces/vxkyyy/vlsi-expert
- Video: Screen recording (see below)

**Tags:** CrewAI, AMD Developer Cloud, ROCm, Qwen, DeepSeek, DARE, TIES, vLLM

### Step 2: Video Demo (2 min)

```
0:00-0:15: Open HuggingFace Space. Show Gradio interface.
0:15-0:45: Type "8-bit counter" → Model generates Verilog. Show the code.
0:45-1:15: Type "UART transmitter" → Model generates UART with baud rate calc.
1:15-1:40: Explain architecture: "Merged via DARE+TIES, FFN-only, 192 layers"
1:40-2:00: Show AgentIC pipeline running the generated Verilog → GDSII output
```

### Step 3: Social Media (Build in Public)

Post on X/LinkedIn:
```
Built a MoE AI chip designer on @AIatAMD MI300X. FFN-merged Qwen2.5-Coder + 
DeepSeek-R1 via DARE+TIES. One model generates Verilog + analyzes designs.
#AMDDeveloperHackathon @lablabai
```

## Using This Model in AgentIC

### Option A: Direct Loading (Simplest)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model once
model = AutoModelForCausalLM.from_pretrained(
    "vxkyyy/vlsi-moe-ffn-merged",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("vxkyyy/vlsi-moe-ffn-merged")
tokenizer.pad_token = tokenizer.eos_token

def generate_verilog(spec: str, pdk: str = "sky130", freq_mhz: int = 100) -> str:
    prompt = (
        f"Generate correct, synthesizable Verilog RTL for: {spec}\n"
        f"Target: {pdk} PDK at {freq_mhz}MHz\n\n"
        f"### Verilog RTL\nmodule"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=800,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test it
verilog = generate_verilog("8-bit up counter with synchronous reset")
print(verilog)
```

### Option B: AgentIC Integration

Add to `src/agentic/config.py`:
```python
VLSI_EXPERT_CONFIG = {
    "model": "vxkyyy/vlsi-moe-ffn-merged",
    "base_url": "",
    "api_key": "",
}
```

Then in CLI, use the model:
```bash
agentic build --name counter --desc "8-bit up counter" --model vxkyyy/vlsi-moe-ffn-merged
```

## What's Next After Hackathon

1. QLoRA fine-tune on more Verilog pairs to improve synthesis pass rate
2. Add SPDX/SPEF-aware generation (parasitic extraction context)
3. Support for foundry-specific PDK libraries (TSMC 28, Samsung 14)
4. Integrate as `agentic build --expert` official model
