---
license: apache-2.0
library_name: transformers
tags:
- verilog
- vlsi
- chip-design
- eda
- hardware
- rtl
- synthesis
- asic
- fpga
- qwen
- deepseek
- dare
- ties
- moe
- mixture-of-experts
- amd
- rocm
- mi300x
datasets:
- shailja/Verilog_GitHub
pipeline_tag: text-generation
language:
- en
- verilog
metrics:
- pass@1
- synthesis_pass_rate
model-index:
- name: VLSI Expert — FFN-Merged MoE
  results:
  - task:
      type: text-generation
      name: Verilog RTL Generation
    dataset:
      name: VerilogEval
      type: shailja/Verilog_GitHub
    metrics:
    - type: synthesis_pass_rate
      value: pending
      name: Synthesis Pass Rate
widget:
- text: "8-bit up counter with synchronous reset and enable"
  example_title: Counter
- text: "UART transmitter at 115200 baud rate, 8 bits, no parity, 1 stop bit"
  example_title: UART
- text: "16-bit pipelined adder with 2 pipeline stages"
  example_title: Pipelined Adder
---

# VLSI Expert — AI Chip Designer (MoE)

**FFN-merged Mixture-of-Experts model for hardware chip design.**

Two models merged via **DARE+TIES** on FFN layers only:
- **Qwen2.5-Coder-32B-Instruct** — generates Verilog RTL (keeps attention + generation)
- **DeepSeek-R1-Distill-Qwen-32B** — adds chain-of-thought reasoning + error analysis (FFN knowledge only)

## Architecture

```
User: "5-stage RISC-V pipeline"

  ┌─────────────────────────────────────────────┐
  │        Qwen2.5-Coder (Attention: 40 heads)  │ ← Generates code
  │                  +                          │
  │   DeepSeek-R1 (FFN layers: reasoning)       │ ← Analyzes, fixes
  └─────────────────────────────────────────────┘
                         │
                    Verilog + SDC + Analysis

One model, both capabilities. FFN-only merge prevents attention head incompatibility.
```

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "vxkyyy/vlsi-moe-ffn-merged",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("vxkyyy/vlsi-moe-ffn-merged")

spec = "8-bit up counter with synchronous reset"
inputs = tokenizer(spec + "\nmodule", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=300)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Details

| Detail | Value |
|--------|-------|
| Base model | Qwen2.5-Coder-32B-Instruct |
| Knowledge donor | DeepSeek-R1-Distill-Qwen-32B |
| Merge method | DARE (90% delta dropout) + TIES (sign voting) |
| Merge scope | FFN layers only (gate_proj, up_proj, down_proj) |
| FFN layers merged | 192 of 771 |
| Attention layers kept from Coder | 579 of 771 (no compatibility issues) |
| Training data | 641 Verilog pairs from shailja/Verilog_GitHub |
| Hardware | AMD Instinct MI300X (192 GB VRAM) |
| Software stack | ROCm 7.2, vLLM 0.17.1, PyTorch, HuggingFace Transformers |

## Technical Novelty

1. **FFN-only knowledge transfer:** Unlike standard weight averaging, we merge only FFN layers where domain expertise lives. Attention layers stay untouched — no generation crashes.

2. **EDA-verified training data:** All training pairs pass Icarus Verilog syntax validation. The model learns from real, working chip designs.

3. **CoT reasoning capability:** DeepSeek-R1's chain-of-thought reasoning is preserved in FFN layers. The model analyzes designs before generating code.

## Performance

Evaluated through [AgentIC](https://github.com/Vickyrrrrrr/AgentIC) — a 27-stage autonomous RTL-to-GDSII pipeline.

| Design | Counter | UART | Pipelined Adder | ALU | SPI |
|--------|---------|------|-----------------|-----|-----|
| Syntax | ✅ | ✅ | ✅ | ✅ | ✅ |
| Simulation | pending | pending | pending | pending | pending |

## Built For

AMD Developer Hackathon 2026 — Track 1: AI Agents & Agentic Workflows

## License

Apache 2.0 (same as base models). Training data from shailja/Verilog_GitHub (MIT).
