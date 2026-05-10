---
title: vlsi-moe-yarn
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: true
license: mit
tags:
- vlsi
- chip-design
- eda
- amd
- rocm
- agentic
---

# vlsi-moe-yarn · SiliconSmith AI

A domain-specialized language model for **VLSI and chip design reasoning**,
built on a custom Qwen-based architecture with reasoning-optimized FFN layers
and YaRN 262K-token context extension. The backbone model powering the
**AgentIC** chip design agentic pipeline.

---

## What This Model Does

This model is not a general assistant. It is purpose-built to reason about:

| Domain | Examples |
|---|---|
| RTL Design | Verilog/SystemVerilog review, bug detection, design patterns |
| Architecture | NoC topology, pipeline microarchitecture, memory hierarchy |
| Timing & CDC | Clock-domain crossing analysis, synchronizer design, hold/setup violations |
| Constraints | SDC/UPF generation, floorplan guidance, power intent |
| Documentation | Q&A over full datasheets and specifications (up to 262K tokens) |

---

## How This Model Is Used — The AgentIC Pipeline

This model serves as the **reasoning engine** inside
[AgentIC](https://github.com/Vickyrrrrrr/AgentIC), a multi-agent
orchestration framework for chip design. Each agent in the pipeline calls
`vlsi-moe-yarn` via an OpenAI-compatible API for its specific sub-task.

### Pipeline Flow

```
User Task (e.g. "Design a RISC-V ALU with testbench")
        │
        ▼
┌─────────────────────────────────────────┐
│           AgentIC Orchestrator          │
│        (orchestrator.py)                │
└────────────────┬────────────────────────┘
                 │ routes to agents
    ┌────────────┼──────────────────┐
    ▼            ▼                    ▼
architect.py  designer.py        sdc_agent.py
(block-level  (RTL Verilog/      (SDC timing
 planning)     SV generation)    constraints)
    │            │                    │
    └────────────┼──────────────────┘
                 ▼
        testbench_designer.py
        (testbench generation)
                 │
                 ▼
           verifier.py
        (functional verification)
                 │
                 ▼
           doc_agent.py
        (design documentation)
                 │
                 ▼
         Final Design Package
```

Every agent above calls `vlsi-moe-yarn` as its LLM backend.

### Agent Roles

| Agent | File | What it uses this model for |
|---|---|---|
| **Architect** | `architect.py` | High-level block decomposition and architecture decisions |
| **Designer** | `designer.py` | RTL code generation in Verilog/SystemVerilog |
| **SDC Agent** | `sdc_agent.py` | Timing constraint generation and clock definition |
| **Testbench Designer** | `testbench_designer.py` | Automated testbench and stimulus generation |
| **Verifier** | `verifier.py` | Functional verification reasoning and coverage analysis |
| **Doc Agent** | `doc_agent.py` | Design documentation and specification writing |

The **Orchestrator** (`orchestrator.py`) manages task decomposition,
agent routing, context passing, and result aggregation. It uses the
262K-token context window to carry full design state across agents
without losing earlier decisions.

---

## Why This Model for AgentIC

Standard general-purpose LLMs fail at chip design tasks because:
- They lack RTL-specific vocabulary and design rule knowledge
- They lose coherence on long design specs (most cap at 8K–32K tokens)
- They hallucinate plausible but incorrect timing numbers and constraints

`vlsi-moe-yarn` addresses all three:
- **Trained on VLSI domain data** via knowledge distillation
- **262,144 token context** via YaRN — holds an entire SoC spec in one pass
- **Reasoning FFN layers** — 10% of FFN blocks replaced with reasoning-optimized variants for deeper logical inference on structured hardware problems

---

## Connect to the Model

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://YOUR_SERVER:8000/v1",
    api_key="EMPTY",
)

response = client.chat.completions.create(
    model="vlsi-moe-yarn",
    messages=[
        {"role": "system", "content": "You are a VLSI design expert specializing in RTL design and chip architecture."},
        {"role": "user",   "content": "Generate a parameterized synchronous FIFO in SystemVerilog with configurable depth and width."}
    ],
    max_tokens=1024,
)
print(response.choices[0].message.content)
```

---

## Model Architecture

| Property | Value |
|---|---|
| Base | Qwen encoder-decoder backbone |
| Modification | 10% of FFN layers → reasoning-optimized FFN blocks |
| Context extension | YaRN (Yet another RoPE extensioN) |
| Max context | 262,144 tokens |
| dtype | bfloat16 · fp8 KV cache |
| Training | Knowledge distillation on VLSI-domain corpus |
| Inference engine | vLLM v0.17.1 on AMD Instinct MI300X |

---

## Links

- ⚙️ **AgentIC pipeline:** [github.com/Vickyrrrrrr/AgentIC](https://github.com/Vickyrrrrrr/AgentIC)
- 🔬 **vlsi-expert repo:** [github.com/Vickyrrrrrr/vlsi-expert](https://github.com/Vickyrrrrrr/vlsi-expert)
- 🏆 **Hackathon:** [AMD Developer Hackathon 2026](https://lablab.ai/ai-hackathons/amd-developer)

---

## License

MIT
