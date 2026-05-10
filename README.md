# 🔬 SiliconSmith AI — Agentic VLSI Copilot on AMD MI300X

> An agentic chip-design assistant powered by a custom long-context model, served with vLLM on AMD Instinct MI300X via ROCm. Built for the [AMD Developer Hackathon](https://lablab.ai/ai-hackathons/amd-developer) — **AI Agents & Agentic Workflows** track.

---

## 🧠 What It Does

SiliconSmith AI is an **agentic VLSI copilot** that helps chip designers reason through complex problems:

- 🏗️ Architecture exploration and block-level design
- 🔌 RTL understanding and code explanation
- ⚡ Clock-domain crossing, power, and timing analysis
- 📐 Constraint reasoning (SDC, UPF, floorplan)
- 📄 Long-context design document Q&A (up to 262,144 tokens)
- 🤖 Agentic tool use: retrieval, memory, multi-step reasoning

---

## ⚡ Why AMD

This project is built end-to-end on the **AMD AI stack**:

| Component | Technology |
|---|---|
| Hardware | AMD Instinct MI300X (192GB HBM3) |
| Software | ROCm + vLLM OpenAI-compatible server |
| Platform | AMD Developer Cloud (DigitalOcean Droplet, 1-click vLLM) |
| Inference | vLLM `v0.17.1` with `bfloat16` + `fp8` KV cache |
| Context | 262,144 tokens max context length |

We don't just _use_ AMD hardware — we built around it. The entire serving stack, model configuration, and agentic access pattern is optimized for MI300X capabilities.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Developer PC                        │
│  ┌─────────────┐    SSH Tunnel     ┌─────────────────┐  │
│  │ Python Agent│◄─ :8000 forward ─►│  Local :8000    │  │
│  │  / CLI App  │                   └────────┬────────┘  │
└──┴─────────────┴───────────────────────────┼────────────┘
                                             │
                           ┌─────────────────▼──────────────────┐
                           │        AMD Developer Cloud          │
                           │   DigitalOcean vLLM Droplet         │
                           │                                     │
                           │  ┌──────────────────────────────┐  │
                           │  │     Docker Container (rocm)  │  │
                           │  │                              │  │
                           │  │  vLLM Serve (port 8000)      │  │
                           │  │  Model: vlsi-moe-yarn        │  │
                           │  │  Max context: 262,144 tokens │  │
                           │  │  dtype: bfloat16             │  │
                           │  │  KV cache: fp8               │  │
                           │  └──────────────────────────────┘  │
                           │                                     │
                           │  ┌──────────────────────────────┐  │
                           │  │  AMD Instinct MI300X GPU     │  │
                           │  │  192GB HBM3 Memory           │  │
                           │  │  ROCm Software Stack         │  │
                           │  └──────────────────────────────┘  │
                           └─────────────────────────────────────┘
```

---

## 🚀 Quickstart

### 1. Start the vLLM server on the AMD box

```bash
# Enter the ROCm Docker container
docker exec -it rocm /bin/bash
cd /app

# Set AMD/ROCm environment
export IFACE=eth0
export GLOO_SOCKET_IFNAME="$IFACE"
export NCCL_SOCKET_IFNAME="$IFACE"
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
export VLLM_ROCM_USE_AITER=1

# Launch vLLM OpenAI-compatible server
vllm serve /app/vlsi-moe-yarn \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --tensor-parallel-size 1 \
  --host 0.0.0.0 \
  --port 8000
```

### 2. Open the SSH tunnel from your PC

```bash
# In a terminal on your local machine — keep this window open
ssh -L 8000:127.0.0.1:8000 root@YOUR_DROPLET_IP
```

### 3. Verify the connection

```bash
curl http://127.0.0.1:8000/v1/models
# Should return JSON with "/app/vlsi-moe-yarn"
```

### 4. Run the VLSI agent client

```bash
pip install openai
python serving/test_client.py
```

---

## 📁 Repository Structure

```
vlsi-expert/
├── README.md                  # This file
├── serving/
│   ├── launch.sh              # One-command vLLM server launch
│   └── test_client.py         # Minimal OpenAI-compatible client demo
├── docs/
│   ├── architecture.md        # Full system design + AMD stack details
│   └── demo.md                # Example prompts + expected outputs
├── app.py                     # Main agent application
├── bridge.py                  # Model bridge / API routing
├── chip.py                    # Chip design task tools
├── factory.py                 # Agent factory / orchestration
├── distill.py                 # Knowledge distillation pipeline
├── config.py                  # Configuration
├── requirements.txt           # Python dependencies
└── setup.sh                   # Environment setup script
```

---

## 🤖 Example Prompts

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)

# Example 1: Architecture reasoning
resp = client.chat.completions.create(
    model="/app/vlsi-moe-yarn",
    messages=[{
        "role": "user",
        "content": "Explain the tradeoffs between synchronous and asynchronous FIFO design for a 200MHz → 400MHz clock domain crossing in a NoC."
    }],
    max_tokens=512,
)
print(resp.choices[0].message.content)

# Example 2: RTL review
resp = client.chat.completions.create(
    model="/app/vlsi-moe-yarn",
    messages=[{
        "role": "user",
        "content": "Review this Verilog module for timing violations and suggest fixes: [paste RTL here]"
    }],
    max_tokens=1024,
)
print(resp.choices[0].message.content)
```

---

## 🏆 Hackathon Track

This project is submitted to the **AMD Developer Hackathon** ([lablab.ai/ai-hackathons/amd-developer](https://lablab.ai/ai-hackathons/amd-developer)).

**Primary Track:** AI Agents & Agentic Workflows  
**Secondary:** AMD GPU / ROCm Inference Stack  
**Prize Path:** Build in Public (`#AMDDevHackathon`)

**Why this project matters:**
- Chip design is one of the most knowledge-intensive fields — an agent with 262K context can hold an entire design specification in memory
- Built entirely on AMD infrastructure to show what the MI300X stack enables
- Long-context MoE model + ROCm + vLLM = production-grade VLSI reasoning at scale

---

## 📊 Performance

| Metric | Value |
|---|---|
| Max context length | 262,144 tokens |
| Model dtype | bfloat16 |
| KV cache | fp8 (memory efficient) |
| Hardware | AMD Instinct MI300X, 192GB HBM3 |
| Inference engine | vLLM v0.17.1 |
| API compatibility | OpenAI-compatible (`/v1/chat/completions`) |

---

## 🛠️ Tech Stack

- **AMD Instinct MI300X** — GPU hardware
- **ROCm** — AMD GPU software stack
- **vLLM** — High-throughput LLM serving
- **Docker** — Containerized deployment
- **Python + OpenAI SDK** — Agent client
- **SSH tunneling** — Secure local access to remote model

---

## 📝 License

MIT License — See [LICENSE](LICENSE) for details.

---

<p align="center">
Built with ❤️ on AMD MI300X for the AMD Developer Hackathon 2026
</p>
