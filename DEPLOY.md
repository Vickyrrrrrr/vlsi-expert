# AMD GPU Deployment Guide

## Overview

Deploy VLSI Expert on **AMD Developer Cloud** (or any MI300X VPS) and call it from your local machine.

```
┌─────────────────────┐      HTTP/API      ┌──────────────────────────┐
│ Your Laptop / WSL   │  ───────────────►  │ AMD MI300X VPS           │
│ AgentIC, chip.py    │    API Key         │ Model Server (Port 8000) │
│ Local .env          │                    │ ROCm + PyTorch           │
└─────────────────────┘                    └──────────────────────────┘
```

---

## Option A: AMD Developer Cloud (Recommended)

### 1. Launch Instance

From [AMD Developer Cloud](https://www.amd.com/en/developer/resources/amd-developer-cloud.html):
- **OS**: Ubuntu 22.04
- **GPU**: 1× AMD Instinct MI300X (192GB VRAM)
- **Disk**: 720GB NVMe
- **ROCm**: Pre-installed 6.2

### 2. One-Command Setup

```bash
git clone https://github.com/Vickyrrrrrr/vlsi-expert.git
cd vlsi-expert
chmod +x setup.sh
./setup.sh
```

This handles everything: venv, ROCm PyTorch, model download (~66GB), server start.

### 3. Verify Server

```bash
curl http://localhost:8000/health
# → {"status":"ok","model_loaded":true,"backend":"fastapi+transformers"}
```

### 4. Expose Port (if needed)

If you want to call from outside without SSH tunnel:
```bash
# Open firewall (careful!)
sudo ufw allow 8000/tcp
# Or use cloud provider security group
```

**Better**: Use SSH tunnel (see below).

---

## Option B: Your Own MI300X Server

If you have direct access to MI300X hardware:

```bash
# 1. Install ROCm (if not present)
# https://rocm.docs.amd.com/projects/install-on-linux/

# 2. Install PyTorch for ROCm
pip install torch==2.5.1+rocm6.2 torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# 3. Clone and setup
git clone https://github.com/Vickyrrrrrr/vlsi-expert.git
cd vlsi-expert
./setup.sh
```

---

## Connect from Local Machine

### Method 1: SSH Tunnel (Secure)

```bash
# Terminal 1: Create tunnel
ssh -N -L 8000:localhost:8000 -i ~/.ssh/id_ed25519 ubuntu@YOUR_VPS_IP

# Terminal 2: Use localhost
export VLSI_EXPERT_HOST=localhost
export VLSI_EXPERT_PORT=8000
python scripts/chip.py "8-bit counter"
```

### Method 2: Direct HTTP

```bash
export VLSI_EXPERT_HOST=YOUR_VPS_IP
export VLSI_EXPERT_PORT=8000
export VLSI_EXPERT_KEY=agentic-vlsi-expert-secure
python scripts/chip.py "8-bit counter"
```

### Method 3: AgentIC Pipeline

Create `~/.env` or export:
```bash
export LLM_BASE_URL=http://YOUR_VPS_IP:8000/v1
export LLM_API_KEY=agentic-vlsi-expert-secure
export LLM_MODEL=vlsi-expert
export SKIP_OPENLANE=1
```

Then:
```bash
agentic build --name counter --desc "8-bit up counter with synchronous reset" --skip-openlane
```

---

## Server Options

| Server | Command | Speed | When to Use |
|--------|---------|-------|-------------|
| **Auto-detect** | `python scripts/serve.py` | Best available | Default |
| **vLLM** | `python scripts/serve_vllm.py --local` | 10× faster | If vLLM ROCm works |
| **FastAPI** | `python scripts/serve_fastapi.py --local` | 5-10s/token | Reliable fallback |

---

## Monitoring

```bash
# GPU usage
watch -n 1 rocm-smi

# Server logs
tail -f server.log

# Test endpoint
python scripts/serve.py --test
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Out of memory" on model load | Model is 66GB. Ensure MI300X has free VRAM. Kill other GPU processes. |
| vLLM won't start | ROCm vLLM binary may be missing. Fallback: `python scripts/serve_fastapi.py` |
| "Invalid API key" | Check `VLSI_API_KEY` env var matches on both client and server |
| Connection refused | Ensure server is running: `curl http://VPS_IP:8000/health` |
| Download too slow | Model is 66GB. Use `screen` or `tmux` so download survives disconnect. |
| Firewall blocks port | Use SSH tunnel instead of opening ports |

---

## Cost

- **AMD Developer Cloud**: $100 free credits (covers ~40 hours of MI300X)
- **HuggingFace Hub**: Free model hosting
- **Total hackathon cost**: $0

---

## Security Notes

- Change default API key: `export VLSI_API_KEY=your-secure-key`
- Use SSH tunnel instead of public ports when possible
- The server binds to `0.0.0.0` — restrict with firewall rules if publicly exposed
