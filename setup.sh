#!/bin/bash
# ── VLSI Expert Quick Start ──
# Run on AMD VPS: source setup.sh

echo "============================================"
echo "  VLSI Expert — Full Stack Setup"
echo "============================================"

# Activate environment
source ~/vlsi-env/bin/activate

# Pull latest code
cd ~/vlsi-expert && git pull

# Install deps (idempotent)
pip install vllm agentic-ic -q 2>/dev/null

# Start vLLM with your model
echo ""
echo "[1/2] Starting vLLM server..."
python scripts/serve.py &
sleep 5

# Set AgentIC env vars to use your model
export LLM_BASE_URL=http://localhost:7860/v1
export LLM_MODEL=vlsi-expert
export LLM_API_KEY=agentic-vlsi-expert-secure
export OPENLANE_ROOT=/root/vlsi-expert
export SKIP_OPENLANE=1

echo ""
echo "[2/2] Ready!"
echo ""
echo "  Test:  python scripts/chip.py '8-bit counter'"
echo "  Build: agentic build --name counter --desc '8-bit counter' --skip-openlane"
echo "============================================"
