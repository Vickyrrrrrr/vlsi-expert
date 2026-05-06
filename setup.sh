#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  VLSI Expert — One-Command Setup for New AMD VPS
# ═══════════════════════════════════════════════════════════════════
#
#  Run this on your fresh AMD MI300X VPS (Ubuntu 22.04 + ROCm 6.2+)
#
#  What it does:
#    1. Creates Python venv
#    2. Installs ROCm PyTorch + dependencies
#    3. Downloads model from HuggingFace (~66 GB)
#    4. Starts API server (vLLM or FastAPI fallback)
#
#  Usage:
#    chmod +x setup.sh
#    ./setup.sh
#
#  Or step-by-step:
#    ./setup.sh --venv-only
#    ./setup.sh --download
#    ./setup.sh --serve
# ═══════════════════════════════════════════════════════════════════

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Config ─────────────────────────────────────────────────────────
VENV_DIR="${HOME}/vlsi-env"
MODEL_ID="${HF_MODEL:-vxkyyy/vlsi-moe-ffn-merged-formal}"
LOCAL_MODEL_DIR="models/vlsi-moe-ffn-merged-formal"
PORT="${VLSI_PORT:-8000}"
API_KEY="${VLSI_API_KEY:-agentic-vlsi-expert-secure}"

# ── Colors ─────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log()  { echo -e "${BLUE}[SETUP]${NC} $1"; }
ok()   { echo -e "${GREEN}[OK]${NC}  $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERR]${NC}  $1"; }

# ── Step 0: Detect GPU ─────────────────────────────────────────────
log "Detecting AMD GPU..."
if command -v rocm-smi &> /dev/null; then
    GPU=$(rocm-smi --showproductname | grep -v "=" | head -1 | xargs)
    ok "GPU detected: $GPU"
else
    err "rocm-smi not found. Is ROCm installed?"
    echo "   Install: https://rocm.docs.amd.com/projects/install-on-linux/"
    exit 1
fi

# ── Step 1: Create venv ────────────────────────────────────────────
setup_venv() {
    log "Setting up Python virtual environment..."
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
        ok "Created venv at $VENV_DIR"
    else
        ok "Venv already exists at $VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip setuptools wheel -q
    ok "Venv activated"
}

# ── Step 2: Install dependencies ───────────────────────────────────
install_deps() {
    log "Installing dependencies..."
    source "$VENV_DIR/bin/activate"

    # ROCm PyTorch (pre-installed on AMD Developer Cloud images)
    python -c "import torch; assert torch.cuda.is_available(), 'ROCm PyTorch not found'" 2>/dev/null && {
        ok "ROCm PyTorch already installed"
    } || {
        warn "Installing ROCm PyTorch..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
    }

    # Core ML stack
    pip install -q transformers accelerate bitsandbytes peft datasets trl huggingface_hub

    # Server stack
    pip install -q fastapi uvicorn pydantic requests

    # vLLM (optional — may fail on ROCm, we handle it)
    pip install -q vllm 2>/dev/null && ok "vLLM installed" || warn "vLLM install failed (will use FastAPI fallback)"

    ok "Dependencies installed"
}

# ── Step 3: Download model ─────────────────────────────────────────
download_model() {
    log "Downloading model from HuggingFace Hub..."
    log "Model: $MODEL_ID"
    log "This will download ~66 GB. Time: 20-60 minutes."
    source "$VENV_DIR/bin/activate"
    python scripts/download_model.py --model "$MODEL_ID" --local-dir "$LOCAL_MODEL_DIR"
    ok "Model downloaded"
}

# ── Step 4: Start server ───────────────────────────────────────────
start_server() {
    source "$VENV_DIR/bin/activate"
    export VLSI_PORT="$PORT"
    export VLSI_API_KEY="$API_KEY"

    log "Starting server on port $PORT..."

    # Try vLLM first
    if python -c "import vllm" 2>/dev/null; then
        ok "Using vLLM (production)"
        echo ""
        echo "═════════════════════════════════════════════════════════════"
        echo "  VLSI Expert Server Starting (vLLM)"
        echo "  Endpoint: http://0.0.0.0:$PORT/v1/chat/completions"
        echo "  API Key:  $API_KEY"
        echo "═════════════════════════════════════════════════════════════"
        echo ""
        python scripts/serve_vllm.py --local --port "$PORT"
    else
        warn "vLLM not available, falling back to FastAPI + transformers"
        echo ""
        echo "═════════════════════════════════════════════════════════════"
        echo "  VLSI Expert Server Starting (FastAPI Fallback)"
        echo "  Endpoint: http://0.0.0.0:$PORT/v1/chat/completions"
        echo "  API Key:  $API_KEY"
        echo "═════════════════════════════════════════════════════════════"
        echo ""
        python scripts/serve_fastapi.py --local --port "$PORT"
    fi
}

# ── Main ───────────────────────────────────────────────────────────
MODE="${1:-all}"

case "$MODE" in
    --venv-only)
        setup_venv
        ;;
    --deps)
        setup_venv
        install_deps
        ;;
    --download)
        setup_venv
        download_model
        ;;
    --serve)
        start_server
        ;;
    all|*)
        echo ""
        echo "╔═══════════════════════════════════════════════════════════════╗"
        echo "║     VLSI Expert — Full Setup for AMD MI300X VPS              ║"
        echo "╚═══════════════════════════════════════════════════════════════╝"
        echo ""
        setup_venv
        install_deps
        if [ ! -d "$LOCAL_MODEL_DIR" ]; then
            download_model
        else
            ok "Model already exists at $LOCAL_MODEL_DIR"
        fi
        start_server
        ;;
esac
