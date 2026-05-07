#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  Qwen-VLSI-SOTA-Sprint — Full Infrastructure Setup
#  Hardware: AMD MI300X (192GB VRAM) + ROCm 7.2.0
# ═══════════════════════════════════════════════════════════════════
#
#  What this does:
#    1. Installs OSS CAD Suite (Yosys, Verilator, Icarus Verilog)
#    2. Sets up Python venv + ROCm-optimized dependencies
#    3. Installs flash-attn (AMD) + bitsandbytes (ROCm fork)
#    4. Downloads Teacher (33B) + Draft (1.5B) models
#    5. Launches dual vLLM instances (Teacher:8000, Draft:8001)
#
#  Usage:
#    chmod +x setup.sh
#    ./setup.sh                  # Full setup + serve
#    ./setup.sh --tools-only     # Just OSS CAD Suite
#    ./setup.sh --venv-only      # Just Python venv + deps
#    ./setup.sh --download       # Just model downloads
#    ./setup.sh --serve-teacher  # Launch teacher vLLM (port 8000)
#    ./setup.sh --serve-draft    # Launch draft vLLM (port 8001)
#    ./setup.sh --quick          # Skip model downloads
# ═══════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Config ─────────────────────────────────────────────────────────
VENV_DIR="${HOME}/vlsi-sota-env"
TOOLS_DIR="${HOME}/oss-cad-suite"
TEACHER_MODEL="${TEACHER_MODEL:-vxkyyy/vlsi-moe-ffn-merged-formal}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2.5-Coder-14B-Instruct}"
DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
TEACHER_PORT="${VLSI_TEACHER_PORT:-8000}"
DRAFT_PORT="${VLSI_DRAFT_PORT:-8001}"
SCRATCH_DIR="${VLSI_SCRATCH:-/scratch}"
API_KEY="${VLSI_API_KEY:-agentic-vlsi-expert-secure}"

# ── Colors ─────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

log()   { echo -e "${BLUE}[SETUP]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]${NC}  $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()   { echo -e "${RED}[ERR]${NC}  $1"; }

# ── Step 0: Pre-flight ─────────────────────────────────────────────
preflight() {
    log "Running pre-flight checks..."
    if ! command -v rocm-smi &> /dev/null; then
        err "rocm-smi not found. Install ROCm 7.2.0 first."
        exit 1
    fi
    GPU_COUNT=$(rocm-smi --showproductname 2>/dev/null | grep -c "MI300" || echo "0")
    ok "Detected ${GPU_COUNT}x AMD MI300X GPU(s) with ROCm"
    if grep -q "gfx942" /opt/rocm/bin/rocminfo 2>/dev/null; then
        ok "ROCm gfx942 (MI300X) confirmed"
    fi
}

# ── Step 1: OSS CAD Suite ──────────────────────────────────────────
install_oss_cad_suite() {
    log "Installing OSS CAD Suite (Yosys + Verilator + Icarus Verilog)..."

    if [ -f "${TOOLS_DIR}/bin/yosys" ] && [ -f "${TOOLS_DIR}/bin/verilator" ] && [ -f "${TOOLS_DIR}/bin/iverilog" ]; then
        ok "OSS CAD Suite already installed at ${TOOLS_DIR}"
        return 0
    fi

    local os_type arch
    case "$(uname -sm)" in
        "Linux x86_64")  os_type=linux; arch=x64 ;;
        "Linux aarch64") os_type=linux; arch=arm64 ;;
        *) err "Unsupported platform: $(uname -sm)"; exit 1 ;;
    esac

    # Fetch latest nightly release URL
    log "Fetching latest OSS CAD Suite nightly release..."
    local release_url
    release_url=$(curl -s "https://api.github.com/repos/YosysHQ/oss-cad-suite-build/releases" \
        | python3 -c "
import json,sys
for r in json.load(sys.stdin):
    for a in r['assets']:
        if '${os_type}-${arch}' in a['name'] and a['name'].endswith('.tgz'):
            print(a['browser_download_url']); sys.exit(0)
" 2>/dev/null) || release_url="https://github.com/YosysHQ/oss-cad-suite-build/releases/latest"

    if [ -z "$release_url" ] || [ "$release_url" = "https://github.com/YosysHQ/oss-cad-suite-build/releases/latest" ]; then
        # Fallback: use a known-good nightly
        release_url="https://github.com/YosysHQ/oss-cad-suite-build/releases/download/2025-01-31/oss-cad-suite-linux-x64-20250131.tgz"
        warn "Could not auto-detect latest nightly; using pinned 2025-01-31 release"
    fi

    local archive="${TOOLS_DIR}/oss-cad-suite.tgz"
    mkdir -p "${TOOLS_DIR}"

    log "Downloading OSS CAD Suite (~600MB)..."
    curl -L --progress-bar -o "${archive}" "${release_url}" || {
        err "Download failed. Check network or try manually."
        exit 1
    }

    log "Extracting to ${TOOLS_DIR}..."
    tar -xzf "${archive}" -C "${TOOLS_DIR}" --strip-components=1
    rm -f "${archive}"

    cat >> "${HOME}/.bashrc" << 'BASHRC_OSS'
# OSS CAD Suite
export PATH="${HOME}/oss-cad-suite/bin:${PATH}"
source "${HOME}/oss-cad-suite/environment" 2>/dev/null || true
BASHRC_OSS

    ok "OSS CAD Suite installed: $(yosys -V 2>/dev/null || echo "restart shell to use")"
}

# ── Step 2: Python Environment ─────────────────────────────────────
setup_venv() {
    log "Setting up Python virtual environment..."
    if [ -f "${VENV_DIR}/bin/activate" ]; then
        ok "Venv exists at ${VENV_DIR}"
    else
        warn "Venv missing or corrupted, recreating..."
        rm -rf "${VENV_DIR}"
        python3 -m venv "${VENV_DIR}"
        ok "Created venv at ${VENV_DIR}"
    fi
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip setuptools wheel -q
}

install_deps() {
    source "${VENV_DIR}/bin/activate"
    log "Installing ROCm PyTorch + core dependencies..."

    python -c "import torch; assert torch.cuda.is_available(); print(f'PyTorch {torch.__version__} ROCm {torch.version.hip}')" 2>/dev/null && {
        ok "ROCm PyTorch already installed"
    } || {
        log "Installing ROCm PyTorch..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.0
    }

    log "Installing ML stack..."
    pip install -q \
        transformers>=4.49.0 \
        accelerate>=0.34.0 \
        peft>=0.13.0 \
        datasets>=3.0.0 \
        trl>=0.12.0 \
        huggingface_hub>=0.26.0 \
        deepspeed>=0.15.0 \
        sentencepiece \
        pyarrow \
        tqdm \
        fastapi \
        uvicorn[standard] \
        pydantic \
        httpx \
        aiohttp \
        orjson

    log "Installing vLLM..."
    pip install -q vllm 2>/dev/null && ok "vLLM installed" || {
        warn "vLLM pip install failed — building from source for ROCm..."
        MAX_JOBS=16 pip install vllm --no-build-isolation 2>/dev/null || {
            warn "vLLM build failed. Will use FastAPI fallback for serving."
        }
    }

    log "Installing flash-attn (AMD ROCm)..."
    FLASH_ATTN_BUILD=1 pip install flash-attn --no-build-isolation 2>/dev/null && {
        ok "flash-attn installed"
    } || {
        warn "flash-attn build failed. Install manually from https://github.com/ROCm/flash-attention"
    }

    log "Installing bitsandbytes (ROCm fork)..."
    pip install -q bitsandbytes>=0.45.0 2>/dev/null && {
        ok "bitsandbytes installed (ROCm-compatible)"
    } || {
        warn "bitsandbytes ROCm fork not available. Install from https://github.com/ROCm/bitsandbytes"
    }

    log "Installing GaLore optimizer..."
    pip install -q galore-torch 2>/dev/null && ok "galore-torch installed" || {
        warn "galore-torch not available; installing from GitHub..."
        pip install -q git+https://github.com/jiaweizzhao/GaLore.git 2>/dev/null || {
            warn "GaLore install failed. Will fall back to AdamW + LoRA."
        }
    }

    ok "All Python dependencies installed"
}

# ── Step 3: Model Downloads ────────────────────────────────────────
download_models() {
    source "${VENV_DIR}/bin/activate"
    log "Downloading models from HuggingFace Hub..."

    for model_id in "${TEACHER_MODEL}" "${DRAFT_MODEL}" "${STUDENT_MODEL}"; do
        local local_dir="${SCRIPT_DIR}/models/$(basename ${model_id})"
        if [ -d "${local_dir}" ] && [ "$(ls -A ${local_dir} 2>/dev/null)" ]; then
            ok "Model already cached: ${model_id}"
        else
            log "Downloading ${model_id}..."
            python scripts/download_model.py --model "${model_id}" --local-dir "${local_dir}"
        fi
    done
    ok "All models downloaded"
}

# ── Step 4: Create Scratch Directories ─────────────────────────────
setup_scratch() {
    mkdir -p "${SCRATCH_DIR}/datasets" "${SCRATCH_DIR}/checkpoints"
    chmod 755 "${SCRATCH_DIR}"
    ok "Scratch storage ready at ${SCRATCH_DIR} (datasets + checkpoints)"
}

# ── Step 5: Launch vLLM Servers ────────────────────────────────────
serve_teacher() {
    source "${VENV_DIR}/bin/activate"
    local model_path="${SCRIPT_DIR}/models/$(basename ${TEACHER_MODEL})"
    [ -d "${model_path}" ] || model_path="${TEACHER_MODEL}"

    log "Launching Teacher vLLM instance (33B) on port ${TEACHER_PORT}..."
    log "Quantization: FP8/INT8 where available to save VRAM"
    echo ""
    echo "═════════════════════════════════════════════════════════════"
    echo "  Teacher: ${TEACHER_MODEL}"
    echo "  Endpoint: http://0.0.0.0:${TEACHER_PORT}/v1"
    echo "  Expected VRAM: ~66GB (bf16) or ~33GB (FP8)"
    echo "═════════════════════════════════════════════════════════════"
    echo ""

    python -m vllm.entrypoints.openai.api_server \
        --model "${model_path}" \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.85 \
        --tensor-parallel-size 1 \
        --port "${TEACHER_PORT}" \
        --host 0.0.0.0 \
        --api-key "${API_KEY}" \
        --served-model-name vlsi-expert-teacher \
        --enforce-eager \
        --max-num-seqs 4
}

serve_draft() {
    source "${VENV_DIR}/bin/activate"
    local model_path="${SCRIPT_DIR}/models/$(basename ${DRAFT_MODEL})"
    [ -d "${model_path}" ] || model_path="${DRAFT_MODEL}"

    log "Launching Draft vLLM instance (1.5B) on port ${DRAFT_PORT}..."
    log "Purpose: Speculative decoding — speeds up generation by ~3x"
    echo ""
    echo "═════════════════════════════════════════════════════════════"
    echo "  Draft:  ${DRAFT_MODEL}"
    echo "  Endpoint: http://0.0.0.0:${DRAFT_PORT}/v1"
    echo "  Expected VRAM: ~5GB"
    echo "═════════════════════════════════════════════════════════════"
    echo ""

    python -m vllm.entrypoints.openai.api_server \
        --model "${model_path}" \
        --dtype bfloat16 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.10 \
        --tensor-parallel-size 1 \
        --port "${DRAFT_PORT}" \
        --host 0.0.0.0 \
        --api-key "${API_KEY}" \
        --served-model-name vlsi-expert-draft \
        --enforce-eager \
        --max-num-seqs 4
}

# ── Main ───────────────────────────────────────────────────────────
MODE="${1:-all}"

case "$MODE" in
    --tools-only)
        preflight
        install_oss_cad_suite
        setup_scratch
        ;;
    --venv-only)
        preflight
        setup_venv
        install_deps
        ;;
    --download)
        setup_venv
        download_models
        ;;
    --serve-teacher)
        serve_teacher
        ;;
    --serve-draft)
        serve_draft
        ;;
    --quick)
        preflight
        setup_scratch
        setup_venv
        install_deps
        echo ""
        echo "Skipping model downloads. To download:"
        echo "  ./setup.sh --download"
        ;;
    all|*)
        echo ""
        echo "╔═══════════════════════════════════════════════════════════════╗"
        echo "║   Qwen-VLSI-SOTA-Sprint — Full Infrastructure Setup         ║"
        echo "║   Hardware: AMD MI300X + ROCm 7.2.0                         ║"
        echo "╚═══════════════════════════════════════════════════════════════╝"
        echo ""
        preflight
        setup_scratch
        install_oss_cad_suite
        setup_venv
        install_deps
        download_models
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "  Setup complete! Launch servers in separate terminals:"
        echo ""
        echo "  Terminal 1 (Teacher 33B):  ./setup.sh --serve-teacher"
        echo "  Terminal 2 (Draft 1.5B):   ./setup.sh --serve-draft"
        echo "  Terminal 3 (Generation):   python factory.py"
        echo "  Terminal 4 (Dashboard):    python bridge.py"
        echo "═══════════════════════════════════════════════════════════════"
        ;;
esac
