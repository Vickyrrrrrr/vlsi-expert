#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  Qwen-VLSI-SOTA-Sprint — Full Infrastructure Setup
#  Hardware: AMD MI300X (192GB VRAM) + ROCm 7.2.0
# ═══════════════════════════════════════════════════════════════════
#
#  What this does:
#    1. Installs OSS CAD Suite (Yosys, Verilator, Icarus Verilog)
#    2. Sets up Python venv + ROCm-optimized dependencies
#    3. Installs vLLM (ROCm wheel) + flash-attn + bitsandbytes
#    4. Downloads Teacher (33B) + Student (14B) + Draft (1.5B) models
#    5. Launches vLLM Teacher (port 8000) and Draft (port 8001)
#
#  Usage:
#    chmod +x setup.sh
#    ./setup.sh                  # Full setup
#    ./setup.sh --tools-only     # Just OSS CAD Suite
#    ./setup.sh --venv-only      # Just Python venv + deps
#    ./setup.sh --download       # Just model downloads
#    ./setup.sh --serve-teacher  # Launch Teacher vLLM (port 8000)
#    ./setup.sh --serve-draft    # Launch Draft vLLM (port 8001)
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
    ok "Detected ${GPU_COUNT}x AMD MI300X GPU(s)"
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

    log "Fetching latest OSS CAD Suite nightly..."
    local release_url
    release_url=$(curl -s "https://api.github.com/repos/YosysHQ/oss-cad-suite-build/releases" \
        | python3 -c "
import json,sys
for r in json.load(sys.stdin):
    for a in r['assets']:
        if '${os_type}-${arch}' in a['name'] and a['name'].endswith('.tgz'):
            print(a['browser_download_url']); sys.exit(0)
" 2>/dev/null) || true

    if [ -z "$release_url" ]; then
        release_url="https://github.com/YosysHQ/oss-cad-suite-build/releases/download/2025-01-31/oss-cad-suite-linux-x64-20250131.tgz"
        warn "Could not auto-detect latest; using pinned 2025-01-31 release"
    fi

    local archive="${TOOLS_DIR}/oss-cad-suite.tgz"
    mkdir -p "${TOOLS_DIR}"

    log "Downloading OSS CAD Suite (~600MB)..."
    curl -L --progress-bar -o "${archive}" "${release_url}" || {
        err "Download failed."
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

    ok "OSS CAD Suite installed — restart shell or: source ~/oss-cad-suite/environment"
}

# ── Step 2: Python Environment ─────────────────────────────────────
setup_venv() {
    log "Setting up Python virtual environment..."
    if [ -f "${VENV_DIR}/bin/activate" ]; then
        ok "Venv exists at ${VENV_DIR}"
    else
        rm -rf "${VENV_DIR}"
        python3 -m venv "${VENV_DIR}"
        ok "Created venv at ${VENV_DIR}"
    fi
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip setuptools wheel -q
}

install_deps() {
    source "${VENV_DIR}/bin/activate"

    # ── ROCm PyTorch ──
    log "Installing PyTorch for ROCm..."
    python -c "import torch; assert torch.cuda.is_available(); print(f'PyTorch {torch.__version__}')" 2>/dev/null && {
        ok "PyTorch ROCm already installed"
    } || {
        pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.0
        python -c "import torch; assert torch.cuda.is_available(), 'PyTorch not seeing GPU'" || {
            err "PyTorch ROCm install failed. GPU not detected."
            exit 1
        }
        ok "PyTorch ROCm installed"
    }

    # ── Core ML stack ──
    log "Installing ML dependencies..."
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
        tqdm

    # ── API stack ──
    pip install -q \
        fastapi \
        uvicorn[standard] \
        pydantic \
        httpx \
        aiohttp \
        orjson

    # ── vLLM (ROCm-native wheel) ──
    log "Installing vLLM (ROCm wheel)..."
    pip install -q vllm \
        --index-url https://download.vllm.ai/wheels/rocm \
        --extra-index-url https://pypi.org/simple 2>/dev/null && {
        ok "vLLM installed (ROCm wheel)"
    } || {
        warn "vLLM ROCm wheel failed. Trying fallback FastAPI mode..."
    }

    # ── flash-attn (AMD) ──
    log "Installing flash-attn..."
    FLASH_ATTN_BUILD=1 pip install flash-attn --no-build-isolation -q 2>/dev/null && {
        ok "flash-attn installed"
    } || {
        warn "flash-attn build failed (non-critical). Install from https://github.com/ROCm/flash-attention"
    }

    # ── bitsandbytes (ROCm) ──
    log "Installing bitsandbytes..."
    pip install -q bitsandbytes>=0.45.0 2>/dev/null && {
        ok "bitsandbytes installed"
    } || {
        warn "bitsandbytes ROCm fork not available (non-critical)"
    }

    # ── GaLore optimizer ──
    log "Installing GaLore..."
    pip install -q galore-torch 2>/dev/null || {
        pip install -q git+https://github.com/jiaweizzhao/GaLore.git 2>/dev/null || {
            warn "GaLore not available. distill.py will use AdamW."
        }
    }

    ok "All Python dependencies installed"
}

# ── Step 2.5: CUDA Ghosting Shim ────────────────────────────────
fix_cuda_ghosting() {
    log "Creating libcuda.so.1 shim for ROCm compatibility..."
    source "${VENV_DIR}/bin/activate"

    local hip_lib
    hip_lib=$(find /opt/rocm/lib -name "libamdhip64.so.6" 2>/dev/null | head -n 1)
    [ -z "$hip_lib" ] && hip_lib=$(find /opt/rocm/lib -name "libamdhip64.so*" 2>/dev/null | head -n 1)

    if [ -z "$hip_lib" ]; then
        warn "libamdhip64.so not found — skipping CUDA shim"
        return 0
    fi

    mkdir -p "${VENV_DIR}/lib/stubs"
    ln -sf "$hip_lib" "${VENV_DIR}/lib/stubs/libcuda.so.1"
    ln -sf "$hip_lib" "${VENV_DIR}/lib/stubs/libcuda.so"

    if ! grep -q "${VENV_DIR}/lib/stubs" "${VENV_DIR}/bin/activate" 2>/dev/null; then
        echo "export LD_LIBRARY_PATH=\"${VENV_DIR}/lib/stubs:\$LD_LIBRARY_PATH\"" >> "${VENV_DIR}/bin/activate"
    fi

    ok "CUDA shim ready — libcuda.so.1 → ${hip_lib}"
}

# ── Step 3: Model Downloads ────────────────────────────────────────
download_models() {
    source "${VENV_DIR}/bin/activate"
    log "Downloading models from HuggingFace Hub..."

    for model_id in "${TEACHER_MODEL}" "${DRAFT_MODEL}"; do
        local local_dir="${SCRIPT_DIR}/models/$(basename ${model_id})"
        if [ -d "${local_dir}" ] && [ "$(ls -A ${local_dir} 2>/dev/null)" ]; then
            ok "Model cached: ${model_id}"
        else
            log "Downloading ${model_id}..."
            python download_model.py --model "${model_id}" --local-dir "${local_dir}"
        fi
    done

    # Student model is optional — only needed for distill.py
    local student_dir="${SCRIPT_DIR}/models/$(basename ${STUDENT_MODEL})"
    if [ -d "${student_dir}" ] && [ "$(ls -A ${student_dir} 2>/dev/null)" ]; then
        ok "Student model cached: ${STUDENT_MODEL}"
    else
        warn "Student model not downloaded. Run ./setup.sh --download to get it."
        warn "Or: python download_model.py --model ${STUDENT_MODEL}"
    fi

    ok "Core models ready"
}

# ── Step 4: Scratch Directories ────────────────────────────────────
setup_scratch() {
    mkdir -p "${SCRATCH_DIR}/datasets" "${SCRATCH_DIR}/checkpoints"
    chmod 755 "${SCRATCH_DIR}"
    ok "Scratch storage ready at ${SCRATCH_DIR}"
}

# ── Step 5: Launch Servers ─────────────────────────────────────────
serve_teacher() {
    source "${VENV_DIR}/bin/activate"

    export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
    export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-9.4.2}"
    export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx942}"
    export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-rocm}"
    export VLLM_ROCM_USE_AITER_FP4BMM=0
    export VLLM_SKIP_P2P_CHECK=1
    export HSA_ENABLE_SDMA=0

    local model_path="${SCRIPT_DIR}/models/$(basename ${TEACHER_MODEL})"
    [ -d "${model_path}" ] || model_path="${TEACHER_MODEL}"

    log "Launching Teacher vLLM (33B) on port ${TEACHER_PORT}..."
    echo ""
    echo "═════════════════════════════════════════════════════════════"
    echo "  Teacher: ${TEACHER_MODEL}"
    echo "  Endpoint: http://0.0.0.0:${TEACHER_PORT}/v1/chat/completions"
    echo "  VRAM: ~66GB (bf16)"
    echo "═════════════════════════════════════════════════════════════"
    echo ""

    python -m vllm.entrypoints.openai.api_server \
        --model "${model_path}" \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.90 \
        --tensor-parallel-size 1 \
        --port "${TEACHER_PORT}" \
        --host 0.0.0.0 \
        --api-key "${API_KEY}" \
        --served-model-name vlsi-expert-teacher \
        --enforce-eager \
        --max-num-seqs 4 || {
        warn "vLLM failed. Falling back to FastAPI server..."
        if [ -f "${SCRIPT_DIR}/serve_teacher.py" ]; then
            python "${SCRIPT_DIR}/serve_teacher.py" --port "${TEACHER_PORT}"
        else
            err "No fallback server found."
            exit 1
        fi
    }
}

serve_draft() {
    source "${VENV_DIR}/bin/activate"

    export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
    export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-9.4.2}"
    export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx942}"
    export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-rocm}"
    export VLLM_ROCM_USE_AITER_FP4BMM=0
    export VLLM_SKIP_P2P_CHECK=1
    export HSA_ENABLE_SDMA=0

    local model_path="${SCRIPT_DIR}/models/$(basename ${DRAFT_MODEL})"
    [ -d "${model_path}" ] || model_path="${DRAFT_MODEL}"

    log "Launching Draft vLLM (1.5B) on port ${DRAFT_PORT}..."
    echo ""
    echo "═════════════════════════════════════════════════════════════"
    echo "  Draft: ${DRAFT_MODEL}"
    echo "  Endpoint: http://0.0.0.0:${DRAFT_PORT}/v1/chat/completions"
    echo "  VRAM: ~5GB"
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
        --max-num-seqs 4 || {
        warn "Draft vLLM failed. This is optional — factory.py uses Teacher only."
    }
}
# ═══════════════════════════════════════════════════════════════════

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
        fix_cuda_ghosting
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
        fix_cuda_ghosting
        echo ""
        echo "Model downloads skipped. To download models:"
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
        fix_cuda_ghosting
        download_models
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "  Setup complete! Launch services in separate terminals:"
        echo ""
        echo "  Terminal 1 (Teacher 33B):  ./setup.sh --serve-teacher"
        echo "  Terminal 2 (Draft 1.5B):   ./setup.sh --serve-draft"
        echo "  Terminal 3 (Generation):   python factory.py"
        echo "  Terminal 4 (Distillation): python distill.py"
        echo "═══════════════════════════════════════════════════════════════"
        ;;
esac
