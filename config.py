import os
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
SCRATCH_DIR = Path(os.environ.get("VLSI_SCRATCH", "/scratch"))
CHECKPOINT_DIR = SCRATCH_DIR / "checkpoints"
DATASET_DIR = SCRATCH_DIR / "datasets"
EXPERT_DISTILLED_DIR = MODELS_DIR / "qwen-vlsi-sota-14b-ternary"

TEACHER_MODEL_ID = "vxkyyy/vlsi-moe-ffn-merged-formal"
STUDENT_MODEL_ID = "Qwen/Qwen2.5-Coder-14B-Instruct"
DRAFT_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

TEACHER_PORT = int(os.environ.get("VLSI_TEACHER_PORT", "8000"))
DRAFT_PORT = int(os.environ.get("VLSI_DRAFT_PORT", "8001"))
BRIDGE_PORT = int(os.environ.get("VLSI_BRIDGE_PORT", "8002"))

TEACHER_URL = f"http://localhost:{TEACHER_PORT}/v1"
DRAFT_URL = f"http://localhost:{DRAFT_PORT}/v1"

API_KEY = os.environ.get("VLSI_API_KEY", "agentic-vlsi-expert-secure")

MAX_REFACTOR_RETRIES = 3
GENERATION_MAX_TOKENS = 2048
GENERATION_TEMPERATURE = 0.2

VERILATOR_ARGS = ["--lint-only", "-Wall"]
IVERILOG_ARGS = ["-g2012"]
YOSYS_FORMAL_SCRIPT = "read_verilog -sv design.sv; prep -top {top}; check; write_smt2 -bv -stbv -wires design.smt2"
Z3_TIMEOUT_MS = 30000

DISTILL_BATCH_SIZE = int(os.environ.get("DISTILL_BATCH_SIZE", "2"))
DISTILL_GRAD_ACCUM = int(os.environ.get("DISTILL_GRAD_ACCUM", "16"))
DISTILL_LR = float(os.environ.get("DISTILL_LR", "2e-5"))
DISTILL_KD_TEMP = float(os.environ.get("DISTILL_KD_TEMP", "4.0"))
DISTILL_KD_ALPHA = float(os.environ.get("DISTILL_KD_ALPHA", "0.9"))
GALORE_RANK = int(os.environ.get("GALORE_RANK", "128"))
GALORE_UPDATE_PROJ_GAP = int(os.environ.get("GALORE_UPDATE_PROJ_GAP", "200"))

PHASE1_HOURS = 6
PHASE2_HOURS = 18
PHASE3_HOURS = 24

CONCURRENT_GENERATORS = int(os.environ.get("CONCURRENT_GENERATORS", "16"))

for d in [DATA_DIR, MODELS_DIR, SCRATCH_DIR, CHECKPOINT_DIR, DATASET_DIR, EXPERT_DISTILLED_DIR]:
    d.mkdir(parents=True, exist_ok=True)
