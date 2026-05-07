# Starting the Full Training Pipeline

**Assumes**: AMD MI300X droplet with ROCm 7.2.0, 192GB VRAM, 240GB RAM, 5TB NVMe scratch.

---

## What Each File Does

| File | Role |
|------|------|
| `setup.sh` | Installs OSS CAD Suite, ROCm deps, models. Everything needed. |
| `factory.py` | Teacher generates RTL+SVA → verilator/iverilog/yosys/z3 verify → refactor on failure → save triplets to Parquet |
| `distill.py` | Loads triplets → KD with KL divergence + GaLore + BitNet QAT → export to GGUF |
| `bridge.py` | Hot-swap API + dashboard. AgentIC switches from 33B→14B transparently. |
| `config.py` | All paths, ports, hyperparams in one place. |

---

## TL;DR — Full Run

```bash
git clone https://github.com/Vickyrrrrrr/vlsi-expert.git
cd vlsi-expert

# Step 1: Install everything (OSS CAD, deps, download models)
./setup.sh

# Step 2: In separate tmux/screen terminals:
#   Terminal A: Teacher vLLM (33B on port 8000)
./setup.sh --serve-teacher
#   Terminal B: Draft vLLM (1.5B on port 8001) — optional, for spec decode
#   ./setup.sh --serve-draft
#   Terminal C: Bridge API (port 8002)
python bridge.py

# Step 3: Generate verified RTL training data
python factory.py
#   → /scratch/datasets/reasoning_triplets.parquet

# Step 4: Distill (you have ~4 hours — run what you can)
python distill.py --phase 1    # Precompute teacher logits
python distill.py --phase 2    # KD distillation (runs until you stop or time expires)
#   → /scratch/checkpoints/phase2_stepN/

# If you get to phase 3:
python distill.py --phase 3    # BitNet ternary QAT

# Export:
python distill.py --export-only
#   → models/qwen-vlsi-sota-14b-ternary/
```

---

## With Only 4 Hours of Compute

**Priority order:**

1. `./setup.sh` — skip model downloads if you already have them (`./setup.sh --quick`)
2. `./setup.sh --serve-teacher` — must have Teacher vLLM running for factory
3. `python factory.py --concurrency 4` — generate as many verified triplets as possible
4. `python distill.py --phase 2` — start KD distillation directly (skip phase 1 precomputation — the teacher forward pass runs on the fly in train_step)

**Phase 1 can be skipped.** When you run phase 2 without phase 1, the teacher forward pass happens inside each training step instead of being precomputed. Slower per step, but works identically.

---

## Minimum Viable Setup (if nothing is installed)

```bash
# Run this and you have everything:
./setup.sh
# (~30 min install + model download)

# Launch teacher:
./setup.sh --serve-teacher &
sleep 60  # wait for model to load

# Start generating:
python factory.py --concurrency 4
# Let it run 1-2 hours to build a decent Parquet file

# Start distilling:
python distill.py --phase 2
# Runs until you kill it or time runs out
# Checkpoints auto-save every 500 steps
```

---

## Verification

```bash
# Check teacher is up:
curl http://localhost:8000/health

# Check bridge:
curl http://localhost:8002/v1/dashboard | python -m json.tool

# Check generated triplets:
python -c "
import pyarrow.parquet as pq
t = pq.read_table('/scratch/datasets/reasoning_triplets.parquet')
print(f'{len(t)} triplets')
print(f'Passed: {sum(1 for s in t.column(\"verification_stage\").to_pylist() if s == \"all-passed\")}')
"

# Check distill progress:
ls -lh /scratch/checkpoints/
```

---

## Config

All knobs are in `config.py` and `.env`. Key ones:

```bash
# In .env or export:
VLSI_SCRATCH=/scratch          # Where datasets + checkpoints go
CONCURRENT_GENERATORS=4        # Fewer for stability
DISTILL_BATCH_SIZE=1           # Lower if OOM
DISTILL_GRAD_ACCUM=32          # Higher to compensate
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| vLLM OOM | Teacher needs ~66GB. Kill other GPU processes. Or reduce `--gpu-memory-utilization` in setup.sh |
| verilator not found | `export PATH=$HOME/oss-cad-suite/bin:$PATH` |
| factory.py hangs | Teacher vLLM not running. Check `curl http://localhost:8000/health` |
| distill.py OOM during training | Lower `DISTILL_BATCH_SIZE` to 1, increase `DISTILL_GRAD_ACCUM` |
| GaLore import error | `pip install galore-torch` or run with `--no-galileo` |
| Empty Parquet after factory | All verifications failed (normal for complex designs early on). Add more prompts. |
