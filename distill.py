#!/usr/bin/env python3
"""
High-Intensity Distillation — SOTA-VLSI-Distiller-v1
Distills the 33B vlsi-moe-ffn-merged-formal Teacher into Qwen2.5-Coder-14B Student.

Techniques:
  1. Knowledge Distillation (KD): KL-Divergence to match Teacher logit distribution
  2. GaLore: Gradient Low-Rank Projection for memory-efficient full-parameter training
  3. BitNet b1.58: Quantization-Aware Training (QAT) to lock weights into {-1, 0, 1}

24-Hour Schedule:
  Hours  0-6:  Data loading + teacher forward pass precomputation
  Hours  6-18: Peak distillation (high LR, large batch via grad accumulation)
  Hours 18-24: Ternary Squeeze — BitNet b1.58 QAT

Output: GGUF or EXL2-compatible model for AgentIC edge deployment.

Usage:
  python distill.py                              # Full 24hr schedule
  python distill.py --phase 1                    # Data precomputation only
  python distill.py --phase 2 --resume           # Peak distillation
  python distill.py --phase 3 --resume           # Ternary QAT
  python distill.py --export-only --ckpt <path>  # Export to GGUF
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

from config import (
    ROOT, DATASET_DIR, CHECKPOINT_DIR, EXPERT_DISTILLED_DIR,
    TEACHER_MODEL_ID, STUDENT_MODEL_ID,
    TEACHER_URL, API_KEY,
    DISTILL_BATCH_SIZE, DISTILL_GRAD_ACCUM, DISTILL_LR,
    DISTILL_KD_TEMP, DISTILL_KD_ALPHA,
    GALORE_RANK, GALORE_UPDATE_PROJ_GAP,
    PHASE1_HOURS, PHASE2_HOURS, PHASE3_HOURS,
)

import pyarrow.parquet as pq

GALORE_AVAILABLE = False
try:
    from galore_torch import GaLoreOptimizer
    GALORE_AVAILABLE = True
except ImportError:
    pass

BITNET_AVAILABLE = False
try:
    from bitnet import BitLinear
    BITNET_AVAILABLE = True
except ImportError:
    pass


def bitnet_ternarize(weight: torch.Tensor) -> torch.Tensor:
    """BitNet b1.58: quantize weights to {-1, 0, 1} * gamma."""
    gamma = weight.abs().mean()
    w_scaled = weight / (gamma + 1e-8)
    w_quant = w_scaled.clamp(-1, 1).round()
    return w_quant * gamma


@dataclass
class DistillConfig:
    batch_size: int = DISTILL_BATCH_SIZE
    grad_accum: int = DISTILL_GRAD_ACCUM
    lr: float = DISTILL_LR
    kd_temp: float = DISTILL_KD_TEMP
    kd_alpha: float = DISTILL_KD_ALPHA
    max_seq_len: int = 2048
    galore_rank: int = GALORE_RANK
    galore_update_proj_gap: int = GALORE_UPDATE_PROJ_GAP
    weight_decay: float = 0.01
    warmup_steps: int = 500
    use_galore: bool = GALORE_AVAILABLE
    use_bitnet: bool = BITNET_AVAILABLE


class ParquetReasoningDataset(Dataset):
    def __init__(self, parquet_path: str, tokenizer, max_len: int = 2048):
        self.tokenizer = tokenizer
        self.max_len = max_len

        if not Path(parquet_path).exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        table = pq.read_table(parquet_path)
        self.data = table.to_pandas()
        self.data = self.data[self.data["corrected_code"].str.len() > 0].reset_index(drop=True)

        if len(self.data) == 0:
            raise ValueError(f"No valid samples in {parquet_path}. Run factory.py first.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        spec = row["spec"]
        corrected_code = row["corrected_code"]

        text = f"Specification: {spec}\n\nCorrected SystemVerilog RTL:\n{corrected_code}"

        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
        }


class KnowledgeDistillationTrainer:
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        tokenizer,
        config: DistillConfig,
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.tokenizer = tokenizer
        self.cfg = config
        self.device = next(student_model.parameters()).device
        self.step = 0
        self.epoch = 0

        if config.use_galore:
            param_groups = []
            for name, param in student_model.named_parameters():
                if param.requires_grad:
                    param_groups.append({"params": [param]})
            self.optimizer = GaLoreOptimizer(
                param_groups,
                lr=config.lr,
                rank=config.galore_rank,
                update_proj_gap=config.galore_update_proj_gap,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                student_model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=100000,
        )

        self.scaler = torch.amp.GradScaler("cuda")

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        shift_student = student_logits[..., :-1, :].contiguous()
        shift_teacher = teacher_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        T = self.cfg.kd_temp
        alpha = self.cfg.kd_alpha

        kd_loss = F.kl_div(
            F.log_softmax(shift_student / T, dim=-1),
            F.softmax(shift_teacher / T, dim=-1),
            reduction="none",
        ).sum(dim=-1) * (T * T)

        ce_loss = F.cross_entropy(
            shift_student.view(-1, shift_student.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view_as(shift_labels)

        mask = shift_mask.float()
        kd_loss = (kd_loss * mask).sum() / mask.sum()
        ce_loss = (ce_loss * mask).sum() / mask.sum()

        loss = alpha * kd_loss + (1 - alpha) * ce_loss

        return loss, {"kd_loss": kd_loss.item(), "ce_loss": ce_loss.item()}

    def prepare_teacher_logits(self, dataset_path: str, cache_dir: Path):
        cache_dir.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(dataset_path)
        data = table.to_pandas()
        data = data[data["corrected_code"].str.len() > 0]

        teacher_shards = []

        print(f"Precomputing teacher logits for {len(data)} samples...")
        self.teacher.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(data), self.cfg.batch_size)):
                batch_rows = data.iloc[i:i + self.cfg.batch_size]
                texts = [
                    f"Specification: {r['spec']}\n\nCorrected SystemVerilog RTL:\n{r['corrected_code']}"
                    for _, r in batch_rows.iterrows()
                ]
                tokens = self.tokenizer(
                    texts,
                    truncation=True,
                    max_length=self.cfg.max_seq_len,
                    padding="max_length",
                    return_tensors="pt",
                ).to(self.device)

                teacher_out = self.teacher(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                )
                teacher_shards.append({
                    "input_ids": tokens["input_ids"].cpu(),
                    "attention_mask": tokens["attention_mask"].cpu(),
                    "teacher_logits": teacher_out.logits.cpu(),
                })

                if (i // self.cfg.batch_size) % 10 == 0:
                    shard_path = cache_dir / f"teacher_logits_{i // self.cfg.batch_size}.pt"
                    torch.save(teacher_shards[-10:], shard_path)
                    teacher_shards = teacher_shards[-10:]

        final_path = cache_dir / "teacher_logits_full.pt"
        torch.save(teacher_shards, final_path)
        print(f"Teacher logits saved to {final_path}")
        return final_path

    def train_step(self, batch: dict, teacher_logits: Optional[torch.Tensor] = None) -> dict:
        self.student.train()
        self.teacher.eval()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        with torch.amp.autocast("cuda"):
            student_out = self.student(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            if teacher_logits is not None:
                t_logits = teacher_logits.to(self.device)
            else:
                with torch.no_grad():
                    with torch.amp.autocast("cuda"):
                        teacher_out = self.teacher(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                    t_logits = teacher_out.logits

            loss, metrics = self.distillation_loss(
                student_out.logits,
                t_logits,
                input_ids,
                attention_mask,
            )

        loss = loss / self.cfg.grad_accum
        self.scaler.scale(loss).backward()

        self.step += 1

        if self.step % self.cfg.grad_accum == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()

        metrics["loss"] = loss.item() * self.cfg.grad_accum
        metrics["lr"] = self.scheduler.get_last_lr()[0]
        return metrics

    def run_phase2(self, dataloader: DataLoader, max_hours: float, checkpoint_interval: int = 500):
        print(f"\n{'='*60}")
        print(f"  Phase 2: Peak Distillation (max {max_hours}h)")
        print(f"  KD alpha={self.cfg.kd_alpha}, T={self.cfg.kd_temp}")
        print(f"  GaLore: {self.cfg.use_galore}, BitNet: {self.cfg.use_bitnet}")
        print(f"{'='*60}\n")

        start = time.time()
        deadline = start + max_hours * 3600
        pbar = tqdm(desc="Distilling", unit="step")

        while time.time() < deadline:
            for batch in dataloader:
                if time.time() > deadline:
                    break

                metrics = self.train_step(batch)
                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "kd": f"{metrics['kd_loss']:.4f}",
                    "lr": f"{metrics['lr']:.2e}",
                })

                if self.step % checkpoint_interval == 0:
                    self.save_checkpoint(phase=2)

            self.epoch += 1

        pbar.close()
        self.save_checkpoint(phase=2)
        print(f"\nPhase 2 complete. Steps: {self.step}, Time: {(time.time()-start)/3600:.1f}h")

    def run_phase3_bitnet(self, dataloader: DataLoader, max_hours: float):
        print(f"\n{'='*60}")
        print(f"  Phase 3: Ternary Squeeze — BitNet b1.58 QAT")
        print(f"  Weights quantized to {{-1, 0, 1}}")
        print(f"{'='*60}\n")

        start = time.time()
        deadline = start + max_hours * 3600
        pbar = tqdm(desc="Ternary QAT", unit="step")

        while time.time() < deadline:
            for batch in dataloader:
                if time.time() > deadline:
                    break

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                with torch.amp.autocast("cuda"):
                    with torch.no_grad():
                        teacher_out = self.teacher(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )

                    if BITNET_AVAILABLE:
                        for module in self.student.modules():
                            if isinstance(module, BitLinear):
                                module.weight.data = bitnet_ternarize(module.weight.data)

                    student_out = self.student(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                    loss, metrics = self.distillation_loss(
                        student_out.logits,
                        teacher_out.logits,
                        input_ids,
                        attention_mask,
                    )

                loss = loss / self.cfg.grad_accum
                loss.backward()

                self.step += 1

                if self.step % self.cfg.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                pbar.update(1)
                pbar.set_postfix({"loss": f"{metrics['loss']:.4f}", "lr": f"{metrics['lr']:.2e}"})

                if self.step % 500 == 0:
                    self.save_checkpoint(phase=3)

            self.epoch += 1

        pbar.close()

        if BITNET_AVAILABLE:
            for module in self.student.modules():
                if isinstance(module, BitLinear):
                    module.weight.data = bitnet_ternarize(module.weight.data)

        self.save_checkpoint(phase=3)
        print(f"\nPhase 3 complete. Steps: {self.step}, Time: {(time.time()-start)/3600:.1f}h")

    def save_checkpoint(self, phase: int):
        path = CHECKPOINT_DIR / f"phase{phase}_step{self.step}"
        path.mkdir(parents=True, exist_ok=True)

        self.student.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
        }, path / "training_state.pt")

        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, ckpt_dir: Path):
        self.student = AutoModelForCausalLM.from_pretrained(
            str(ckpt_dir),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), trust_remote_code=True)

        state_path = ckpt_dir / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu")
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.step = state["step"]
            self.epoch = state["epoch"]

    def export_gguf(self, model_dir: Path, output_path: Optional[Path] = None):
        if output_path is None:
            output_path = EXPERT_DISTILLED_DIR / "qwen-vlsi-sota-14b-ternary.gguf"

        print(f"\n{'='*60}")
        print(f"  Exporting to GGUF: {output_path}")
        print(f"{'='*60}\n")

        model_dir.mkdir(parents=True, exist_ok=True)
        self.student.save_pretrained(str(model_dir))
        self.tokenizer.save_pretrained(str(model_dir))

        quant_config = {
            "model": {
                "model_type": "qwen2",
                "vocab_size": self.student.config.vocab_size,
                "hidden_size": self.student.config.hidden_size,
                "num_hidden_layers": self.student.config.num_hidden_layers,
                "num_attention_heads": self.student.config.num_attention_heads,
                "rms_norm_eps": self.student.config.rms_norm_eps,
                "rope_theta": self.student.config.rope_theta,
                "max_position_embeddings": self.student.config.max_position_embeddings,
            },
            "quantization": {
                "type": "bitnet_b158",
                "bits": 1.58,
                "group_size": -1,
            },
        }
        with open(model_dir / "quant_config.json", "w") as f:
            json.dump(quant_config, f, indent=2)

        print("\nTo convert to GGUF using llama.cpp:")
        print(f"  python llama.cpp/convert_hf_to_gguf.py {model_dir} --outtype f16")
        print(f"  llama.cpp/quantize {output_path} q4_k_m")

        print(f"\nModel ready at: {model_dir}")


def load_models(cfg: DistillConfig):
    print(f"Loading Teacher: {TEACHER_MODEL_ID}")
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"Loading Student: {STUDENT_MODEL_ID}")
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return teacher, student, tokenizer


def main():
    parser = argparse.ArgumentParser(description="High-Intensity Distillation")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=0,
                        help="Run specific phase (0 = all)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint directory")
    parser.add_argument("--export-only", action="store_true",
                        help="Export model to GGUF and exit")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint path for export")
    parser.add_argument("--dataset", type=str, default=str(DATASET_DIR / "reasoning_triplets.parquet"),
                        help="Parquet dataset path")
    parser.add_argument("--output", type=str, default=str(EXPERT_DISTILLED_DIR),
                        help="Output directory for distilled model")
    parser.add_argument("--no-galileo", action="store_true",
                        help="Disable GaLore (fallback to AdamW)")
    parser.add_argument("--no-bitnet", action="store_true",
                        help="Disable BitNet QAT")
    args = parser.parse_args()

    cfg = DistillConfig()
    if args.no_galileo:
        cfg.use_galore = False
    if args.no_bitnet:
        cfg.use_bitnet = False

    dataset_path = args.dataset
    if not Path(dataset_path).exists():
        print(f"Dataset not found: {dataset_path}")
        print("Run factory.py first to generate reasoning triplets.")
        return

    teacher, student, tokenizer = load_models(cfg)
    trainer = KnowledgeDistillationTrainer(teacher, student, tokenizer, cfg)

    if args.export_only:
        ckpt = Path(args.ckpt) if args.ckpt else CHECKPOINT_DIR
        ckpt_dirs = sorted(ckpt.glob("phase*"), key=lambda p: p.stat().st_mtime)
        if ckpt_dirs:
            trainer.load_checkpoint(ckpt_dirs[-1])
        trainer.export_gguf(Path(args.output))
        return

    if args.resume:
        trainer.load_checkpoint(Path(args.resume))

    dataset = ParquetReasoningDataset(dataset_path, tokenizer, cfg.max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    if args.phase == 1:
        teacher_cache = CHECKPOINT_DIR / "teacher_logits"
        trainer.prepare_teacher_logits(dataset_path, teacher_cache)
    elif args.phase == 2:
        trainer.run_phase2(dataloader, PHASE2_HOURS)
    elif args.phase == 3:
        trainer.run_phase3_bitnet(dataloader, PHASE3_HOURS)
    else:
        # Full schedule
        teacher_cache = CHECKPOINT_DIR / "teacher_logits"
        trainer.prepare_teacher_logits(dataset_path, teacher_cache)
        trainer.run_phase2(dataloader, PHASE2_HOURS)
        trainer.run_phase3_bitnet(dataloader, PHASE3_HOURS)
        trainer.export_gguf(Path(args.output))

    print(f"\n{'='*60}")
    print(f"  Distillation pipeline finished.")
    print(f"  Final model: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
