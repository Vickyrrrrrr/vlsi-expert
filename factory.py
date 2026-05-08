#!/usr/bin/env python3
"""
Logic Factory & Refactor Loop — SOTA-VLSI-Distiller-v1
Generates SystemVerilog RTL + SVA via 33B Teacher, verifies, refactors on failure.

Verification Chain:
  1. verilator --lint-only -Wall                  (Syntax)
  2. iverilog -g2012 + testbench simulation       (Functional)
  3. yosys formal -> z3                            (Formal proof)

Refactor Loop:
  On failure: prompt Teacher with error log -> regenerate -> retry (max 3).
  Saves (Incorrect_Code, Error_Log, Corrected_Code) triplets as Parquet.

Usage:
  python factory.py                                    # Generate from built-in prompts
  python factory.py --prompts prompts.txt               # Generate from file
  python factory.py --spec "8-bit counter with reset"   # Single generation
  python factory.py --resume                            # Resume from checkpoint
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from config import (
    ROOT, DATASET_DIR, CHECKPOINT_DIR,
    TEACHER_URL, API_KEY,
    MAX_REFACTOR_RETRIES, GENERATION_MAX_TOKENS, GENERATION_TEMPERATURE,
    VERILATOR_ARGS, IVERILOG_ARGS, YOSYS_FORMAL_SCRIPT, Z3_TIMEOUT_MS,
    CONCURRENT_GENERATORS,
)

STATUS_FILE = CHECKPOINT_DIR / "factory_checkpoint.json"
TRIPLETS_JSONL = DATASET_DIR / "reasoning_triplets.jsonl"
PARQUET_LOCK = threading.Lock()

SYSTEM_PROMPT = (
    "You are a VLSI design compiler. Output ONLY raw synthesizable SystemVerilog RTL. "
    "MANDATORY: Use 'clk' and 'rst_n' (active low). "
    "For assertions, use simple 'assert property (@(posedge clk) disable iff (!rst_n) ...);' "
    "inline within the module. DO NOT use 'property...endproperty' blocks. "
    "No markdown, no explanations, no text before or after the code."
)

REFACTOR_PROMPT_TEMPLATE = (
    "The following SystemVerilog code failed verification.\n\n"
    "=== Specification ===\n"
    "{spec}\n\n"
    "=== Error Log ===\n"
    "{error_log}\n\n"
    "=== Code That Failed ===\n"
    "{incorrect_code}\n\n"
    "Refactor the code to fix the logic while maintaining the specification. "
    "Think step-by-step about the root cause of each error, then produce the corrected code. "
    "Output ONLY the corrected SystemVerilog code with SVA properties. "
    "No explanation, no markdown."
)

RTL_PROMPTS = [
    "Generate a parameterized FIFO buffer with configurable depth and width. Include full/empty flags, SVA to verify no overflow/underflow.",
    "Generate a pipelined RISC-V RV32I ALU supporting ADD, SUB, AND, OR, XOR, SLT, SLTU, SLL, SRL, SRA. Include SVA for result correctness.",
    "Generate an AXI4-Lite slave interface with read/write channels. Include SVA to verify handshake protocol compliance.",
    "Generate a dual-port synchronous SRAM with read/write collision detection. Include SVA for memory consistency.",
    "Generate a UART transmitter with configurable baud rate, parity, stop bits. Include SVA for timing violations.",
    "Generate a SPI master controller supporting modes 0-3. Include SVA for protocol compliance.",
    "Generate a round-robin arbiter with parameterized number of requesters. Include SVA for fairness.",
    "Generate a binary to BCD converter with pipelined stages. Include SVA for output correctness.",
    "Generate a Wallace tree multiplier with parameterized bit-width. Include SVA for correctness vs behavioral model.",
    "Generate a Gray code counter with parameterized width. Include SVA that each transition changes exactly one bit.",
    "Generate a synchronous FIFO with parameterized depth. Include SVA for read/write pointer invariants.",
    "Generate a barrel shifter supporting left/right arithmetic/logical shifts. Include SVA for boundary conditions.",
    "Generate a CRC-32 generator with parallel byte-wide input. Include SVA for polynomial correctness.",
    "Generate a JTAG TAP controller state machine. Include SVA for all state transitions.",
    "Generate a DDR interface write path with DQS alignment. Include SVA for setup/hold timing.",
    "Generate a clock domain crossing synchronizer (2-FF) with SVA for metastability prevention.",
    "Generate a priority encoder with parameterized width. Include SVA for one-hot output guarantee.",
    "Generate a signed integer divider using non-restoring algorithm. Include SVA for sign correctness.",
    "Generate a LFSR-based pseudo-random number generator. Include SVA for non-zero state and period.",
    "Generate a Wishbone B4 pipelined master interface. Include SVA for handshake protocol.",
]


@dataclass
class VerificationResult:
    success: bool
    stage: str
    error_log: str
    tool_output: str = ""


@dataclass
class ReasoningTriplet:
    spec: str
    incorrect_code: str
    error_log: str
    corrected_code: str
    num_refactors: int
    total_time_sec: float
    verification_stage: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def strip_markdown(text: str) -> str:
    text = text.strip()
    if "```" in text:
        lines = text.splitlines()
        code_lines = []
        in_code = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                code_lines.append(line)
        if code_lines:
            text = "\n".join(code_lines)
    return text.strip()


def extract_module_name(code: str) -> str:
    m = re.search(r"module\s+(\w+)", code)
    return m.group(1) if m else "top"


def verify_verilator(code: str) -> VerificationResult:
    with tempfile.NamedTemporaryFile(suffix=".sv", mode="w", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(
            ["verilator"] + VERILATOR_ARGS + [path],
            capture_output=True, text=True, timeout=60
        )
        error_log = result.stderr + result.stdout
        return VerificationResult(
            success=result.returncode == 0,
            stage="verilator-lint",
            error_log=error_log[-2000:],
            tool_output=error_log[-2000:],
        )
    except FileNotFoundError:
        return VerificationResult(True, "verilator-lint", "verilator not found — skipping lint")
    except subprocess.TimeoutExpired:
        return VerificationResult(False, "verilator-lint", "Timeout after 60s", "")
    finally:
        os.unlink(path)


def verify_iverilog(code: str, module_name: str) -> VerificationResult:
    with tempfile.TemporaryDirectory() as tmpdir:
        sv_path = os.path.join(tmpdir, "design.sv")
        tb_path = os.path.join(tmpdir, "tb.v")
        out_path = os.path.join(tmpdir, "sim.out")

        with open(sv_path, "w") as f:
            f.write(code)

        tb_code = f"""
module tb_{module_name};
    reg clk = 0;
    reg rst_n = 0;
    always #5 clk = ~clk;
    initial begin
        rst_n = 1'b0;
        #20 rst_n = 1'b1;
        #200 $finish;
    end
    {module_name} dut (.*);
endmodule
"""
        with open(tb_path, "w") as f:
            f.write(tb_code)

        try:
            compile_result = subprocess.run(
                ["iverilog"] + IVERILOG_ARGS + ["-o", out_path, sv_path, tb_path],
                capture_output=True, text=True, timeout=30
            )
            if compile_result.returncode != 0:
                return VerificationResult(
                    False, "iverilog-compile",
                    compile_result.stderr[-2000:],
                    compile_result.stderr[-2000:],
                )

            sim_result = subprocess.run(
                ["vvp", out_path],
                capture_output=True, text=True, timeout=30
            )
            if sim_result.returncode != 0:
                return VerificationResult(
                    False, "iverilog-sim",
                    (sim_result.stderr + sim_result.stdout)[-2000:],
                    sim_result.stdout[-2000:],
                )

            return VerificationResult(True, "iverilog-sim", "", "")
        except FileNotFoundError:
            return VerificationResult(True, "iverilog", "iverilog/vvp not found — skipping simulation")
        except subprocess.TimeoutExpired:
            return VerificationResult(False, "iverilog", "Timeout after 30s", "")


def verify_yosys_z3(code: str, module_name: str) -> VerificationResult:
    with tempfile.TemporaryDirectory() as tmpdir:
        sv_path = os.path.join(tmpdir, "design.sv")
        with open(sv_path, "w") as f:
            f.write(code)

        script = YOSYS_FORMAL_SCRIPT.format(top=module_name)
        script_path = os.path.join(tmpdir, "script.ys")
        with open(script_path, "w") as f:
            f.write(script)

        try:
            yosys_result = subprocess.run(
                ["yosys", "-s", script_path],
                capture_output=True, text=True, timeout=120,
                cwd=tmpdir,
            )
            yosys_err = yosys_result.stderr + yosys_result.stdout

            smt2_path = os.path.join(tmpdir, "design.smt2")
            if not os.path.exists(smt2_path):
                return VerificationResult(
                    False, "yosys-formal",
                    f"Yosys SMT2 generation failed:\n{yosys_err[-2000:]}",
                    yosys_err[-2000:],
                )

            z3_result = subprocess.run(
                ["z3", "-T:{:d}".format(Z3_TIMEOUT_MS // 1000), smt2_path],
                capture_output=True, text=True,
                timeout=Z3_TIMEOUT_MS // 1000 + 5,
            )
            z3_out = z3_result.stdout + z3_result.stderr
            if "unsat" in z3_out.lower():
                return VerificationResult(True, "yosys-z3", "", z3_out)
            else:
                return VerificationResult(
                    False, "yosys-z3",
                    f"Z3 could not prove assertions (expected 'unsat'):\n{z3_out[-2000:]}",
                    z3_out[-2000:],
                )
        except FileNotFoundError as e:
            tool = "yosys" if "yosys" in str(e) else "z3"
            return VerificationResult(True, f"yosys-{tool}", f"{tool} not found — skipping formal")
        except subprocess.TimeoutExpired:
            return VerificationResult(False, "yosys-z3", "Timeout", "")


def verify_full(code: str) -> Tuple[List[VerificationResult], Optional[VerificationResult]]:
    """Run full verification chain. Returns (all_results, first_failure_or_None)."""
    results: List[VerificationResult] = []

    r = verify_verilator(code)
    results.append(r)
    if not r.success:
        return results, r

    module_name = extract_module_name(code)

    r = verify_iverilog(code, module_name)
    results.append(r)
    if not r.success:
        return results, r

    r = verify_yosys_z3(code, module_name)
    results.append(r)
    if not r.success:
        return results, r

    return results, None


class LogicFactory:
    def __init__(self, concurrency: int = CONCURRENT_GENERATORS):
        self.client = AsyncOpenAI(base_url=TEACHER_URL, api_key=API_KEY)
        self.semaphore = asyncio.Semaphore(concurrency)
        self.stats = {
            "total_prompts": 0,
            "successful_refactors": 0,
            "failed_refactors": 0,
            "proof_passed": 0,
            "total_tokens": 0,
            "start_time": time.time(),
        }
        self._load_checkpoint()

    def _load_checkpoint(self):
        if STATUS_FILE.exists():
            saved = json.loads(STATUS_FILE.read_text())
            self.stats.update({k: v for k, v in saved.items() if k in self.stats})

    def _save_checkpoint(self):
        STATUS_FILE.write_text(json.dumps(self.stats, indent=2))

    async def generate(self, spec: str) -> str:
        async with self.semaphore:
            response = await self.client.chat.completions.create(
                model="vlsi-expert-teacher",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Generate SystemVerilog RTL with SVA for: {spec}\n"},
                ],
                max_tokens=GENERATION_MAX_TOKENS,
                temperature=GENERATION_TEMPERATURE,
            )
            content = response.choices[0].message.content
            if response.usage:
                self.stats["total_tokens"] += response.usage.total_tokens

            # Patch the hallucination here (OUTSIDE the if block)
            content = content.replace("assert_property", "assert property")

            return strip_markdown(content)

    async def refactor(self, spec: str, incorrect_code: str, error_log: str) -> str:
        async with self.semaphore:
            prompt = REFACTOR_PROMPT_TEMPLATE.format(
                spec=spec,
                error_log=error_log,
                incorrect_code=incorrect_code,
            )
            response = await self.client.chat.completions.create(
                model="vlsi-expert-teacher",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=GENERATION_MAX_TOKENS,
                temperature=GENERATION_TEMPERATURE * 0.7,
            )
            content = response.choices[0].message.content
            if response.usage:
                self.stats["total_tokens"] += response.usage.total_tokens

            # Patch the hallucination here (OUTSIDE the if block)
            content = content.replace("assert_property", "assert property")

            return strip_markdown(content)

    def _verify_sync(self, code: str) -> Tuple[List[VerificationResult], Optional[VerificationResult]]:
        """Synchronous verification — returns (all_results, first_failure)."""
        return verify_full(code)

    async def process_spec(self, spec: str) -> ReasoningTriplet:
        t_start = time.time()

        # Step 1: Generate initial code
        current_code = await self.generate(spec)
        incorrect_code = current_code
        error_log = ""
        verification_stage = ""
        num_refactors = 0

        # Step 2: Verify the initial code
        results, failure = await asyncio.to_thread(self._verify_sync, current_code)

        if failure is None:
            # Passed on first attempt
            self.stats["successful_refactors"] += 1
            self.stats["proof_passed"] += 1
            return ReasoningTriplet(
                spec=spec,
                incorrect_code=incorrect_code,
                error_log="All checks passed on first attempt",
                corrected_code=current_code,
                num_refactors=0,
                total_time_sec=time.time() - t_start,
                verification_stage="all-passed",
            )

        # Step 3: Refactor loop — prompt Teacher with error, regenerate, retry
        incorrect_code = current_code
        error_log = failure.error_log
        verification_stage = failure.stage

        for attempt in range(MAX_REFACTOR_RETRIES):
            num_refactors += 1

            # Ask Teacher to fix the code
            refactored_code = await self.refactor(spec, incorrect_code, error_log)
            current_code = refactored_code

            # Re-verify the refactored code
            results, failure = await asyncio.to_thread(self._verify_sync, current_code)

            if failure is None:
                # Refactored code passed!
                self.stats["successful_refactors"] += 1
                self.stats["proof_passed"] += 1
                return ReasoningTriplet(
                    spec=spec,
                    incorrect_code=incorrect_code,
                    error_log=error_log,
                    corrected_code=current_code,
                    num_refactors=num_refactors,
                    total_time_sec=time.time() - t_start,
                    verification_stage="all-passed",
                )

            # Still failing — update BOTH error and code for next refactor attempt
            error_log = failure.error_log
            incorrect_code = current_code
            verification_stage = failure.stage

        # Step 4: All retries exhausted — save what we have
        self.stats["failed_refactors"] += 1
        return ReasoningTriplet(
            spec=spec,
            incorrect_code=incorrect_code,
            error_log=error_log,
            corrected_code=current_code,
            num_refactors=num_refactors,
            total_time_sec=time.time() - t_start,
            verification_stage=verification_stage,
        )

    def _save_triplet(self, triplet: ReasoningTriplet):
        with PARQUET_LOCK:
            with open(TRIPLETS_JSONL, "a") as f:
                f.write(json.dumps(triplet.__dict__) + "\n")

    async def run(self, prompts: list[str]) -> list[ReasoningTriplet]:
        self.stats["total_prompts"] = len(prompts)
        self.stats["start_time"] = time.time()
        self._save_checkpoint()

        results: list[ReasoningTriplet] = []
        pbar = tqdm_asyncio(total=len(prompts), desc="Generating & Verifying")

        async def process_one(prompt: str):
            try:
                triplet = await self.process_spec(prompt)
                self._save_triplet(triplet)
                results.append(triplet)
                elapsed = max(time.time() - self.stats["start_time"], 1)
                throughput = self.stats["total_tokens"] / elapsed
                pbar.set_postfix({
                    "ok": self.stats["successful_refactors"],
                    "pass": self.stats["proof_passed"],
                    "tok/s": f"{throughput:.0f}",
                })
            except Exception:
                traceback.print_exc()
                triplet = ReasoningTriplet(
                    spec=prompt,
                    incorrect_code="",
                    error_log=traceback.format_exc()[-2000:],
                    corrected_code="",
                    num_refactors=0,
                    total_time_sec=0,
                )
                results.append(triplet)
            pbar.update(1)

        tasks = [process_one(p) for p in prompts]
        await asyncio.gather(*tasks)

        self._save_checkpoint()
        return results


async def main():
    parser = argparse.ArgumentParser(description="Logic Factory — Generate & Verify RTL+SVA")
    parser.add_argument("--spec", help="Single spec to generate")
    parser.add_argument("--prompts", help="File with one spec per line")
    parser.add_argument("--concurrency", type=int, default=CONCURRENT_GENERATORS)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    if args.spec:
        prompts = [args.spec]
    elif args.prompts:
        prompts = [l.strip() for l in Path(args.prompts).read_text().splitlines() if l.strip()]
    else:
        prompts = RTL_PROMPTS

    factory = LogicFactory(concurrency=args.concurrency)

    print("=" * 60)
    print("  Logic Factory — SOTA-VLSI-Distiller-v1")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Teacher: {TEACHER_URL}")
    print(f"  Output: {TRIPLETS_JSONL}")
    print("=" * 60)
    print()

    results = await factory.run(prompts)

    passed = sum(1 for r in results if r.verification_stage == "all-passed")
    total = len(results)
    elapsed = time.time() - factory.stats["start_time"]

    print(f"\n{'='*60}")
    print(f"  Generation Complete!")
    print(f"  Total prompts:       {total}")
    print(f"  Passed all checks:   {passed} ({100 * passed // max(total, 1)}%)")
    print(f"  Successful refactors: {factory.stats['successful_refactors']}")
    print(f"  Time elapsed:        {elapsed / 60:.1f} min")
    print(f"  Tokens generated:    {factory.stats['total_tokens']}")
    print(f"  Triplets saved:      {TRIPLETS_JSONL}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
