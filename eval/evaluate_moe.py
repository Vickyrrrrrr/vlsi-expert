#!/usr/bin/env python3
"""
Evaluate VLSI Expert MoE model through AgentIC pipeline via vLLM.
Uses the OpenAI-compatible API served by vLLM on the MI300X.

This is the FINAL evaluation script for the hackathon submission.
"""

import json
import time
import sys
import requests
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "AgentIC" / "src"))

# ── Config ────────────────────────────────────────────────────────────
VLLM_URL = "http://localhost:8000/v1"

TEST_DESIGNS = [
    {"id": "counter_8bit", "spec": "8-bit up counter with synchronous reset and enable"},
    {"id": "fifo_depth8", "spec": "Synchronous FIFO with depth 8, 8-bit data, empty and full flags"},
    {"id": "uart_tx", "spec": "UART transmitter at 115200 baud rate, 8N1"},
    {"id": "spi_master", "spec": "SPI master controller with configurable clock divider and CPOL/CPHA"},
    {"id": "alu_16bit", "spec": "16-bit ALU with add, subtract, AND, OR, XOR, comparison, and shift operations"},
    {"id": "fsm_1011", "spec": "FSM that detects the sequence 1011 on a single-bit input with overlapping"},
    {"id": "pwm_gen", "spec": "PWM generator with 8-bit duty cycle control and programmable period"},
    {"id": "crc8", "spec": "8-bit CRC generator with CRC-8-CCITT polynomial 0x07"},
    {"id": "gray_counter", "spec": "4-bit Gray code counter with synchronous reset"},
    {"id": "edge_detect", "spec": "Rising and falling edge detector on a single-bit input signal"},
    {"id": "mult_pipe", "spec": "8x8 pipelined multiplier with 2 pipeline stages and registered outputs"},
    {"id": "adder_pipe", "spec": "16-bit pipelined adder with 2 pipeline stages"},
    {"id": "arbiter_rr", "spec": "Round-robin arbiter for 4 requestors with grant acknowledgment"},
    {"id": "lfsr_prbs", "spec": "16-bit LFSR for pseudo-random sequence generation with configurable taps"},
    {"id": "sync_2ff", "spec": "2 flip-flop synchronizer for CDC between asynchronous clock domains"},
    {"id": "debounce", "spec": "Button debouncer with 20ms debounce period and edge detection"},
    {"id": "divider_restoring", "spec": "8-bit restoring divider producing quotient and remainder"},
    {"id": "i2c_master", "spec": "I2C master controller with start, stop, and single-byte write"},
    {"id": "shift_reg_wide", "spec": "16-bit shift register with parallel load, serial in, serial out"},
    {"id": "riscv_simple", "spec": "Single-cycle RISC-V processor supporting RV32I base integer instructions"},
]


def call_vllm(prompt: str, max_tokens: int = 2048, temperature: float = 0.2) -> str:
    """Call vLLM endpoint and return generated text."""
    payload = {
        "model": "vlsi-moe-merged",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": ["</s>", "```\n", "\n\n\n"],
    }

    try:
        resp = requests.post(f"{VLLM_URL}/completions", json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["text"]
        return f"[vLLM error HTTP {resp.status_code}]"
    except requests.exceptions.ConnectionError:
        return "[vLLM not running — start with: python scripts/serve_vllm.py]"


def extract_verilog(text: str) -> str:
    """Extract Verilog from LLM output (in ```verilog fence or raw)."""
    if "```verilog" in text:
        return text.split("```verilog")[-1].split("```")[0].strip()
    if "module " in text and "endmodule" in text:
        start = text.index("module ")
        end = text.rindex("endmodule") + len("endmodule")
        return text[start:end]
    return text.strip()[:8000]


def evaluate_design(spec: str, model_name: str = "vlsi-moe") -> Dict:
    """Evaluate one design through AgentIC pipeline via vLLM with CoT prompting."""
    print(f"  [{model_name}] Testing: {spec[:50]}...", end=" ", flush=True)
    start = time.time()

    # Advanced CoT prompt — triggers DeepSeek-R1 reasoning in the merged model
    coder_prompt = (
        "You are an expert VLSI design engineer. Analyze the design requirements "
        "carefully, think through the architecture, then generate complete, "
        "synthesizable Verilog RTL with a self-checking testbench. "
        "Target: SkyWater 130nm PDK. Output the code in a ```verilog fence.\n\n"
        f"### Design Specification\n{spec}\n\n"
        "### Design Analysis\nDesign Type:"
    )

    result = {
        "design_id": spec[:30].replace(" ", "_"),
        "spec": spec,
        "model": model_name,
        "syntax_pass": False,
        "sim_pass": False,
        "synth_pass": False,
        "gdsii_pass": False,
        "build_duration_s": 0,
        "verilog_lines": 0,
        "error": "",
    }

    try:
        response = call_vllm(coder_prompt)
        verilog = extract_verilog(response)
        result["verilog_lines"] = len(verilog.split("\n"))

        # Run through AgentIC pipeline
        from agentic.orchestrator import BuildOrchestrator
        from crewai import LLM

        llm = LLM(
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            base_url=f"{VLLM_URL}",
            api_key="not-needed",
        )

        orch = BuildOrchestrator(
            name=result["design_id"],
            desc=spec,
            llm=llm,
            skip_openlane=True,
            max_retries=2,
            verbose=False,
            pdk_profile="sky130",
        )

        orch.run()
        result["syntax_pass"] = orch.artifacts.get("syntax_pass", False)
        result["sim_pass"] = orch.artifacts.get("sim_pass", False)
        result["synth_pass"] = "synth_metrics" in orch.artifacts
        result["build_duration_s"] = time.time() - start
        result["retries"] = orch.retry_count

        status = "✅" if result["synth_pass"] else "❌"
        print(f"{status} ({result['build_duration_s']:.0f}s)")

    except Exception as e:
        result["error"] = str(e)[:200]
        result["build_duration_s"] = time.time() - start
        print(f"❌ {str(e)[:80]}")

    return result


def main():
    print("=" * 60)
    print("  VLSI Expert MoE — AgentIC Pipeline Evaluation")
    print(f"  vLLM endpoint: {VLLM_URL}")
    print(f"  Test designs: {len(TEST_DESIGNS)}")
    print("=" * 60)
    print()

    results = []
    for design in TEST_DESIGNS:
        result = evaluate_design(design["spec"])
        results.append(result)

    # Print summary
    print("\n" + "=" * 60)
    synth_passes = sum(1 for r in results if r["synth_pass"])
    sim_passes = sum(1 for r in results if r["sim_pass"])
    avg_lines = sum(r.get("verilog_lines", 0) for r in results) / len(results)
    avg_time = sum(r["build_duration_s"] for r in results) / len(results)

    print(f"\n  RESULTS (VLSI Expert MoE)")
    print(f"  Synthesis pass: {synth_passes}/{len(results)} ({100*synth_passes/len(results):.0f}%)")
    print(f"  Simulation pass: {sim_passes}/{len(results)} ({100*sim_passes/len(results):.0f}%)")
    print(f"  Avg Verilog lines: {avg_lines:.0f}")
    print(f"  Avg build time: {avg_time:.0f}s")

    # Per-design breakdown
    print(f"\n  Per-design:")
    for r in results:
        status = "✅" if r["synth_pass"] else "❌"
        print(f"    {status} {r['design_id']:<20} {r['verilog_lines']:>4} lines  {r['build_duration_s']:.0f}s")

    # Save
    out_path = Path(__file__).parent / "moe_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
