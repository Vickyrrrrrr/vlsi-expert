#!/usr/bin/env python3
"""
VLSI Expert — Gradio Demo for HuggingFace Spaces
Loads both LoRA adapters + routes tasks to the right model head.
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── Model paths (update these after training) ──────────────────────────
CODER_BASE = "Qwen/Qwen2.5-Coder-32B-Instruct"
CODER_LORA = "Vickyrrrrrr/vlsi-coder-lora"
INSTRUCT_BASE = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
INSTRUCT_LORA = "Vickyrrrrrr/vlsi-instruct-lora"

# ── Load models once on startup ────────────────────────────────────────
print("Loading CODER model (Qwen2.5-Coder-32B + VLSI adapter)...")
coder_base = AutoModelForCausalLM.from_pretrained(
    CODER_BASE,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
coder_model = PeftModel.from_pretrained(coder_base, CODER_LORA)
coder_tok = AutoTokenizer.from_pretrained(CODER_BASE, trust_remote_code=True)

print("Loading INSTRUCT model (DeepSeek-R1-32B + VLSI adapter)...")
instruct_base = AutoModelForCausalLM.from_pretrained(
    INSTRUCT_BASE,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
instruct_model = PeftModel.from_pretrained(instruct_base, INSTRUCT_LORA)
instruct_tok = AutoTokenizer.from_pretrained(INSTRUCT_BASE, trust_remote_code=True)

print("✅ Both models loaded!")


def generate_chip(
    spec: str,
    pdk: str,
    freq_mhz: int,
    width: int = 8,
) -> tuple:
    """Generate chip design from specification."""
    # Route to CODER for RTL generation
    coder_prompt = (
        f"Generate synthesizable Verilog RTL for the following specification. "
        f"PDK: {pdk}. Target clock: {freq_mhz}MHz. Data width: {width}-bit.\n\n"
        f"Specification: {spec}\n\n"
        f"Output ONLY the Verilog code in a ```verilog fence:"
    )

    inputs = coder_tok(coder_prompt, return_tensors="pt").to(coder_model.device)

    with torch.no_grad():
        outputs = coder_model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.2,
            do_sample=True,
            top_p=0.9,
            pad_token_id=coder_tok.eos_token_id,
        )

    verilog = coder_tok.decode(outputs[0], skip_special_tokens=True)
    verilog = verilog.split("```verilog")[-1].split("```")[0].strip() if "```verilog" in verilog else verilog

    # Generate SDC constraints via INSTRUCT
    sdc_prompt = (
        f"Generate SDC timing constraints for this Verilog module. "
        f"Clock: {freq_mhz}MHz. PDK: {pdk}.\n\n"
        f"```verilog\n{verilog[:2000]}\n```\n\n"
        f"Output ONLY SDC commands:"
    )

    sdc_inputs = instruct_tok(sdc_prompt, return_tensors="pt").to(instruct_model.device)
    with torch.no_grad():
        sdc_outputs = instruct_model.generate(
            **sdc_inputs, max_new_tokens=512, temperature=0.1, do_sample=True
        )
    sdc = instruct_tok.decode(sdc_outputs[0], skip_special_tokens=True)

    # Build summary report
    report = f"""## Synthesis Report

**PDK:** {pdk}
**Target Clock:** {freq_mhz}MHz
**Data Width:** {width}-bit
**Verilog Lines:** {len(verilog.split(chr(10)))}

**Model Details:**
- Coder: Qwen2.5-Coder-32B + VLSI QLoRA adapter (rank 64)
- Instruct: DeepSeek-R1-Distill-Qwen-32B + VLSI QLoRA adapter (rank 32)
- Trained on: 500+ verified Verilog pairs (VerilogEval v2, RTLLM)
- Training method: QLoRA (4-bit quantization)
- Each adapter: ~80 MB"""

    return verilog, sdc, report


# ── Gradio Interface ───────────────────────────────────────────────────
with gr.Blocks(title="VLSI Expert — AI Chip Designer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# VLSI Expert — AI Chip Designer")
    gr.Markdown(
        "Two specialized models for chip design: **Coder** (RTL generation) + "
        "**Instruct** (error fixing, SDC, timing analysis). "
        "QLoRA fine-tuned on 500+ verified public Verilog pairs.\n\n"
        "[GitHub](https://github.com/Vickyrrrrrr/vlsi-expert) | "
        "[HuggingFace](https://huggingface.co/Vickyrrrrrr)"
    )

    with gr.Row():
        with gr.Column(scale=2):
            spec = gr.Textbox(
                label="Chip Specification",
                placeholder="Example: 8-bit up counter with synchronous reset and enable signal.\n"
                           "Example: 32-bit pipelined multiplier with bypass and hazard detection.\n"
                           "Example: UART transmitter at 115200 baud, 8 data bits, no parity, 1 stop bit.",
                lines=4,
            )
            with gr.Row():
                pdk = gr.Dropdown(
                    ["sky130", "gf180mcu", "asap7", "nangate45", "freepdk45"],
                    value="sky130",
                    label="Target PDK",
                )
                freq = gr.Slider(
                    10, 2000, value=100, step=10,
                    label="Target Frequency (MHz)",
                )
                width = gr.Slider(
                    4, 64, value=8, step=4,
                    label="Data Width (bits)",
                )
            btn = gr.Button("Generate Chip Design", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### How it works")
            gr.Markdown(
                "1. **Coder model** (Qwen2.5-Coder-32B) generates the Verilog RTL\n"
                "2. **Instruct model** (DeepSeek-R1-32B) generates SDC constraints\n"
                "3. Both models use QLoRA adapters trained on public Verilog benchmarks\n"
                "4. Total LoRA weights: ~160 MB for both models"
            )

    with gr.Tabs():
        with gr.TabItem("Verilog RTL"):
            rtl_out = gr.Code(label="Generated Verilog", language="verilog", lines=20)
        with gr.TabItem("SDC Constraints"):
            sdc_out = gr.Code(label="SDC Constraints", language="tcl", lines=15)
        with gr.TabItem("Design Report"):
            report_out = gr.Markdown(label="Report")

    btn.click(
        fn=generate_chip,
        inputs=[spec, pdk, freq, width],
        outputs=[rtl_out, sdc_out, report_out],
    )

if __name__ == "__main__":
    demo.launch()
