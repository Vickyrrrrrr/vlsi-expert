#!/usr/bin/env python3
"""VLSI Expert — Gradio Demo. Uses locally merged FFN model."""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "models/vlsi-moe-ffn-merged/merged"

print(f"Loading VLSI Expert model from {MODEL_PATH}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,
)
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tok.pad_token = tok.eos_token
print("✅ Model loaded!")


def design_chip(spec: str, pdk: str, freq: int, width: int = 8) -> tuple:
    """Generate Verilog RTL + SDC from specification."""
    prompt = (
        f"Generate correct, synthesizable Verilog RTL for the following specification.\n"
        f"Target: {pdk} PDK at {freq}MHz, {width}-bit data width.\n\n"
        f"### Specification\n{spec}\n\n### Verilog RTL\nmodule"
    )

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=800, temperature=0.2,
            do_sample=True, pad_token_id=tok.eos_token_id, use_cache=False,
        )

    verilog = tok.decode(out[0], skip_special_tokens=True)

    # Generate SDC constraints
    sdc_prompt = (
        f"Generate SDC timing constraints for this module. Clock: {freq}MHz. PDK: {pdk}.\n\n"
        f"```verilog\n{verilog[:1500]}\n```\n\nSDC constraints:"
    )
    sdc_inputs = tok(sdc_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        sdc_out = model.generate(
            **sdc_inputs, max_new_tokens=300, temperature=0.1,
            do_sample=True, pad_token_id=tok.eos_token_id, use_cache=False,
        )
    sdc = tok.decode(sdc_out[0], skip_special_tokens=True)

    report = (
        f"**Design Generated**\n\n"
        f"- PDK: {pdk}\n- Clock: {freq}MHz\n- Width: {width}-bit\n"
        f"- Lines: {len(verilog.splitlines())}\n\n"
        f"**Model:** FFN-merged Qwen2.5-Coder-32B + DeepSeek-R1-32B\n"
        f"**Method:** DARE+TIES FFN-only merge | MI300X | ROCm 7.2"
    )
    return verilog, sdc, report


with gr.Blocks(title="VLSI Expert — AI Chip Designer") as demo:
    gr.Markdown("# VLSI Expert — AI Chip Designer")
    gr.Markdown(
        "FFN-merged: Qwen2.5-Coder (generation) + DeepSeek-R1 (reasoning). "
        "DARE+TIES merge on FFN layers only. One model, both capabilities. "
        "[GitHub](https://github.com/Vickyrrrrrr/vlsi-expert)"
    )

    with gr.Row():
        with gr.Column(scale=2):
            spec = gr.Textbox(
                label="Chip Specification",
                placeholder="8-bit up counter with synchronous reset and enable",
                lines=3,
            )
            with gr.Row():
                pdk = gr.Dropdown(["sky130", "gf180mcu", "asap7", "nangate45"], value="sky130", label="PDK")
                freq = gr.Slider(10, 2000, value=100, step=10, label="Clock (MHz)")
                width = gr.Slider(4, 64, value=8, step=4, label="Data Width")
            btn = gr.Button("Generate Chip Design", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### Architecture")
            gr.Markdown(
                "**Coder** (Qwen2.5-Coder-32B) — generates Verilog RTL\n"
                "**Reason** (DeepSeek-R1-32B) — analyzes, fixes errors\n"
                "FFN-only merge keeps attention from Coder\n"
                "Both models' knowledge in one"
            )

    with gr.Tabs():
        with gr.TabItem("Verilog RTL"):
            rtl = gr.Code(label="Generated Verilog", language="c", lines=18)
        with gr.TabItem("SDC Constraints"):
            sdc_out = gr.Code(label="SDC Constraints", lines=10)
        with gr.TabItem("Report"):
            report_out = gr.Markdown()

    btn.click(fn=design_chip, inputs=[spec, pdk, freq, width], outputs=[rtl, sdc_out, report_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
