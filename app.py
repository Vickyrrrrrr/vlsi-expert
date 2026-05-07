import os
import sys

# --- Hugging Face Storage Fix ---
if os.path.exists("/data") and os.access("/data", os.W_OK):
    # Redirect all caches to persistent storage to avoid 50GB eviction limit
    os.environ["HF_HOME"] = "/data/.cache"
    os.environ["HF_HUB_CACHE"] = "/data/.cache/huggingface/hub"
    os.environ["TRANSFORMERS_CACHE"] = "/data/.cache/huggingface/hub"
    os.environ["HF_DATASETS_CACHE"] = "/data/.cache/huggingface/datasets"
    os.environ["TORCH_HOME"] = "/data/.cache/torch"
    os.environ["TMPDIR"] = "/data/tmp"
    os.makedirs("/data/.cache", exist_ok=True)
    os.makedirs("/data/tmp", exist_ok=True)

import gradio as gr
import httpx
import json
import asyncio
from pathlib import Path
from datetime import datetime

# Import project config
try:
    from config import BRIDGE_PORT, TEACHER_MODEL_ID, STUDENT_MODEL_ID, DRAFT_MODEL_ID, MODELS_DIR
except ImportError:
    # Fallback if config not in path
    BRIDGE_PORT = 8002
    TEACHER_MODEL_ID = "vxkyyy/vlsi-moe-ffn-merged-formal"
    STUDENT_MODEL_ID = "Qwen/Qwen2.5-Coder-14B-Instruct"
    DRAFT_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    MODELS_DIR = Path("models")

# Constants
BRIDGE_URL = f"http://localhost:{BRIDGE_PORT}"

# --- CSS for Premium Look ---
CUSTOM_CSS = """
.container { max-width: 1200px; margin: auto; }
.header { text-align: center; margin-bottom: 2rem; }
.header h1 { font-size: 2.5rem; color: #2D3436; }
.metric-card { 
    background: #f8f9fa; 
    border-radius: 12px; 
    padding: 1.5rem; 
    border-left: 5px solid #00D1B2;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}
.status-ok { color: #00D1B2; font-weight: bold; }
.status-warn { color: #FFD166; font-weight: bold; }
.status-err { color: #EF476F; font-weight: bold; }
.chat-container { border-radius: 15px; overflow: hidden; }
"""

# --- Background Processes ---
def start_bridge():
    import subprocess
    import time
    
    # Check if already running
    try:
        with httpx.Client() as client:
            resp = client.get(f"{BRIDGE_URL}/health")
            if resp.status_code == 200:
                print("Bridge already running.")
                return
    except:
        pass
        
    print("Starting AgentIC Bridge...")
    subprocess.Popen([sys.executable, "bridge.py"], start_new_session=True)
    time.sleep(2) # Give it time to bind

# --- API Helpers ---
async def get_metrics():
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{BRIDGE_URL}/v1/dashboard")
            if resp.status_code == 200:
                return resp.json().get("dashboard", {})
    except Exception:
        return None

async def chat_with_model(message, history, model="vlsi-expert"):
    messages = []
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{BRIDGE_URL}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 2048,
                }
            )
            if resp.status_code == 200:
                result = resp.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"Error: {resp.status_code} - {resp.text[:200]}"
    except Exception as e:
        return f"Connection Error: {str(e)}. Is bridge.py running?"

# --- Gradio UI ---
def create_ui():
    with gr.Blocks(css=CUSTOM_CSS, title="Qwen-VLSI-SOTA-Sprint Dashboard") as demo:
        with gr.Column(elem_classes="container"):
            # Header
            gr.Markdown(
                """
                # 🚀 Qwen-VLSI-SOTA-Sprint
                ### SOTA SystemVerilog Distillation — Teacher (33B) to Student (14B Ternary)
                """,
                elem_classes="header"
            )
            
            with gr.Tabs():
                # Tab 1: Dashboard
                with gr.TabItem("📊 Training Dashboard"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            active_model = gr.Label(label="Active Serving Model", value="Searching...")
                            teacher_status = gr.Markdown("Teacher vLLM: `Checking...`")
                            draft_status = gr.Markdown("Draft vLLM: `Checking...`")
                            
                        with gr.Column(scale=2):
                            with gr.Row():
                                samples = gr.Number(label="Verified Samples", value=0)
                                refactors = gr.Number(label="Successful Refactors", value=0)
                                density = gr.Textbox(label="Proof Density (%)", value="0%")
                    
                    with gr.Row():
                        phase_info = gr.Textbox(label="Current Phase", value="Initializing...")
                        checkpoint_size = gr.Textbox(label="Latest Checkpoint Size", value="0 GB")

                    refresh_btn = gr.Button("🔄 Refresh Metrics", variant="primary")

                # Tab 2: Designer (Chat)
                with gr.TabItem("💻 VLSI Designer"):
                    gr.Markdown("Ask the expert to generate RTL, SVA properties, or fix bugs.")
                    chatbot = gr.Chatbot(height=500, show_label=False)
                    msg = gr.Textbox(placeholder="E.g., Design an AXI4-Lite slave for a 1KB SRAM...", label="Specification")
                    
                    with gr.Row():
                        submit = gr.Button("Generate", variant="primary")
                        clear = gr.Button("Clear History")

                # Tab 3: Model Management
                with gr.TabItem("⚙️ Model Config"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Serving Status")
                            model_selector = gr.Radio(choices=["teacher", "draft", "student"], label="Select Active Model", value="teacher")
                            swap_btn = gr.Button("Apply Hot-Swap", variant="primary")
                            swap_output = gr.Markdown("")
                        
                        with gr.Column():
                            gr.Markdown("### Storage Status")
                            storage_info = gr.Markdown("Checking storage...")
                            check_storage_btn = gr.Button("Check Disk Space & Models")
                    
                    gr.Markdown("---")
                    gr.Markdown("#### 💡 HF Spaces Tip")
                    gr.Markdown("If you get 'Storage Limit Exceeded', ensure you have **Persistent Storage** enabled in Space Settings and the app will automatically use `/data` for the 100GB+ of model weights.")

            # Logic
            async def update_dashboard():
                m = await get_metrics()
                if not m:
                    return {
                        active_model: "Bridge Offline",
                        teacher_status: "Teacher vLLM: 🔴 `Disconnected`",
                        draft_status: "Draft vLLM: 🔴 `Disconnected`",
                        samples: 0,
                        refactors: 0,
                        density: "0%",
                        phase_info: "Bridge not reachable at port 8002",
                        checkpoint_size: "N/A"
                    }
                
                stats = m.get("sampling_stats", {})
                ckpt = m.get("checkpoint") or {}
                
                return {
                    active_model: m.get("active_model", "Unknown").upper(),
                    teacher_status: f"Teacher vLLM: {'🟢 `Online`' if m.get('teacher_reachable') else '🔴 `Offline`'}",
                    draft_status: f"Draft vLLM: {'🟢 `Online`' if m.get('draft_reachable') else '🔴 `Offline`'}",
                    samples: stats.get("total_samples", 0),
                    refactors: stats.get("successful_refactors", 0),
                    density: stats.get("proof_density", "0%"),
                    phase_info: m.get("phase", "Unknown"),
                    checkpoint_size: f"{ckpt.get('size_gb', 0):.2f} GB" if ckpt else "None"
                }

            async def handle_swap(model_name):
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        resp = await client.post(f"{BRIDGE_URL}/v1/swap", json={"model": model_name})
                        if resp.status_code == 200:
                            res = resp.json()
                            return f"✅ **Swapped to {model_name}**. VRAM Saved: {res.get('vram_saved_gb', 0)} GB"
                        return f"❌ **Error**: {resp.json().get('detail', 'Unknown error')}"
                except Exception as e:
                    return f"❌ **Connection Error**: {str(e)}"

            async def check_storage():
                import shutil
                total, used, free = shutil.disk_usage("/")
                data_total, data_used, data_free = (0,0,0)
                if os.path.exists("/data"):
                    data_total, data_used, data_free = shutil.disk_usage("/data")
                
                models_found = []
                for m in ["vlsi-moe-ffn-merged-formal", "Qwen2.5-Coder-14B-Instruct", "Qwen2.5-Coder-1.5B-Instruct"]:
                    if (MODELS_DIR / m).exists():
                        models_found.append(f"✅ {m}")
                    else:
                        models_found.append(f"❌ {m}")
                
                status = f"**Root Disk**: {free/1e9:.1f}GB free / {total/1e9:.1f}GB total\n\n"
                if os.path.exists("/data"):
                    status += f"**Persistent Storage (/data)**: {data_free/1e9:.1f}GB free / {data_total/1e9:.1f}GB total\n\n"
                else:
                    status += "**Persistent Storage**: Not detected (using ephemeral root)\n\n"
                
                status += "**Models on Disk**:\n" + "\n".join(models_found)
                return status

            refresh_btn.click(update_dashboard, outputs=[active_model, teacher_status, draft_status, samples, refactors, density, phase_info, checkpoint_size])
            swap_btn.click(handle_swap, inputs=[model_selector], outputs=[swap_output])
            check_storage_btn.click(check_storage, outputs=[storage_info])
            
            def respond(message, history):
                bot_message = asyncio.run(chat_with_model(message, history))
                history.append((message, bot_message))
                return "", history

            submit.click(respond, [msg, chatbot], [msg, chatbot])
            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)

            # Auto-refresh on load
            demo.load(update_dashboard, outputs=[active_model, teacher_status, draft_status, samples, refactors, density, phase_info, checkpoint_size])

    return demo

if __name__ == "__main__":
    # Ensure /data exists if on HF
    if os.path.exists("/data"):
        os.makedirs("/data/models", exist_ok=True)
        os.makedirs("/data/checkpoints", exist_ok=True)
    
    # Start the backend bridge
    start_bridge()
    
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7860)
