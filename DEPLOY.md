# AMD GPU Deployment Guide

## Setup AMD Developer Cloud Instance

### 1. Launch MI300X Instance
```bash
# From AMD Developer Cloud Console
# Select: Ubuntu 22.04 + ROCm 6.2 + MI300X
# Instance: 1× MI300X (192 GB VRAM, 20 vCPU)
# Boot disk: 720 GB NVMe
```

### 2. SSH into Instance
```bash
ssh -i your-key.pem ubuntu@<instance-ip>
```

### 3. Install Dependencies (5 min)
```bash
# ROCm PyTorch (pre-installed on AMD images)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# ML stack
pip install transformers accelerate bitsandbytes peft datasets trl
pip install gradio huggingface_hub

# Verify GPU is visible
python -c "import torch; print(f'GPU: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}')"
# Expected: GPU: 1 x AMD Instinct MI300X
```

### 4. Clone & Setup
```bash
git clone https://github.com/Vickyrrrrrr/vlsi-expert.git
cd vlsi-expert

# Download base models (one-time, ~64GB each)
python -c "
from transformers import AutoModelForCausalLM
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-32B-Instruct')
AutoModelForCausalLM.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-32B')
"
```

### 5. Run Training (Day 1)
```bash
# Collect data
python scripts/collect_data.py

# Train CODER head (8 hours)
python scripts/train_coder.py

# Train INSTRUCT head (6 hours)
python scripts/train_instruct.py
```

### 6. Test with AgentIC (Day 2)
```bash
# Clone AgentIC
git clone https://github.com/Vickyrrrrrr/AgentIC.git
cd AgentIC
pip install -e .

# Run evaluation: 20 test designs through full pipeline
cd ../vlsi-expert
python eval/evaluate.py
```

### 7. Upload to HuggingFace (Day 2)
```bash
huggingface-cli login
huggingface-cli upload Vickyrrrrrr/vlsi-coder-lora models/vlsi-coder-lora
huggingface-cli upload Vickyrrrrrr/vlsi-instruct-lora models/vlsi-instruct-lora
```

### 8. Launch Demo
```bash
python gradio/app.py
# → http://localhost:7860
```

## Cost: $0
- AMD Developer Cloud: $100 free credits cover everything
- HuggingFace Hub: free model hosting
- HuggingFace Spaces: free Gradio hosting
- Total cost for hackathon: $0
