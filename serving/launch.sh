#!/bin/bash
# SiliconSmith AI — vLLM Server Launch Script
# Run this INSIDE the ROCm Docker container on your AMD MI300X droplet
# Usage: bash serving/launch.sh

set -e

echo "================================================"
echo "  SiliconSmith AI — vLLM Server Launch"
echo "  Hardware: AMD Instinct MI300X + ROCm"
echo "================================================"

# Network interface config
export IFACE=eth0
export GLOO_SOCKET_IFNAME="$IFACE"
export NCCL_SOCKET_IFNAME="$IFACE"
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

# ROCm / aiter optimization
export VLLM_ROCM_USE_AITER=1

MODEL_PATH="/app/vlsi-moe-yarn"
PORT=8000

echo "Model path:     $MODEL_PATH"
echo "Host IP:        $VLLM_HOST_IP"
echo "Port:           $PORT"
echo "Max context:    262144 tokens"
echo "dtype:          bfloat16"
echo "KV cache dtype: fp8"
echo ""
echo "Starting vLLM server..."
echo "API will be available at: http://$VLLM_HOST_IP:$PORT/v1"
echo "================================================"

vllm serve "$MODEL_PATH" \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --tensor-parallel-size 1 \
  --host 0.0.0.0 \
  --port "$PORT"
