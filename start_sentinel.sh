#!/usr/bin/env bash
# SENTINEL launcher — starts the whole stack headless, in the correct order,
# each in its own tmux session (survives SSH drops).
#
# Order matters on 8 GB unified memory:
#   1) free the desktop GUI (~2.5 GB)
#   2) start llama-server FIRST so it claims GPU memory
#   3) then the engine (TensorFlow/PyTorch load on top)
#   4) then the dashboard
#
# Adjust the paths below to match your device.

MODEL=~/models/lfm2-vl/LFM2-VL-1.6B-Q4_0.gguf
MMPROJ=~/models/lfm2-vl/mmproj-LFM2-VL-1.6B-Q8_0.gguf
LLAMA=~/llama.cpp/build/bin/llama-server
PYTHON=~/onvif_env/bin/python   # venv python (has cv2, ultralytics, deepface, dotenv, flask)

echo "[1/4] Going headless (free desktop RAM)..."
sudo systemctl stop display-manager 2>/dev/null || true
tmux kill-server 2>/dev/null || true

echo "[2/4] Starting llama-server (VLM)..."
tmux new-session -d -s llm -c ~ "GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 $LLAMA \
  --model $MODEL --mmproj $MMPROJ \
  --host 127.0.0.1 --port 8080 --n-gpu-layers 999 --ctx-size 2048 --parallel 1"

echo "    Waiting for llama-server to listen..."
for i in $(seq 1 45); do
  if curl -s http://127.0.0.1:8080/v1/models >/dev/null 2>&1; then
    echo "    ✅ llama-server UP"; break
  fi
  sleep 2
done

echo "[3/4] Starting surveillance engine..."
tmux new-session -d -s engine -c ~ "$PYTHON ~/surveillance4_1.py"

echo "[4/4] Starting dashboard..."
tmux new-session -d -s dash -c ~ "$PYTHON ~/dashboard.py"

sleep 3
tmux ls
echo ""
echo "Dashboard:  http://<jetson-ip>:5000   (login: DASH_USER / DASH_PASS from .env)"
echo "Stop all:   tmux kill-server"
echo "Watch:      tmux attach -t engine   (detach: Ctrl-b then d)"
