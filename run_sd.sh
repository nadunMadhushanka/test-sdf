#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Enforce strict offline operation
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=================================================="
echo " Starting Stable Diffusion (Realistic Vision v5.1)"
echo " Mode: Strictly Offline (127.0.0.1:7860)"
echo " GPU: NVIDIA RTX 3090"
echo "=================================================="

# Activate virtual environment
source "$DIR/venv/bin/activate"

# Launch browser after a short delay if in desktop session
(
  sleep 2
  if which google-chrome >/dev/null 2>&1; then
    google-chrome "http://127.0.0.1:7860" >/dev/null 2>&1 &
  elif which firefox >/dev/null 2>&1; then
    firefox "http://127.0.0.1:7860" >/dev/null 2>&1 &
  fi
) &

# Run local offline FastAPI server
exec python3 server.py
