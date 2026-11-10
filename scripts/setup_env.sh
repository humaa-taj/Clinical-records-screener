#!/usr/bin/env bash
# Linux/macOS setup: creates .venv, installs dependencies and downloads what the app needs.
# Usage: bash scripts/setup_env.sh        (CPU PyTorch, works on any machine)
#        CRS_CUDA=1 bash scripts/setup_env.sh   (CUDA 12.8 PyTorch for an NVIDIA GPU)
set -euo pipefail
cd "$(dirname "$0")/.."

[ -x .venv/bin/python ] || python3 -m venv .venv
PY=.venv/bin/python

$PY -m pip install --upgrade pip
if [ "${CRS_CUDA:-0}" = "1" ]; then
  $PY -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
else
  $PY -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
fi
$PY -m pip install -r requirements.txt

# Qwen2.5-1.5B (target) and Qwen2.5-0.5B (draft) at the locked revisions, plus NLTK punkt_tab.
$PY scripts/fetch_resources.py --app
# Fine-tuned LoRA adapters from Hugging Face at the revisions in adapters.lock.json.
$PY scripts/fetch_adapters.py

$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "Setup complete. Start the app with: ./start_app.sh"
