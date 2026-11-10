#!/usr/bin/env bash
# Starts CRS on http://127.0.0.1:8600 and opens it in the browser. Linux/macOS twin of start_app.bat.
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "The Python environment is missing. Run: bash scripts/setup_env.sh"; exit 1
fi
if [ ! -f models/qwen2.5-1.5b/config.json ]; then
  echo "The model files are missing. Run: bash scripts/setup_env.sh"; exit 1
fi
if [ ! -f app/web/dist/index.html ]; then
  echo "The built interface is missing from app/web/dist."; exit 1
fi

echo "Starting CRS. Your browser will open shortly."
( sleep 3; xdg-open http://127.0.0.1:8600 >/dev/null 2>&1 || open http://127.0.0.1:8600 >/dev/null 2>&1 ) &
.venv/bin/python -m app.server
echo "CRS has stopped."
