#!/usr/bin/env bash
# One-time setup for Wake Word Studio in WSL (Ubuntu): a Python 3.12 environment at ~/ww/.venv.
#   wsl -d Ubuntu -- bash "/mnt/c/Users/<you>/Desktop/Wake Word/training/studio/setup_wsl.sh"
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
mkdir -p ~/ww
cd ~/ww
[ -d .venv ] || uv venv -p 3.12 .venv
. .venv/bin/activate
uv pip install --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu -r "$here/requirements.txt"
python -c "import torch, onnxruntime, piper, fastapi; print('Studio environment ready:', torch.__version__, onnxruntime.__version__)"
