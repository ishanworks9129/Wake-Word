#!/usr/bin/env bash
# Starts Wake Word Studio in WSL. Used by the VS Code task "Studio: start"; run from the repo root:
#   wsl -d Ubuntu --cd "<repo>" -- bash training/studio/start_wsl.sh ~/ww/studio_base
set -euo pipefail
base="${1:-$HOME/ww/studio_base}"
base="${base/#\~/$HOME}"
cd "$(dirname "$0")/.."
if [ ! -f ~/ww/.venv/bin/activate ]; then
  echo "No Studio environment yet: run the task 'Studio: set up WSL environment (once)' first." >&2
  exit 1
fi
. ~/ww/.venv/bin/activate
exec python -m studio.app --base "$base" --template configs/uno.yaml --jobs ~/ww/studio-jobs --host 0.0.0.0
