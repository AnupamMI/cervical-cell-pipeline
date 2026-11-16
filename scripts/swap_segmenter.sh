#!/usr/bin/env bash
# One command to swap the web app's segmenter: re-export ONNX -> parity test -> redeploy the Space.
#   scripts/swap_segmenter.sh [--weights models/seg/V4/best.pt] [--variant V1] [--no-deploy]
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
py="$root/.venv/bin/python"; [ -x "$py" ] || py="$root/.venv/Scripts/python.exe"; [ -x "$py" ] || py=python3
exec "$py" "$root/scripts/swap_segmenter.py" "$@"
