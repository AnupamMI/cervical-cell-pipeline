#!/usr/bin/env bash
# Sync the web app + current ONNX bundle to the private Hugging Face Space, wait for the build, smoke-test it.
#   HF_TOKEN must be set.   scripts/deploy_space.sh [--space AnupamZZZZ/cervical-cell-demo] [extra deploy_space.py args]
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
py="$root/.venv/bin/python"; [ -x "$py" ] || py="$root/.venv/Scripts/python.exe"; [ -x "$py" ] || py=python3
exec "$py" "$root/scripts/deploy_space.py" --wait --smoke "$@"
