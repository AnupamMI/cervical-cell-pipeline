#!/bin/bash
# Step 3: pick the winner (rules in pick_winner.py) -> results/winner.json + a section appended to RESULTS.md
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"$PY" "$ROOT/steps/pick_winner.py"
test -s "$ROOT/results/winner.json"
