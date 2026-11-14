#!/bin/bash
# Step 2: T2 independence check (only if its outputs are missing) + final evaluation -> results/RESULTS.md
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ ! -f "$ROOT/t2_verify/t2_verify_summary.json" ]; then
  (cd "$ROOT/t2_verify" && "$PY" -I verify_t2.py --eval "$HOME/Downloads/nc_ratio_new_evaluation" --out .)
fi
echo "[step2] T2 independence check: $(python3 -c "import json;print(json.load(open('$ROOT/t2_verify/t2_verify_summary.json'))['class_counts'])")"
cd "$ROOT/phase2"
"$PY" eval_final.py
test -s "$ROOT/results/RESULTS.md" && test -s "$ROOT/results/summary.csv"
echo "[step2] RESULTS.md written"
