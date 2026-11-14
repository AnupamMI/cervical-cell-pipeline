#!/bin/bash
# Step 6: morning report -> ~/Downloads/ANUPAM KA HOOD/MORNING_REPORT.md and reports/MORNING_REPORT.md in the repo (pushed).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$HOME/Downloads/cervical-cell-pipeline"
"$PY" "$ROOT/steps/morning_report.py" || exit 1
if [ -d "$REPO/.git" ]; then
  cd "$REPO"
  git add reports/MORNING_REPORT.md && git commit -q -m "reports: morning report $(date +%F)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && echo "[step6] committed" || echo "[step6] nothing to commit"
  if timeout 600 git push -q origin HEAD 2>&1; then echo "[step6] pushed"; else echo "[step6] push FAILED (report is committed locally)"; fi
fi
exit 0
