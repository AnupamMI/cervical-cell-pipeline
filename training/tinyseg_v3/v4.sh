#!/bin/bash
# V4 = V0 recipe (raw RGB, 40 epochs, seed 42, from scratch) WITH aug_v2.photometric fully on (colour, blur, noise).
# Detached supervisor (tmux is not installed; same setsid + systemd-inhibit mechanism as overnight.sh).
#   1. train V4 -> phase2/runs/V4 (never retrains a finished run; one retry into V4_r1 on crash / NaN)
#   2. evaluate V4 vs v2_1 exactly as Step 2 (eval_final.py --models V4,v2_1 --out results_v4); waits until the
#      v0.3 release job (lock file .v03_busy) is finished so ms/cell is timed on an idle machine
#   3. decide (steps/decide_v4.py): V4 replaces v2_1 only if primary higher AND CIs don't overlap -> results/RESULTS_V4.md
#   4. only if V4 wins: wait for tag v0.3, then steps/release.py --tag v0.4 with V4 installed (tests, samples, smoke)
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
export PY="$HOME/Downloads/ANUPAM KA HOOD/seg_audit/venv/bin/python"
LOG="$ROOT/v4.log"; ST="$ROOT/v4_status"; mkdir -p "$ST"
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
status() { printf '{"step": "%s", "state": "%s", "reason": "%s", "time": "%s"}\n' "$1" "$2" "$3" "$(date -Iseconds)" > "$ST/$1.json"; log "STATUS $1: $2 $3"; }
log "===== v4.sh started (pid $$) ====="
df -BG --output=avail "$ROOT" | tail -1 | tr -d ' G' | { read g; [ "$g" -lt 20 ] && log "WARNING: only ${g} GB free (need >= 20); continuing, nothing removed" || log "disk: ${g} GB free"; }

# 1. train
run=""
for d in V4 V4_r1; do [ -f phase2/runs/$d/done.json ] && run=$d && break; done
if [ -z "$run" ]; then
  for d in V4 V4_r1; do
    [ -e phase2/runs/$d ] && { log "phase2/runs/$d exists without done.json: not overwritten, trying next"; continue; }
    log "training V4 -> phase2/runs/$d"
    ( cd phase2 && "$PY" train_v3.py --variant V0 --photometric --epochs 40 --workers 10 --seed 42 --out runs/$d > runs_$d.log 2>&1 ) &
    tp=$!
    while kill -0 $tp 2>/dev/null; do
      sleep 600; l=$(grep -E "^ep " phase2/runs/$d/log.txt 2>/dev/null | tail -1); log "training $d: ${l:0:160}"
    done
    wait $tp; rc=$?
    if [ -f phase2/runs/$d/done.json ]; then run=$d; break; fi
    log "training $d failed (rc $rc): $(tail -3 phase2/runs_$d.log | tr '\n' ' ' | cut -c1-300)"
  done
fi
if [ -z "$run" ]; then status train failed "V4 failed twice"; exit 1; fi
status train ok "$run"

# 2. evaluate
while [ -e "$ROOT/.v03_busy" ]; do sleep 60; done
if [ -f results_v4/per_cell.csv ] && [ -f results_v4/eval.json ]; then log "results_v4 exists: evaluation not rerun"
else
  ( cd phase2 && "$PY" eval_final.py --models V4,v2_1 --out ../results_v4 ) >> "$LOG" 2>&1 || { status eval failed "eval_final.py exit"; exit 1; }
fi
status eval ok ""

# 3. decide
"$PY" steps/decide_v4.py --run "phase2/runs/$run" >> "$LOG" 2>&1 || { status decide failed "decide_v4.py exit"; exit 1; }
status decide ok "$(python3 -c 'import json;print(json.load(open("results_v4/decision.json"))["winner"])')"

# 4. v0.4 only if V4 won
if [ "$(python3 -c 'import json;print(json.load(open("results_v4/decision.json"))["winner"])')" = "V4" ]; then
  until git -C "$HOME/Downloads/cervical-cell-pipeline" ls-remote --tags origin 2>/dev/null | grep -q "refs/tags/v0.3$"; do
    [ -e "$ROOT/.v03_failed" ] && { status release_v04 skipped "v0.3 release failed; v0.4 not built on top of it"; exit 1; }
    sleep 120
  done
  "$PY" steps/release.py --tag v0.4 --install-run "phase2/runs/$run" --name V4 --readme-head results_v4/README_HEAD_v04.md >> "$LOG" 2>&1 \
    && status release_v04 ok "" || status release_v04 failed "release.py exit (main untouched if before publish)"
else
  status release_v04 skipped "V4 did not replace v2_1 (see results/RESULTS_V4.md)"
fi
log "===== v4.sh finished ====="
