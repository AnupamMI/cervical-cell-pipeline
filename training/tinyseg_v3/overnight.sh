#!/bin/bash
# Overnight supervisor for TinyUNet v3 (Steps 1-6). Runs detached (setsid + systemd-inhibit), independent of
# the Claude Code session. Log: tinyseg_v3/overnight.log   Status: tinyseg_v3/overnight_status/<step>.json
#
#   Step 1  train   wait for the running training; restart unfinished variants (never a finished one), 1 retry each
#   Step 2  eval    steps/step2_eval.sh          (needs: step1 produced at least V0)
#   Step 3  winner  steps/step3_winner.sh        (needs: step2)
#   Step 4  pipe_v3 steps/step4_pipeline_v3.sh   (needs: step3)
#   Step 5  push    steps/step5_push_v02.sh      (needs: step4)
#   Step 6  report  steps/step6_morning_report.sh (always)
# Never deletes or overwrites runs, originals, data or zips. Never prints credentials.
set -u
ROOT="$HOME/Downloads/ANUPAM KA HOOD/tinyseg_v3"
P2="$ROOT/phase2"
export PY="$HOME/Downloads/ANUPAM KA HOOD/seg_audit/venv/bin/python"
LOG="$ROOT/overnight.log"
ST="$ROOT/overnight_status"
mkdir -p "$ST"
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
status() { # step state reason
  printf '{"step": "%s", "state": "%s", "reason": "%s", "time": "%s"}\n' "$1" "$2" "$3" "$(date -Iseconds)" > "$ST/$1.json"
  log "STATUS $1: $2 ${3:+($3)}"
}
ok() { [ -f "$ST/$1.json" ] && grep -q '"state": "ok"' "$ST/$1.json"; }

log "===== overnight.sh started (pid $$) ====="
log "detached with setsid + systemd-inhibit (tmux/screen not installed; installing tmux needs a sudo password, so not installed)"
FREE_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
if [ "$FREE_GB" -lt 20 ]; then log "WARNING: only ${FREE_GB} GB free (< 20 GB); continuing, nothing removed"; else log "disk: ${FREE_GB} GB free"; fi

# ---------------------------------------------------------------- Step 1: training
train_alive() { pgrep -f "train_v3.py" >/dev/null || pgrep -f "phase2/run_all.sh|./run_all.sh" >/dev/null; }
done_dir() { for d in "$P2/runs/$1" "$P2/runs/$1_r1" "$P2/runs/$1_r2"; do [ -f "$d/done.json" ] && { echo "$d"; return 0; }; done; return 1; }
new_dir() { for d in "$P2/runs/$1" "$P2/runs/$1_r1" "$P2/runs/$1_r2" "$P2/runs/$1_r3"; do [ -e "$d" ] || { echo "$d"; return 0; }; done; return 1; }

last_progress=0
while train_alive; do
  now=$(date +%s)
  if [ $((now - last_progress)) -ge 600 ]; then
    for v in V0 V3 V1 V2; do
      for d in "$P2/runs/$v" "$P2/runs/$v"_r*; do
        [ -f "$d/log.txt" ] || continue
        [ -f "$d/done.json" ] && continue
        n=$(wc -l < "$d/log.txt"); s=$(tail -1 "$d/log.txt" | grep -oE '[0-9]+s$' | tr -d s)
        [ -n "$s" ] && log "training $v ($(basename "$d")): epoch $n/40, last epoch ${s}s, ETA this variant ~$(( (40 - n) * s / 60 )) min"
      done
    done
    last_progress=$now
  fi
  sleep 60
done
log "no training process alive"

for v in V0 V3 V1 V2; do
  if d=$(done_dir $v); then log "train $v: already finished in $(basename "$d") (not retrained)"; continue; fi
  tries=0
  while [ $tries -lt 2 ]; do
    d=$(new_dir $v) || { log "train $v: no free run directory"; break; }
    tries=$((tries + 1))
    log "train $v: attempt $tries -> $(basename "$d") (from scratch, 40 epochs, seed 42)"
    ( cd "$P2" && "$PY" train_v3.py --variant $v --epochs 40 --workers 10 --out "$d" >> "$P2/runs_$(basename "$d").log" 2>&1 )
    rc=$?
    if [ -f "$d/done.json" ]; then log "train $v: finished (rc $rc)"; break; fi
    log "train $v: FAILED attempt $tries (rc $rc; $( [ -f "$d/failed.json" ] && cat "$d/failed.json" ))"
  done
  done_dir $v >/dev/null || log "train $v: marked FAILED after $tries attempt(s)"
done
if "$PY" "$ROOT/steps/train_report.py" >> "$LOG" 2>&1 && done_dir V0 >/dev/null; then status step1_train ok ""; else
  if done_dir V0 >/dev/null; then status step1_train ok "train_report.py failed"; else status step1_train failed "V0 did not finish"; fi
fi

# ---------------------------------------------------------------- Steps 2-6
run_step() { # name script needs...
  local name=$1 script=$2; shift 2
  for dep in "$@"; do ok "$dep" || { status "$name" skipped "dependency $dep not ok"; return; }; done
  if [ ! -f "$ROOT/steps/$script" ]; then status "$name" failed "script steps/$script missing"; return; fi
  log "START $name ($script)"
  if bash "$ROOT/steps/$script" >> "$LOG" 2>&1; then status "$name" ok ""; else status "$name" failed "steps/$script exit code $?"; fi
}
run_step step2_eval step2_eval.sh step1_train
run_step step3_winner step3_winner.sh step2_eval
run_step step4_pipeline_v3 step4_pipeline_v3.sh step3_winner
run_step step5_push_v02 step5_push_v02.sh step4_pipeline_v3
run_step step6_morning_report step6_morning_report.sh
log "===== overnight.sh finished ====="
