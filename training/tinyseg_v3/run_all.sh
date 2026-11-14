#!/bin/bash
# Train V0, V3 (no cache needed), then V1, V2 (wait for the cache build to finish). 40 epochs each, seed 42.
set -u
cd "$(dirname "$0")"
PY=../../seg_audit/venv/bin/python
for v in V0 V3; do
  [ -f runs/$v/done.json ] || $PY train_v3.py --variant $v --epochs 40 --workers 10 --out runs/$v > runs_$v.log 2>&1
done
while [ ! -f cache/fallbacks.json ]; do sleep 60; done
for v in V1 V2; do
  [ -f runs/$v/done.json ] || $PY train_v3.py --variant $v --epochs 40 --workers 10 --out runs/$v > runs_$v.log 2>&1
done
echo ALL_DONE
