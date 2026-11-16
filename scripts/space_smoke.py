#!/usr/bin/env python3
"""Smoke test of the deployed (private) Space: health, frontend, every sample with and without consensus,
download links, the 25 MB limit. Prints time per FOV (server-side and wall clock) and writes a JSON report.

    python scripts/space_smoke.py [--space AnupamZZZZ/cervical-cell-demo] [--json reports/space_smoke.json]

Auth: HF_TOKEN (private Spaces accept it as a bearer token; it is never printed).
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent.parent


def host(space):
    owner, name = space.split("/")
    return f"https://{owner.lower()}-{name.lower().replace('_', '-').replace('.', '-')}.hf.space"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--space", default="AnupamZZZZ/cervical-cell-demo")
    ap.add_argument("--json", default=str(HERE / "reports" / "space_smoke.json"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    base = host(a.space)
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    fails, rep = [], {"space": a.space, "url": base, "runs": []}

    def check(cond, msg):
        print(("  ok    " if cond else "  FAIL  ") + msg, flush=True)
        if not cond:
            fails.append(msg)

    # the container may still be loading models / warming up right after RUNNING
    for _ in range(40):
        try:
            h = s.get(base + "/api/health", timeout=30)
            if h.status_code == 200 and h.json().get("status") == "ok":
                break
        except requests.RequestException:
            pass
        time.sleep(6)
    hj = h.json()
    check(h.status_code == 200 and hj["status"] == "ok", "health ok")
    check(hj["bundle_check"] == [], "bundle matches config.yaml")
    check(all(m["onnx_sha256"] == m["onnx_sha256_manifest"] for m in hj["models"].values()), "ONNX SHA256s match manifest")
    rep["health"] = {k: hj[k] for k in ("models", "versions", "providers", "consensus_available", "typical_ms")}
    page = s.get(base + "/", timeout=30)
    check(page.status_code == 200 and "<div id=\"root\">" in page.text, "frontend served")
    samples = s.get(base + "/api/samples", timeout=30).json()["samples"]
    check(len(samples) == 10, f"{len(samples)} samples")
    for smp in samples[: a.limit or None]:
        for cons in (True, False):
            t0 = time.perf_counter()
            r = s.post(base + f"/api/infer?consensus={str(cons).lower()}", data={"sample": smp["name"]}, timeout=300)
            wall = time.perf_counter() - t0
            if r.status_code != 200:
                check(False, f"{smp['name']} consensus={cons}: HTTP {r.status_code} {r.text[:200]}")
                continue
            j = r.json()
            tm = j["timings_ms"]
            rep["runs"].append({"sample": smp["name"], "consensus": cons, "cells": len(j["cells"]),
                                "status": j["counts"]["status"], "server_ms": tm["request"], "wall_s": round(wall, 2),
                                "stages_ms": {k: tm[k] for k in ("detect", "consensus", "segment", "gate")}})
            print(f"  {smp['name']} consensus={'on ' if cons else 'off'}  {len(j['cells']):3d} cells  "
                  f"server {tm['request'] / 1000:5.2f}s  wall {wall:5.2f}s  "
                  f"(detect {tm['detect']:.0f} ms, consensus {tm['consensus']:.0f}, segment {tm['segment']:.0f}, gate {tm['gate']:.0f})",
                  flush=True)
            if smp is samples[0] and cons:
                for k, url in j["downloads"].items():
                    d = s.get(base + url, timeout=60)
                    check(d.status_code == 200 and len(d.content) > 100, f"download {k} ({len(d.content)} bytes)")
    big = s.post(base + "/api/infer", files={"file": ("big.png", b"\0" * (25 * 1024 * 1024 + 1))}, timeout=120)
    check(big.status_code == 413, f"25 MB limit -> {big.status_code}")
    for cons in (True, False):
        xs = [r["server_ms"] / 1000 for r in rep["runs"] if r["consensus"] == cons]
        ws = [r["wall_s"] for r in rep["runs"] if r["consensus"] == cons]
        if xs:
            rep[f"time_per_fov_consensus_{'on' if cons else 'off'}"] = {
                "server_median_s": round(statistics.median(xs), 2), "server_max_s": round(max(xs), 2),
                "wall_median_s": round(statistics.median(ws), 2), "n": len(xs)}
            print(f"time per FOV, consensus {'on ' if cons else 'off'}: server median {statistics.median(xs):.2f}s "
                  f"(max {max(xs):.2f}s), wall median {statistics.median(ws):.2f}s  [n={len(xs)}]")
    rep["passed"] = not fails
    Path(a.json).write_text(json.dumps(rep, indent=1))
    print(("PASS" if not fails else f"FAIL ({len(fails)})") + f" -> {a.json}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
