#!/usr/bin/env python3
"""Time per FOV of the web app engine (ONNX Runtime, CPU) on samples/, for every detector combination:
all three detectors on, consensus (YOLO26l) off, clusters (detector C) off, both off.

    python tools/bench_app.py [--threads 0] [--repeat 1] [--json reports/app_timing.json]

--threads 0 = ONNX Runtime default (all cores); 2 = what the Space's CPU basic hardware gets (NCMASK_THREADS=2).
One warm-up run per combination first. Times are the engine's process_fov call (decode excluded).
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

COMBOS = {"all_on": (True, True), "consensus_off": (False, True), "clusters_off": (True, False), "both_off": (False, False)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--json", default=str(HERE / "reports" / "app_timing.json"))
    a = ap.parse_args(argv)
    from app.backend.engine import Engine
    eng = Engine(device="cpu", threads=a.threads)
    files = sorted((HERE / "samples").glob("*.tiff"))
    imgs = [eng.decode(f)[0] for f in files]
    out = {"cpu": platform.processor() or platform.machine(), "cores": os.cpu_count(), "threads": a.threads or "default",
           "n_fovs": len(imgs), "repeat": a.repeat, "combos": {}}
    for name, (cons, clus) in COMBOS.items():
        eng.infer(imgs[0], cons, clusters=clus)    # warm-up
        tot, stages = [], {k: [] for k in ("detect", "consensus", "clusters", "segment", "gate")}
        for _ in range(a.repeat):
            for img in imgs:
                t = time.perf_counter()
                r = eng.infer(img, cons, clusters=clus)
                tot.append(time.perf_counter() - t)
                for k in stages:
                    stages[k].append(r["timings_ms"][k])
        out["combos"][name] = {"consensus": cons, "clusters": clus, "median_s": round(statistics.median(tot), 2),
                               "mean_s": round(statistics.mean(tot), 2), "max_s": round(max(tot), 2),
                               "stage_median_ms": {k: round(statistics.median(v)) for k, v in stages.items()}}
        c = out["combos"][name]
        print(f"{name:14s} median {c['median_s']:5.2f}s  mean {c['mean_s']:5.2f}s  max {c['max_s']:5.2f}s  "
              + "  ".join(f"{k} {v}" for k, v in c["stage_median_ms"].items()), flush=True)
    Path(a.json).write_text(json.dumps(out, indent=1))
    print("->", a.json)


if __name__ == "__main__":
    main()
