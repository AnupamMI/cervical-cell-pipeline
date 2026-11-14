#!/usr/bin/env python3
"""Compare a pipeline output folder with the committed reference outputs (samples/expected).

Pass when, for every FOV in the reference: the output exists, has the same number of cells, every reference cell is
matched one-to-one to an output cell (detector box IoU >= --box-iou, Hungarian), and each matched pair has
cell-mask Dice >= --min-dice and nucleus-mask Dice >= --min-dice (both empty counts as 1.0).
Small GPU-to-GPU differences (Jetson vs Colab T4) are expected to stay above 0.99.

    python scripts/check_against_samples.py --output out/ [--expected samples/expected] [--limit N]
Exit code 0 = pass, 1 = fail.
"""
import argparse, csv, sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]


def rows(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True)
    ap.add_argument("--expected", default=str(ROOT / "samples/expected"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N reference FOVs (sorted)")
    ap.add_argument("--box-iou", type=float, default=0.9)
    ap.add_argument("--min-dice", type=float, default=0.99)
    a = ap.parse_args()
    exp, out = Path(a.expected), Path(a.output)
    fovs = sorted(p.name for p in (exp / "fovs").iterdir() if (p / "done.json").exists())
    if a.limit:
        fovs = fovs[:a.limit]
    ok, worst = True, 1.0
    for fov in fovs:
        e, o = exp / "fovs" / fov, out / "fovs" / fov
        if not (o / "done.json").exists():
            print(f"FAIL {fov}: missing in output"); ok = False; continue
        er, orr = rows(e / "cells.csv"), rows(o / "cells.csv")
        if len(er) != len(orr):
            print(f"FAIL {fov}: {len(orr)} cells, expected {len(er)}"); ok = False; continue
        if not er:
            print(f"ok   {fov}: 0 cells"); continue
        B = lambda r: [float(r[k]) for k in ("box_x1", "box_y1", "box_x2", "box_y2")]
        M = np.array([[iou(B(x), B(y)) for y in orr] for x in er])
        ri, ci = linear_sum_assignment(-M)
        if (M[ri, ci] < a.box_iou).any() or len(ri) != len(er):
            print(f"FAIL {fov}: {(M[ri, ci] < a.box_iou).sum()} cell(s) without a matching box (IoU >= {a.box_iou})"); ok = False; continue
        labs = {k: (cv2.imread(str(e / f"{k}_labels.png"), cv2.IMREAD_UNCHANGED), cv2.imread(str(o / f"{k}_labels.png"), cv2.IMREAD_UNCHANGED))
                for k in ("cell", "nucleus")}
        ds = []
        for i, j in zip(ri, ci):
            ie, io = int(er[i]["cell_id"]), int(orr[j]["cell_id"])
            for k, (le, lo) in labs.items():
                ds.append(dice(le == ie, lo == io))
        mn = min(ds); worst = min(worst, mn)
        bad = sum(d < a.min_dice for d in ds)
        print(f"{'ok  ' if not bad else 'FAIL'} {fov}: {len(er)} cells, min mask Dice {mn:.4f}" + (f", {bad} mask(s) < {a.min_dice}" if bad else ""))
        ok &= not bad
    print(f"\n{'PASS' if ok else 'FAIL'}: {len(fovs)} FOV(s) checked, worst mask Dice {worst:.4f}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
