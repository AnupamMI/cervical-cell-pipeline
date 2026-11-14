#!/usr/bin/env python3
"""Compare a pipeline output folder with the committed reference outputs (samples/expected).

Pass when, for every FOV in the reference: the output exists, has the same number of cells, every reference cell is
matched one-to-one to an output cell (detector box IoU >= --box-iou, Hungarian), the POOLED mask Dice over all matched
cells of the FOV is >= --min-dice (0.99) for the cell masks and for the nucleus masks, and EVERY matched cell has mask
Dice >= --min-cell-dice (0.95; both empty counts as 1.0). Cells between the two thresholds are listed as warnings.
Why pooled + per-cell floor: across devices (Jetson GPU vs CPU vs Colab T4) detector boxes move by < 0.2 px, which
shifts the crop rounding and changes a few pixels of small nuclei (measured on the Jetson, GPU vs CPU: identical cell
counts, pooled Dice > 0.99, single nucleus masks down to 0.984). A real regression still fails both rules.

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
    ap.add_argument("--min-dice", type=float, default=0.99, help="pooled Dice per FOV and structure")
    ap.add_argument("--min-cell-dice", type=float, default=0.95, help="floor for every single matched cell")
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
        ds, pooled = [], {}
        for k, (le, lo) in labs.items():
            inter = tot = 0
            for i, j in zip(ri, ci):
                x, y = le == int(er[i]["cell_id"]), lo == int(orr[j]["cell_id"])
                inter += int((x & y).sum()); tot += int(x.sum() + y.sum())
                ds.append((dice(x, y), k, er[i]["cell_id"]))
            pooled[k] = 1.0 if tot == 0 else 2 * inter / tot
        mn = min(d for d, *_ in ds); worst = min(worst, mn)
        floor_bad = [x for x in ds if x[0] < a.min_cell_dice]
        warn = [x for x in ds if a.min_cell_dice <= x[0] < a.min_dice]
        pool_bad = [k for k, v in pooled.items() if v < a.min_dice]
        fail = bool(floor_bad or pool_bad)
        print(f"{'FAIL' if fail else 'ok  '} {fov}: {len(er)} cells, pooled Dice cell {pooled['cell']:.4f} nucleus {pooled['nucleus']:.4f}, "
              f"min single-mask Dice {mn:.4f}" + (f"; {len(warn)} mask(s) in [{a.min_cell_dice}, {a.min_dice})" if warn else "")
              + (f"; {len(floor_bad)} mask(s) < {a.min_cell_dice}" if floor_bad else ""))
        ok &= not fail
    print(f"\n{'PASS' if ok else 'FAIL'}: {len(fovs)} FOV(s) checked, worst mask Dice {worst:.4f}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
