#!/usr/bin/env python3
"""3-panel review sheets + a summary table for a finished pipeline-v3 run (read-only on the run).

  left    YOLO26s boxes (id conf); solid = consensus (YOLO26l agrees), dashed = s_only
  middle  FULL measurement masks: cell outline (a colour per cell, neighbours differ), nucleus filled yellow
  right   status: cell filled green (measurable) / orange (review) / red (excluded) + first reason code

    python tools/make_sheets.py --run runs/prod_5fov --out reports/sheets_5fov [--why why.json]
Writes <out>/<fov>.png (full resolution), <out>/summary.md and <out>/summary.csv.
"""
import argparse, csv, json, sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ncmask.fov import draw_dashed  # noqa: E402
from ncmask.imageio import load_bgr8  # noqa: E402

PAL = [(0, 255, 0), (255, 255, 0), (0, 140, 255), (255, 0, 0), (255, 255, 255), (0, 0, 0), (0, 0, 255), (0, 120, 0),
       (255, 0, 255), (128, 128, 0)]
STATUS = {"measurable": (0, 200, 0), "review": (0, 165, 255), "excluded": (0, 0, 255)}


def label(img, txt, org, scale=0.7):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def full_masks(coco, shape):
    """cell_id -> (cell mask, nucleus mask) from the FULL-layer COCO polygons."""
    out = {}
    for a in coco["annotations"]:
        cid = a["attributes"]["cell_id"]
        m = np.zeros(shape, np.uint8)
        for poly in a["segmentation"]:
            cv2.fillPoly(m, [np.round(np.array(poly).reshape(-1, 2)).astype(np.int32)], 1)
        c, n = out.get(cid, (np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)))
        out[cid] = (m | c, n) if a["category_id"] == 2 else (c, m | n)
    return out


def colours(masks):
    ids = sorted(masks); col, k = {}, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    for i in ids:
        near = set()
        di = cv2.dilate(masks[i][0], k)
        for j in col:
            if (di & masks[j][0]).any():
                near.add(col[j])
        col[i] = next((c for c in range(len(PAL)) if c not in near), i % len(PAL))
    return {i: PAL[c] for i, c in col.items()}


def sheet(fov_dir, why=""):
    done = json.loads((fov_dir / "done.json").read_text())
    img, _ = load_bgr8(done["source"])
    rows = list(csv.DictReader(open(fov_dir / "cells.csv")))
    coco = json.loads((fov_dir / "cells_full.coco.json").read_text())
    excl = cv2.imread(str(fov_dir / "cell_labels.png"), cv2.IMREAD_UNCHANGED)
    H, W = img.shape[:2]
    m = full_masks(coco, (H, W)); cols = colours(m)
    p1, p2, p3 = img.copy(), img.copy(), img.copy()
    for r in rows:
        x1, y1, x2, y2 = (int(round(float(r[k]))) for k in ("box_x1", "box_y1", "box_x2", "box_y2"))
        if r["det_tag"] == "s_only":
            draw_dashed(p1, np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]), (255, 255, 0), 3)
        else:
            cv2.rectangle(p1, (x1, y1), (x2, y2), (255, 255, 0), 3)
        label(p1, f"{r['cell_id']} {float(r['conf']):.2f}", (x1 + 4, max(22, y1 + 24)))
    nuc_any = np.zeros((H, W), bool)
    for cid, (c, n) in m.items():
        nuc_any |= n > 0
    p2[nuc_any] = (0.55 * p2[nuc_any] + 0.45 * np.array([0, 255, 255])).astype(np.uint8)
    for cid, (c, n) in m.items():
        cs, _ = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(p2, cs, -1, cols[cid], 3, cv2.LINE_AA)
    over = p3.copy()
    for r in rows:
        cid = int(r["cell_id"]); col = STATUS[r["status"]]
        region = excl == cid
        if not region.any() and cid in m:
            region = m[cid][0] > 0
        over[region] = col
    p3 = cv2.addWeighted(over, 0.35, p3, 0.65, 0)
    for r in rows:
        cid = int(r["cell_id"]); col = STATUS[r["status"]]
        if cid in m:
            cs, _ = cv2.findContours(m[cid][0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(p3, cs, -1, col, 3, cv2.LINE_AA)
        x1, y1 = int(float(r["box_x1"])), int(float(r["box_y1"]))
        reason = r["reasons"].split(";")[0] if r["reasons"] else "ok"
        label(p3, f"{cid} {reason}", (x1 + 4, max(22, y1 + 24)), 0.6)
    sc = Counter(r["status"] for r in rows)
    gap = np.full((H, 20, 3), 255, np.uint8)
    head = np.full((120, 3 * W + 40, 3), 255, np.uint8)
    cv2.putText(head, f"{fov_dir.name}  |  {len(rows)} cells: measurable {sc['measurable']}, review {sc['review']}, "
                f"excluded {sc['excluded']}  |  image QC: {','.join(done.get('fov_qc') or []) or 'ok'}" + (f"  |  {why}" if why else ""),
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(head, "LEFT: YOLO26s boxes (dashed = s_only)   MIDDLE: full masks (outline = cell, yellow = nucleus)   "
                "RIGHT: status green measurable / orange review / red excluded + first reason",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 60), 2, cv2.LINE_AA)
    return np.vstack([head, np.hstack([p1, gap, p2, gap, p3])]), rows, sc, done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--out", required=True); ap.add_argument("--why", default=None)
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    why = json.loads(Path(a.why).read_text()) if a.why else {}
    table = []
    for d in sorted(p for p in (run / "fovs").iterdir() if (p / "done.json").exists()):
        img, rows, sc, done = sheet(d, why.get(d.name, ""))
        cv2.imwrite(str(out / f"{d.name}.png"), img)
        rc = Counter(x for r in rows for x in r["reasons"].split(";") if x)
        table.append({"fov": d.name, "why": why.get(d.name, ""), "cells": len(rows), "measurable": sc["measurable"],
                      "review": sc["review"], "excluded": sc["excluded"],
                      "top_reasons": ", ".join(f"{k} {v}" for k, v in rc.most_common(4)),
                      "image_qc": ",".join(done.get("fov_qc") or []) or "ok", "s_per_fov": round(done.get("t_total_s", 0), 2),
                      "sheet": str(out / f"{d.name}.png")})
        print(d.name, dict(sc), flush=True)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    md = ["| FOV | why picked | cells | measurable | review | excluded | top reasons | image QC | s/FOV |", "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {t['fov']} | {t['why']} | {t['cells']} | {t['measurable']} | {t['review']} | {t['excluded']} | {t['top_reasons']} | "
           f"{t['image_qc']} | {t['s_per_fov']} |" for t in table]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
