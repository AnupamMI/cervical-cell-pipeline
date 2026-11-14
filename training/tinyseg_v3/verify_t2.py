"""T2 (NC003, 92 cells) independence check against NCNet V3's SAVED predictions. Read-only; NCNet is not run.

    python verify_t2.py --eval ~/Downloads/nc_ratio_new_evaluation --out .

Per cell and per structure (nucleus, cell):
  1. Dice(GT mask, NCNet mask): NCNet post-processed (predicted_masks, centre component) and raw (predicted_masks_raw).
  2. Vertices of the GT polygon (.txt, x*W, y*H): share lying <= 0.5 px from NCNet's boundary (centres of mask pixels
     with a 4-neighbour outside the mask; exact Euclidean distance via KD-tree), against raw and post-processed masks;
     control: the same against the GT's own mask. Vertex-spacing stats: share of integer vertices, step length median
     and coefficient of variation, share of steps that are exactly 1 or sqrt(2) px (pixel-contour tracing).
  3. Class: identical (min Dice >= 0.99) | corrected (0.90-0.99) | independent (< 0.90, or freehand-like:
     < 50 % of vertices on NCNet's boundary AND < 50 % integer vertices). Min Dice = min(nucleus, cell) vs post-processed.
  4. Rasterisation: GT .txt polygons re-filled with Pillow ImageDraw.polygon(fill=1, outline=1) (the protocol's rule)
     must equal the saved GT masks pixel for pixel.
Writes t2_verify.csv, t2_verify_summary.json, overlays/*.png.
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree


def read_polys(txt, w, h):
    out = {0: [], 1: []}
    for line in Path(txt).read_text().splitlines():
        v = line.split()
        if v:
            out[int(v[0])].append(np.array(v[1:], float).reshape(-1, 2) * (w, h))
    return out


def pil_raster(polys, w, h):
    im = Image.new("L", (w, h), 0); d = ImageDraw.Draw(im)
    for p in polys:
        d.polygon([tuple(x) for x in p], fill=1, outline=1)
    return np.array(im, bool)


def boundary_pts(m):
    m = m.astype(np.uint8)
    inner = m & ~cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    ys, xs = np.nonzero(inner)
    return np.c_[xs, ys].astype(float)


def on_boundary(verts, mask, tol=0.5):
    b = boundary_pts(mask)
    if not len(b) or not len(verts):
        return np.nan, np.nan
    d, _ = cKDTree(b).query(verts)
    return float((d <= tol).mean()), float(np.median(d))


def spacing(verts):
    st = np.hypot(*np.diff(np.vstack([verts, verts[:1]]), axis=0).T)
    unit = (np.abs(st - 1) < 1e-3) | (np.abs(st - np.sqrt(2)) < 1e-3)
    integer = (np.abs(verts - np.round(verts)) < 1e-3).all(1)
    return {"n_vertices": len(verts), "step_median": float(np.median(st)), "step_cv": float(st.std() / st.mean()),
            "unit_steps": float(unit.mean()), "integer_vertices": float(integer.mean())}


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * (a & b).sum() / s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--eval", required=True); ap.add_argument("--out", default=".")
    a = ap.parse_args()
    E, O = Path(a.eval).expanduser(), Path(a.out)
    (O / "overlays").mkdir(parents=True, exist_ok=True)
    gt = pd.read_csv(E / "results/ground_truth_nc_ratios.csv")
    rows = []
    for _, r in gt.iterrows():
        name, stem = r.original_filename, Path(r.original_filename).stem
        w, h = int(r.image_width), int(r.image_height)
        polys = read_polys(E / "ground_truth_annotations" / f"{stem}.txt", w, h)
        rec = {"sample_id": r.sample_id, "file": name, "w": w, "h": h}
        for c, s in ((0, "nucleus"), (1, "cell")):
            g = cv2.imread(str(E / "ground_truth_masks" / s / name), 0) > 0
            pp = cv2.imread(str(E / "predicted_masks" / s / name), 0) > 0
            pr = cv2.imread(str(E / "predicted_masks_raw" / s / name), 0) > 0
            assert g.shape == pp.shape == pr.shape == (h, w), name
            v = np.vstack(polys[c]) if polys[c] else np.zeros((0, 2))
            rr = pil_raster(polys[c], w, h)
            rec[f"{s}_n_polygons"] = len(polys[c])
            rec[f"{s}_raster_equal"] = bool((rr == g).all())
            rec[f"{s}_raster_diff_px"] = int((rr != g).sum())
            rec[f"{s}_dice_ncnet"] = dice(g, pp)
            rec[f"{s}_dice_ncnet_raw"] = dice(g, pr)
            rec[f"{s}_vtx_on_ncnet_raw"], rec[f"{s}_vtx_dist_med_raw"] = on_boundary(v, pr)
            rec[f"{s}_vtx_on_ncnet"], _ = on_boundary(v, pp)
            rec[f"{s}_vtx_on_own_gt"], _ = on_boundary(v, g)
            for k, x in spacing(v).items():
                rec[f"{s}_{k}"] = x
        md = min(rec["nucleus_dice_ncnet"], rec["cell_dice_ncnet"])
        on = np.nanmean([rec["nucleus_vtx_on_ncnet_raw"], rec["cell_vtx_on_ncnet_raw"]])
        integ = np.mean([rec["nucleus_integer_vertices"], rec["cell_integer_vertices"]])
        freehand = on < 0.5 and integ < 0.5
        rec["min_dice"], rec["vtx_on_ncnet_mean"], rec["freehand_like"] = md, on, bool(freehand)
        rec["class"] = "independent" if (md < 0.90 or freehand) else "identical" if md >= 0.99 else "corrected"
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(O / "t2_verify.csv", index=False)
    q = lambda x: {k: round(float(v), 4) for k, v in zip(("min", "p5", "p25", "median", "p75", "p95", "max"),
                                                       np.nanpercentile(x, [0, 5, 25, 50, 75, 95, 100]))}
    summ = {"n": len(df), "class_counts": df["class"].value_counts().to_dict(),
            "raster_check": {s: {"all_equal": bool(df[f"{s}_raster_equal"].all()), "cells_differing": int((~df[f"{s}_raster_equal"]).sum()),
                                 "max_diff_px": int(df[f"{s}_raster_diff_px"].max())} for s in ("nucleus", "cell")}}
    for s in ("nucleus", "cell"):
        for k in ("dice_ncnet", "dice_ncnet_raw", "vtx_on_ncnet_raw", "vtx_on_ncnet", "vtx_on_own_gt", "vtx_dist_med_raw",
                  "integer_vertices", "unit_steps", "step_median", "step_cv", "n_vertices"):
            summ[f"{s}_{k}"] = q(df[f"{s}_{k}"])
    (O / "t2_verify_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    # overlays: up to 2 per class (lowest / median min-Dice), GT green+red fill, NCNet magenta outline
    picks = []
    for c in ("identical", "corrected", "independent"):
        g = df[df["class"] == c].sort_values("min_dice")
        if len(g):
            picks += [g.iloc[0], g.iloc[len(g) // 2]] if len(g) > 1 else [g.iloc[0]]
    for k, r in enumerate(picks, 1):
        img = cv2.imread(str(E / "selected_images" / r.file))
        ov = img.copy()
        for s, col_gt, col_nn in (("cell", (0, 200, 0), (255, 0, 255)), ("nucleus", (0, 0, 255), (255, 255, 0))):
            gm = (cv2.imread(str(E / "ground_truth_masks" / s / r.file), 0) > 0).astype(np.uint8)
            nm = (cv2.imread(str(E / "predicted_masks" / s / r.file), 0) > 0).astype(np.uint8)
            for m, col, th in ((gm, col_gt, 2), (nm, col_nn, 1)):
                cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(ov, cs, -1, col, th)
        z = 2 if max(img.shape[:2]) < 450 else 1
        ov = cv2.resize(ov, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
        cap = np.full((48, ov.shape[1], 3), 255, np.uint8)
        cv2.putText(cap, f"{r.sample_id} [{r['class']}] Dice cell {r.cell_dice_ncnet:.3f} nuc {r.nucleus_dice_ncnet:.3f}",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(cap, f"vtx on NCNet edge: cell {r.cell_vtx_on_ncnet_raw:.0%} nuc {r.nucleus_vtx_on_ncnet_raw:.0%}  "
                    f"GT=green/red(thick) NCNet=magenta/cyan(thin)", (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.imwrite(str(O / "overlays" / f"{k}_{r['class']}_{r.sample_id}.png"), np.vstack([cap, ov]))


if __name__ == "__main__":
    main()
