"""Same metrics as verify_t2.py on the 100-cell set of "check the segmentation" (50 ZIP01 = known hand-drawn,
50 ZIP02), against NCNet V3's SAVED predictions in ~/Downloads/nc_ratio_model_predictions. Read-only.
Also, for every set incl. T2: per-cell count of vertices > 1 px and > 3 px from NCNet's raw boundary.
Writes control.csv and control_summary.json."""
import json, sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_t2 import boundary_pts, dice, on_boundary, read_polys, spacing  # noqa: E402

H = Path.home() / "Downloads"
GT = H / "Anupam_AI/check the segmentation/nc_ratio_ground_truth"
PR = H / "nc_ratio_model_predictions"
T2 = H / "nc_ratio_new_evaluation"


def far(verts, mask):
    b = boundary_pts(mask)
    d, _ = cKDTree(b).query(verts)
    return int((d > 1).sum()), int((d > 3).sum()), float(d.max())


def cell_metrics(setname, name, txt, gt_masks, pr_post, pr_raw):
    im = cv2.imread(str(gt_masks["cell"]), 0); h, w = im.shape
    polys = read_polys(txt, w, h)
    r = {"set": setname, "file": name}
    for c, s in ((0, "nucleus"), (1, "cell")):
        g = cv2.imread(str(gt_masks[s]), 0) > 0
        pp = cv2.imread(str(pr_post[s]), 0) > 0; pr = cv2.imread(str(pr_raw[s]), 0) > 0
        v = np.vstack(polys[c])
        r[f"{s}_dice_ncnet"] = dice(g, pp)
        r[f"{s}_vtx_on_ncnet_raw"], r[f"{s}_vtx_dist_med_raw"] = on_boundary(v, pr)
        r[f"{s}_vtx_gt1px"], r[f"{s}_vtx_gt3px"], r[f"{s}_vtx_dist_max"] = far(v, pr)
        for k, x in spacing(v).items():
            r[f"{s}_{k}"] = x
    return r


def main():
    rows = []
    for sub in ("cervical_segmentation_01", "cervical_segmentation_02"):
        for txt in sorted((GT / "ground_truth_annotations" / sub).glob("*.txt")):
            n = txt.stem + ".png"
            rows.append(cell_metrics("ZIP01 (hand-drawn ref.)" if sub.endswith("01") else "ZIP02", n, txt,
                                     {s: GT / "ground_truth_masks" / s / sub / n for s in ("cell", "nucleus")},
                                     {s: PR / "predicted_masks" / s / sub / n for s in ("cell", "nucleus")},
                                     {s: PR / "predicted_masks_raw" / s / sub / n for s in ("cell", "nucleus")}))
    for txt in sorted((T2 / "ground_truth_annotations").glob("*.txt")):
        n = txt.stem + ".png"
        rows.append(cell_metrics("T2 NC003", n, txt, {s: T2 / "ground_truth_masks" / s / n for s in ("cell", "nucleus")},
                                 {s: T2 / "predicted_masks" / s / n for s in ("cell", "nucleus")},
                                 {s: T2 / "predicted_masks_raw" / s / n for s in ("cell", "nucleus")}))
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).resolve().parent / "control.csv", index=False)
    out = {}
    for st, g in df.groupby("set"):
        o = {"cells": len(g)}
        for s in ("nucleus", "cell"):
            o[s] = {"dice_median": round(g[f"{s}_dice_ncnet"].median(), 4),
                    "vtx_on_ncnet_median": round(g[f"{s}_vtx_on_ncnet_raw"].median(), 4),
                    "vtx_on_ncnet_min": round(g[f"{s}_vtx_on_ncnet_raw"].min(), 4),
                    "vtx_dist_median_px": round(g[f"{s}_vtx_dist_med_raw"].median(), 4),
                    "integer_vertices_median": round(g[f"{s}_integer_vertices"].median(), 4),
                    "step_cv_median": round(g[f"{s}_step_cv"].median(), 3),
                    "cells_with_vertex_gt_1px": int((g[f"{s}_vtx_gt1px"] > 0).sum()),
                    "cells_with_vertex_gt_3px": int((g[f"{s}_vtx_gt3px"] > 0).sum()),
                    "max_vertex_dist_px_max": round(g[f"{s}_vtx_dist_max"].max(), 2)}
        out[st] = o
    (Path(__file__).resolve().parent / "control_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
