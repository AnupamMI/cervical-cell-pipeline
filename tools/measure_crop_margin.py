"""Measure the margin between the COCO cell box and the crop border in the TinyUNet training crops.

margin per side = gap (px) between the cell-area bbox and that crop border, / max(cell box w, h).
Crops cut at a FOV edge have a smaller margin on that side, so the median over all sides is the
estimate of the padding that was used; the spread shows how consistent it was.

    python tools/measure_crop_margin.py <tinyseg_retrain/data/annotations/instances_default.json> [--out margin.json]
"""
import argparse, json
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coco")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = json.load(open(a.coco))
    ims = {im["id"]: im for im in d["images"]}
    cells = defaultdict(list)
    for an in d["annotations"]:
        if an["category_id"] == 2:
            cells[an["image_id"]].append(an)
    rows = []
    for iid, anns in cells.items():
        if len(anns) != 1:      # only crops with exactly one labelled cell
            continue
        im = ims[iid]; W, H = im["width"], im["height"]
        # bbox from the polygon (the COCO bbox field is rounded by CVAT)
        pts = np.concatenate([np.array(p, float).reshape(-1, 2) for p in anns[0]["segmentation"]])
        x1, y1 = pts.min(0); x2, y2 = pts.max(0)
        s = max(x2 - x1, y2 - y1)
        rows.append([x1 / s, y1 / s, (W - x2) / s, (H - y2) / s, s, W, H])
    r = np.array(rows)
    sides = r[:, :4]
    q = lambda v: {k: round(float(x), 4) for k, x in zip(("p5", "p25", "median", "p75", "p95"), np.percentile(v, [5, 25, 50, 75, 95]))}
    res = {
        "n_crops": len(r), "n_crops_skipped_not_1_cell": sum(1 for v in cells.values() if len(v) != 1),
        "left": q(sides[:, 0]), "top": q(sides[:, 1]), "right": q(sides[:, 2]), "bottom": q(sides[:, 3]),
        "all_sides": q(sides.ravel()),
        "max_side_per_crop": q(sides.max(1)),       # least affected by FOV-edge truncation
        "share_sides_within_0.01_of_median": round(float((np.abs(sides - np.median(sides)) <= 0.01).mean()), 4),
        "share_crops_with_a_side_below_0.12": round(float((sides.min(1) < 0.12).mean()), 4),
        "cell_box_max_side_px": q(r[:, 4]),
    }
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
