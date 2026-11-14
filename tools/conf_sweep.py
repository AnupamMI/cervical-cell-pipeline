"""Detector confidence sweep on one FOV: box counts per conf, and whether given regions get a box (and at what conf).

    python tools/conf_sweep.py --image <fov> --confs 0.10 0.15 0.25 --region 430 760 720 950 --region 1050 540 1190 740
"""
import argparse, copy, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ncmask import config as cfgmod, geometry  # noqa: E402
from ncmask.detect import Detector  # noqa: E402
from ncmask.imageio import load_bgr8  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--confs", type=float, nargs="+", required=True)
    ap.add_argument("--region", type=float, nargs=4, action="append", default=[], metavar=("X1", "Y1", "X2", "Y2"))
    ap.add_argument("--config", default=str(Path(__file__).parents[1] / "config.yaml"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = cfgmod.load(a.config)
    import torch
    img, _ = load_bgr8(a.image)
    res = {"image": a.image, "regions": a.region, "by_conf": {}}
    for cf in sorted(a.confs):
        c = copy.deepcopy(cfg); c["detector"]["conf"] = cf
        det = Detector(c, "cuda" if torch.cuda.is_available() else "cpu")
        b, s, info = det(img)
        keep, _ = geometry.containment_suppress(b, s, c["detector"]["containment_thr"] or 1.01,
                                                c["detector"]["containment_min_area_ratio"]) if len(b) else (np.zeros(0, bool), None)
        hits = []
        for r in a.region:
            iou = geometry.iou_matrix(b, [r])[:, 0] if len(b) else np.zeros(0)
            cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
            inside = (cx >= r[0]) & (cx <= r[2]) & (cy >= r[1]) & (cy <= r[3])
            cand = np.where((iou >= 0.3) | inside)[0]
            if len(cand):
                j = cand[np.argmax(iou[cand])]
                hits.append({"found": True, "conf": round(float(s[j]), 4), "iou_with_region": round(float(iou[j]), 3),
                             "box": [round(float(v), 1) for v in b[j]], "kept_after_containment": bool(keep[j])})
            else:
                hits.append({"found": False})
        res["by_conf"][str(cf)] = {"boxes_after_nms": int(len(b)), "boxes_after_containment": int(keep.sum()), "regions": hits}
        print(cf, json.dumps(res["by_conf"][str(cf)]), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
