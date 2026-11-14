"""Compare full-frame vs tiled detection against GT boxes (uses the pipeline's own Detector).

GT: YOLO txt labels normalised to a square-padded FOV (yolo26_cell_detection layout: original W x H padded
top/bottom to square, labels relative to the square).

    python tools/eval_detector_tiling.py --images <full-res FOV dir> --labels <labels dir> --pad-top 254 \
        --square 2028 --n 40 --out reports/detector_tiling.json
"""
import argparse, copy, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ncmask import config as cfgmod, geometry  # noqa: E402
from ncmask.detect import Detector  # noqa: E402
from ncmask.imageio import load_bgr8  # noqa: E402


def gt_boxes(txt, S, pad_top):
    rows = [l.split() for l in Path(txt).read_text().splitlines() if l.strip()]
    b = np.array([[float(v) for v in r[1:5]] for r in rows]).reshape(-1, 4) * S
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2 - pad_top,
                     b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2 - pad_top], 1)


def match(pred, conf, gt, thr=0.5):
    """Greedy by confidence; returns matched-GT bool array and TP count."""
    hit = np.zeros(len(gt), bool); tp = 0
    if len(pred) and len(gt):
        iou = geometry.iou_matrix(pred, gt)
        for i in np.argsort(-conf, kind="stable"):
            j = np.where(~hit & (iou[i] >= thr))[0]
            if len(j):
                hit[j[np.argmax(iou[i, j])]] = True; tp += 1
    return hit, tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True); ap.add_argument("--labels", required=True)
    ap.add_argument("--pad-top", type=float, default=254); ap.add_argument("--square", type=float, default=2028)
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--config", default=str(Path(__file__).parents[1] / "config.yaml"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = cfgmod.load(a.config)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    labels = sorted(Path(a.labels).glob("*.txt"))
    imgs = {p.stem: p for p in Path(a.images).iterdir()}
    pairs = [(imgs[l.stem], l) for l in labels if l.stem in imgs]
    pairs = pairs[::max(1, len(pairs) // a.n)][:a.n]
    data = [(load_bgr8(p)[0], gt_boxes(l, a.square, a.pad_top)) for p, l in pairs]
    allgt = np.concatenate([g for _, g in data])
    size = np.sqrt((allgt[:, 2] - allgt[:, 0]) * (allgt[:, 3] - allgt[:, 1]))
    edges = np.percentile(size, [0, 20, 40, 60, 80, 100])
    res = {"n_fovs": len(data), "n_gt": int(len(allgt)), "gt_size_sqrt_wh_px_quintile_edges": [round(float(e), 1) for e in edges],
           "iou_thr": 0.5, "conf": cfg["detector"]["conf"], "modes": {}}
    for name, mode, tile, ov in (("full_frame", "off", 1024, 512), ("tiled_1024_ov512", "on", 1024, 512),
                                 ("tiled_1280_ov640", "on", 1280, 640)):
        c = copy.deepcopy(cfg); c["detector"]["tiling"].update(mode=mode, tile_px=tile, overlap_px=ov)
        det = Detector(c, dev)
        tp = fp = fn = 0; hits = []; t0 = time.time()
        for img, g in data:
            b, cf, _ = det(img)
            h, t = match(b, cf, g)
            tp += t; fp += len(b) - t; fn += len(g) - t; hits.append(h)
        hits = np.concatenate(hits)
        q = np.clip(np.searchsorted(edges, size, side="right") - 1, 0, 4)
        res["modes"][name] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(tp / max(tp + fp, 1), 4),
                              "recall": round(tp / max(tp + fn, 1), 4),
                              "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
                              "recall_by_gt_size_quintile": [round(float(hits[q == k].mean()), 4) for k in range(5)],
                              "s_per_fov": round((time.time() - t0) / len(data), 3)}
        print(name, res["modes"][name], flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
