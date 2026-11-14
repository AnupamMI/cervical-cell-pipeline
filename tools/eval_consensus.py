"""Step 1 of the consensus stage: evaluate A, B, A∩B, A∪B against CVAT GT boxes. Read-only on existing runs.

    python tools/eval_consensus.py --images data/images --gt data/annotations.xml --out runs/consensus_eval

A = config detector (YOLO26s, imgsz 640). B = --b-weights at --b-imgsz 1280. Both run once at --run-conf (0.10)
through the pipeline's own Detector (Ultralytics + class-agnostic NMS 0.7), boxes cached in boxes.json; metrics are
then computed at --a-conf / --b-conf by filtering (equivalent to re-running at the higher threshold).
A∩B = A boxes Hungarian-matched to a B box (IoU >= --match-iou), with A's geometry and conf.
A∪B = all A boxes + the unmatched B boxes. GT matching: greedy by confidence, IoU >= 0.5, one-to-one.
"""
import argparse, copy, json, sys, time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ncmask import config as cfgmod, geometry  # noqa: E402
from ncmask.consensus import match_boxes  # noqa: E402
from ncmask.imageio import load_bgr8  # noqa: E402


def read_cvat(path):
    """{image name: (N,4) boxes} for every <image>, all labels merged (as in the detectors' training)."""
    out = {}
    for im in ET.parse(path).getroot().iter("image"):
        bx = [[float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")] for b in im.iter("box")]
        out[im.get("name")] = np.array(bx, float).reshape(-1, 4)
    return out


def gt_match(pred, conf, gt, thr=0.5):
    """Greedy by conf -> (tp bool per pred, hit bool per gt)."""
    tp, hit = np.zeros(len(pred), bool), np.zeros(len(gt), bool)
    if len(pred) and len(gt):
        iou = geometry.iou_matrix(pred, gt)
        for i in np.argsort(-conf, kind="stable"):
            j = np.where(~hit & (iou[i] >= thr))[0]
            if len(j):
                k = j[np.argmax(iou[i, j])]; hit[k] = tp[i] = True
    return tp, hit


def prf(tp, fp, fn):
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / max(p + r, 1e-9), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True); ap.add_argument("--gt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(Path(__file__).parents[1] / "config.yaml"))
    ap.add_argument("--b-weights", default=str(Path(__file__).parents[1] / "models/detector_b/yolo26l_1280_best.pt"))
    ap.add_argument("--b-imgsz", type=int, default=1280)
    ap.add_argument("--run-conf", type=float, default=0.10)
    ap.add_argument("--a-conf", type=float, default=0.15); ap.add_argument("--b-conf", type=float, nargs="+", default=[0.15, 0.25])
    ap.add_argument("--match-iou", type=float, default=0.5); ap.add_argument("--gt-iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    gt = read_cvat(a.gt)
    files = sorted(p for p in Path(a.images).iterdir() if p.name in gt)
    if a.limit:
        files = files[:a.limit]
    cache = out / "boxes.json"
    if cache.exists():
        boxes = json.loads(cache.read_text())
    else:
        import torch
        from ncmask.detect import Detector
        cfg = cfgmod.load(a.config)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ca = copy.deepcopy(cfg); ca["detector"]["conf"] = a.run_conf
        cb = copy.deepcopy(cfg); cb["detector"].update(conf=a.run_conf, weights=a.b_weights, imgsz=a.b_imgsz)
        dets = {"A": Detector(ca, dev), "B": Detector(cb, dev)}
        print("A", dets["A"].imgsz, dets["A"].names, "| B", dets["B"].imgsz, dets["B"].names, flush=True)
        boxes, t0, tm = {}, time.time(), {"A": 0.0, "B": 0.0}
        for k, p in enumerate(files, 1):
            img, _ = load_bgr8(p)
            boxes[p.name] = {}
            for n, d in dets.items():
                t = time.perf_counter(); b, c, _ = d(img); tm[n] += time.perf_counter() - t
                boxes[p.name][n] = {"box": np.round(b, 2).tolist(), "conf": np.round(c, 5).tolist()}
            if k % 100 == 0:
                print(f"{k}/{len(files)} {time.time() - t0:.0f}s", flush=True)
        boxes["_timing_s_per_fov"] = {n: round(v / len(files), 4) for n, v in tm.items()}
        cache.write_text(json.dumps(boxes))
    timing = boxes.pop("_timing_s_per_fov", None)

    def get(name, n, thr):
        b = np.array(boxes[name][n]["box"], float).reshape(-1, 4); c = np.array(boxes[name][n]["conf"], float)
        return b[c >= thr], c[c >= thr]

    res = {"n_fovs": len(files), "n_gt": int(sum(len(gt[p.name]) for p in files)), "gt_per_fov_mean": None,
           "timing_s_per_fov": timing, "settings": vars(a), "by_b_conf": {}}
    res["gt_per_fov_mean"] = round(res["n_gt"] / max(len(files), 1), 2)
    per_fov_rows = []
    for bconf in a.b_conf:
        tot = {k: [0, 0, 0] for k in ("A", "B", "A∩B", "A∪B")}
        counts = {"A": 0, "B": 0, "consensus": 0, "s_only": 0, "l_only": 0,
                  "s_only_tp": 0, "s_only_fp": 0, "l_only_tp": 0, "l_only_fp": 0}
        for p in files:
            g = gt[p.name]
            A, Ac = get(p.name, "A", a.a_conf); B, Bc = get(p.name, "B", bconf)
            pairs, am, bm = match_boxes(A, B, a.match_iou)
            sets = {"A": (A, Ac), "B": (B, Bc), "A∩B": (A[am], Ac[am]),
                    "A∪B": (np.concatenate([A, B[~bm]]), np.concatenate([Ac, Bc[~bm]]))}
            r = {}
            for k, (bx, cf) in sets.items():
                tp, hit = gt_match(bx, cf, g, a.gt_iou)
                tot[k][0] += tp.sum(); tot[k][1] += (~tp).sum(); tot[k][2] += (~hit).sum()
                r[k] = (tp, hit)
            tpA = r["A"][0]
            tpB = gt_match(B, Bc, g, a.gt_iou)[0]
            counts["A"] += len(A); counts["B"] += len(B); counts["consensus"] += int(am.sum())
            counts["s_only"] += int((~am).sum()); counts["l_only"] += int((~bm).sum())
            counts["s_only_tp"] += int((tpA & ~am).sum()); counts["s_only_fp"] += int((~tpA & ~am).sum())
            counts["l_only_tp"] += int((tpB & ~bm).sum()); counts["l_only_fp"] += int((~tpB & ~bm).sum())
            if bconf == a.b_conf[0]:
                n = max(len(g), 1)
                per_fov_rows.append({"fov": p.stem, "gt": len(g), "A": len(A), "B": len(B), "consensus": int(am.sum()),
                                     "s_only": int((~am).sum()), "l_only": int((~bm).sum()),
                                     "recall_A": r["A"][1].sum() / n, "recall_AB": r["A∩B"][1].sum() / n,
                                     "s_only_tp": int((tpA & ~am).sum()), "s_only_fp": int((~tpA & ~am).sum()),
                                     "l_only_tp": int((tpB & ~bm).sum())})
        res["by_b_conf"][str(bconf)] = {"metrics": {k: prf(*v) for k, v in tot.items()}, "box_counts": counts}
        print(f"B conf {bconf}:", json.dumps(res["by_b_conf"][str(bconf)]), flush=True)
    import pandas as pd
    pf = pd.DataFrame(per_fov_rows)
    pf["recall_loss"] = pf.recall_A - pf.recall_AB
    pf.round(4).to_csv(out / "per_fov.csv", index=False)
    q = pf.recall_loss
    res["per_fov_recall_loss_A_minus_AB"] = {"mean": round(float(q.mean()), 4), "median": round(float(q.median()), 4),
                                             "p90": round(float(q.quantile(0.9)), 4), "max": round(float(q.max()), 4),
                                             "fovs_with_loss": int((q > 0).sum()), "fovs_with_loss_ge_0.10": int((q >= 0.10).sum())}
    (out / "metrics.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res["per_fov_recall_loss_A_minus_AB"]))


if __name__ == "__main__":
    main()
