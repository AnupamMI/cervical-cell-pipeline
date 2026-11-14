#!/usr/bin/env python3
"""Step 2: final evaluation of V0-V3 + old v2_1 on T1, T1b, T2 (crop-based), with GT boxes and YOLO26s boxes.

Test sets (sources = the crops; T2 has no source FOVs, so it is CROP-BASED everywhere):
  T1   clean 212 (tinyseg_retrain test ∩ YOLO test), COCO GT                   groups: same-cell-merged FOV groups
  T1b  50 ZIP01 cells of "check the segmentation", PNG GT                      (old v2_1 in-sample: never ranked)
  T2   your 92 NC003 cells, PNG GT, reported as T2 full and T2 corrected+independent (t2_verify classes)
Boxes: 'gt'  = GT cell bounding box
       'det' = YOLO26s (models/detector of nc_mask_pipeline, conf 0.15, imgsz 640, NMS 0.7) on the crop placed at
               native resolution in a 2028x1520 canvas (detect_at_fov_scale: the detector's training scale);
               the box with the highest IoU to the GT cell box, if IoU >= 0.5; otherwise the cell is 'unmatched'
               (counted, no masks scored). Crops without a GT cell polygon have no GT box and are skipped in 'det'.
Per cell: aug_v2.pipeline_crop(box, margin 0.17, mirror padding) -> model input variant -> sigmoid ->
nc_mask_pipeline postprocess (centre component, no clip inside the crop, holes filled, nucleus in cell, blobs < 20 px
removed) -> back to source coordinates. Metrics: nucleus/cell Dice, IoU, boundary F1 at 2 px, N:C MAE
(nucleus / (cell - nucleus), cells with both GT structures); failures: no nucleus, multi-nucleus (>= 2 nucleus blobs
>= 20 px), N:C > 1; params; ms per cell (variant + input + GPU inference batch 1 synced + post-processing; median,
5 warm-up cells; the machine is idle when this runs after training).
95 % CI: cluster bootstrap over groups (2,000 resamples, seed 0). '†' = CI overlaps the best model's CI.
Writes results/{per_cell.csv, summary.csv, eval.json, RESULTS.md, overlays/<set>/*.png}.
"""
import json, sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
sys.path.insert(0, str(HERE))
import eval_v3 as E  # noqa: E402  (tested helpers: loaders, Model, metrics, cluster CI)
from eval_v3 import aug_v2, postprocess, MARGIN  # noqa: E402

NCM = E.NCM
T2E = E.T2E
MODELS = ["V0", "V1", "V2", "V3", "v2_1"]
SETS = ["T1", "T1b", "T2"]
FAILS = ["fail_no_nucleus", "fail_multi_nucleus", "fail_nc_gt_1"]
METRICS = ["nuc_dice", "nuc_iou", "nuc_bf2", "cell_dice", "cell_iou", "cell_bf2", "nc_abs_err"]


def run_dir(v):
    for d in (HERE / "runs" / v, HERE / "runs" / f"{v}_r1", HERE / "runs" / f"{v}_r2"):
        if (d / "done.json").exists():
            return d
    return None


def load_T2_crops():
    gt = pd.read_csv(T2E / "results/ground_truth_nc_ratios.csv")
    cls = pd.read_csv(ROOT / "t2_verify/t2_verify.csv").set_index("file")["class"].to_dict()
    out = []
    for _, r in gt.iterrows():
        n = r.original_filename
        nm = cv2.imread(str(T2E / "ground_truth_masks/nucleus" / n), 0) > 0
        cm = cv2.imread(str(T2E / "ground_truth_masks/cell" / n), 0) > 0
        out.append({"name": n, "group": n.split("_a")[0], "S": cv2.imread(str(T2E / "selected_images" / n)),
                    "nuc": nm.astype(np.uint8), "cell": cm.astype(np.uint8), "t2_class": cls.get(n)})
    return out


def gt_box(cell):
    ys, xs = np.nonzero(cell)
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


FOV_W, FOV_H = 2028, 1520


def detect_at_fov_scale(det, crop):
    """YOLO26s was trained on 2028x1520 FOVs run at 640 (a cell is ~95 px there). A ~450 px crop fed directly is
    upscaled to fill 640 px, far outside that scale, and is mostly missed. So the crop is placed at its NATIVE
    resolution in the centre of a 2028x1520 canvas filled with the crop's median border colour, the detector runs
    full-frame exactly as on a real FOV, and boxes are shifted back to crop coordinates (clipped to the crop)."""
    h, w = crop.shape[:2]
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    canvas = np.empty((FOV_H, FOV_W, 3), np.uint8); canvas[:] = np.median(border, 0).astype(np.uint8)
    y0, x0 = (FOV_H - h) // 2, (FOV_W - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = crop
    b, c, info = det(canvas)
    if len(b):
        b = b - (x0, y0, x0, y0)
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, w); b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
        keep = (b[:, 2] - b[:, 0] > 1) & (b[:, 3] - b[:, 1] > 1)
        b, c = b[keep], c[keep]
    return b, c, info


def segment(model, S, box, sync):
    H_, W_ = S.shape[:2]
    t0 = time.perf_counter()
    win, _, cbox = aug_v2.pipeline_crop(S, [], box)
    prob = model(win)
    if sync:
        torch.cuda.synchronize()
    m = max(box[2] - box[0], box[3] - box[1])
    r = postprocess(prob, win.shape[0], win.shape[1], cbox, 0.5, 0.5, MARGIN * m, 20)
    ms = (time.perf_counter() - t0) * 1000
    mg = MARGIN * m; X0, Y0 = int(np.floor(box[0] - mg)), int(np.floor(box[1] - mg))
    pn, pc = np.zeros((H_, W_), bool), np.zeros((H_, W_), bool)
    ys, xs = slice(max(0, Y0), min(H_, Y0 + win.shape[0])), slice(max(0, X0), min(W_, X0 + win.shape[1]))
    wy, wx = slice(ys.start - Y0, ys.stop - Y0), slice(xs.start - X0, xs.stop - X0)
    pn[ys, xs] = r["nucleus"][wy, wx]; pc[ys, xs] = r["cell"][wy, wx]
    return pn, pc, r["n_nuc"], ms


def score(rec, pn, pc, n_nuc):
    out = {"fail_no_nucleus": int(not pn.any()), "fail_multi_nucleus": int(n_nuc > 1)}
    ncp = E.nc(pn, pc)
    out["fail_nc_gt_1"] = int(np.isfinite(ncp) and ncp > 1)
    for s, p, g in (("nuc", pn, rec["nuc"]), ("cell", pc, rec["cell"])):
        if g is not None and g.any():
            d, i = E.dice_iou(p, g)
            out[f"{s}_dice"], out[f"{s}_iou"], out[f"{s}_bf2"] = d, i, E.bf_score(p, g)
    if rec["cell"] is not None and rec["cell"].any() and rec["nuc"] is not None and rec["nuc"].any():
        out["nc_gt"], out["nc_pred"] = E.nc(rec["nuc"] > 0, rec["cell"] > 0), ncp
        out["nc_abs_err"] = abs(ncp - out["nc_gt"]) if np.isfinite(ncp) else np.nan
    return out


def overlay_row(rec, preds, names):
    S = rec["S"]
    def draw(img, nuc, cell, cn, cc):
        o = img.copy()
        for m, col, th in ((cell, cc, 2), (nuc, cn, 2)):
            if m is not None and m.any():
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(o, cs, -1, col, th)
        return o
    tiles = [("GT", draw(S, rec["nuc"], rec["cell"], (0, 0, 255), (0, 200, 0)))]
    for n in names:
        if n in preds:
            pn, pc, d = preds[n]
            tiles.append((f"{n} nuc {d:.3f}" if d == d else n, draw(S, pn, pc, (255, 255, 0), (255, 0, 255))))
    h = 220
    row = []
    for t, im in tiles:
        im = cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
        cap = np.full((24, im.shape[1], 3), 255, np.uint8)
        cv2.putText(cap, t, (3, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        row.append(np.vstack([cap, im])); row.append(np.full((h + 24, 4, 3), 255, np.uint8))
    return np.hstack(row)


def fmt(r, m, best_idx, i, digits=3):
    v, lo, hi = r[m], r[m + "_lo"], r[m + "_hi"]
    if v != v:
        return "n/a"
    star = "" if i == best_idx else (" †" if r.get(m + "_overlaps_best") else "")
    b = "**" if i == best_idx else ""
    return f"{b}{v:.{digits}f}{b} [{lo:.{digits}f}, {hi:.{digits}f}]{star}"


def main():
    global RES
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", default=None, help="test mode: output dir; V0 = current best.pt, 12 cells/set")
    a = ap.parse_args()
    if a.smoke:
        RES = Path(a.smoke)
    RES.mkdir(parents=True, exist_ok=True); (RES / "overlays").mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sync = device == "cuda"
    torch.backends.cudnn.benchmark = False
    log = {"device": device, "started": time.strftime("%F %T"), "models": {}, "notes": []}
    models = {}
    for v in MODELS:
        d = Path(NCM / "models/seg/v2_1") if v == "v2_1" else run_dir(v)
        if a.smoke and v == "V0" and d is None and (HERE / "runs/V0/best.pt").exists():
            d = HERE / "runs/V0"
        if d is None:
            log["notes"].append(f"{v}: no finished run, not evaluated"); continue
        m = E.Model(v, d, device)
        models[v] = m
        log["models"][v] = {"dir": str(d), "params": int(sum(p.numel() for p in m.net.parameters())), "in_ch": m.in_ch,
                            "variant": m.variant}
    if "V0" not in models:
        sys.exit("V0 missing: cannot evaluate")
    data = {"T1": E.load_T1(), "T1b": E.load_T1b(), "T2": load_T2_crops()}
    if a.smoke:
        data = {k: v[:12] for k, v in data.items()}
    # detector boxes (YOLO26s, the pipeline's own detector code and config)
    sys.path.insert(0, str(NCM))
    from ncmask import config as ncfg
    from ncmask.detect import Detector
    cfg = ncfg.load(NCM / "config.yaml")
    det = Detector(cfg, device)
    log["detector"] = {"weights": cfg["detector"]["weights"], "conf": cfg["detector"]["conf"], "imgsz": det.imgsz}
    for ts, recs in data.items():
        for rec in recs:
            has_cell = rec["cell"] is not None and rec["cell"].any()
            rec["gt_box"] = gt_box(rec["cell"]) if has_cell else aug_v2.base_box(rec["S"], None)
            rec["det_box"], rec["det_iou"], rec["det_conf"] = None, 0.0, None
            if has_cell:
                b, c, _ = detect_at_fov_scale(det, rec["S"])
                if len(b):
                    ious = [iou(rec["gt_box"], x) for x in b]; k = int(np.argmax(ious))
                    rec["det_iou"] = ious[k]
                    if ious[k] >= 0.5:
                        rec["det_box"], rec["det_conf"] = tuple(float(v) for v in b[k]), float(c[k])
    rows, ov = [], {}
    for ts, recs in data.items():
        for proto in ("gt", "det"):
            for name, m in models.items():
                warm = [r for r in recs if (r["det_box"] if proto == "det" else r["gt_box"]) is not None][:5]
                for r in warm:
                    segment(m, r["S"], r["det_box"] if proto == "det" else r["gt_box"], sync)
                for k, rec in enumerate(recs):
                    base = {"set": ts, "protocol": proto, "model": name, "name": rec["name"], "group": rec["group"],
                            "t2_class": rec.get("t2_class"), "det_iou": rec["det_iou"]}
                    box = rec["det_box"] if proto == "det" else rec["gt_box"]
                    if box is None:
                        base["status"] = "no_gt_cell" if not (rec["cell"] is not None and rec["cell"].any()) else "unmatched"
                        rows.append(base); continue
                    pn, pc, n_nuc, ms = segment(m, rec["S"], box, sync)
                    base.update(status="scored", ms=ms, **score(rec, pn, pc, n_nuc))
                    rows.append(base)
                    if proto == "det":
                        ov.setdefault((ts, rec["name"]), {})[name] = (pn, pc, base.get("nuc_dice", np.nan))
                print(f"{ts} {proto} {name}: done", flush=True)
    per = pd.DataFrame(rows)
    per["subset"] = "all"
    sub = per[(per.set == "T2") & per.t2_class.isin(["corrected", "independent"])].copy(); sub["subset"] = "corrected+independent"
    per = pd.concat([per, sub], ignore_index=True)
    per.to_csv(RES / "per_cell.csv", index=False)
    # summary with CIs
    srows = []
    for (ts, subset, proto, name), g in per.groupby(["set", "subset", "protocol", "model"], sort=False):
        s = g[g.status == "scored"]
        r = {"set": ts, "subset": subset, "protocol": proto, "model": name, "n_cells": len(g), "n_scored": len(s),
             "n_unmatched": int((g.status == "unmatched").sum()), "n_no_gt_cell": int((g.status == "no_gt_cell").sum()),
             "n_groups": int(s.group.nunique()), "params": log["models"][name]["params"],
             "ms_median": float(s.ms.median()) if len(s) else np.nan,
             "in_sample": bool(name == "v2_1" and ts == "T1b")}
        for mtr in METRICS:
            r[mtr], r[mtr + "_lo"], r[mtr + "_hi"] = E.cluster_ci(s, mtr)
        for f in FAILS:
            r[f] = int(s[f].sum()) if f in s else 0
        srows.append(r)
    S_ = pd.DataFrame(srows)
    for mtr in METRICS:
        S_[mtr + "_overlaps_best"] = False
        S_[mtr + "_best"] = False
        for _, g in S_.groupby(["set", "subset", "protocol"]):
            g = g[~g.in_sample & g[mtr].notna()]
            if g.empty:
                continue
            b = g[mtr].idxmin() if mtr == "nc_abs_err" else g[mtr].idxmax()
            S_.loc[b, mtr + "_best"] = True
            for i in g.index:
                if i != b:
                    S_.loc[i, mtr + "_overlaps_best"] = bool(S_.loc[i, mtr + "_hi"] >= S_.loc[b, mtr + "_lo"] and S_.loc[i, mtr + "_lo"] <= S_.loc[b, mtr + "_hi"])
    S_.to_csv(RES / "summary.csv", index=False)
    # overlays: 10 per set (5 random + the 5 lowest V0 nucleus Dice with detector boxes), detector protocol
    rng = np.random.default_rng(0)
    for ts, recs in data.items():
        d0 = per[(per.set == ts) & (per.subset == "all") & (per.protocol == "det") & (per.model == "V0") & (per.status == "scored")]
        worst = list(d0.sort_values("nuc_dice").name[:5])
        rest = [r["name"] for r in recs if r["name"] not in worst and (ts, r["name"]) in ov]
        pick = worst + list(rng.choice(rest, min(5, len(rest)), replace=False))
        out = RES / "overlays" / ts; out.mkdir(parents=True, exist_ok=True)
        by = {r["name"]: r for r in recs}
        for k, n in enumerate(pick, 1):
            if (ts, n) in ov:
                cv2.imwrite(str(out / f"{k:02d}_{'worstV0' if n in worst else 'random'}_{Path(n).stem}.png"),
                            overlay_row(by[n], ov[(ts, n)], list(models)))
    log["finished"] = time.strftime("%F %T")
    (RES / "eval.json").write_text(json.dumps(log, indent=1))
    write_results_md(S_, per, log)
    print("eval done")


def write_results_md(S_, per, log):
    tr = (RES / "train_report.md").read_text() if (RES / "train_report.md").exists() else "(train report missing)\n"
    t2v = json.loads((ROOT / "t2_verify/t2_verify_summary.json").read_text()) if (ROOT / "t2_verify/t2_verify_summary.json").exists() else {}
    L = ["# TinyUNet v3: results", "",
         f"Evaluation run {log['started']} – {log['finished']} on {log['device']} (idle machine, after training).", "",
         "Test sets: **T1** = clean 212 · **T1b** = 50 ZIP01 cells (old v2_1 is *in-sample* here: shown, never ranked) · "
         "**T2** = your 92 NC003 cells, **crop-based** (no source FOVs exist: pipeline crop = cell box + 0.17, mirror "
         "padding, on each crop itself), as *T2 full* and *T2 corrected+independent*.",
         "Boxes: **gt** = GT cell box; **det** = YOLO26s (conf 0.15) box with IoU ≥ 0.5 to the GT box (unmatched GT counted separately). "
         "The detector runs on each crop placed at native resolution in a 2028×1520 canvas (median border colour), i.e. at its "
         "FOV training scale; fed directly, a ~450 px crop is upscaled ~1.5× past that scale and the cell is mostly missed.",
         "Cells: `[low, high]` = 95 % cluster-bootstrap CI over FOV groups; **bold** = best; **†** = its CI overlaps the best model's CI "
         "(difference not established).", "", "## Training", "", tr, ""]
    if t2v:
        L += ["## T2 independence check (your GT vs NCNet V3's saved masks; NCNet not run)", "",
              f"Classes (min Dice rule): {t2v['class_counts']}. Raster check: masks == polygons for all 92 "
              f"({t2v['raster_check']['cell']['all_equal'] and t2v['raster_check']['nucleus']['all_equal']}). "
              f"Median vertices on NCNet's boundary: cell {t2v['cell_vtx_on_ncnet_raw']['median']:.1%}, nucleus "
              f"{t2v['nucleus_vtx_on_ncnet_raw']['median']:.1%} (hand-drawn ZIP01 control: ~28 % / ~36 %). Details: "
              "`tinyseg_v3/t2_verify/T2_VERIFY_REPORT.md`.", ""]
    for ts in SETS:
        for subset in (["all", "corrected+independent"] if ts == "T2" else ["all"]):
            for proto in ("det", "gt"):
                g = S_[(S_.set == ts) & (S_.subset == subset) & (S_.protocol == proto)].reset_index(drop=True)
                if g.empty:
                    continue
                title = f"{ts}{' crop-based' if ts == 'T2' else ''} – {subset} – {'YOLO26s boxes' if proto == 'det' else 'GT boxes'}"
                L += [f"### {title}", "",
                      f"cells {g.n_cells[0]}, scored {int(g.n_scored.max())}, unmatched GT {int(g.n_unmatched.max())}, no GT cell {int(g.n_no_gt_cell.max())}", "",
                      "| model | nucleus Dice | nucleus IoU | nucleus BF@2px | cell Dice | cell IoU | cell BF@2px | N:C MAE | no nucleus | multi-nucleus | N:C>1 | params | ms/cell |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
                for i, r in g.iterrows():
                    bests = {m: (g.index[g[m + "_best"]][0] if g[m + "_best"].any() else -1) for m in METRICS}
                    name = r.model + (" (in-sample, not ranked)" if r.in_sample else "")
                    L.append(f"| {name} | " + " | ".join(fmt(r, m, bests[m], i, 4 if m == "nc_abs_err" else 3) for m in METRICS)
                             + f" | {r.fail_no_nucleus} | {r.fail_multi_nucleus} | {r.fail_nc_gt_1} | {r.params:,} | {r.ms_median:.1f} |")
                L.append("")
    L += ["Overlays (YOLO26s boxes; GT green/red, prediction magenta/cyan): `tinyseg_v3/results/overlays/<set>/`.", ""]
    (RES / "RESULTS.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
