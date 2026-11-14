#!/usr/bin/env python3
"""Evaluate TinyUNet v3 variants (V0-V3) + the old v2_1 on T1, T1b and T2. Read-only on all data.

Per test cell (pipeline protocol):
  source S = the crop (T1, T1b) or the source FOV (T2, located by exact template match), GT masks in S coordinates
  box = GT cell bounding box (primary; 'gt' protocol) or the evaluate_v2 seeded jittered box (seed 777; 'jitter')
  window = aug_v2.pipeline_crop(S, box): box + 0.17 * max side, mirror padding outside S (training/eval code itself)
  input variant applied to the window (V1/V2 Macenko [+CLAHE] with the train reference; V3 adds the H channel)
  model (tinyseg.TinyUNet, sigmoid) -> nc_mask_pipeline postprocess (centre component, no clip inside the crop
  [tolerance = margin], holes filled, nucleus within cell, nucleus blobs < 20 px removed) -> back to S coordinates
Metrics per cell (only for structures present in GT): Dice, IoU, boundary F1 at 2 px, N:C = nucleus/(cell - nucleus)
abs error (cells with both GT structures), failures (empty cell / empty nucleus prediction, Dice < 0.5), ms per cell
(variant transform + input + GPU inference, batch 1, synced + post-processing; median after 5 warm-up cells).
95 % CI: cluster bootstrap over FOV groups (2,000 resamples, seed 0).

    python eval_v3.py --models V0=runs/V0 V1=runs/V1 V2=runs/V2 V3=runs/V3 v2_1=<old> --sets T1 T1b [T2] \
                      [--t2-fov-dir DIR] --out eval/
"""
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
H = Path.home() / "Downloads"
TR = H / "ANUPAM KA HOOD/seg_audit/extracted/tinyseg_retrain/tinyseg_retrain"
NCM = H / "ANUPAM KA HOOD/nc_mask_pipeline"
EX = HERE.parent / "explore"
CHK = H / "Anupam_AI/check the segmentation/nc_ratio_ground_truth"
T2E = H / "nc_ratio_new_evaluation"
sys.path.insert(0, str(TR)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(NCM))
import aug_v2, data, tinyseg  # noqa: E402
import preprocess as pp  # noqa: E402
from ncmask.postprocess import postprocess  # noqa: E402

MARGIN = aug_v2.PAD


# ------------------------------------------------------------------ test sets: list of dict(name, group, S, nuc, cell)
def load_T1():
    split = json.load(open(HERE / "split_v3.json"))
    grp = pd.read_csv(EX / "fov_groups.csv").set_index("name").group.to_dict()
    by = {i["file"]: i for i in data.load_coco(TR / "data")}
    out = []
    for n in split["t1_clean212"]:
        img, nuc, cell = aug_v2.load_item(by[n])
        out.append({"name": n, "group": grp[n], "S": img, "nuc": nuc, "cell": cell})
    return out


def load_T1b():
    grp = pd.read_csv(EX / "fov_groups.csv").set_index("name").group.to_dict()
    out = []
    sub = "cervical_segmentation_01"
    for p in sorted((CHK / "images" / sub).glob("*.png")):
        n = p.name
        nm = cv2.imread(str(CHK / "ground_truth_masks/nucleus" / sub / n), 0) > 0
        cm = cv2.imread(str(CHK / "ground_truth_masks/cell" / sub / n), 0) > 0
        out.append({"name": n, "group": grp[n], "S": cv2.imread(str(p)), "nuc": nm.astype(np.uint8), "cell": cm.astype(np.uint8)})
    return out


def load_T2(fov_dir):
    """Locate each NC003 crop in its source FOV (exact pixel match required) and paste its GT masks there."""
    fov_dir = Path(fov_dir)
    gt = pd.read_csv(T2E / "results/ground_truth_nc_ratios.csv")
    cls = pd.read_csv(HERE.parent / "t2_verify/t2_verify.csv").set_index("file")["class"].to_dict()
    out, cache = [], {}
    for _, r in gt.iterrows():
        n = r.original_filename; fov = n.split("_a")[0]
        cands = [p for p in fov_dir.iterdir() if p.stem == fov]
        if not cands:
            raise FileNotFoundError(f"no FOV {fov}.* in {fov_dir}")
        if fov not in cache:
            cache[fov] = cv2.imread(str(cands[0]))
        S = cache[fov]; crop = cv2.imread(str(T2E / "selected_images" / n))
        res = cv2.matchTemplate(S, crop, cv2.TM_SQDIFF); _, _, (x, y), _ = cv2.minMaxLoc(res)
        if not np.array_equal(S[y:y + crop.shape[0], x:x + crop.shape[1]], crop):
            raise ValueError(f"{n}: crop not found pixel-exact in {cands[0]}")
        masks = {}
        for s in ("nucleus", "cell"):
            m = np.zeros(S.shape[:2], np.uint8)
            m[y:y + crop.shape[0], x:x + crop.shape[1]] = cv2.imread(str(T2E / "ground_truth_masks" / s / n), 0) > 0
            masks[s] = m
        out.append({"name": n, "group": fov, "S": S, "nuc": masks["nucleus"], "cell": masks["cell"],
                    "t2_class": cls.get(n), "crop_xy": (x, y, crop.shape[1], crop.shape[0])})
    return out


# ------------------------------------------------------------------ models
class Model:
    def __init__(self, name, run_dir, device, weights=None):
        cfg = json.loads((Path(run_dir) / "config.json").read_text())
        self.name, self.in_ch, self.size = name, cfg["in_ch"], cfg["size"]
        self.variant = cfg.get("variant", "V0")
        self.net = tinyseg.load(str(weights or Path(run_dir) / "best.pt"), device, self.in_ch)
        self.device = device
        self.ref = json.load(open(HERE / "stain_ref.json")) if self.variant in ("V1", "V2") else None
        h = json.load(open(HERE / "h_stats.json")) if self.in_ch == 4 else None
        self.hmu, self.hsd = (h["mean"], h["std"]) if h else (0, 1)
        self.fallbacks = 0

    @torch.no_grad()
    def __call__(self, window):
        w, ok = pp.transform_source(self.variant, window, self.ref)
        self.fallbacks += (not ok)
        x = tinyseg.to_input(w, self.size)
        if self.in_ch == 4:
            hs = pp.h_channel(cv2.resize(w, (self.size, self.size), interpolation=cv2.INTER_AREA))
            x = np.concatenate([x, ((hs - self.hmu) / self.hsd)[..., None]], -1)
        t = torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float32).copy()).to(self.device)
        return torch.sigmoid(self.net(t))[0].cpu().numpy()


# ------------------------------------------------------------------ metrics
def dice_iou(p, g):
    p, g = p.astype(bool), g.astype(bool); i = (p & g).sum(); s = p.sum() + g.sum(); u = (p | g).sum()
    return (1.0 if s == 0 else 2 * i / s), (1.0 if u == 0 else i / u)


def bf_score(p, g, tol=2):
    k = np.ones((3, 3), np.uint8)
    bp = p.astype(np.uint8) - cv2.erode(p.astype(np.uint8), k); bg = g.astype(np.uint8) - cv2.erode(g.astype(np.uint8), k)
    if not bp.any() or not bg.any():
        return float(bp.any() == bg.any())
    dg = cv2.distanceTransform(1 - bg, cv2.DIST_L2, 5); dp = cv2.distanceTransform(1 - bp, cv2.DIST_L2, 5)
    prec = (dg[bp > 0] <= tol).mean(); rec = (dp[bg > 0] <= tol).mean()
    return 0.0 if prec + rec == 0 else float(2 * prec * rec / (prec + rec))


def nc(nuc, cell):
    n, c = int(nuc.sum()), int(cell.sum())
    return n / (c - n) if c > n else np.nan


def run_cell(model, rec, protocol, k, sync):
    S = rec["S"]; H_, W_ = S.shape[:2]
    cell_gt = rec["cell"] if rec["cell"] is not None and rec["cell"].any() else None
    box = aug_v2.base_box(S, cell_gt) if cell_gt is not None else aug_v2.base_box(S, None)
    if protocol == "jitter":
        box = aug_v2.jitter_box(box, np.random.default_rng((777, k)))
    t0 = time.perf_counter()
    win, _, cbox = aug_v2.pipeline_crop(S, [], box)
    prob = model(win)
    if sync: torch.cuda.synchronize()
    m = box[2] - box[0], box[3] - box[1]
    r = postprocess(prob, win.shape[0], win.shape[1], cbox, 0.5, 0.5, MARGIN * max(m), 20)
    ms = (time.perf_counter() - t0) * 1000
    # back to S coordinates (same rounding as pipeline_crop)
    mg = MARGIN * max(m); X0, Y0 = int(np.floor(box[0] - mg)), int(np.floor(box[1] - mg))
    pn, pc = np.zeros((H_, W_), bool), np.zeros((H_, W_), bool)
    ys, xs = slice(max(0, Y0), min(H_, Y0 + win.shape[0])), slice(max(0, X0), min(W_, X0 + win.shape[1]))
    wy, wx = slice(ys.start - Y0, ys.stop - Y0), slice(xs.start - X0, xs.stop - X0)
    pn[ys, xs] = r["nucleus"][wy, wx]; pc[ys, xs] = r["cell"][wy, wx]
    out = {"name": rec["name"], "group": rec["group"], "ms": ms, "t2_class": rec.get("t2_class"),
           "fail_empty_cell": int(cell_gt is not None and not pc.any()),
           "fail_empty_nuc": int(rec["nuc"] is not None and rec["nuc"].any() and not pn.any())}
    for s, p, g in (("nuc", pn, rec["nuc"]), ("cell", pc, cell_gt)):
        if g is not None and g.any():
            d, i = dice_iou(p, g); out[f"{s}_dice"], out[f"{s}_iou"], out[f"{s}_bf2"] = d, i, bf_score(p, g)
            out[f"fail_{s}_dice_lt_0.5"] = int(d < 0.5)
    if cell_gt is not None and rec["nuc"] is not None and rec["nuc"].any():
        out["nc_gt"], out["nc_pred"] = nc(rec["nuc"] > 0, cell_gt > 0), nc(pn, pc)
        out["nc_abs_err"] = abs(out["nc_pred"] - out["nc_gt"]) if np.isfinite(out["nc_pred"]) else np.nan
    return out


# ------------------------------------------------------------------ CIs
def cluster_ci(df, col, n=2000, seed=0, agg=np.nanmean):
    d = df[["group", col]].dropna()
    if d.empty:
        return np.nan, np.nan, np.nan
    groups = d.group.unique(); by = {g: d[col][d.group == g].values for g in groups}
    rng = np.random.default_rng(seed)
    stats = [agg(np.concatenate([by[g] for g in rng.choice(groups, len(groups))])) for _ in range(n)]
    return float(agg(d[col].values)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


METRICS = ["nuc_dice", "nuc_iou", "nuc_bf2", "cell_dice", "cell_iou", "cell_bf2", "nc_abs_err"]
FAILS = ["fail_empty_cell", "fail_empty_nuc", "fail_nuc_dice_lt_0.5", "fail_cell_dice_lt_0.5"]


def summarise(per):
    rows = []
    for (ts, sub, proto, model), g in per.groupby(["set", "subset", "protocol", "model"], sort=False):
        r = {"set": ts, "subset": sub, "protocol": proto, "model": model, "n": len(g), "n_groups": g.group.nunique()}
        for m in METRICS:
            r[m], r[m + "_lo"], r[m + "_hi"] = cluster_ci(g, m)
        for f in FAILS:
            r[f] = int(g[f].fillna(0).sum()) if f in g else 0
        r["ms_median"] = float(g.ms.median())
        rows.append(r)
    s = pd.DataFrame(rows)
    # flag: CI overlaps the best model's CI (per set/subset/protocol and metric); lower is better for nc_abs_err
    for m in METRICS:
        s[m + "_overlaps_best"] = False
        for _, g in s.groupby(["set", "subset", "protocol"]):
            b = g[m].idxmin() if m == "nc_abs_err" else g[m].idxmax()
            lo, hi = s.loc[b, m + "_lo"], s.loc[b, m + "_hi"]
            for i in g.index:
                s.loc[i, m + "_overlaps_best"] = bool(i != b and s.loc[i, m + "_hi"] >= lo and s.loc[i, m + "_lo"] <= hi)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="name=run_dir (run_dir has config.json + best.pt)")
    ap.add_argument("--sets", nargs="+", default=["T1", "T1b"])
    ap.add_argument("--protocols", nargs="+", default=["gt", "jitter"])
    ap.add_argument("--t2-fov-dir", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = False
    sets = {"T1": load_T1, "T1b": load_T1b, "T2": lambda: load_T2(a.t2_fov_dir)}
    data_ = {s: sets[s]() for s in a.sets}
    models = []
    for spec in a.models:
        n, d = spec.split("=", 1)
        models.append(Model(n, d, device))
    rows = []
    for ts, recs in data_.items():
        for proto in a.protocols:
            for mdl in models:
                for k in range(5):                                       # warm-up
                    run_cell(mdl, recs[k % len(recs)], proto, k, device == "cuda")
                for k, rec in enumerate(recs):
                    r = run_cell(mdl, rec, proto, k, device == "cuda")
                    r.update(set=ts, protocol=proto, model=mdl.name)
                    rows.append(r)
                print(f"{ts} {proto} {mdl.name}: done", flush=True)
    per = pd.DataFrame(rows)
    per["subset"] = "all"
    if "T2" in data_:
        sub = per[(per.set == "T2") & (per.t2_class.isin(["corrected", "independent"]))].copy(); sub["subset"] = "corrected+independent"
        sub2 = per[(per.set == "T2") & (per.t2_class == "identical")].copy(); sub2["subset"] = "identical"
        per = pd.concat([per, sub, sub2], ignore_index=True)
    per.to_csv(out / "per_cell.csv", index=False)
    s = summarise(per)
    s.to_csv(out / "summary.csv", index=False)
    (out / "fallbacks.json").write_text(json.dumps({m.name: m.fallbacks for m in models}))
    pd.set_option("display.width", 250)
    print(s[["set", "subset", "protocol", "model", "n", "nuc_dice", "cell_dice", "nuc_bf2", "cell_bf2", "nc_abs_err", "ms_median"]].round(4).to_string())


if __name__ == "__main__":
    main()
