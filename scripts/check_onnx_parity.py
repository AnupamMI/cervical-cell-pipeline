#!/usr/bin/env python3
"""ONNX bundle vs PyTorch, full pipeline (ncmask.fov.process_fov) on every FOV in samples/:

  boxes      detector A, B and C separately: same count, one-to-one Hungarian match, every pair IoU >= 0.99
             (C: cluster / debris only, and every matched pair has the same class)
  cells      same number of cells, every cell matched (box IoU >= 0.99), same det_tag
  masks      Dice >= 0.999 per FOV (pooled cell and nucleus foreground) and per cell (cell mask; nucleus Dice per
             cell is reported, and must be >= 0.99 where either nucleus has >= 50 px)
  gate       identical status for every matched cell (reasons compared and reported too), identical
             near_cluster / near_debris, and (PyTorch) identical statuses with detector C on and off

    python scripts/check_onnx_parity.py [--samples samples] [--bundle models/onnx] [--limit N] [--json out.json]

Exit 0 = all checks pass. Needs torch + ultralytics (the training environment).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

BOX_IOU, DICE_FOV, DICE_CELL, DICE_NUC_CELL, NUC_MIN_PX = 0.99, 0.999, 0.999, 0.99, 50


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def match(a, b, thr):
    from ncmask.consensus import match_boxes
    pairs, am, bm = match_boxes(np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4), thr)
    return pairs, bool(am.all() and bm.all())


def compare_fov(img, fid, pt, ox, cfg, ocfg):
    from ncmask.fov import process_fov
    res = {"fov": fid, "fail": []}
    for sec in ("detector", "detector_b", "detector_c"):
        if pt.get(sec) is None:
            continue
        (b1, c1, k1, _), (b2, c2, k2, _) = pt[sec].detect(img), ox[sec].detect(img)
        pairs, ok = match(b1, b2, BOX_IOU)
        res[sec] = {"n_pt": len(b1), "n_onnx": len(b2), "min_iou": round(min((p[2] for p in pairs), default=1.0), 6)}
        if len(b1) != len(b2) or not ok:
            res["fail"].append(f"{sec}: {len(b1)} vs {len(b2)} boxes or a pair IoU < {BOX_IOU}")
        if any(k1[i] != k2[j] for i, j, _ in pairs):
            res["fail"].append(f"{sec}: a matched pair has different classes")
        if sec == "detector_c":
            res[sec]["classes_pt"] = sorted({pt[sec].names[int(k)] for k in k1})
    t = time.perf_counter()
    r1, _, cl1, nl1, _, i1 = process_fov(img, fid, fid, pt["detector"], pt["seg"], cfg, pt["detector_b"], pt.get("detector_c"))
    res["t_pt_s"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    r2, _, cl2, nl2, _, i2 = process_fov(img, fid, fid, ox["detector"], ox["seg"], ocfg, ox["detector_b"], ox.get("detector_c"))
    res["t_onnx_s"] = round(time.perf_counter() - t, 2)
    if pt.get("detector_c") is not None:
        # detector C is information only: the gate must not see it (statuses / reasons / flags identical without it)
        r0 = process_fov(img, fid, fid, pt["detector"], pt["seg"], cfg, pt["detector_b"])[0]
        key = lambda rows: [(r["cell_id"], r["status"], r["reasons"], r["flags"]) for r in rows]
        res["c_changes_gate"] = key(r0) != key(r1)
        if res["c_changes_gate"]:
            res["fail"].append("detector C changed a status / reason / flag")
        res["c_boxes"] = {"n_pt": len(i1["c_boxes"]), "n_onnx": len(i2["c_boxes"])}
    box = lambda rows: [[r[k] for k in ("box_x1", "box_y1", "box_x2", "box_y2")] for r in rows]
    pairs, ok = match(box(r1), box(r2), BOX_IOU)
    res["cells"] = {"n_pt": len(r1), "n_onnx": len(r2)}
    if len(r1) != len(r2) or not ok:
        res["fail"].append(f"cells: {len(r1)} vs {len(r2)} or unmatched")
    res["dice_cell_fov"] = round(dice(cl1 > 0, cl2 > 0), 6)
    res["dice_nuc_fov"] = round(dice(nl1 > 0, nl2 > 0), 6)
    if min(res["dice_cell_fov"], res["dice_nuc_fov"]) < DICE_FOV:
        res["fail"].append(f"FOV Dice < {DICE_FOV}")
    dc, dn, st_diff, rs_diff, tag_diff = [], [], [], [], []
    for i, j, _ in pairs:
        a, b = r1[i], r2[j]
        dc.append(dice(cl1 == a["cell_id"], cl2 == b["cell_id"]))
        n1, n2 = nl1 == a["cell_id"], nl2 == b["cell_id"]
        d = dice(n1, n2)
        dn.append(d)
        if max(n1.sum(), n2.sum()) >= NUC_MIN_PX and d < DICE_NUC_CELL:
            res["fail"].append(f"cell {a['cell_id']}: nucleus Dice {d:.4f}")
        if a["status"] != b["status"]:
            st_diff.append((a["cell_id"], a["status"], b["status"]))
        if a["reasons"] != b["reasons"]:
            rs_diff.append((a["cell_id"], a["reasons"], b["reasons"]))
        if a["det_tag"] != b["det_tag"]:
            tag_diff.append((a["cell_id"], a["det_tag"], b["det_tag"]))
        if (a["near_cluster"] != "") != (b["near_cluster"] != "") or (a["near_debris"] != "") != (b["near_debris"] != ""):
            res["fail"].append(f"cell {a['cell_id']}: near_cluster / near_debris differ")
    res["min_dice_cell"] = round(min(dc, default=1.0), 6)
    res["min_dice_nucleus"] = round(min(dn, default=1.0), 6)
    if res["min_dice_cell"] < DICE_CELL:
        res["fail"].append(f"a cell mask Dice {res['min_dice_cell']} < {DICE_CELL}")
    res["status_diff"], res["reasons_diff"], res["det_tag_diff"] = st_diff, rs_diff, tag_diff
    if st_diff:
        res["fail"].append(f"status differs for {len(st_diff)} cell(s)")
    if tag_diff:
        res["fail"].append(f"det_tag differs for {len(tag_diff)} cell(s)")
    res["status_pt"], res["status_onnx"] = i1["status_counts"], i2["status_counts"]
    return res


def load_models(config, bundle):
    from ncmask import config as cfgmod
    from ncmask.detect import Detector
    from ncmask.onnx_backend import OnnxDetector, OnnxSegModel, load_bundle_config
    from ncmask.segment import SegModel
    cfg, ocfg = cfgmod.load(config), load_bundle_config(config, bundle)
    if cfg["consensus"]["detection_mode"] != ocfg["consensus"]["detection_mode"]:
        raise SystemExit(f"detection_mode differs: PyTorch {cfg['consensus']['detection_mode']} vs bundle "
                         f"{ocfg['consensus']['detection_mode']} (detector_b missing on one side?)")
    cons = cfg["consensus"]["detection_mode"] != "single"
    use_c = bool((cfg.get("detector_c") or {}).get("enabled"))
    if use_c != bool((ocfg.get("detector_c") or {}).get("enabled")):
        raise SystemExit("detector C is on in config.yaml but not in the bundle (or vice versa): re-run scripts/export_onnx.py")
    pt = {"detector": Detector(cfg, "cpu"), "detector_b": Detector(cfg, "cpu", "detector_b") if cons else None,
          "detector_c": Detector(cfg, "cpu", "detector_c") if use_c else None, "seg": SegModel(cfg, "cpu")}
    ox = {"detector": OnnxDetector(ocfg, "cpu"), "detector_b": OnnxDetector(ocfg, "cpu", "detector_b") if cons else None,
          "detector_c": OnnxDetector(ocfg, "cpu", "detector_c") if use_c else None, "seg": OnnxSegModel(ocfg, "cpu")}
    return cfg, ocfg, pt, ox


def run(samples, bundle, config, limit=0):
    from ncmask.imageio import load_bgr8
    from ncmask.onnx_backend import bundle_manifest
    man = bundle_manifest(bundle)
    if man is None:
        raise SystemExit(f"no manifest.json in {bundle}: run scripts/export_onnx.py first")
    cfg, ocfg, pt, ox = load_models(config, bundle)
    files = sorted(p for p in Path(samples).iterdir() if p.suffix.lower() in cfg["input"]["extensions"])
    files = files[:limit] if limit else files
    out = []
    for p in files:
        img, _ = load_bgr8(p, cfg["input"]["sixteen_bit"])
        r = compare_fov(img, p.stem, pt, ox, cfg, ocfg)
        out.append(r)
        print(f"{p.stem}: cells {r['cells']['n_pt']}/{r['cells']['n_onnx']}  "
              f"A {r['detector']['min_iou']:.6f}" + (f"  B {r['detector_b']['min_iou']:.6f}" if "detector_b" in r else "") +
              (f"  C {r['detector_c']['n_pt']}/{r['detector_c']['n_onnx']} {r['detector_c']['min_iou']:.6f}" if "detector_c" in r else "") +
              f"  Dice fov {r['dice_cell_fov']:.6f}/{r['dice_nuc_fov']:.6f}  min cell {r['min_dice_cell']:.6f}"
              f"  min nuc {r['min_dice_nucleus']:.6f}  status diffs {len(r['status_diff'])}"
              f"  t pt {r['t_pt_s']}s onnx {r['t_onnx_s']}s  {'FAIL ' + '; '.join(r['fail']) if r['fail'] else 'ok'}",
              flush=True)
    return {"bundle_models": {k: {"source": v["source"], "onnx_sha256": v["onnx_sha256"]} for k, v in man["models"].items()},
            "detection_mode": cfg["consensus"]["detection_mode"],
            "detector_c": bool((cfg.get("detector_c") or {}).get("enabled")), "thresholds": {
                "box_iou": BOX_IOU, "dice_fov": DICE_FOV, "dice_cell": DICE_CELL, "dice_nucleus_cell": DICE_NUC_CELL},
            "fovs": out, "passed": all(not r["fail"] for r in out) and bool(out)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default=str(HERE / "samples"))
    ap.add_argument("--bundle", default=str(HERE / "models" / "onnx"))
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", default=str(HERE / "reports" / "onnx_parity.json"))
    a = ap.parse_args(argv)
    rep = run(a.samples, a.bundle, a.config, a.limit)
    Path(a.json).write_text(json.dumps(rep, indent=1, default=str))
    print(("PASS" if rep["passed"] else "FAIL") + f"  ({len(rep['fovs'])} FOVs) -> {a.json}")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
