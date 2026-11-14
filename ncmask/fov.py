"""One FOV (pipeline v3):
Stage 0 image QC (flags only) -> Stage 1 detect (YOLO26s; YOLO26l consensus tag) -> Stage 2 containment suppression,
cell_at_edge / crop_clamped, box size vs training -> Stage 3 crop (margin, mirror padding) + segment + centre component,
nucleus within cell -> Stage 4 two layers: FULL per-cell masks (measurement, overlaps allowed) + EXCLUSIVE label map
(display; higher confidence wins) -> occluded_frac, no cell dropped for overlap -> Stage 5 gate (measurable / review /
excluded + reasons) -> outputs."""
import json
import logging
import time

import cv2
import numpy as np

from . import geometry
from .consensus import match_boxes
from .postprocess import postprocess
from .quality import gate, image_qc, nucleus_touches_cell_boundary

log = logging.getLogger("ncmask.fov")

CELL_COLS = ["fov", "cell_id", "status", "reasons", "box_x1", "box_y1", "box_x2", "box_y2", "conf", "det_tag", "b_conf",
             "ab_iou", "cell_area_px", "nucleus_area_px", "nc_ratio", "excl_cell_area_px", "excl_nucleus_area_px",
             "occluded_frac", "flags", "fov_qc", "crop_x0", "crop_y0", "crop_x1", "crop_y1", "clipped_frac"]
STATUS_BGR = {"measurable": (0, 200, 0), "review": (0, 165, 255), "excluded": (0, 0, 255)}
L_ONLY_COLS = ["fov", "b_conf", "box_x1", "box_y1", "box_x2", "box_y2", "max_iou_with_A"]
# COCO categories follow the training annotations (instances_default.json)
CATEGORIES = [{"id": 1, "name": "nucleus-area", "supercategory": ""}, {"id": 2, "name": "cell-area", "supercategory": ""}]


def resolve_overlaps(shape, items, drop_lost_frac=None):
    """items: list of dicts with cell_id, conf, win (CropWindow), cell, nucleus (crop-sized bool masks).
    Paints cells in priority order (higher conf first, ties -> lower cell_id); a pixel already owned by a
    higher-priority cell is not taken. A cell losing MORE than drop_lost_frac of its pixels is dropped (not painted).
    This single pass is exact: a cell's loss depends only on higher-priority cells, which are final when it is painted.
    Returns (cell_labels int32, nucleus_labels, lost px per id, won px per id = pixels this cell kept that a kept
    lower-priority cell also claimed, dropped {id: lost fraction})."""
    cell_lab = np.zeros(shape, np.int32)
    nuc_lab = np.zeros(shape, np.int32)
    lost, won, dropped = {}, {}, {}
    for it in sorted(items, key=lambda d: (-d["conf"], d["cell_id"])):
        w, cid = it["win"], it["cell_id"]
        sl = (slice(w.y0, w.y1), slice(w.x0, w.x1))
        region = cell_lab[sl]
        owners = region[it["cell"] & (region > 0)]
        area = int(it["cell"].sum())
        if drop_lost_frac is not None and area and owners.size / area > drop_lost_frac:
            dropped[cid] = owners.size / area
            continue
        lost[cid] = int(owners.size)
        for o, n in zip(*np.unique(owners, return_counts=True)):
            won[int(o)] = won.get(int(o), 0) + int(n)
        mine = it["cell"] & (region == 0)
        region[mine] = cid
        nuc_lab[sl][it["nucleus"] & mine] = cid
    return cell_lab, nuc_lab, lost, won, dropped


def polygons(mask, offset, eps):
    """External contours of a bool mask -> COCO polygons (lists of x,y floats) in FOV coords, >= 3 points each."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        if eps > 0:
            c = cv2.approxPolyDP(c, eps, True)
        if len(c) >= 3:
            out.append([float(v) for v in (c[:, 0, :] + offset).ravel()])
    return out


def draw_dashed(img, pts, colour, th, on=14, off=10):
    """Closed polyline pts (N,2) drawn as dashes `on` px long separated by `off` px (one polyline per dash)."""
    pts = np.vstack([pts, pts[:1]]).astype(np.float64)
    pos = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    phase = np.mod(pos, on + off) < on
    start = None
    for k in range(len(pts)):
        if phase[k] and start is None:
            start = k
        if start is not None and (not phase[k] or k == len(pts) - 1):
            seg = np.round(pts[start:k + 1]).astype(np.int32)
            if len(seg) >= 2:
                cv2.polylines(img, [seg], False, colour, th, cv2.LINE_AA)
            start = None


INFO_FLAGS = {"crop_clamped", "overlap_won"}   # informational: don't mark the cell as flagged in the overlay


def is_flagged(flags):
    return any(f and f not in INFO_FLAGS for f in flags.split(";"))


def draw_overlay(img, rows, cell_lab, nuc_lab, max_side):
    ov = img.copy()
    nuc_any = nuc_lab > 0
    red = np.zeros_like(ov); red[..., 2] = 255
    ov[nuc_any] = (0.55 * ov[nuc_any] + 0.45 * red[nuc_any]).astype(np.uint8)
    for r in rows:
        cid = r["cell_id"]
        x1, y1, x2, y2 = (int(round(r[k])) for k in ("box_x1", "box_y1", "box_x2", "box_y2"))
        flagged = r.get("status", "measurable") != "measurable"
        cv2.rectangle(ov, (x1, y1), (x2 - 1, y2 - 1), STATUS_BGR.get(r.get("status"), (255, 255, 0)), 2)
        if r["excl_cell_area_px"]:
            sl = (slice(r["crop_y0"], r["crop_y1"]), slice(r["crop_x0"], r["crop_x1"]))
            cs, _ = cv2.findContours((cell_lab[sl] == cid).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if r.get("det_tag") == "s_only":   # detector A only: dashed outline
                for c in cs:
                    draw_dashed(ov, c[:, 0, :] + (r["crop_x0"], r["crop_y0"]), (0, 220, 0), 2)
            else:
                cv2.drawContours(ov, cs, -1, (0, 220, 0), 2, offset=(r["crop_x0"], r["crop_y0"]))
        txt = f"{cid} {r['conf']:.2f}" + (f" {r['reasons'].split(';')[0]}" if flagged else "")
        org = (x1 + 3, max(14, y1 + 16))
        cv2.putText(ov, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(ov, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if max_side and max(ov.shape[:2]) > max_side:
        s = max_side / max(ov.shape[:2])
        ov = cv2.resize(ov, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return ov


def clip_tol_px(cfg, box):
    """postprocess.clip_tolerance_px in px for this box: 'margin' = crop.margin * max(box w, h) (the crop window)."""
    t = cfg["postprocess"]["clip_tolerance_px"]
    return cfg["crop"]["margin"] * max(box[2] - box[0], box[3] - box[1]) if t == "margin" else float(t)


def extract_crop(img, box, cfg, seg):
    """Crop for one detector box -> (crop BGR uint8, box in crop coords, CropWindow, crop origin in FOV px,
    crop sides lying on the FOV edge). clamp: the window cut at the FOV edge. reflect: aug_v2.pipeline_crop itself
    (reflect padding outside the FOV, as in training / evaluate_v2 'jitter')."""
    H, W = img.shape[:2]
    win = geometry.crop_window(box, W, H, cfg["crop"]["margin"])
    if cfg["crop"]["edge_mode"] == "reflect":
        crop, _, cbox = seg.aug_v2.pipeline_crop(img, [], box)
        origin = (win.ux0, win.uy0)
        if crop.shape[:2] != (win.uy1 - win.uy0, win.ux1 - win.ux0):
            raise RuntimeError("aug_v2.pipeline_crop window differs from geometry.crop_window")
    else:
        crop, cbox, origin = img[win.y0:win.y1, win.x0:win.x1], win.box, (win.x0, win.y0)
    return np.ascontiguousarray(crop), tuple(float(v) for v in cbox), win, origin, win.edge_sides


def process_fov(img, fov_id, src_name, detector, seg, cfg, detector_b=None):
    """Returns (rows, coco dict, cell_lab, nuc_lab, overlay or None, info). Raises on internal errors."""
    t0 = time.perf_counter()
    H, W = img.shape[:2]
    qc_flags, qc_metrics = image_qc(img, cfg["qc"])                       # Stage 0: flag, never skip
    boxes, conf, dinfo = detector(img)
    mode = cfg.get("consensus", {}).get("detection_mode", "single")
    dropped, l_only = [], []   # dropped: removed from the outputs, with the reason (done.json + run dropped.csv)
    tag = np.array(["single"] * len(boxes), dtype=object)
    bconf, abiou = np.full(len(boxes), np.nan), np.full(len(boxes), np.nan)
    if mode != "single":
        if detector_b is None:
            raise ValueError(f"detection_mode {mode} needs detector B")
        bb, bc, binfo = detector_b(img)
        pairs, am, bm = match_boxes(boxes, bb, cfg["consensus"]["match_iou"])
        tag = np.where(am, "consensus", "s_only").astype(object)
        for i, j, v in pairs:
            bconf[i], abiou[i] = bc[j], v
        iou_ab = geometry.iou_matrix(bb, boxes) if len(bb) and len(boxes) else np.zeros((len(bb), 0))
        for j in np.where(~bm)[0]:
            l_only.append({"fov": fov_id, "b_conf": round(float(bc[j]), 5), **{k: round(float(v), 2) for k, v in
                           zip(("box_x1", "box_y1", "box_x2", "box_y2"), bb[j])},
                           "max_iou_with_A": round(float(iou_ab[j].max()), 4) if iou_ab.shape[1] else 0.0})
        dinfo["b"] = {**binfo, "n_b": int(len(bb)), "consensus": int(am.sum()), "s_only": int((~am).sum()),
                      "l_only": int((~bm).sum())}
        if mode == "consensus_filter":
            for j in np.where(~am)[0]:
                dropped.append({"reason": "s_only_filtered", "cell_id": "", "conf": round(float(conf[j]), 5),
                                "box": [round(float(v), 2) for v in boxes[j]], "detail": "no matching detector-B box"})
            boxes, conf, tag, bconf, abiou = boxes[am], conf[am], tag[am], bconf[am], abiou[am]
    t_det = time.perf_counter() - t0
    pp, sc, dc = cfg["postprocess"], cfg["seg"], cfg["detector"]
    if dc["containment_thr"] and len(boxes):
        keep, into = geometry.containment_suppress(boxes, conf, dc["containment_thr"], dc["containment_min_area_ratio"])
        for j in np.where(~keep)[0]:
            dropped.append({"reason": "contained", "cell_id": "", "conf": round(float(conf[j]), 5),
                            "box": [round(float(v), 2) for v in boxes[j]],
                            "detail": f"{into[j]:.2f} of its area inside a higher-confidence box"})
            log.info("%s: dropped contained box conf %.3f (%.0f%% inside a higher-conf box)", fov_id, conf[j], 100 * into[j])
        boxes, conf, tag, bconf, abiou = boxes[keep], conf[keep], tag[keep], bconf[keep], abiou[keep]
    dinfo["n_contained_dropped"] = int(sum(d["reason"] == "contained" for d in dropped))
    min_px = float(dc["min_box_px"])
    e = float(pp["cell_at_edge_px"])
    items, rows_pre = [], []
    crops, cboxes, metas = [], [], []
    for i, (b, c) in enumerate(zip(boxes, conf), 1):
        rec = {"cell_id": i, "conf": float(c), "box": [float(v) for v in b], "flags": [], "det_tag": tag[i - 1],
               "b_conf": "" if np.isnan(bconf[i - 1]) else round(float(bconf[i - 1]), 5),
               "ab_iou": "" if np.isnan(abiou[i - 1]) else round(float(abiou[i - 1]), 4)}
        if b[0] <= e or b[1] <= e or b[2] >= W - e or b[3] >= H - e:
            rec["flags"].append("cell_at_edge")
        bw, bh = b[2] - b[0], b[3] - b[1]
        if not cfg["size"]["box_area_min"] <= bw * bh <= cfg["size"]["box_area_max"]:
            rec["flags"].append("box_size_out_of_range")
        if bw < min_px or bh < min_px:
            rec["flags"].append("box_too_small")
            rows_pre.append((rec, None)); continue
        crop, cbox, win, origin, edge_sides = extract_crop(img, b, cfg, seg)
        if win.truncated:
            rec["flags"].append("crop_clamped")
        crops.append(crop); cboxes.append(cbox)
        metas.append((rec, win, origin, edge_sides))
        rows_pre.append((rec, win))
    t1 = time.perf_counter()
    probs = seg.predict(crops, cboxes) if crops else []
    t_seg = time.perf_counter() - t1
    for (rec, win, origin, edge_sides), crop, cbox, prob in zip(metas, crops, cboxes, probs):
        tol = clip_tol_px(cfg, rec["box"])
        oy, ox = win.y0 - origin[1], win.x0 - origin[0]       # in-FOV part of the (possibly padded) crop
        r = postprocess(prob, crop.shape[0], crop.shape[1], cbox, sc["thr_nucleus"], sc["thr_cell"],
                        tol, pp["nucleus_min_component_px"], edge_sides, valid=(ox, oy, ox + win.w, oy + win.h))
        rec["clipped_frac"] = r["clipped_frac"]
        cell, nuc = r["cell"], r["nucleus"]
        if cell.shape != (win.h, win.w):  # reflect mode, padded on any side: keep only the part inside the FOV
            oy, ox = win.y0 - origin[1], win.x0 - origin[0]
            cell, nuc = cell[oy:oy + win.h, ox:ox + win.w], nuc[oy:oy + win.h, ox:ox + win.w]
            if not cell.any() and "empty_mask" not in r["flags"]:
                r["flags"] = [f for f in r["flags"] if f not in ("no_nucleus", "multi_nucleus")] + ["empty_mask"]
        rec["flags"] += r["flags"]
        items.append({"cell_id": rec["cell_id"], "conf": rec["conf"], "win": win, "cell": cell, "nucleus": nuc})
    if pp["drop_lost_frac"] is not None:
        raise ValueError("pipeline v3 never drops cells for overlap: postprocess.drop_lost_frac must be null")
    cell_lab, nuc_lab, lost, won, dropped_ov = resolve_overlaps((H, W), items, None)
    full = {it["cell_id"]: it for it in items}
    for rec, win in rows_pre:
        if rec["cell_id"] in dropped_ov:
            f = dropped_ov[rec["cell_id"]]
            dropped.append({"reason": "overlap_lost", "cell_id": rec["cell_id"], "conf": round(rec["conf"], 5),
                            "box": [round(v, 2) for v in rec["box"]],
                            "detail": f"lost {f:.2f} of its pixels to higher-confidence cells"})
            log.info("%s: dropped cell %d conf %.3f (lost %.0f%% of its pixels to higher-conf cells)",
                     fov_id, rec["cell_id"], rec["conf"], 100 * f)
    rows_pre = [(rec, win) for rec, win in rows_pre if rec["cell_id"] not in dropped_ov]
    t2 = time.perf_counter()

    eps = float(cfg["output"]["coco_polygon_epsilon_px"])
    rows, anns, anns_full = [], [], []
    coco = {"info": {"description": f"nc_mask_pipeline v3 EXCLUSIVE layer (display) for {src_name}"},
            "images": [{"id": 1, "file_name": src_name, "width": W, "height": H}],
            "categories": CATEGORIES, "annotations": anns}
    coco_full = {"info": {"description": f"nc_mask_pipeline v3 FULL layer (measurement; masks may overlap) for {src_name}"},
                 "images": [{"id": 1, "file_name": src_name, "width": W, "height": H}],
                 "categories": CATEGORIES, "annotations": anns_full}
    for rec, win in rows_pre:
        cid = rec["cell_id"]
        x1, y1, x2, y2 = rec["box"]
        flags = list(rec["flags"])
        if win is not None:
            sl = (slice(win.y0, win.y1), slice(win.x0, win.x1))
            cm, nm = cell_lab[sl] == cid, nuc_lab[sl] == cid                 # exclusive (display)
            fc, fn = full[cid]["cell"], full[cid]["nucleus"]                 # full (measurement)
            if lost.get(cid, 0):
                flags.append("overlap_lost")
            if won.get(cid, 0):
                flags.append("overlap_won")
            ca, na = int(fc.sum()), int(fn.sum())
            eca, ena = int(cm.sum()), int(nm.sum())
            occl = lost.get(cid, 0) / ca if ca else 0.0
            touch = nucleus_touches_cell_boundary(fc, fn)
            crop_xy = (win.x0, win.y0, win.x1, win.y1)
        else:
            cm = nm = fc = fn = None; ca = na = eca = ena = 0; occl = 0.0; touch = False; crop_xy = ("", "", "", "")
        nc = na / (ca - na) if ca > na and na > 0 else float("nan")
        status, reasons = gate(flags, ca, na, nc, occl, rec["det_tag"], rec["conf"], touch, cfg["size"], cfg["gate"])
        row = {"fov": fov_id, "cell_id": cid, "status": status, "reasons": ";".join(reasons),
               "box_x1": round(x1, 2), "box_y1": round(y1, 2), "box_x2": round(x2, 2), "box_y2": round(y2, 2),
               "conf": round(rec["conf"], 5), "det_tag": rec["det_tag"], "b_conf": rec["b_conf"], "ab_iou": rec["ab_iou"],
               "cell_area_px": ca, "nucleus_area_px": na, "nc_ratio": round(nc, 6) if nc == nc else "",
               "excl_cell_area_px": eca, "excl_nucleus_area_px": ena, "occluded_frac": round(occl, 4),
               "flags": ";".join(flags), "fov_qc": ";".join(qc_flags),
               "crop_x0": crop_xy[0], "crop_y0": crop_xy[1], "crop_x1": crop_xy[2], "crop_y1": crop_xy[3],
               "clipped_frac": round(rec.get("clipped_frac", 0.0), 5)}
        rows.append(row)
        fattrs = {"cell_id": cid, "status": status, "reasons": reasons, "conf": row["conf"], "det_tag": rec["det_tag"],
                  "occluded_frac": row["occluded_frac"], "nc_ratio": row["nc_ratio"]}
        if ca:
            anns_full.append({"id": len(anns_full) + 1, "image_id": 1, "category_id": 2, "iscrowd": 0, "area": ca,
                              "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                              "segmentation": polygons(fc, (win.x0, win.y0), eps), "attributes": fattrs})
        if na:
            anns_full.append({"id": len(anns_full) + 1, "image_id": 1, "category_id": 1, "iscrowd": 0, "area": na,
                              "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                              "segmentation": polygons(fn, (win.x0, win.y0), eps), "attributes": {"cell_id": cid}})
        ca, na = eca, ena                                                       # exclusive layer below
        attrs = {"cell_id": cid, "conf": row["conf"], "status": status, "reasons": reasons, "flags": flags,
                 "det_box_xyxy": [row[k] for k in ("box_x1", "box_y1", "box_x2", "box_y2")],
                 "occluded_frac": row["occluded_frac"], "det_tag": rec["det_tag"], "b_conf": rec["b_conf"],
                 "ab_iou": rec["ab_iou"]}
        anns.append({"id": len(anns) + 1, "image_id": 1, "category_id": 2, "iscrowd": 0, "area": ca,
                     "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                     "segmentation": polygons(cm, (win.x0, win.y0), eps) if ca else [], "attributes": attrs})
        if na:
            ys, xs = np.nonzero(nm)
            anns.append({"id": len(anns) + 1, "image_id": 1, "category_id": 1, "iscrowd": 0, "area": na,
                         "bbox": [float(win.x0 + xs.min()), float(win.y0 + ys.min()),
                                  float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)],
                         "segmentation": polygons(nm, (win.x0, win.y0), eps), "attributes": {"cell_id": cid}})
    overlay = draw_overlay(img, rows, cell_lab, nuc_lab, cfg["output"]["overlay_max_side"]) \
        if cfg["output"]["overlay"] else None
    info = {"detector": dinfo, "n_cells": len(rows), "dropped": dropped, "l_only": l_only, "detection_mode": mode,
            "fov_qc": qc_flags, "fov_qc_metrics": qc_metrics, "coco_full": coco_full,
            "status_counts": {k: sum(r["status"] == k for r in rows) for k in ("measurable", "review", "excluded")},
            "n_dropped_overlap": len(dropped_ov), "t_detect_s": round(t_det, 4), "t_seg_s": round(t_seg, 4),
            "t_total_s": round(time.perf_counter() - t0, 4), "t_post_s": round(t2 - t1 - t_seg, 4)}
    return rows, coco, cell_lab, nuc_lab, overlay, info


def write_fov_outputs(d, rows, coco, cell_lab, nuc_lab, overlay, meta):
    """Write all per-FOV files into folder d (which the caller renames into place when complete)."""
    import csv
    if cell_lab.max() > 65535:
        raise ValueError("more than 65535 cells: cannot store as 16-bit label PNG")
    for name, lab in (("cell_labels.png", cell_lab), ("nucleus_labels.png", nuc_lab)):
        if not cv2.imwrite(str(d / name), lab.astype(np.uint16)):
            raise IOError(f"could not write {d / name}")
    if overlay is not None and not cv2.imwrite(str(d / "overlay.png"), overlay):
        raise IOError(f"could not write {d / 'overlay.png'}")
    (d / "cells.coco.json").write_text(json.dumps(coco))                       # exclusive layer (display)
    (d / "cells_full.coco.json").write_text(json.dumps(meta.pop("coco_full")))   # full layer (measurement)
    with open(d / "cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CELL_COLS); w.writeheader(); w.writerows(rows)
    if meta.get("detection_mode", "single") != "single":
        with open(d / "l_only.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=L_ONLY_COLS); w.writeheader(); w.writerows(meta.get("l_only", []))
    (d / "done.json").write_text(json.dumps(meta, indent=1))  # written last: marks the FOV as complete
