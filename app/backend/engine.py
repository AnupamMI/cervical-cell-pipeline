"""Inference engine for the web app: the ONNX bundle + the unchanged ncmask pipeline (ncmask.fov.process_fov).

One FOV per call. Stage timings come from wrapping the three model callables:
  detect     YOLO26s (detector A)
  consensus  YOLO26l @1280 + A<->B matching (tag only, never filters)
  segment    crop + TinyUNet forward for every box
  gate       everything else in process_fov (post-processing, overlaps, gate, COCO polygons, overlay)
"""
import copy
import csv
import hashlib
import io
import json
import logging
import os
import platform
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from ncmask.fov import CELL_COLS, process_fov
from ncmask.imageio import ImageLoadError, load_bgr8
from ncmask.onnx_backend import (BUNDLE_FILES, OnnxDetector, OnnxSegModel, bundle_manifest, load_bundle_config,
                                 providers)

log = logging.getLogger("app.engine")
ROOT = Path(__file__).resolve().parents[2]
EXPECTED_WH = (2028, 1520)
DISPLAY_EPS_PX = 1.0       # polygon simplification for the JSON / viewer only (downloads keep exact outlines)
MAX_PIXELS = 40_000_000    # decompression-bomb guard (a 2028x1520 FOV is 3.1 MP)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class Timed:
    """Wraps a model callable / object and accumulates wall time."""

    def __init__(self, fn):
        self.fn, self.t = fn, 0.0

    def __call__(self, *a, **k):
        t = time.perf_counter()
        try:
            return self.fn(*a, **k)
        finally:
            self.t += time.perf_counter() - t


class TimedSeg:
    """SegModel proxy: times predict(), passes everything else (aug_v2 for the crops, sizes) through."""

    def __init__(self, seg):
        self._seg, self.t = seg, 0.0

    def predict(self, crops, boxes):
        t = time.perf_counter()
        try:
            return self._seg.predict(crops, boxes)
        finally:
            self.t += time.perf_counter() - t

    def __getattr__(self, k):
        return getattr(self._seg, k)


class Engine:
    def __init__(self, config=None, bundle=None, device=None, threads=None):
        self.config_path = Path(config or os.environ.get("NCMASK_CONFIG", ROOT / "config.yaml")).resolve()
        self.bundle = Path(bundle or os.environ.get("NCMASK_BUNDLE", ROOT / "models" / "onnx")).resolve()
        self.manifest = bundle_manifest(self.bundle)
        if self.manifest is None:
            raise RuntimeError(f"no ONNX bundle at {self.bundle} (run scripts/export_onnx.py)")
        cfg = load_bundle_config(self.config_path, self.bundle)
        cfg["output"]["overlay"] = True
        dev = device or os.environ.get("NCMASK_DEVICE", cfg["device"])
        threads = int(threads if threads is not None else os.environ.get("NCMASK_THREADS", "0"))
        self.providers = providers(dev)
        self.detector = OnnxDetector(cfg, dev, threads=threads)
        self.consensus_available = cfg["consensus"]["detection_mode"] != "single"
        self.consensus_note = cfg["consensus"].get("auto_off")
        self.detector_b = OnnxDetector(cfg, dev, "detector_b", threads=threads) if self.consensus_available else None
        self.seg = OnnxSegModel(cfg, dev, threads=threads)
        # two configs, identical except the detection mode; consensus off = detector A only (B never runs)
        self.cfg_on = cfg
        self.cfg_off = copy.deepcopy(cfg)
        self.cfg_off["consensus"]["detection_mode"] = "single"
        self.mismatch = self._check_bundle(cfg)
        self.health = self._health(cfg)
        self.typical_ms = None

    def _check_bundle(self, cfg):
        """The bundle must come from the weights config.yaml names (path; SHA256 too where the .pt is present)."""
        out = []
        for sec, src in cfg["_source_weights"].items():
            m = self.manifest["models"].get(sec)
            if m is None:
                if sec != "detector_b":
                    out.append(f"{sec}: not in the bundle")
                continue
            if Path(src).as_posix() != m["source"]:
                out.append(f"{sec}: config.yaml names {src}, the bundle was exported from {m['source']}")
            pt = self.config_path.parent / src
            if pt.is_file() and sha256(pt) != m["source_sha256"]:
                out.append(f"{sec}: {src} changed since export (re-run scripts/export_onnx.py)")
        for msg in out:
            log.warning("bundle check: %s", msg)
        return out

    def _health(self, cfg):
        import onnxruntime
        models = {}
        for sec, m in self.manifest["models"].items():
            p = self.bundle / m["onnx"]
            models[sec] = {"onnx": m["onnx"], "onnx_sha256": sha256(p), "onnx_sha256_manifest": m["onnx_sha256"],
                           "source": m["source"], "source_sha256": m["source_sha256"],
                           **{k: m[k] for k in ("imgsz", "variant", "in_ch", "size") if k in m}}
        models["detector"]["role"] = "YOLO26s cell detector (the only source of candidate cells)"
        if "detector_b" in models:
            models["detector_b"]["role"] = "YOLO26l @1280 consensus check (tags consensus / s_only, never filters)"
        models["seg"]["role"] = "TinyUNet cell + nucleus segmenter"
        return {"models": models, "consensus_available": self.consensus_available,
                "consensus_note": self.consensus_note, "bundle_check": self.mismatch,
                "providers": self.providers, "exported_utc": self.manifest.get("created_utc"),
                "export_versions": self.manifest.get("versions"),
                "versions": {"python": platform.python_version(), "onnxruntime": onnxruntime.__version__,
                             "numpy": np.__version__, "opencv": cv2.__version__},
                "thresholds": {"detector_conf": cfg["detector"]["conf"], "match_iou": cfg["consensus"]["match_iou"],
                               "seg_thr": [cfg["seg"]["thr_nucleus"], cfg["seg"]["thr_cell"]]}}

    # ------------------------------------------------------------------ decoding
    def decode(self, path):
        """File on disk -> (BGR uint8, image info). Raises ImageLoadError with a user-facing message."""
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        try:
            with Image.open(path) as im:
                w, h = im.size
        except Exception:
            w = h = None   # PIL can't parse every TIFF flavour; cv2 decides below
        if w and h and w * h > MAX_PIXELS:
            raise ImageLoadError(f"image too large: {w}x{h} (limit {MAX_PIXELS / 1e6:.0f} MP; this demo is for single FOVs)")
        img, info = load_bgr8(path, self.cfg_on["input"]["sixteen_bit"])
        if img.shape[0] * img.shape[1] > MAX_PIXELS:
            raise ImageLoadError(f"image too large: {img.shape[1]}x{img.shape[0]}")
        return img, info

    # ------------------------------------------------------------------ inference
    def infer(self, img, consensus=True, name="upload"):
        use_b = bool(consensus) and self.consensus_available
        cfg = self.cfg_on if use_b else self.cfg_off
        det, detb, seg = Timed(self.detector), Timed(self.detector_b) if use_b else None, TimedSeg(self.seg)
        t0 = time.perf_counter()
        rows, coco, cell_lab, nuc_lab, overlay, info = process_fov(img, "fov", name, det, seg, cfg, detb)
        total = time.perf_counter() - t0
        timings = {"detect": det.t, "consensus": detb.t if detb else 0.0, "segment": seg.t}
        # the rest: QC, A<->B matching (sub-ms), crops, post-processing, overlaps, gate, polygons, overlay
        timings["gate"] = max(0.0, total - sum(timings.values()))
        timings["total"] = total
        return {"rows": rows, "coco": coco, "coco_full": info.pop("coco_full"), "overlay": overlay, "info": info,
                "consensus_used": use_b, "timings_ms": {k: round(1000 * v, 1) for k, v in timings.items()},
                "cell_lab": cell_lab}

    def warmup(self, img):
        """One run of each mode on a sample: loads kernels and measures typical stage times for the UI's progress."""
        out = {}
        for mode in (True, False):
            r = self.infer(img, mode)
            out["consensus" if mode else "no_consensus"] = r["timings_ms"]
        self.typical_ms = out
        return out


# ---------------------------------------------------------------------- result shaping
def _polys(anns, cat, eps):
    """cell_id -> list of simplified polygons [[x, y], ...] for COCO annotations of category cat."""
    out = {}
    for a in anns:
        if a["category_id"] != cat or not a["segmentation"]:
            continue
        ps = []
        for seg in a["segmentation"]:
            pts = np.asarray(seg, np.float32).reshape(-1, 1, 2)
            if eps > 0 and len(pts) > 8:
                pts = cv2.approxPolyDP(pts, eps, True)
            if len(pts) >= 3:
                ps.append([[round(float(x), 1), round(float(y), 1)] for x, y in pts.reshape(-1, 2)])
        out.setdefault(a["attributes"]["cell_id"], []).extend(ps)
    return out


def _num(v):
    if v == "" or v is None:
        return None
    return float(v)


def to_json(res, img_shape, image_info, rid, expires_utc, consensus_requested, engine):
    H, W = img_shape[:2]
    rows, info = res["rows"], res["info"]
    cells_p = _polys(res["coco"]["annotations"], 2, DISPLAY_EPS_PX)
    nuc_p = _polys(res["coco"]["annotations"], 1, DISPLAY_EPS_PX)
    cells = []
    for r in rows:
        cid = r["cell_id"]
        cells.append({
            "id": cid, "box": [r["box_x1"], r["box_y1"], r["box_x2"], r["box_y2"]], "conf": r["conf"],
            "det_tag": r["det_tag"], "b_conf": _num(r["b_conf"]), "ab_iou": _num(r["ab_iou"]),
            "status": r["status"], "reasons": [x for x in r["reasons"].split(";") if x],
            "flags": [x for x in r["flags"].split(";") if x],
            "cell_area_px": r["cell_area_px"], "nucleus_area_px": r["nucleus_area_px"],
            "nc_ratio": _num(r["nc_ratio"]), "occluded_frac": r["occluded_frac"],
            "cell_polygons": cells_p.get(cid, []), "nucleus_polygons": nuc_p.get(cid, [])})
    reasons = Counter(x for c in cells for x in c["reasons"])
    warnings = []
    if (W, H) != EXPECTED_WH:
        warnings.append(f"Image is {W}x{H}; the models were built for {EXPECTED_WH[0]}x{EXPECTED_WH[1]} FOVs "
                        "(~0.155 µm/px). Results may be unreliable.")
    if image_info.get("grey"):
        warnings.append("Greyscale image: the models are colour-trained, expect fewer / poorer cells.")
    if consensus_requested and not res["consensus_used"]:
        warnings.append("Consensus check unavailable on this deployment (YOLO26l not loaded); det_tag is 'single'.")
    qc = info["fov_qc"]
    return {
        "id": rid, "expires_utc": expires_utc,
        "size": {"width": W, "height": H, "expected": list(EXPECTED_WH), "is_expected": (W, H) == EXPECTED_WH},
        "image_qc": {"ok": not qc, "flags": qc, "metrics": {k: round(v, 5) for k, v in info["fov_qc_metrics"].items()}},
        "consensus": {"requested": bool(consensus_requested), "used": res["consensus_used"],
                      "l_only": len(info["l_only"]), "l_only_boxes": [
                          [d["box_x1"], d["box_y1"], d["box_x2"], d["box_y2"]] for d in info["l_only"]]},
        "cells": cells,
        "counts": {"total": len(cells), "status": info["status_counts"],
                   "reasons": dict(reasons.most_common()),
                   "det_tag": dict(Counter(c["det_tag"] for c in cells)),
                   "dropped": dict(Counter(d["reason"] for d in info["dropped"]))},
        "timings_ms": res["timings_ms"],
        "warnings": warnings,
        "models": {k: {"source": v["source"], "onnx_sha256": v["onnx_sha256"][:12]}
                   for k, v in engine.health["models"].items()},
        "downloads": {k: f"/api/infer/{rid}/{k}" for k in ("overlay.png", "cells.csv", "coco.json", "image.jpg")},
    }


def cells_csv(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CELL_COLS)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode()


def encode_png(img):
    ok, b = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("PNG encode failed")
    return b.tobytes()


def encode_jpg(img, q=90):
    ok, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return b.tobytes()


def coco_bytes(coco):
    return json.dumps(coco).encode()
