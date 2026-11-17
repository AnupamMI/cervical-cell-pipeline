"""Inference engine for the web app: the ONNX bundle + the unchanged ncmask pipeline (ncmask.fov.process_fov).

One FOV per call. Stage timings come from wrapping the model callables:
  detect     YOLO26s (detector A)
  consensus  YOLO26l @1280 + A<->B matching (tag only, never filters)
  clusters   detector C (YOLO26s @1024, cluster / debris boxes; information only, never changes a status)
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

from ncmask.fov import C_OBJECTS, CELL_COLS, process_fov
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


class TimedDetC:
    """Detector C proxy (process_fov calls .detect for the class ids); times detect(), passes names through."""

    def __init__(self, det):
        self._det, self.t = det, 0.0

    def detect(self, img):
        t = time.perf_counter()
        try:
            return self._det.detect(img)
        finally:
            self.t += time.perf_counter() - t

    def __getattr__(self, k):
        return getattr(self._det, k)


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
        dcc = cfg.get("detector_c") or {}
        self.clusters_available = bool(dcc.get("enabled"))
        self.clusters_note = dcc.get("auto_off")
        self.detector_c = OnnxDetector(cfg, dev, "detector_c", threads=threads) if self.clusters_available else None
        self.seg = OnnxSegModel(cfg, dev, threads=threads)
        # two configs, identical except the detection mode; consensus off = detector A only (B never runs)
        # configs per (consensus, vote); consensus off = detector A only (B never runs)
        self.cfg_on = cfg
        self._cfgs = {}
        for cons in (True, False):
            for vote in (True, False):
                c = copy.deepcopy(cfg)
                if not cons:
                    c["consensus"]["detection_mode"] = "single"
                c["voting"]["enabled"] = vote
                self._cfgs[cons, vote] = c
        self.mismatch = self._check_bundle(cfg)
        self.health = self._health(cfg)
        self.typical_ms = None

    def _check_bundle(self, cfg):
        """The bundle must come from the weights config.yaml names (path; SHA256 too where the .pt is present)."""
        out = []
        for sec, src in cfg["_source_weights"].items():
            m = self.manifest["models"].get(sec)
            if m is None:
                if sec not in ("detector_b", "detector_c"):
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
        if "detector_c" in models:
            dcc = cfg["detector_c"]
            models["detector_c"].update({
                "role": "detector C: YOLO26s @1024 cluster / debris boxes (experimental, information only: "
                        "near_cluster / near_debris tags, never changes a cell's status)",
                "names": self.manifest["models"]["detector_c"].get("names"), "classes_used": list(dcc["classes"]),
                "conf": dcc["conf"], "near_frac": dcc["near_frac"],
                "test_map50": {"cell": 0.829, "cluster": 0.503, "debris": 0.539}})
        return {"models": models, "consensus_available": self.consensus_available,
                "consensus_note": self.consensus_note, "clusters_available": self.clusters_available,
                "clusters_note": self.clusters_note, "bundle_check": self.mismatch,
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
    def infer(self, img, consensus=True, name="upload", clusters=True, vote=True):
        use_b = bool(consensus) and self.consensus_available
        use_c = bool(clusters) and self.clusters_available
        cfg = self._cfgs[use_b, bool(vote)]
        det, detb, seg = Timed(self.detector), Timed(self.detector_b) if use_b else None, TimedSeg(self.seg)
        detc = TimedDetC(self.detector_c) if use_c else None
        t0 = time.perf_counter()
        rows, coco, cell_lab, nuc_lab, overlay, info = process_fov(img, "fov", name, det, seg, cfg, detb, detc)
        total = time.perf_counter() - t0
        timings = {"detect": det.t, "consensus": detb.t if detb else 0.0, "clusters": detc.t if detc else 0.0,
                   "segment": seg.t}
        # the rest: QC, A<->B matching (sub-ms), crops, post-processing, overlaps, gate, polygons, overlay
        timings["gate"] = max(0.0, total - sum(timings.values()))
        timings["total"] = total
        return {"rows": rows, "coco": coco, "coco_full": info.pop("coco_full"), "overlay": overlay, "info": info,
                "consensus_used": use_b, "clusters_used": use_c, "vote_used": bool(vote), "timings_ms": {k: round(1000 * v, 1) for k, v in timings.items()},
                "cell_lab": cell_lab}

    def warmup(self, img):
        """One run of each mode on a sample: loads kernels and measures typical stage times for the UI's progress."""
        out = {}
        for mode in (True, False):
            r = self.infer(img, mode, clusters=True)
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
            "votes": int(r["votes"].split("/")[0]), "voters": r["voters"],
            "near_cluster": [x for x in r["near_cluster"].split(";") if x],
            "near_debris": [x for x in r["near_debris"].split(";") if x],
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
    if res.get("clusters_requested") and not res["clusters_used"]:
        warnings.append("Detector C (clusters & debris) unavailable on this deployment.")
    cb = info.get("c_boxes", [])
    qc = info["fov_qc"]
    return {
        "id": rid, "expires_utc": expires_utc,
        "size": {"width": W, "height": H, "expected": list(EXPECTED_WH), "is_expected": (W, H) == EXPECTED_WH},
        "image_qc": {"ok": not qc, "flags": qc, "metrics": {k: round(v, 5) for k, v in info["fov_qc_metrics"].items()}},
        "consensus": {"requested": bool(consensus_requested), "used": res["consensus_used"],
                      "l_only": len(info["l_only"]), "l_only_boxes": [
                          [d["box_x1"], d["box_y1"], d["box_x2"], d["box_y2"]] for d in info["l_only"]]},
        "clusters": {"requested": bool(res.get("clusters_requested")), "used": res["clusters_used"],
                     "info_only": True, "conf": engine.health["models"].get("detector_c", {}).get("conf"),
                     "near_frac": engine.health["models"].get("detector_c", {}).get("near_frac"),
                     "boxes": [{"id": c["cell_id"], "class": c["object"], "conf": c["conf"],
                                "box": [c["box_x1"], c["box_y1"], c["box_x2"], c["box_y2"]]} for c in cb]},
        "voting": {"used": res.get("vote_used", False), "detectors_run": info["voting"]["detectors_run"],
                   "votes_needed": info["voting"]["votes_needed"], "skip_in_cluster_debris":
                       bool(engine.cfg_on["voting"].get("skip_in_cluster_debris", True)),
                   "bc_only": sum(c["det_tag"] == "bc_only" for c in cells)},
        "cells": cells,
        "counts": {"total": len(cells), "status": info["status_counts"],
                   "votes": dict(Counter(f"{c['votes']}/{info['voting']['detectors_run']}" for c in cells)),
                   "clusters": {o: sum(c["object"] == o for c in cb) for o in C_OBJECTS},
                   "near": {"cluster": sum(bool(c["near_cluster"]) for c in cells),
                            "debris": sum(bool(c["near_debris"]) for c in cells)},
                   "reasons": dict(reasons.most_common()),
                   "det_tag": dict(Counter(c["det_tag"] for c in cells)),
                   "dropped": dict(Counter(d["reason"] for d in info["dropped"]))},
        "timings_ms": res["timings_ms"],
        "warnings": warnings,
        "models": {k: {"source": v["source"], "onnx_sha256": v["onnx_sha256"][:12]}
                   for k, v in engine.health["models"].items()},
        "downloads": {k: f"/api/infer/{rid}/{k}" for k in ("overlay.png", "cells.csv", "coco.json", "image.jpg")},
    }


def cells_csv(rows, c_boxes=()):
    """Cell rows (object = cell), then detector-C rows (object = cluster | debris, cell_id C1.., no status)."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CELL_COLS)
    w.writeheader()
    w.writerows(rows)
    w.writerows(c_boxes)
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
