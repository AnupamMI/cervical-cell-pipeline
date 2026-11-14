"""YOLO26 cell detector: full-frame or tiled inference, then class-agnostic NMS.

Images are passed to Ultralytics as BGR uint8 numpy arrays (Ultralytics' convention for numpy input; it converts
to RGB internally, as for its own cv2-loaded training images).
"""
import logging

import numpy as np

from . import geometry

log = logging.getLogger("ncmask.detect")


def checkpoint_info(path):
    """Metadata of an Ultralytics checkpoint (unpickles it: only use on trusted model files)."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ck.get("ema") or ck.get("model")
    ta = ck.get("train_args") or {}
    return {
        "ultralytics_version": ck.get("version"), "date": ck.get("date"),
        "names": dict(m.names) if m is not None else None,
        "end2end": bool(getattr(m, "yaml", {}).get("end2end", False)) if m is not None else None,
        "train_args": {k: ta.get(k) for k in ("model", "data", "imgsz", "epochs", "batch", "seed", "iou", "max_det",
                                              "rect", "fliplr", "flipud", "mosaic", "hsv_h", "hsv_s", "hsv_v", "scale")},
        "train_metrics": ck.get("train_metrics"),
    }


class Detector:
    def __init__(self, cfg, device, section="detector"):
        from ultralytics import YOLO
        d = dict(cfg[section])
        d.setdefault("tiling", {"mode": "off"})   # detector_b has no tiling settings: always full-frame
        self.cfg = d
        self.model = YOLO(d["weights"], task="detect")
        ta = (getattr(self.model, "ckpt", None) or {}).get("train_args", {}) or {}
        self.imgsz = int(d["imgsz"]) or int(ta.get("imgsz", 640))
        if ta.get("imgsz") and int(ta["imgsz"]) != self.imgsz:
            log.warning("detector.imgsz=%s differs from the training imgsz %s", self.imgsz, ta["imgsz"])
        self.names = dict(self.model.names)
        self.device = 0 if device == "cuda" else "cpu"

    def _predict(self, imgs):
        res = self.model.predict(imgs, imgsz=self.imgsz, conf=self.cfg["conf"], iou=self.cfg["nms_iou"],
                                 max_det=int(self.cfg["max_det"]), device=self.device,
                                 agnostic_nms=True, verbose=False)
        out = []
        for r in res:
            b = r.boxes
            out.append((b.xyxy.cpu().numpy().astype(np.float64).reshape(-1, 4), b.conf.cpu().numpy().astype(np.float64)))
        return out

    def use_tiling(self, W, H):
        t = self.cfg["tiling"]
        return t["mode"] == "on" or (t["mode"] == "auto" and max(W, H) > t["trigger_factor"] * self.imgsz)

    @property
    def weights(self):
        return self.cfg["weights"]

    def __call__(self, img):
        """img BGR uint8 -> (boxes (N,4) float FOV px xyxy, conf (N,), info dict). Sorted by conf desc, then y, x."""
        H, W = img.shape[:2]
        tiled = self.use_tiling(W, H)
        if not tiled:
            boxes, conf = self._predict([img])[0]
            n_raw = len(boxes)
        else:
            t = self.cfg["tiling"]
            tl = geometry.tiles(W, H, int(t["tile_px"]), int(t["overlap_px"]))
            res = self._predict([np.ascontiguousarray(img[y0:y1, x0:x1]) for x0, y0, x1, y1 in tl])
            bs, cs = [], []
            e = t["edge_px"]
            for (x0, y0, x1, y1), (b, c) in zip(tl, res):
                if len(b) == 0:
                    continue
                # a box touching an inner tile edge is a cut cell: the overlapping tile sees it whole
                inner = ((b[:, 0] <= e) & (x0 > 0)) | ((b[:, 1] <= e) & (y0 > 0)) | \
                        ((b[:, 2] >= (x1 - x0) - e) & (x1 < W)) | ((b[:, 3] >= (y1 - y0) - e) & (y1 < H))
                b, c = b[~inner] + (x0, y0, x0, y0), c[~inner]
                bs.append(b); cs.append(c)
            boxes = np.concatenate(bs) if bs else np.zeros((0, 4))
            conf = np.concatenate(cs) if cs else np.zeros(0)
            n_raw = len(boxes)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
        keep = geometry.nms(boxes, conf, self.cfg["nms_iou"]) if len(boxes) else np.zeros(0, int)
        boxes, conf = boxes[keep], conf[keep]
        order = np.lexsort((boxes[:, 0], boxes[:, 1], -conf)) if len(boxes) else np.zeros(0, int)
        return boxes[order], conf[order], {"tiled": tiled, "n_raw": int(n_raw), "n_after_nms": int(len(boxes))}
