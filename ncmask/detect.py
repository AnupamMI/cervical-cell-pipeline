"""YOLO26 detectors: full-frame or tiled inference, then NMS.

Detector A / B (cells): class-agnostic NMS. Detector C (cluster / debris, information only): per-class NMS and only
the classes named in detector_c.classes (its `cell` class is dropped before NMS, as Ultralytics' `classes=` does).

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
        self._setup_classes()

    def _setup_classes(self):
        """detector_c: class_conf {name: conf} (else classes) -> ids of this checkpoint; per-class NMS, predicted at
        the lowest of the thresholds, then each class cut at its own. A / B: all classes, agnostic."""
        cc = self.cfg.get("class_conf")
        want = list(cc) if cc else self.cfg.get("classes")
        self.agnostic = want is None
        self.class_conf = None
        if want is None:
            self.class_ids = None
            return
        inv = {v: int(k) for k, v in self.names.items()}
        missing = [c for c in want if c not in inv]
        if missing:
            raise ValueError(f"{self.cfg['weights']}: classes {missing} not in the checkpoint (has {self.names})")
        self.class_ids = sorted(inv[c] for c in want)
        if cc:
            self.class_conf = {inv[k]: float(v) for k, v in cc.items()}
            self.cfg["conf"] = min(self.class_conf.values())

    def _predict(self, imgs):
        """-> list of (boxes (N,4) xyxy, conf (N,), cls (N,) int) per image."""
        res = self.model.predict(imgs, imgsz=self.imgsz, conf=self.cfg["conf"], iou=self.cfg["nms_iou"],
                                 max_det=int(self.cfg["max_det"]), device=self.device,
                                 agnostic_nms=self.agnostic, classes=self.class_ids, verbose=False)
        out = []
        for r in res:
            b = r.boxes
            out.append((b.xyxy.cpu().numpy().astype(np.float64).reshape(-1, 4), b.conf.cpu().numpy().astype(np.float64),
                        b.cls.cpu().numpy().astype(np.int64).reshape(-1)))
        return out

    def _nms(self, boxes, conf, cls):
        """Indices kept, highest conf first: agnostic (A / B) or within each class (C)."""
        if not len(boxes):
            return np.zeros(0, int)
        if self.agnostic:
            return geometry.nms(boxes, conf, self.cfg["nms_iou"])
        keep = np.concatenate([np.where(cls == c)[0][geometry.nms(boxes[cls == c], conf[cls == c], self.cfg["nms_iou"])]
                               for c in np.unique(cls)])
        return keep[np.argsort(-conf[keep], kind="stable")]

    def use_tiling(self, W, H):
        t = self.cfg["tiling"]
        return t["mode"] == "on" or (t["mode"] == "auto" and max(W, H) > t["trigger_factor"] * self.imgsz)

    @property
    def weights(self):
        return self.cfg["weights"]

    def __call__(self, img):
        """img BGR uint8 -> (boxes (N,4) float FOV px xyxy, conf (N,), info dict). Sorted by conf desc, then y, x."""
        boxes, conf, _, info = self.detect(img)
        return boxes, conf, info

    def detect(self, img):
        """As __call__, plus the class id of every box: (boxes, conf, cls, info)."""
        H, W = img.shape[:2]
        tiled = self.use_tiling(W, H)
        if not tiled:
            boxes, conf, cls = self._predict([img])[0]
            n_raw = len(boxes)
        else:
            t = self.cfg["tiling"]
            tl = geometry.tiles(W, H, int(t["tile_px"]), int(t["overlap_px"]))
            res = self._predict([np.ascontiguousarray(img[y0:y1, x0:x1]) for x0, y0, x1, y1 in tl])
            bs, cs, ks = [], [], []
            e = t["edge_px"]
            for (x0, y0, x1, y1), (b, c, k) in zip(tl, res):
                if len(b) == 0:
                    continue
                # a box touching an inner tile edge is a cut cell: the overlapping tile sees it whole
                inner = ((b[:, 0] <= e) & (x0 > 0)) | ((b[:, 1] <= e) & (y0 > 0)) | \
                        ((b[:, 2] >= (x1 - x0) - e) & (x1 < W)) | ((b[:, 3] >= (y1 - y0) - e) & (y1 < H))
                b, c, k = b[~inner] + (x0, y0, x0, y0), c[~inner], k[~inner]
                bs.append(b); cs.append(c); ks.append(k)
            boxes = np.concatenate(bs) if bs else np.zeros((0, 4))
            conf = np.concatenate(cs) if cs else np.zeros(0)
            cls = np.concatenate(ks) if ks else np.zeros(0, np.int64)
            n_raw = len(boxes)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
        keep = self._nms(boxes, conf, cls)
        boxes, conf, cls = boxes[keep], conf[keep], cls[keep]
        if self.class_conf:   # per-class thresholds (same result as before NMS: NMS here is within each class)
            ok = np.array([c > self.class_conf[int(k)] for c, k in zip(conf, cls)], bool) if len(conf) else np.zeros(0, bool)
            boxes, conf, cls = boxes[ok], conf[ok], cls[ok]
        order = np.lexsort((boxes[:, 0], boxes[:, 1], -conf)) if len(boxes) else np.zeros(0, int)
        return boxes[order], conf[order], cls[order], {"tiled": tiled, "n_raw": int(n_raw), "n_after_nms": int(len(boxes))}
