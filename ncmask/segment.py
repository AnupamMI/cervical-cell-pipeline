"""TinyUNet segmentation inside detector boxes, using the training code itself (seg.code_dir).

Preprocessing is the training/eval code's own functions, imported, not re-written:
  * crop:  aug_v2.pipeline_crop (edge_mode reflect), or the same window clamped to the FOV (edge_mode clamp)
  * input: tinyseg.to_input(crop, size)  (INTER_AREA resize, BGR->RGB, /255, (x-0.5)/0.25)
  * model: tinyseg.load(weights, device, in_ch); in_ch/size from config.json next to the weights, else 3 / 160,
           exactly as evaluate_v2.load_model
  * box channel (in_ch 4): evaluate_v2.predict builds it inline (not importable: evaluate_v2 loads a hard-coded
    /home/nvidia path at import), so `_box_channel` repeats those 3 lines; tests/test_parity.py checks the whole
    input tensor against the original evaluate_v2.predict.
"""
import importlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("ncmask.segment")


def import_training_code(code_dir):
    """Import tinyseg / aug_v2 from code_dir. Refuses if differently-located modules of those names are loaded."""
    code_dir = str(Path(code_dir).resolve())
    for name in ("tinyseg", "aug_v2", "data"):
        m = sys.modules.get(name)
        if m is not None and str(Path(getattr(m, "__file__", "") or "").resolve().parent) != code_dir:
            raise ImportError(f"a different module '{name}' is already imported from {getattr(m, '__file__', '?')}")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    tinyseg = importlib.import_module("tinyseg")
    aug_v2 = importlib.import_module("aug_v2")
    return tinyseg, aug_v2


def _box_channel(crop_hw, box, size):
    """Same lines as evaluate_v2.predict: 1 inside round(box) (lower bounds clamped at 0), INTER_NEAREST to size."""
    import cv2
    m = np.zeros(crop_hw, np.uint8)
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    m[max(0, y1):y2, max(0, x1):x2] = 1
    return cv2.resize(m, (size,) * 2, interpolation=cv2.INTER_NEAREST)[..., None]


class SegModel:
    def __init__(self, cfg, device):
        s = cfg["seg"]
        self.tinyseg, self.aug_v2 = import_training_code(s["code_dir"])
        wpath = Path(s["weights"])
        cfg_file = Path(s["model_config"]) if s["model_config"] else wpath.with_name("config.json")
        mc = json.loads(cfg_file.read_text()) if cfg_file.is_file() else {}
        self.model_config_file = str(cfg_file) if cfg_file.is_file() else None
        self.in_ch = int(mc.get("in_ch", 3))
        self.size = int(mc.get("size", self.tinyseg.SIZE))
        if self.in_ch not in (3, 4):
            raise ValueError(f"unsupported in_ch {self.in_ch}")
        self.device = device
        self.model = self.tinyseg.load(str(wpath), device, self.in_ch)
        self.bs = int(s["batch_size"])
        self.margin = float(cfg["crop"]["margin"])
        self.edge_mode = cfg["crop"]["edge_mode"]
        if self.edge_mode == "reflect" and abs(self.margin - float(self.aug_v2.PAD)) > 1e-12:
            raise ValueError(f"crop.edge_mode reflect uses aug_v2.pipeline_crop, whose margin is fixed at "
                             f"aug_v2.PAD={self.aug_v2.PAD}; crop.margin is {self.margin}")
        log.info("seg model %s: in_ch=%d size=%d device=%s", wpath, self.in_ch, self.size, device)

    def make_input(self, crop, box):
        """One crop (BGR uint8) + detector box in crop coords -> float32 HxWxC network input."""
        x = self.tinyseg.to_input(crop, self.size)
        if self.in_ch == 4:
            x = np.concatenate([x, _box_channel(crop.shape[:2], box, self.size)], -1)
        return x

    @torch.no_grad()
    def predict(self, crops, boxes):
        """crops: list of BGR uint8 arrays; boxes: detector box per crop in crop coords -> (N, 2, size, size)
        sigmoid probabilities (channel 0 nucleus, 1 cell)."""
        outs = []
        for b in range(0, len(crops), self.bs):
            xs = [self.make_input(c, bx) for c, bx in zip(crops[b:b + self.bs], boxes[b:b + self.bs])]
            x = torch.from_numpy(np.stack(xs).transpose(0, 3, 1, 2).astype(np.float32).copy())
            outs.append(torch.sigmoid(self.model(x.to(self.device))).cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros((0, 2, self.size, self.size), np.float32)
