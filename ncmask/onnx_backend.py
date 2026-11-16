"""ONNX Runtime backends for the three models: drop-in replacements for ncmask.detect.Detector and
ncmask.segment.SegModel. torch-free at inference (the web app ships without torch / ultralytics).

  OnnxDetector(Detector)  only `_predict` is new: the Ultralytics predict path for an end-to-end YOLO26 head, re-done
                          in numpy: rect LetterBox (auto=True, stride from the export metadata, pad 114, INTER_LINEAR),
                          BGR->RGB, /255 -> ONNX (1, 300, 6) -> conf > thr, [:max_det] -> scale_boxes + clip.
                          Tiling, NMS and ordering are Detector.__call__ itself, unchanged.
  OnnxSegModel(SegModel)  only model loading and the forward pass are new (the ONNX graph contains the sigmoid);
                          crop, input variants (Macenko, H channel, box channel) and batching are SegModel's own code.

The ONNX files are made by scripts/export_onnx.py; tests/test_onnx_parity.py checks them against PyTorch.
"""
import contextlib
import json
import logging
import sys
import types
from pathlib import Path

import cv2
import numpy as np

from .detect import Detector
from .segment import SegModel

log = logging.getLogger("ncmask.onnx")


def providers(device="auto"):
    """ONNX Runtime execution providers: CUDA first if available (device auto|cuda), else CPU."""
    import onnxruntime as ort
    av = ort.get_available_providers()
    if device in ("auto", "cuda") and "CUDAExecutionProvider" in av:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "cuda":
        raise RuntimeError("device cuda requested but onnxruntime has no CUDAExecutionProvider "
                           f"(available: {av}); install onnxruntime-gpu")
    return ["CPUExecutionProvider"]


def session(path, device="auto", threads=0):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = int(threads)
    return ort.InferenceSession(str(path), so, providers=providers(device))


@contextlib.contextmanager
def torch_shim():
    """Lets the vendored training code (tinyseg.py imports torch at module level; aug_v2 -> data -> tinyseg) be
    imported where torch is not installed. Only module-level names are provided (nn.Module as a base class,
    no_grad as a decorator); nothing in them is called by the ONNX path (to_input, pipeline_crop are numpy/cv2).
    The stub is removed from sys.modules on exit, so no other library ever sees a fake torch."""
    try:
        import torch  # noqa: F401
        yield False
        return
    except ImportError:
        pass

    def missing(*a, **k):
        raise RuntimeError("torch is not installed: this is the ONNX Runtime build")

    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")
    fn = types.ModuleType("torch.nn.functional")
    nn.Module = type("Module", (), {"__init__": missing})
    nn.functional = fn
    torch.nn = nn
    torch.no_grad = lambda: (lambda f: f)
    torch.load = missing
    names = {"torch": torch, "torch.nn": nn, "torch.nn.functional": fn}
    sys.modules.update(names)
    try:
        yield True
    finally:
        for n in names:
            if sys.modules.get(n) is names[n]:
                del sys.modules[n]


def onnx_metadata(sess):
    """Ultralytics writes its export metadata (imgsz, stride, names, end2end, ...) as ONNX metadata_props."""
    m = dict(sess.get_modelmeta().custom_metadata_map)
    out = {}
    for k, v in m.items():
        try:
            import ast
            out[k] = ast.literal_eval(v)
        except Exception:
            out[k] = v
    return out


def letterbox_params(shape, new_shape, stride, auto=True):
    """ultralytics.data.augment.LetterBox.get_params (center=True, scaleup=True, scale_fill=False)."""
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = max(round(shape[1] * r), 1), max(round(shape[0] * r), 1)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    return new_unpad, round(dh - 0.1), round(dh + 0.1), round(dw - 0.1), round(dw + 0.1)


def letterbox(img, new_shape, stride, auto=True):
    new_unpad, top, bottom, left, right = letterbox_params(img.shape[:2], new_shape, stride, auto)
    if img.shape[1::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def scale_boxes(img1_shape, boxes, img0_shape):
    """ultralytics.utils.ops.scale_boxes (ratio_pad=None, padding=True, xyxy) + clip_boxes, numpy."""
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    new_h, new_w = (max(round(x * gain), 1) for x in img0_shape[:2])
    gain_y, gain_x = new_h / img0_shape[0], new_w / img0_shape[1]
    pad_x, pad_y = round((img1_shape[1] - new_w) / 2 - 0.1), round((img1_shape[0] - new_h) / 2 - 0.1)
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes[:, [0, 2]] /= gain_x
    boxes[:, [1, 3]] /= gain_y
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img0_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img0_shape[0])
    return boxes


class OnnxDetector(Detector):
    def __init__(self, cfg, device, section="detector", threads=0):  # noqa: super().__init__ loads ultralytics
        d = dict(cfg[section])
        d.setdefault("tiling", {"mode": "off"})
        self.cfg = d
        self.sess = session(d["weights"], device, threads)
        self.meta = onnx_metadata(self.sess)
        # Ultralytics 8.4 runs these YOLO26 checkpoints in PyTorch with head.end2end False (one-to-many head + NMS),
        # so the export must be the same head (scripts/export_onnx.py: nms left at None). An end-to-end export is
        # a different decoder: refuse it rather than silently diverge from the PyTorch pipeline.
        if self.meta.get("end2end", False):
            raise ValueError(f"{d['weights']}: end-to-end (one-to-one) export; re-export with scripts/export_onnx.py")
        exp = self.meta.get("imgsz")
        self.imgsz = int(d["imgsz"]) or int(max(exp) if isinstance(exp, (list, tuple)) else exp)
        self.stride = int(self.meta.get("stride", 32))
        self.names = dict(self.meta.get("names", {0: "cell"}))
        inp = self.sess.get_inputs()[0]
        # dynamic H/W export = symbolic (string / None) dims; then Ultralytics' rect letterbox applies
        self.dynamic = not all(isinstance(v, int) for v in inp.shape[2:])
        self.input_name = inp.name
        self.device = self.sess.get_providers()[0]

    def _predict(self, imgs):
        # Ultralytics: auto (rect) letterbox when all images in the batch share a shape and the model is dynamic
        auto = self.dynamic and len({x.shape for x in imgs}) == 1
        out = []
        for img in imgs:   # batch 1 per image (exact: no cross-image padding)
            lb = letterbox(img, self.imgsz, self.stride, auto)
            x = (lb[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
            pred = self.sess.run(None, {self.input_name: np.ascontiguousarray(x)})[0][0]   # (4 + nc, anchors)
            boxes, conf = self._nms(pred)
            boxes = scale_boxes(lb.shape[:2], boxes, img.shape[:2])
            out.append((boxes.reshape(-1, 4), conf))
        return out

    def _nms(self, pred):
        """ultralytics.utils.nms.non_max_suppression for one image, agnostic, best class only: score = max class
        prob > conf, xywh -> xyxy, at most max_nms (30000) by score, greedy NMS (torchvision convention: suppress
        IoU > iou_thr; geometry.nms), first max_det kept."""
        from . import geometry
        p = pred.T.astype(np.float64)                       # (anchors, 4 + nc)
        score = p[:, 4:].max(1)
        p, score = p[score > self.cfg["conf"]], score[score > self.cfg["conf"]]
        if len(p) > 30000:
            o = np.argsort(-score, kind="stable")[:30000]
            p, score = p[o], score[o]
        xy, wh = p[:, :2], p[:, 2:4] / 2
        boxes = np.concatenate([xy - wh, xy + wh], 1)
        keep = geometry.nms(boxes, score, self.cfg["nms_iou"])[:int(self.cfg["max_det"])] if len(p) else np.zeros(0, int)
        return boxes[keep], score[keep]


class OnnxSegModel(SegModel):
    """seg.weights points at the .onnx file; config.json and stain_ref.json / h_stats.json sit next to it
    (scripts/export_onnx.py copies them from the PyTorch model's folder)."""

    def __init__(self, cfg, device, threads=0):
        self._threads = threads
        with torch_shim():
            super().__init__(cfg, device)

    def _load_model(self, path, device):
        if not str(path).endswith(".onnx"):
            raise ValueError(f"OnnxSegModel needs an .onnx file, got {path}")
        s = session(path, device, self._threads)
        self.input_name = s.get_inputs()[0].name
        shp = s.get_inputs()[0].shape
        if shp[1] != self.in_ch or shp[2] != self.size:
            raise ValueError(f"{path}: input {shp} does not match config.json (in_ch {self.in_ch}, size {self.size})")
        self.device = s.get_providers()[0]
        return s

    def _forward(self, x):
        return self.model.run(None, {self.input_name: x})[0]


# Layout of an ONNX bundle (scripts/export_onnx.py writes it, the web app and the Space read it)
BUNDLE_FILES = {"detector": "detector_a.onnx", "detector_b": "detector_b.onnx", "seg": "seg/seg.onnx"}


def bundle_manifest(bundle_dir):
    p = Path(bundle_dir) / "manifest.json"
    return json.loads(p.read_text()) if p.is_file() else None


def load_bundle_config(config_path, bundle_dir):
    """config.yaml with every model path pointed at the ONNX bundle (all other settings unchanged), validated by
    ncmask.config.load. A missing detector_b.onnx switches consensus off exactly like a missing YOLO26l .pt.
    cfg['_source_weights'] = the .pt paths config.yaml names (relative to its folder), to check against the
    bundle's manifest."""
    import os
    import tempfile

    import yaml

    from . import config as cfgmod
    config_path, bundle = Path(config_path).resolve(), Path(bundle_dir).resolve()
    base = config_path.parent
    raw = yaml.safe_load(config_path.read_text())
    src = {sec: str(raw[sec]["weights"]) for sec in BUNDLE_FILES if isinstance(raw.get(sec), dict)}
    for sec, k in cfgmod.PATH_KEYS:
        if isinstance(raw.get(sec), dict) and raw[sec].get(k) is not None:
            p = Path(str(raw[sec][k])).expanduser()
            raw[sec][k] = str(p if p.is_absolute() else (base / p).resolve())
    for sec, f in BUNDLE_FILES.items():
        if isinstance(raw.get(sec), dict):
            raw[sec]["weights"] = str(bundle / f)
    raw["seg"]["model_config"] = None        # bundle/seg/config.json
    raw["seg"]["preprocessing_dir"] = None   # bundle/seg/ (stain_ref.json / h_stats.json)
    fd, tmp = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(raw, f)
        cfg = cfgmod.load(tmp)
    finally:
        os.unlink(tmp)
    cfg["_config_path"] = str(config_path)
    cfg["_source_weights"] = src
    return cfg
