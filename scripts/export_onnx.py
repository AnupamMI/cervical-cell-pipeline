#!/usr/bin/env python3
"""Export the models named in config.yaml to an ONNX bundle (default models/onnx/, gitignored):

    models/onnx/detector_a.onnx    YOLO26s       (detector.weights)    dynamic H/W, the head PyTorch runs for these
                                                                       checkpoints (end2end False: raw (1, 4 + nc,
                                                                       anchors), NMS in ncmask.onnx_backend)
    models/onnx/detector_b.onnx    YOLO26l @1280 (detector_b.weights)  same; skipped if the .pt is absent
    models/onnx/detector_c.onnx    YOLO26s @1024 (detector_c.weights)  same (3 classes); skipped if detector C is off
    models/onnx/seg/seg.onnx       TinyUNet      (seg.weights)         dynamic batch, sigmoid included
    models/onnx/seg/config.json    in_ch / size / variant (explicit, also for legacy models without one)
    models/onnx/seg/<stain_ref.json | h_stats.json>   preprocessing files the variant needs
    models/onnx/manifest.json      source .pt + ONNX SHA256s, sizes, settings, tool versions

    python scripts/export_onnx.py [--config config.yaml] [--out models/onnx] [--only all|seg|detectors]

Needs torch + ultralytics + onnx (the training environment); the app itself only needs onnxruntime.
Then: python -m pytest tests/test_onnx_parity.py
"""
import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from ncmask import config as cfgmod  # noqa: E402
from ncmask.onnx_backend import BUNDLE_FILES  # noqa: E402

OPSET = 17


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def rel(p):
    p = Path(p).resolve()
    try:
        return p.relative_to(HERE).as_posix()
    except ValueError:
        return str(p)


def export_yolo(weights, imgsz, dst):
    from ultralytics import YOLO
    m = YOLO(weights, task="detect")
    # nms left at None: Ultralytics then exports the head PyTorch inference uses for these checkpoints
    # (head.end2end False -> raw (1, 4 + nc, anchors), NMS in ncmask.onnx_backend). nms=False would force the
    # one-to-one end-to-end head, a different decoder with different boxes.
    f = m.export(format="onnx", imgsz=int(imgsz), dynamic=True, simplify=True, opset=OPSET,
                 batch=1, device="cpu", verbose=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(f), str(dst))
    return {"names": dict(m.names), "imgsz": int(imgsz)}


def export_seg(cfg, out_dir):
    import torch
    from ncmask.segment import SegModel
    seg = SegModel(cfg, "cpu")

    class WithSigmoid(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return torch.sigmoid(self.m(x))

    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.iterdir():           # never mix preprocessing files of two segmenters
        f.unlink()
    x = torch.zeros(2, seg.in_ch, seg.size, seg.size)
    dst = out_dir / "seg.onnx"
    torch.onnx.export(WithSigmoid(seg.model).eval(), (x,), str(dst), input_names=["x"], output_names=["prob"],
                      dynamic_axes={"x": {0: "n"}, "prob": {0: "n"}}, opset_version=OPSET, dynamo=False)
    (out_dir / "config.json").write_text(json.dumps({"in_ch": seg.in_ch, "size": seg.size, "variant": seg.variant}))
    pre = None
    if seg.preprocessing_file:
        shutil.copy2(seg.preprocessing_file, out_dir / Path(seg.preprocessing_file).name)
        pre = {"file": Path(seg.preprocessing_file).name, "source": rel(seg.preprocessing_file),
               "sha256": sha256(seg.preprocessing_file)}
    return {"in_ch": seg.in_ch, "size": seg.size, "variant": seg.variant, "preprocessing": pre,
            "model_config_source": rel(seg.model_config_file) if seg.model_config_file else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--out", default=str(HERE / "models" / "onnx"))
    ap.add_argument("--only", choices=("all", "seg", "detectors"), default="all")
    a = ap.parse_args(argv)
    import onnx
    import onnxruntime
    import torch
    cfg = cfgmod.load(a.config)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads((out / "manifest.json").read_text()) if (out / "manifest.json").is_file() else {}
    man.setdefault("models", {})
    if a.only in ("all", "detectors"):
        for sec in ("detector", "detector_b", "detector_c"):
            off = (cfg["consensus"].get("auto_off") if sec == "detector_b" else
                   None if sec == "detector" or (cfg.get("detector_c") or {}).get("enabled") else
                   (cfg.get("detector_c") or {}).get("auto_off", "detector_c.enabled is false or no detector_c section"))
            if off:
                print(f"skip {sec}: {off}")
                man["models"].pop(sec, None)
                (out / BUNDLE_FILES[sec]).unlink(missing_ok=True)
                continue
            w = cfg[sec]["weights"]
            print(f"export {sec}: {rel(w)} (imgsz {cfg[sec]['imgsz']}) ...")
            info = export_yolo(w, cfg[sec]["imgsz"], out / BUNDLE_FILES[sec])
            import onnxruntime as _ort
            from ncmask.onnx_backend import onnx_metadata
            meta = onnx_metadata(_ort.InferenceSession(str(out / BUNDLE_FILES[sec]), providers=["CPUExecutionProvider"]))
            if meta.get("end2end", False):   # the end-to-end head trap: a different decoder than PyTorch runs
                raise SystemExit(f"{sec}: exported an end-to-end head; PyTorch runs end2end False for this checkpoint")
            info["end2end"] = False
            man["models"][sec] = {"onnx": BUNDLE_FILES[sec], "source": rel(w), "source_sha256": sha256(w), **info}
    if a.only in ("all", "seg"):
        w = cfg["seg"]["weights"]
        print(f"export seg: {rel(w)} ...")
        info = export_seg(cfg, out / Path(BUNDLE_FILES["seg"]).parent)
        man["models"]["seg"] = {"onnx": BUNDLE_FILES["seg"], "source": rel(w), "source_sha256": sha256(w), **info}
    for sec, m in man["models"].items():
        p = out / m["onnx"]
        onnx.checker.check_model(str(p))
        m["onnx_sha256"], m["onnx_bytes"] = sha256(p), p.stat().st_size
        onnxruntime.InferenceSession(str(p), providers=["CPUExecutionProvider"])   # loads
    man.update({"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "opset": OPSET,
                "config": rel(a.config), "versions": {"torch": torch.__version__, "onnx": onnx.__version__,
                                                      "onnxruntime": onnxruntime.__version__}})
    try:
        import ultralytics
        man["versions"]["ultralytics"] = ultralytics.__version__
    except ImportError:
        pass
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    for sec, m in man["models"].items():
        print(f"  {sec:10s} {m['onnx']:22s} {m['onnx_bytes'] / 1e6:7.1f} MB  sha256 {m['onnx_sha256'][:12]}  <- {m['source']}")
    print(f"bundle: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
