"""ONNX bundle (models/onnx) vs PyTorch on samples/: boxes, masks, gate statuses (scripts/check_onnx_parity.py).
Skipped without torch / ultralytics / onnxruntime or without an exported bundle. NCMASK_PARITY_LIMIT=N limits FOVs."""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "models" / "onnx"
need = [m for m in ("torch", "ultralytics", "onnxruntime") if importlib.util.find_spec(m) is None]
pytestmark = [pytest.mark.skipif(bool(need), reason=f"needs {need}"),
              pytest.mark.skipif(not (BUNDLE / "manifest.json").is_file(), reason="no ONNX bundle (scripts/export_onnx.py)")]


def test_bundle_matches_config():
    """The bundle was exported from the weights config.yaml names now (same .pt SHA256)."""
    import hashlib
    import json
    from ncmask import config as cfgmod
    cfg = cfgmod.load(ROOT / "config.yaml")
    man = json.loads((BUNDLE / "manifest.json").read_text())
    secs = ("detector", "seg") + (("detector_b",) if cfg["consensus"]["detection_mode"] != "single" else ()) + \
        (("detector_c",) if (cfg.get("detector_c") or {}).get("enabled") else ())
    for sec in secs:
        h = hashlib.sha256(Path(cfg[sec]["weights"]).read_bytes()).hexdigest()
        assert man["models"][sec]["source_sha256"] == h, f"{sec}: bundle is stale; run scripts/export_onnx.py"


def test_onnx_parity_on_samples():
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_onnx_parity as cp
    rep = cp.run(ROOT / "samples", BUNDLE, ROOT / "config.yaml", int(os.environ.get("NCMASK_PARITY_LIMIT", "0")))
    fails = {r["fov"]: r["fail"] for r in rep["fovs"] if r["fail"]}
    assert rep["passed"], fails
