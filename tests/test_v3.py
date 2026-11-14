"""Pipeline v3: Stage 0 QC, Stage 5 gate, two-layer masks, input variants and fail-loud preprocessing."""
import copy
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from ncmask.quality import gate, image_qc, nucleus_touches_cell_boundary

ROOT = Path(__file__).resolve().parents[1]
PHASE2 = ROOT.parent / "tinyseg_v3/phase2"
FOV = ROOT.parent / "nc_mask_pipeline/data/images/cervical_0001.tiff"


def G(cfg, **kw):
    a = dict(flags=[], cell_area=60000, nuc_area=1500, nc=0.025, occluded_frac=0.0, det_tag="consensus", conf=0.8,
             nuc_touch=False, size=cfg["size"], g=cfg["gate"])
    a.update(kw)
    return gate(**a)


def test_gate_statuses_and_reasons(cfg):
    assert G(cfg) == ("measurable", [])
    for f in ("cell_at_edge", "no_nucleus", "empty_mask", "box_too_small"):
        st, rs = G(cfg, flags=[f]); assert st == "excluded" and f in rs
    assert G(cfg, cell_area=5000)[0] == "excluded" and "implausible_mask" in G(cfg, cell_area=5000)[1]
    assert "implausible_mask" in G(cfg, cell_area=400000)[1]
    assert "implausible_mask" in G(cfg, nuc_area=20000)[1]
    assert "implausible_mask" in G(cfg, nc=1.2)[1] and G(cfg, nc=0.99)[0] == "measurable"
    assert G(cfg, flags=["multi_nucleus"]) == ("review", ["multi_nucleus"])
    assert G(cfg, occluded_frac=0.31) == ("review", ["occluded"]) and G(cfg, occluded_frac=0.3)[0] == "measurable"
    assert G(cfg, det_tag="s_only", conf=0.2) == ("review", ["s_only_low_conf"])
    assert G(cfg, det_tag="s_only", conf=0.3)[0] == "measurable" and G(cfg, det_tag="consensus", conf=0.2)[0] == "measurable"
    assert G(cfg, nuc_touch=True) == ("review", ["nucleus_touches_cell_boundary"])
    assert G(cfg, cell_area=10000) == ("review", ["likely_inflammatory"])
    assert G(cfg, cell_area=200000) == ("review", ["likely_cluster"])
    st, rs = G(cfg, flags=["cell_at_edge", "multi_nucleus"], occluded_frac=0.5)       # excluded wins, all reasons kept
    assert st == "excluded" and rs == ["cell_at_edge", "multi_nucleus", "occluded"]


def test_nucleus_touches_cell_boundary():
    cell = np.zeros((50, 50), bool); cell[10:40, 10:40] = True
    inner = np.zeros_like(cell); inner[20:30, 20:30] = True
    edge = np.zeros_like(cell); edge[10:15, 20:25] = True
    assert not nucleus_touches_cell_boundary(cell, inner)
    assert nucleus_touches_cell_boundary(cell, edge)
    assert not nucleus_touches_cell_boundary(cell, np.zeros_like(cell))


def test_image_qc(cfg):
    if not FOV.exists():
        pytest.skip("sample FOV not available")
    img = cv2.imread(str(FOV))
    flags, m = image_qc(img, cfg["qc"])
    assert flags == [], (flags, m)
    assert "blurry" in image_qc(cv2.GaussianBlur(img, (0, 0), 6), cfg["qc"])[0]
    assert "too_dark" in image_qc((img * 0.6).astype(np.uint8), cfg["qc"])[0]
    blank = np.full_like(img, 235)
    assert "low_stain" in image_qc(blank, cfg["qc"])[0]


def _seg_cfg(cfg, tmp_path, weights, variant_json, prep_files):
    """A model folder in tmp with the given weights, a config.json and the given preprocessing files."""
    d = tmp_path / "model"; d.mkdir()
    shutil.copy(weights, d / "best.pt")
    (d / "config.json").write_text(json.dumps(variant_json))
    for f in prep_files:
        shutil.copy(PHASE2 / f, d / f)
    c = copy.deepcopy(cfg); c["seg"]["weights"] = str(d / "best.pt"); c["seg"]["model_config"] = None
    c["seg"]["variant"] = "auto"; c["seg"]["preprocessing_dir"] = None
    return c


def test_preprocessing_fail_loud_and_variant_mismatch(cfg, tmp_path):
    from ncmask.segment import SegModel
    w = ROOT / "models/seg/v2_1/best.pt"
    c = _seg_cfg(cfg, tmp_path, w, {"in_ch": 3, "size": 224, "variant": "V1"}, [])
    with pytest.raises(FileNotFoundError, match="stain_ref.json"):
        SegModel(c, "cpu")
    c["seg"]["variant"] = "V2"
    with pytest.raises(ValueError, match="variant"):
        SegModel(c, "cpu")


@pytest.mark.parametrize("variant", ["V1", "V2", "V3"])
def test_variant_inputs_equal_training_functions(cfg, tmp_path, variant):
    if not (PHASE2 / "preprocess.py").exists():
        pytest.skip("tinyseg_v3 training code not available")
    import importlib.util
    from ncmask.segment import SegModel
    spec = importlib.util.spec_from_file_location("pp_train", PHASE2 / "preprocess.py")
    pp = importlib.util.module_from_spec(spec); spec.loader.exec_module(pp)
    in_ch = 4 if variant == "V3" else 3
    w = ROOT / ("models/seg/v2/best.pt" if in_ch == 4 else "models/seg/v2_1/best.pt")
    files = ["h_stats.json"] if variant == "V3" else ["stain_ref.json"]
    seg = SegModel(_seg_cfg(cfg, tmp_path, w, {"in_ch": in_ch, "size": 224, "variant": variant}, files), "cpu")
    assert seg.variant == variant
    rng = np.random.default_rng(0)
    crop = np.full((300, 280, 3), 225, np.uint8)
    cv2.circle(crop, (140, 150), 90, (200, 150, 210), -1); cv2.circle(crop, (140, 150), 20, (120, 40, 110), -1)
    crop = np.clip(crop + rng.normal(0, 3, crop.shape), 0, 255).astype(np.uint8)
    x = seg.make_input(crop, (50, 60, 230, 240))
    ref = json.loads((PHASE2 / "stain_ref.json").read_text())
    t, _ = pp.transform_source(variant, crop, ref)
    want = seg.tinyseg.to_input(t, 224)
    if variant == "V3":
        h = json.loads((PHASE2 / "h_stats.json").read_text())
        hs = pp.h_channel(cv2.resize(t, (224, 224), interpolation=cv2.INTER_AREA))
        want = np.concatenate([want, ((hs - h["mean"]) / h["std"])[..., None]], -1)
    assert x.shape == (224, 224, in_ch) and np.array_equal(x, want)
    if variant in ("V1", "V2"):                                    # the training fallback rule is applied and counted
        n0 = seg.fallbacks
        blank = np.full((200, 200, 3), 238, np.uint8)
        xb = seg.make_input(blank, (20, 20, 180, 180))
        assert seg.fallbacks == n0 + 1
        assert np.array_equal(xb, seg.tinyseg.to_input(pp.transform_source(variant, blank, ref)[0], 224))
