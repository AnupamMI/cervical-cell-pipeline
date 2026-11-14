"""Integration tests with the real models (CPU or GPU)."""
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from ncmask import config as cfgmod
from ncmask.runner import RunError, run

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(cfg):
    """The process_fov tests in this file exercise single-detector behaviour (consensus modes: test_consensus.py)."""
    import copy
    c = copy.deepcopy(cfg)
    c["consensus"]["detection_mode"] = "single"
    return c


def real_fov(tmp):
    """A real FOV from the training crops' source if available, else a synthetic one from a training crop."""
    from conftest import TINYSEG_DIR
    crops = sorted((TINYSEG_DIR / "data/images/default").glob("b1_cervical_0001_cell_*.png"))[:4]
    if not crops:
        pytest.skip("no tinyseg crops to build a test FOV")
    bg = np.full((1100, 1500, 3), (236, 226, 238), np.uint8)
    x = 20
    for c in crops:
        im = cv2.imread(str(c))
        h, w = im.shape[:2]
        if x + w > bg.shape[1] or h > bg.shape[0]:
            break
        bg[20:20 + h, x:x + w] = im
        x += w + 20
    p = tmp / "synthetic_fov.png"
    cv2.imwrite(str(p), bg)
    return p


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    d = tmp_path_factory.mktemp("in")
    real_fov(d)
    cv2.imwrite(str(d / "blank.png"), np.full((1520, 2028, 3), 245, np.uint8))     # zero detections
    (d / "corrupt.jpg").write_bytes(b"\xff\xd8\xff\xe0 broken")
    (d / "notes.txt").write_text("not an image")
    return d


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_full_run_zero_detections_corrupt_and_outputs(inputs, tmp_path):
    before = {p.name: digest(p) for p in inputs.iterdir()}
    out = tmp_path / "out"
    m = run(inputs, out, ROOT / "config.yaml")
    assert m["status"] == "finished"
    c = m["counts"]
    assert c["fovs_listed"] == 3 and c["fovs_ok"] == 2 and c["fovs_failed"] == 1 and c["ignored_files"] == 1
    assert m["failures"][0]["fov"] == "corrupt" and m["failures"][0]["stage"] == "load"
    # zero-detection FOV: complete outputs, empty
    z = out / "fovs" / "blank"
    assert (z / "done.json").exists()
    assert cv2.imread(str(z / "cell_labels.png"), cv2.IMREAD_UNCHANGED).max() == 0
    assert json.loads((z / "cells.coco.json").read_text())["annotations"] == []
    assert "zero detections" in (out / "run.log").read_text()
    # real FOV: label maps are 16-bit, nucleus ids are a subset of their cell ids, csv matches coco
    f = out / "fovs" / "synthetic_fov"
    cl = cv2.imread(str(f / "cell_labels.png"), cv2.IMREAD_UNCHANGED)
    nl = cv2.imread(str(f / "nucleus_labels.png"), cv2.IMREAD_UNCHANGED)
    assert cl.dtype == np.uint16 and cl.shape == (1100, 1500) and cl.max() >= 1
    assert (nl[nl > 0] == cl[nl > 0]).all()
    coco = json.loads((f / "cells.coco.json").read_text())
    rows = (f / "cells.csv").read_text().strip().splitlines()[1:]
    assert len([a for a in coco["annotations"] if a["category_id"] == 2]) == len(rows)
    for a in coco["annotations"]:
        if a["category_id"] == 2 and a["area"]:
            assert a["area"] == int((cl == a["attributes"]["cell_id"]).sum())
    man = json.loads((out / "run_manifest.json").read_text())
    assert len(man["models"]["detector"]["sha256"]) == 64 and man["versions"]["torch"]
    # input untouched
    assert {p.name: digest(p) for p in inputs.iterdir()} == before


def test_resume_overwrite_and_config_guard(inputs, tmp_path):
    out = tmp_path / "out"
    run(inputs, out, ROOT / "config.yaml", limit=2)            # blank.png, corrupt.jpg (sorted)
    m = run(inputs, out, ROOT / "config.yaml")
    assert m["counts"]["fovs_skipped_already_done"] == 1        # blank done; corrupt retried (and fails again)
    assert m["counts"]["fovs_ok"] == 1 and m["counts"]["cells_in_cells_csv"] >= 1
    m = run(inputs, out, ROOT / "config.yaml", overwrite=True)
    assert m["counts"]["fovs_skipped_already_done"] == 0 and m["counts"]["fovs_ok"] == 2
    c = yaml.safe_load((ROOT / "config.yaml").read_text())
    c["detector"]["conf"] = 0.3
    c["detector"]["weights"] = str(ROOT / c["detector"]["weights"])
    c["seg"]["weights"] = str(ROOT / c["seg"]["weights"]); c["seg"]["code_dir"] = str(ROOT / c["seg"]["code_dir"])
    c["detector_b"]["weights"] = str(ROOT / c["detector_b"]["weights"])
    alt = tmp_path / "alt.yaml"; alt.write_text(yaml.safe_dump(c))
    with pytest.raises(RunError, match="different config"):
        run(inputs, out, alt)


def test_refuses_unsafe_paths(inputs, tmp_path):
    with pytest.raises(RunError, match="must differ"):
        run(inputs, inputs, ROOT / "config.yaml")
    with pytest.raises(RunError, match="inside the output"):
        run(inputs, inputs.parent, ROOT / "config.yaml")
    with pytest.raises(RunError, match="not found"):
        run(tmp_path / "missing", tmp_path / "o", ROOT / "config.yaml")


def test_deterministic(inputs, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    run(inputs, a, ROOT / "config.yaml")
    run(inputs, b, ROOT / "config.yaml")
    for name in ("cell_labels.png", "nucleus_labels.png", "cells.csv", "cells.coco.json", "overlay.png"):
        assert digest(a / "fovs/synthetic_fov" / name) == digest(b / "fovs/synthetic_fov" / name), name


class StubDetector:
    def __init__(self, boxes, conf):
        self.boxes, self.conf = np.array(boxes, float).reshape(-1, 4), np.array(conf, float)

    def __call__(self, img):
        return self.boxes, self.conf, {"tiled": False, "n_raw": len(self.boxes), "n_after_nms": len(self.boxes)}


class ZeroSeg:
    def __init__(self):
        import aug_v2
        self.aug_v2 = aug_v2

    def predict(self, crops, boxes):
        return np.zeros((len(crops), 2, 224, 224), np.float32)


class BoxSeg(ZeroSeg):
    """Cell probability 1 everywhere (-> cell = box + clip tolerance), nucleus = central 20% of the crop."""
    def predict(self, crops, boxes):
        p = np.zeros((len(crops), 2, 224, 224), np.float32)
        p[:, 1] = 1.0; p[:, 0, 90:134, 90:134] = 1.0
        return p


def flags(rows):
    return {r["cell_id"]: r["flags"].split(";") for r in rows}


@pytest.mark.parametrize("mode", ["reflect", "clamp"])
def test_tiny_edge_and_empty_seg_boxes(cfg, mode):
    import copy
    from ncmask.fov import process_fov
    from ncmask.segment import SegModel
    c = copy.deepcopy(cfg); c["crop"]["edge_mode"] = mode
    img = np.full((400, 600, 3), 230, np.uint8)
    cv2.circle(img, (300, 200), 60, (180, 120, 200), -1); cv2.circle(img, (300, 200), 15, (90, 30, 120), -1)
    det = StubDetector([[230, 130, 370, 270], [0, 0, 80, 70], [100, 300, 108, 309]], [0.9, 0.8, 0.7])
    rows, coco, cl, nl, ov, info = process_fov(img, "t", "t.png", det, ZeroSeg(), c)
    by = {r["cell_id"]: r for r in rows}
    assert "box_too_small" in by[3]["flags"] and by[3]["cell_area_px"] == 0 and by[3]["crop_x0"] == ""
    assert "crop_clamped" in by[2]["flags"] and "cell_at_edge" in by[2]["flags"]
    assert by[2]["crop_x0"] == 0 and by[2]["crop_y0"] == 0          # recorded window is the in-FOV part
    assert all("empty_mask" in by[i]["flags"] for i in (1, 2)) and cl.max() == 0
    seg = SegModel(c, "cpu")
    rows, *_ = process_fov(img, "t", "t.png", StubDetector([], []), seg, c)
    assert rows == []


@pytest.mark.parametrize("mode", ["reflect", "clamp"])
def test_crop_clamped_vs_cell_at_edge(cfg, mode):
    """crop_clamped = padded window passes the FOV edge (info); cell_at_edge = the box itself within 2 px of it."""
    import copy
    from ncmask.fov import process_fov
    c = copy.deepcopy(cfg); c["crop"]["edge_mode"] = mode; c["detector"]["containment_thr"] = None
    W, H = 800, 600
    img = np.full((H, W, 3), 230, np.uint8)
    boxes = [[300, 200, 400, 300],      # 1 interior: no flag
             [10, 200, 110, 300],       # 2 margin (17 px) passes the left edge, box doesn't touch it: crop_clamped only
             [2, 400, 102, 500],        # 3 box 2 px from the left edge: both
             [690, 0, 790, 100],        # 4 box on the top edge: both
             [500, 498, 600, 598],      # 5 bottom exactly 2 px from the edge: both
             [500, 50, 600, 150],       # 6 interior
             [150, 497.5, 250, 597.5]]  # 7 bottom 2.5 px from the edge: crop_clamped only
    det = StubDetector(boxes, [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6])
    rows, coco, cl, nl, ov, info = process_fov(img, "t", "t.png", det, BoxSeg(), c)
    f = flags(rows)
    has = lambda i, x: x in f[i]
    assert not has(1, "crop_clamped") and not has(1, "cell_at_edge")
    assert has(2, "crop_clamped") and not has(2, "cell_at_edge")
    for i in (3, 4, 5):
        assert has(i, "crop_clamped") and has(i, "cell_at_edge"), (i, f[i])
    assert not has(6, "crop_clamped") and not has(6, "cell_at_edge")
    assert has(7, "crop_clamped") and not has(7, "cell_at_edge")
    assert "truncated" not in ";".join(r["flags"] for r in rows)
    # the edge cells are segmented and pasted (reflect mode keeps only the in-FOV part)
    for i in (2, 3, 4, 5):
        assert (cl == i).sum() > 0
    assert cl.shape == (H, W)


def test_containment_and_overlap_drop_in_process_fov(cfg, caplog):
    import copy, logging
    from ncmask.fov import process_fov
    c = copy.deepcopy(cfg)
    # the bare rules: containment without the area-ratio guard, 16 px clip tolerance (geometry below assumes both)
    c["detector"]["containment_min_area_ratio"] = 0.0
    c["postprocess"]["clip_tolerance_px"] = 16
    img = np.full((600, 800, 3), 230, np.uint8)
    boxes = [[100, 100, 300, 300],    # 1 conf .9
             [150, 150, 330, 330],    # .8: 69% inside box 1 -> survives containment; its cell then loses ~74% -> dropped
             [120, 120, 200, 200],    # .7: fully inside box 1 -> dropped by containment (before seg)
             [500, 100, 700, 300],    # .6: far away, kept
             [690, 250, 790, 350]]    # .5: 10% inside box 4 -> kept; overlaps cell 4 a little
    det = StubDetector(boxes, [0.9, 0.8, 0.7, 0.6, 0.5])
    with caplog.at_level(logging.INFO, logger="ncmask.fov"):
        rows, coco, cl, nl, ov, info = process_fov(img, "t", "t.png", det, BoxSeg(), c)
    reasons = [(d["reason"], d["cell_id"], d["conf"]) for d in info["dropped"]]
    assert ("contained", "", 0.7) in reasons
    assert ("overlap_lost", 2, 0.8) in reasons and len(reasons) == 2
    ids = sorted(r["cell_id"] for r in rows)
    assert ids == [1, 3, 4]                       # ids assigned after containment; dropped cell 2 removed
    assert not (cl == 2).any() and set(np.unique(cl)) == {0, 1, 3, 4}
    assert len([a for a in coco["annotations"] if a["category_id"] == 2]) == 3
    assert "overlap_lost" in flags(rows)[4] and "overlap_won" in flags(rows)[3]
    log = caplog.text
    assert "dropped contained box conf 0.700" in log and "dropped cell 2 conf 0.800" in log
    # both rules can be switched off
    c["detector"]["containment_thr"] = None; c["postprocess"]["drop_lost_frac"] = None
    rows, *_ , info = process_fov(img, "t", "t.png", det, BoxSeg(), c)
    assert len(rows) == 5 and info["dropped"] == []


def test_containment_default_keeps_small_cells_on_big_ones(cfg):
    """Default containment_min_area_ratio 0.4: a small box inside a big one is kept, a near-duplicate is dropped."""
    from ncmask.fov import process_fov
    assert cfg["detector"]["containment_min_area_ratio"] == 0.4 and cfg["detector"]["containment_thr"] == 0.8
    img = np.full((600, 800, 3), 230, np.uint8)
    det = StubDetector([[100, 100, 400, 400],      # big, conf .9
                        [150, 150, 230, 230],      # small, fully inside, 7% of the big box's area -> kept
                        [110, 110, 390, 390]],     # near-duplicate, fully inside, 87% of its area -> dropped
                       [0.9, 0.8, 0.7])
    rows, coco, cl, nl, ov, info = process_fov(img, "t", "t.png", det, ZeroSeg(), cfg)
    contained = [d for d in info["dropped"] if d["reason"] == "contained"]
    assert [d["conf"] for d in contained] == [0.7]
    assert sorted(r["conf"] for r in rows) == [0.8, 0.9]


def test_clip_tolerance_margin_does_not_clip(cfg):
    """clip_tolerance_px 'margin' = the crop window: a cell filling the crop is not clipped. 16 px clips it."""
    import copy
    from ncmask.fov import process_fov
    assert cfg["postprocess"]["clip_tolerance_px"] == "margin"
    img = np.full((600, 800, 3), 230, np.uint8)
    det = StubDetector([[300, 200, 500, 400], [0, 0, 150, 120]], [0.9, 0.8])   # interior + edge (mirror-padded)
    for mode in ("reflect", "clamp"):
        c = copy.deepcopy(cfg); c["crop"]["edge_mode"] = mode
        rows, *_ = process_fov(img, "t", "t.png", det, BoxSeg(), c)
        for r in rows:
            assert "box_clipped" not in r["flags"] and r["clipped_frac"] == 0, (mode, r)
        assert rows[0]["cell_area_px"] == (rows[0]["crop_x1"] - rows[0]["crop_x0"]) * (rows[0]["crop_y1"] - rows[0]["crop_y0"])
        c["postprocess"]["clip_tolerance_px"] = 16
        rows, *_ = process_fov(img, "t", "t.png", det, BoxSeg(), c)
        r = rows[0]
        assert "box_clipped" in r["flags"]
        crop_area = (r["crop_x1"] - r["crop_x0"]) * (r["crop_y1"] - r["crop_y0"])
        assert r["cell_area_px"] == 232 * 232 and abs(r["clipped_frac"] - (1 - 232 * 232 / crop_area)) < 1e-4
        # the edge cell: clip measured over its in-FOV part only
        e = rows[1]
        assert "box_clipped" in e["flags"] and 0 < e["clipped_frac"] < 1


def test_grayscale_warning_and_dropped_csv(tmp_path, caplog):
    import logging
    d = tmp_path / "in"; d.mkdir()
    src = real_fov(d)
    g = cv2.cvtColor(cv2.imread(str(src)), cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(d / "grey.png"), g)
    cv2.imwrite(str(d / "grey_as_rgb.png"), cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    m = run(d, tmp_path / "out", ROOT / "config.yaml")
    assert sorted(m["grayscale_fovs"]) == ["grey", "grey_as_rgb"]
    text = (tmp_path / "out" / "run.log").read_text()
    assert text.count("grayscale image") == 2 and "colour-trained" in text
    assert (tmp_path / "out" / "dropped.csv").read_text().startswith("fov,reason,cell_id")


def test_vendored_code_is_verbatim(tinyseg_dir):
    for name in ("tinyseg.py", "aug_v2.py", "data.py"):
        assert digest(ROOT / "vendor/tinyseg_retrain" / name) == digest(tinyseg_dir / name), name
    for v in ("v2_1", "v2"):
        for f in ("best.pt", "config.json"):
            assert digest(ROOT / "models/seg" / v / f) == digest(tinyseg_dir / "runs" / f"finetune_{v}" / f)


def test_config_validation(tmp_path):
    c = yaml.safe_load((ROOT / "config.yaml").read_text())
    c["crop"]["edge_mode"] = "wrap"
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(c))
    with pytest.raises(cfgmod.ConfigError):
        cfgmod.load(p)
    del c["seg"]
    p.write_text(yaml.safe_dump(c))
    with pytest.raises(cfgmod.ConfigError, match="seg"):
        cfgmod.load(p)
