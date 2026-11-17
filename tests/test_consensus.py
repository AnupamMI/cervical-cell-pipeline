"""A<->B cross-check: Hungarian matching and the three detection modes."""
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ncmask.consensus import match_boxes
from test_runner import BoxSeg, StubDetector, real_fov

ROOT = Path(__file__).resolve().parents[1]


def test_match_one_to_one_and_threshold():
    a = [[0, 0, 10, 10], [100, 100, 110, 110], [200, 200, 210, 210]]
    b = [[0, 0, 10, 10], [0, 0, 10, 10.5], [105, 100, 115, 110], [300, 300, 310, 310]]
    pairs, am, bm = match_boxes(a, b, 0.5)
    # a0 can take only one of the two identical B boxes; a1-b2 IoU = 50/150 = 0.33 < 0.5 -> unmatched
    assert [(i, j) for i, j, _ in pairs] == [(0, 0)]
    assert am.tolist() == [True, False, False] and bm.tolist() == [True, False, False, False]
    # threshold is inclusive: IoU exactly 0.5
    p, am, _ = match_boxes([[0, 0, 10, 10]], [[0, 0, 10, 5]], 0.5)
    assert len(p) == 1 and abs(p[0][2] - 0.5) < 1e-12
    p, *_ = match_boxes([[0, 0, 10, 10]], [[0, 0, 10, 5]], 0.51)
    assert p == []
    for x, y in (([], b), (a, []), ([], [])):
        p, am, bm = match_boxes(x, y, 0.5)
        assert p == [] and len(am) == len(x) and len(bm) == len(y)


def test_hungarian_beats_greedy():
    """Greedy (best IoU first) would pair a0-b0 (0.905) and leave a1 unmatched (a1-b1 0.29); Hungarian finds
    a0-b1 (0.6) + a1-b0 (0.6) = two matches."""
    a = np.array([[0, 0, 100, 100], [30, 0, 130, 100]], float)
    b = np.array([[5, 0, 105, 100], [-25, 0, 75, 100]], float)
    from ncmask.geometry import iou_matrix
    iou = iou_matrix(a, b)
    assert iou[0, 0] > iou[0, 1] and iou[0, 0] > iou[1, 0] and iou[1, 1] < 0.5
    assert iou[0, 1] >= 0.5 and iou[1, 0] >= 0.5
    pairs, am, bm = match_boxes(a, b, 0.5)
    assert sorted((i, j) for i, j, _ in pairs) == [(0, 1), (1, 0)] and am.all() and bm.all()


def make_case():
    img = np.full((600, 800, 3), 230, np.uint8)
    A = StubDetector([[100, 100, 250, 250],     # 1 matched by B (shifted 5 px)
                      [400, 100, 550, 250],     # 2 A only
                      [100, 350, 250, 500]],    # 3 matched by B
                     [0.9, 0.8, 0.7])
    B = StubDetector([[105, 100, 255, 250], [100, 360, 250, 510],
                      [600, 400, 700, 500]],    # B only -> l_only
                     [0.85, 0.6, 0.5])
    return img, A, B


@pytest.mark.parametrize("mode", ["single", "consensus_flag", "consensus_filter"])
def test_detection_modes(cfg, mode):
    from ncmask.fov import process_fov
    c = copy.deepcopy(cfg); c["consensus"]["detection_mode"] = mode
    img, A, B = make_case()
    rows, coco, cl, nl, ov, info = process_fov(img, "t", "t.png", A, BoxSeg(), c, None if mode == "single" else B)
    tags = {r["conf"]: r["det_tag"] for r in rows}
    if mode == "single":
        assert tags == {0.9: "single", 0.8: "single", 0.7: "single"} and info["l_only"] == []
        assert all(r["b_conf"] == "" for r in rows)
    elif mode == "consensus_flag":
        assert tags == {0.9: "consensus", 0.8: "s_only", 0.7: "consensus"}       # every A box segmented
        assert all(r["cell_area_px"] > 0 for r in rows)
        by = {r["conf"]: r for r in rows}
        assert by[0.9]["b_conf"] == 0.85 and by[0.7]["b_conf"] == 0.6 and by[0.8]["b_conf"] == ""
        # A's geometry is kept for matched cells
        assert (by[0.9]["box_x1"], by[0.9]["box_x2"]) == (100, 250)
        assert [(l["b_conf"], l["box_x1"]) for l in info["l_only"]] == [(0.5, 600)]
        assert info["detector"]["b"]["consensus"] == 2 and info["detector"]["b"]["s_only"] == 1
        coco_tags = sorted(a["attributes"]["det_tag"] for a in coco["annotations"] if a["category_id"] == 2)
        assert coco_tags == ["consensus", "consensus", "s_only"]
    else:
        assert tags == {0.9: "consensus", 0.7: "consensus"}                      # s_only not segmented
        assert [(d["reason"], d["conf"]) for d in info["dropped"]] == [("s_only_filtered", 0.8)]
        assert not cl[100:250, 420:540].any()
        assert len(info["l_only"]) == 1
        assert sorted(r["cell_id"] for r in rows) == [1, 2]                      # ids assigned after filtering


def test_consensus_auto_off_without_weights(tmp_path):
    """Missing YOLO26l weights -> consensus switches itself off (single), it does not fail."""
    import yaml
    from ncmask import config as cfgmod
    c = yaml.safe_load((ROOT / "config.yaml").read_text())
    for sec, k in (("detector", "weights"), ("seg", "weights"), ("seg", "code_dir"), ("seg", "preprocess_code")):
        c[sec][k] = str(ROOT / c[sec][k])
    c["detector_b"]["weights"] = str(tmp_path / "missing.pt")
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(c))
    cfg = cfgmod.load(p)
    assert cfg["consensus"]["detection_mode"] == "single" and "not found" in cfg["consensus"]["auto_off"]


def test_consensus_mode_requires_detector_b(cfg):
    from ncmask.fov import process_fov
    c = copy.deepcopy(cfg); c["consensus"]["detection_mode"] = "consensus_flag"
    img, A, _ = make_case()
    with pytest.raises(ValueError, match="needs detector B"):
        process_fov(img, "t", "t.png", A, BoxSeg(), c, None)


def test_dashed_outline_for_s_only(cfg):
    from ncmask.fov import draw_dashed
    canvas = np.zeros((200, 200, 3), np.uint8)
    m = np.zeros((200, 200), np.uint8); cv2.rectangle(m, (40, 40), (160, 160), 1, -1)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    draw_dashed(canvas, cs[0][:, 0, :], (0, 255, 0), 2)
    solid = np.zeros_like(canvas); cv2.drawContours(solid, cs, -1, (0, 255, 0), 2, cv2.LINE_AA)
    d, s = (canvas[..., 1] > 0), (solid[..., 1] > 0)
    assert 0.4 * s.sum() < d.sum() < 0.8 * s.sum()          # ~14/24 of the outline is drawn
    assert not (d & ~cv2.dilate(s.astype(np.uint8), np.ones((3, 3))).astype(bool)).any()   # only on the outline
    top = canvas[40, 45:155, 1] > 0                          # along the top edge: alternating dashes and gaps
    runs = np.flatnonzero(np.diff(top.astype(int)))
    assert len(runs) >= 6


def test_run_consensus_flag_writes_tags_and_l_only(tmp_path, cfg):
    from ncmask.runner import run
    if cfg["consensus"].get("auto_off"):
        pytest.skip("YOLO26l weights not present (fetch with scripts/fetch_release_weights.py)")
    d = tmp_path / "in"; d.mkdir(); real_fov(d)
    m = run(d, tmp_path / "out", ROOT / "config.yaml")
    assert m["config"]["consensus"]["detection_mode"] == "consensus_flag"
    assert len(m["models"]["detector_b"]["sha256"]) == 64 and m["models"]["detector_b"]["imgsz"] == 1280
    import pandas as pd
    cells = pd.read_csv(tmp_path / "out" / "cells.csv")
    cells = cells[cells["object"] == "cell"]     # detector-C rows (cluster / debris) are listed too
    assert set(cells["det_tag"]) <= {"consensus", "s_only"} and len(cells) > 0
    assert sum(m["counts"]["det_tags_this_run"].values()) == len(cells)
    lo = (tmp_path / "out" / "l_only.csv").read_text().splitlines()
    assert lo[0] == "fov,b_conf,box_x1,box_y1,box_x2,box_y2,max_iou_with_A"
    assert len(lo) - 1 == m["counts"]["l_only_in_l_only_csv"]
    assert (tmp_path / "out" / "fovs" / "synthetic_fov" / "l_only.csv").exists()
