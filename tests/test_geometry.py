import numpy as np
import pytest

from ncmask import geometry


def test_crop_window_interior_matches_aug_v2_pipeline_crop():
    import aug_v2
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (900, 1200, 3), np.uint8)
    for _ in range(200):
        x1, y1 = rng.uniform(250, 600), rng.uniform(250, 400)
        box = (x1, y1, x1 + rng.uniform(20, 300), y1 + rng.uniform(20, 300))
        win = geometry.crop_window(box, 1200, 900, aug_v2.PAD)
        ref, _, rbox = aug_v2.pipeline_crop(img, [], box)
        assert not win.truncated
        assert np.array_equal(img[win.y0:win.y1, win.x0:win.x1], ref)
        assert np.allclose(win.box, rbox)


def test_crop_window_clamps_and_flags_edges():
    win = geometry.crop_window((0, 10, 100, 60), 500, 400, 0.17)   # m = 17
    assert (win.x0, win.y0, win.x1, win.y1) == (0, 0, 117, 77)
    assert (win.ux0, win.uy0) == (-17, -7)
    assert win.truncated and set(win.edge_sides) == {"l", "t"}
    assert win.box == (0.0, 10.0, 100.0, 60.0)
    win = geometry.crop_window((450, 350, 500, 400), 500, 400, 0.17)
    assert (win.x1, win.y1) == (500, 400) and set(win.edge_sides) == {"r", "b"}
    win = geometry.crop_window((100, 100, 200, 150), 500, 400, 0.17)
    assert not win.truncated and win.edge_sides == ()
    with pytest.raises(ValueError):
        geometry.crop_window((600, 600, 700, 700), 500, 400, 0.17)


def test_box_crop_fov_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(100):
        W, H = 2028, 1520
        x1, y1 = rng.uniform(-20, W - 10), rng.uniform(-20, H - 10)
        box = (max(0, x1), max(0, y1), min(W, x1 + rng.uniform(5, 500)), min(H, y1 + rng.uniform(5, 500)))
        win = geometry.crop_window(box, W, H, 0.17)
        corners = np.array([[box[0], box[1]], [box[2], box[3]]])
        assert np.allclose(geometry.fov_to_crop(corners, win), np.array(win.box).reshape(2, 2))
        assert np.allclose(geometry.crop_to_fov(geometry.fov_to_crop(corners, win), win), corners)
        # a mask pixel at crop (r, c) lands at FOV (r + y0, c + x0)
        m = np.zeros((win.h, win.w), bool)
        r, c = rng.integers(win.h), rng.integers(win.w)
        m[r, c] = True
        full = geometry.paste((H, W), m, win)
        assert full.sum() == 1 and full[r + win.y0, c + win.x0]


def test_clip_rect():
    assert geometry.clip_rect((10.4, 20.6, 50.2, 60.0), 100, 100, 5) == (5, 15, 56, 65)
    assert geometry.clip_rect((2, 2, 98, 98), 100, 100, 16) == (0, 0, 100, 100)


def test_tiles_cover_image():
    for W, H in ((2028, 1520), (640, 480), (1025, 1024)):
        t = geometry.tiles(W, H, 1024, 512)
        cov = np.zeros((H, W), bool)
        for x0, y0, x1, y1 in t:
            assert x1 - x0 <= 1024 and y1 - y0 <= 1024
            cov[y0:y1, x0:x1] = True
        assert cov.all()


def test_containment_suppression():
    b = np.array([[0, 0, 100, 100],        # 0 big, conf .9
                  [10, 10, 50, 50],        # 1 fully inside 0, conf .8 -> dropped
                  [80, 80, 120, 120],      # 2 25% inside 0 -> kept
                  [85, 0, 115, 30],        # 3 50% inside 0 -> kept
                  [12, 12, 48, 48]], float)  # 4 inside 0 (and inside dropped 1) -> dropped
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.95 * 0 + 0.5])
    keep, frac = geometry.containment_suppress(b, s, 0.8)
    assert keep.tolist() == [True, False, True, True, False]
    assert frac[1] == 1.0 and abs(frac[2] - 0.25) < 1e-9 and abs(frac[3] - 0.5) < 1e-9
    # a contained box with HIGHER confidence is kept (only higher-conf boxes suppress)
    keep, _ = geometry.containment_suppress(b[:2], np.array([0.5, 0.9]), 0.8)
    assert keep.tolist() == [True, True]
    # exactly at the threshold -> dropped (>=)
    keep, _ = geometry.containment_suppress(np.array([[0, 0, 10, 10], [0, 0, 10, 12.5]]), np.array([0.9, 0.8]), 0.8)
    assert keep.tolist() == [True, False]
    # min_area_ratio: a small box (16% of the big one's area) on a big one is kept
    keep, _ = geometry.containment_suppress(b[:2], s[:2], 0.8, min_area_ratio=0.4)
    assert keep.tolist() == [True, True]
    assert geometry.containment_suppress(np.zeros((0, 4)), np.zeros(0), 0.8)[0].tolist() == []


def test_nms_deterministic_and_correct():
    b = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60], [0, 0, 10, 10]], float)
    s = np.array([0.9, 0.8, 0.7, 0.9])
    keep = geometry.nms(b, s, 0.5)
    assert keep.tolist() == [0, 2]          # tie 0.9 vs 0.9 -> lower index wins
    assert geometry.nms(np.zeros((0, 4)), np.zeros(0), 0.5).tolist() == []
