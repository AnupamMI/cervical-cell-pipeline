import numpy as np

from ncmask import geometry
from ncmask.fov import resolve_overlaps
from ncmask.postprocess import postprocess, select_component

S = 100  # probability maps given at crop size so cv2.resize is the identity


def probs(cell, nuc):
    return np.stack([nuc.astype(np.float32), cell.astype(np.float32)])


def blank():
    return np.zeros((S, S), bool)


def test_select_component_centre_then_overlap():
    m = blank(); m[40:60, 40:60] = True; m[0:10, 0:10] = True
    assert select_component(m, (50, 50), (30, 30, 70, 70))[45, 45]
    assert not select_component(m, (50, 50), (30, 30, 70, 70))[5, 5]
    # centre on background -> component with the largest overlap with the box
    m = blank(); m[20:45, 20:45] = True; m[70:95, 70:95] = True
    sel = select_component(m, (50, 50), (25, 25, 75, 75))
    assert sel[30, 30] and not sel[80, 80]       # 20x20 overlap beats 5x5
    # nothing overlaps the box -> empty
    m = blank(); m[0:5, 0:5] = True
    assert not select_component(m, (50, 50), (40, 40, 60, 60)).any()


def test_cell_clipped_holes_filled_and_nucleus_inside_cell():
    cell = blank(); cell[10:90, 10:90] = True; cell[45:55, 45:55] = False      # hole
    nuc = blank(); nuc[40:60, 40:60] = True; nuc[0:30, 0:30] = True           # second blob mostly outside the box
    box = (30, 30, 70, 70)
    r = postprocess(probs(cell, nuc), S, S, box, 0.5, 0.5, clip_tol=5, nuc_min=20)
    c = r["cell"]
    assert c[50, 50]                               # hole filled
    assert not c[24, 50] and c[25, 50] and c[74, 50] and not c[75, 50]   # clipped to box +- 5 px
    assert not (r["nucleus"] & ~c).any()           # nucleus only inside the cell
    assert r["nucleus"][50, 50] and not r["nucleus"][5, 5]
    assert r["clip_removed_px"] > 0


def test_box_clipped_and_clipped_frac():
    cell = blank(); cell[10:90, 10:90] = True                        # 6400 px
    nuc = blank(); nuc[45:55, 45:55] = True
    box = (30, 30, 70, 70)
    r = postprocess(probs(cell, nuc), S, S, box, .5, .5, clip_tol=5, nuc_min=20)
    assert "box_clipped" in r["flags"]
    assert r["cell"].sum() == 50 * 50 and abs(r["clipped_frac"] - (6400 - 2500) / 6400) < 1e-9
    # tolerance reaching the crop border: nothing clipped, no flag
    r = postprocess(probs(cell, nuc), S, S, box, .5, .5, clip_tol=40, nuc_min=20)
    assert "box_clipped" not in r["flags"] and r["clipped_frac"] == 0 and r["cell"].sum() == 6400
    # clipped_frac only counts the in-FOV part of a mirror-padded crop (valid rect)
    r = postprocess(probs(cell, nuc), S, S, box, .5, .5, clip_tol=5, nuc_min=20, valid=(0, 0, 50, 100))
    assert abs(r["clipped_frac"] - (40 * 80 - 25 * 50) / (40 * 80)) < 1e-9   # in FOV: cols 10..49 x rows 10..89; kept cols 25..49 x rows 25..74
    empty = postprocess(probs(blank(), nuc), S, S, box, .5, .5, 5, 20)
    assert empty["clipped_frac"] == 0 and "box_clipped" not in empty["flags"]


def test_flags_no_nucleus_multi_nucleus_empty():
    cell = blank(); cell[20:80, 20:80] = True
    box = (20, 20, 80, 80)
    r = postprocess(probs(cell, blank()), S, S, box, 0.5, 0.5, 5, 20)
    assert r["flags"] == ["no_nucleus"]
    nuc = blank(); nuc[30:40, 30:40] = True; nuc[60:70, 60:70] = True; nuc[50, 50] = True  # 2 blobs + 1 px speck
    r = postprocess(probs(cell, nuc), S, S, box, 0.5, 0.5, 5, 20)
    assert "multi_nucleus" in r["flags"] and r["n_nuc"] == 2 and not r["nucleus"][50, 50]   # speck removed
    r = postprocess(probs(blank(), nuc), S, S, box, 0.5, 0.5, 5, 20)
    assert r["flags"] == ["empty_mask"] and not r["nucleus"].any()
    speck = blank(); speck[50, 50] = True
    r = postprocess(probs(cell, speck), S, S, box, 0.5, 0.5, 5, 20)
    assert r["flags"] == ["no_nucleus"]


def test_box_too_tight_only_on_non_fov_sides():
    cell = blank(); cell[30:70, 0:70] = True       # reaches the left crop side
    nuc = blank(); nuc[45:55, 45:55] = True
    box = (20, 30, 70, 70)
    assert "box_too_tight" in postprocess(probs(cell, nuc), S, S, box, .5, .5, 5, 20)["flags"]
    # the left side is the FOV edge (crop clamped there): not "too tight", that is 'truncated'
    assert "box_too_tight" not in postprocess(probs(cell, nuc), S, S, box, .5, .5, 5, 20, fov_edge_sides=("l",))["flags"]


def test_clipping_split_keeps_centre_part():
    cell = blank(); cell[45:55, 0:100] = True; cell[0:100, 0:5] = True   # bar + vertical strip joined at the left
    nuc = blank(); nuc[47:53, 47:53] = True
    box = (30, 30, 70, 70)
    r = postprocess(probs(cell, nuc), S, S, box, .5, .5, 3, 5)
    assert r["cell"][50, 50] and not r["cell"][:, :20].any()


def test_overlap_resolution_higher_conf_wins():
    W = H = 60
    a = geometry.crop_window((5, 5, 35, 35), W, H, 0.0)
    b = geometry.crop_window((25, 25, 55, 55), W, H, 0.0)
    ca, cb = np.ones((a.h, a.w), bool), np.ones((b.h, b.w), bool)
    na = np.zeros_like(ca); na[20:30, 20:30] = True          # A's nucleus partly in the overlap (25..34)
    nb = np.zeros_like(cb); nb[0:5, 0:5] = True               # B's nucleus entirely in the overlap
    items = [{"cell_id": 2, "conf": 0.6, "win": b, "cell": cb, "nucleus": nb},
             {"cell_id": 1, "conf": 0.9, "win": a, "cell": ca, "nucleus": na}]
    cl, nl, lost, won, dropped = resolve_overlaps((H, W), items)
    assert lost == {1: 0, 2: 100} and won == {1: 100} and dropped == {}
    assert (cl[25:35, 25:35] == 1).all()                      # overlap -> higher conf
    assert (cl == 2).sum() == 900 - 100 and (cl == 1).sum() == 900
    assert (nl == 2).sum() == 0 and (nl == 1).sum() == 100    # B lost its whole nucleus
    assert (nl[nl > 0] == cl[nl > 0]).all()                   # every nucleus pixel belongs to its own cell
    # equal confidence -> lower cell_id wins (deterministic)
    items[0]["conf"] = 0.9
    cl2, _, lost2, _, _ = resolve_overlaps((H, W), items)
    assert (cl2[25:35, 25:35] == 1).all() and lost2[2] == 100


def test_drop_cells_that_lost_too_much():
    W = H = 100
    big = geometry.crop_window((10, 10, 70, 70), W, H, 0.0)       # 60x60, conf 0.9
    mid = geometry.crop_window((30, 30, 80, 80), W, H, 0.0)       # 50x50, 40x40 = 64% inside big -> dropped at 0.5
    low = geometry.crop_window((55, 55, 95, 95), W, H, 0.0)       # 40x40, overlaps big (15x15) and mid
    mk = lambda w: (np.ones((w.h, w.w), bool), np.zeros((w.h, w.w), bool))
    items = [{"cell_id": i, "conf": c, "win": w, "cell": mk(w)[0], "nucleus": mk(w)[1]}
             for i, c, w in ((1, 0.9, big), (2, 0.8, mid), (3, 0.7, low))]
    cl, nl, lost, won, dropped = resolve_overlaps((H, W), items, 0.5)
    assert set(dropped) == {2} and abs(dropped[2] - 0.64) < 1e-9
    assert not (cl == 2).any()                                    # dropped cell is not painted at all
    # cell 3 loses only to cell 1 (225 px of 1600), not to the dropped cell 2, and cell 1 'wins' only against 3
    assert lost[3] == 225 and won == {1: 225} and 2 not in lost
    assert (cl[70:80, 70:80] == 3).all()                          # pixels cell 2 would have owned go to cell 3
    # threshold is 'more than': exactly 0.64 is kept at 0.64; None keeps everything
    assert resolve_overlaps((H, W), items, 0.64)[4] == {}
    assert resolve_overlaps((H, W), items, None)[4] == {}
