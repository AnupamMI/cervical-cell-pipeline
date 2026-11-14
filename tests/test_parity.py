"""Seg parity: the pipeline's crop + preprocessing + inference must reproduce the original eval code path.

Original path = tinyseg_retrain/evaluate_v2.py 'jitter' protocol, loaded from the UNMODIFIED original file
(only the hard-coded /home/nvidia/nc_pipeline/run_nc.py path is redirected, in memory, to the local copy):
  aug_v2.pipeline_crop around a seeded jittered GT box -> evaluate_v2.load_model / evaluate_v2.predict
  -> run_nc.postprocess (box-centre blob, clipped to box + 5%).
Pipeline path = ncmask.fov.extract_crop -> ncmask.segment.SegModel.predict (different batch size).

Checked on 20 tinyseg_retrain TEST crops, for v2_1 (3-channel) and v2 (4-channel, box mask):
  crops byte-identical; probabilities ~identical; thresholded maps Dice >= 0.999; and the original eval masks
  recomputed from the pipeline's probabilities Dice >= 0.999. Results -> reports/parity.json.
"""
import copy
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from conftest import NC_PIPELINE_DIR, ROOT

N = 20
MODELS = {"v2_1": "models/seg/v2_1/best.pt", "v2": "models/seg/v2/best.pt"}
RESULTS = {}


def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * (a & b).sum() / s


def load_original_eval(tinyseg_dir):
    src = (tinyseg_dir / "evaluate_v2.py").read_text()
    run_nc = NC_PIPELINE_DIR / "run_nc.py"
    if not run_nc.is_file():
        pytest.skip(f"run_nc.py not found at {run_nc}")
    hard = '"/home/nvidia/nc_pipeline/run_nc.py"'
    assert src.count(hard) == 1
    mod = types.ModuleType("evaluate_v2_original")
    mod.__file__ = str(tinyseg_dir / "evaluate_v2.py")
    exec(compile(src.replace(hard, repr(str(run_nc))), mod.__file__, "exec"), mod.__dict__)
    return mod


@pytest.fixture(scope="module")
def setup(tinyseg_dir, cfg):
    import aug_v2, data, torch                                    # vendored copies (verbatim, see test_runner)
    ev = load_original_eval(tinyseg_dir)
    split = json.load(open(tinyseg_dir / "cache/split.json"))["test"]
    pick = split[::len(split) // N][:N]
    by = {i["file"]: i for i in data.load_coco(tinyseg_dir / "data")}
    items = [by[f] for f in pick]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return ev, items, device


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("edge_mode", ["reflect", "clamp"])
def test_seg_parity(setup, cfg, name, edge_mode):
    import aug_v2
    from ncmask.fov import clip_tol_px, extract_crop
    from ncmask.segment import SegModel
    ev, items, device = setup
    weights = str(ROOT / MODELS[name])

    # ---- original path (evaluate_v2 'jitter')
    m0 = ev.load_model(weights, device)
    o_crops, o_boxes, o_gt, fovs, boxes = [], [], [], [], []
    for k, it in enumerate(items):
        img, nuc, cell = aug_v2.load_item(it)
        H, W = img.shape[:2]
        gn = nuc if nuc is not None else np.zeros((H, W), np.uint8)
        gc = cell if cell is not None else np.zeros((H, W), np.uint8)
        box = aug_v2.jitter_box(aug_v2.base_box(img, cell), np.random.default_rng((777, k)))
        if edge_mode == "clamp":
            # clamp == reflect only where no padding is needed: put the crop on a larger background canvas
            # (both paths see the same canvas) so every window lies inside the "FOV"
            P = 400
            canvas = np.empty((H + 2 * P, W + 2 * P, 3), np.uint8); canvas[:] = np.median(img.reshape(-1, 3), 0)
            canvas[P:P + H, P:P + W] = img
            img, gn, gc = canvas, np.pad(gn, P), np.pad(gc, P)
            box = tuple(v + P for v in box)
        crop, _, jb = aug_v2.pipeline_crop(img, [gn, gc], box)
        o_crops.append(crop); o_boxes.append(jb); fovs.append(img); boxes.append(box)
    o_prob = ev.predict(m0, o_crops, o_boxes)

    def eval_masks(prob, crop, box):
        H, W = crop.shape[:2]
        x1, y1, x2, y2 = box; mg = 0.05 * max(x2 - x1, y2 - y1)
        clip = (max(0, int(x1 - mg)), max(0, int(y1 - mg)), min(W, int(np.ceil(x2 + mg))), min(H, int(np.ceil(y2 + mg))))
        return ev.run_nc.postprocess(prob, H, W, ((x1 + x2) / 2, (y1 + y2) / 2), (0.5, 0.5), clip)[:2]

    # ---- pipeline path (the source crop plays the FOV, the jittered box plays the detector box)
    c = copy.deepcopy(cfg)
    c["seg"]["weights"] = weights
    c["seg"]["model_config"] = None
    c["seg"]["batch_size"] = 7                                   # different batching than the original
    c["crop"]["edge_mode"] = edge_mode
    seg = SegModel(c, device)
    assert seg.in_ch == (4 if name == "v2" else 3) and seg.size == m0.size
    p_crops, p_boxes, same = [], [], []
    for img, box, oc in zip(fovs, boxes, o_crops):
        crop, cbox, win, origin, _ = extract_crop(img, box, c, seg)
        p_crops.append(crop); p_boxes.append(cbox)
        same.append(crop.shape == oc.shape and np.array_equal(crop, oc))
    p_prob = seg.predict(p_crops, p_boxes)

    res = {"n": len(items), "device": device, "crops_identical": int(sum(same))}
    assert all(same), "pipeline crops must be byte-identical to aug_v2.pipeline_crop"
    idx = list(range(len(items)))
    # network input tensors identical
    for i in idx:
        x0 = seg.make_input(p_crops[i], p_boxes[i])
        xo = __import__("tinyseg").to_input(o_crops[i], m0.size)
        if m0.in_ch == 4:
            import cv2
            mm = np.zeros(o_crops[i].shape[:2], np.uint8)
            x1, y1, x2, y2 = (int(round(v)) for v in o_boxes[i]); mm[max(0, y1):y2, max(0, x1):x2] = 1
            xo = np.concatenate([xo, cv2.resize(mm, (m0.size,) * 2, interpolation=cv2.INTER_NEAREST)[..., None]], -1)
        assert x0.dtype == np.float32 and np.array_equal(x0, xo)
    diff = float(np.abs(p_prob[idx] - o_prob[idx]).max())
    d_thr, d_eval = [], []
    for i in idx:
        for ch in (0, 1):
            d_thr.append(dice(p_prob[i, ch] > 0.5, o_prob[i, ch] > 0.5))
        on, oc_ = eval_masks(o_prob[i], o_crops[i], o_boxes[i])
        pn, pc = eval_masks(p_prob[i], p_crops[i], p_boxes[i])
        d_eval += [dice(pn, on), dice(pc, oc_)]
    res.update({"compared": len(idx), "max_abs_prob_diff": diff, "min_dice_thresholded": float(min(d_thr)),
                "min_dice_eval_masks": float(min(d_eval)), "mean_dice_eval_masks": float(np.mean(d_eval))})
    # informational: the pipeline's own post-processing (by design not run_nc's) vs the eval masks
    from ncmask.postprocess import postprocess
    dn, dc = [], []
    for i in idx:
        H, W = p_crops[i].shape[:2]
        r = postprocess(p_prob[i], H, W, p_boxes[i], 0.5, 0.5, clip_tol_px(c, p_boxes[i]),
                        c["postprocess"]["nucleus_min_component_px"])
        on, oc_ = eval_masks(o_prob[i], o_crops[i], o_boxes[i])
        dn.append(dice(r["nucleus"], on)); dc.append(dice(r["cell"], oc_))
    res["info_pipeline_postprocess_vs_eval"] = {"nucleus_dice_mean": float(np.mean(dn)), "nucleus_dice_min": float(np.min(dn)),
                                                "cell_dice_mean": float(np.mean(dc)), "cell_dice_min": float(np.min(dc))}
    RESULTS[f"{name}/{edge_mode}"] = res
    out = ROOT / "reports" / "parity.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update(RESULTS); out.write_text(json.dumps(prev, indent=1))
    assert diff < 1e-4
    assert min(d_thr) >= 0.999 and min(d_eval) >= 0.999
