"""Seg parity for the CONFIGURED model (the TinyUNet v3 winner, or v2_1): the pipeline's crop + preprocessing +
inference must reproduce the evaluation code path (tinyseg_v3/phase2/eval_v3.py: aug_v2.pipeline_crop on the GT box,
then eval_v3.Model with the training preprocessing) on 20 T1 crops: identical crops, identical network input tensors,
and (both at batch size 1, as the evaluation ran; GPU kernels differ between batch sizes) max |prob diff| < 1e-5 and
thresholded maps Dice 1.0. This is the path the RESULTS.md numbers were measured with."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PHASE2 = ROOT.parent / "tinyseg_v3/phase2"


def test_parity_configured_model(cfg):
    if not (PHASE2 / "eval_v3.py").exists() or not (PHASE2 / "split_v3.json").exists():
        pytest.skip("tinyseg_v3 evaluation code / split not available")
    from ncmask.fov import extract_crop
    from ncmask.segment import SegModel
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    c = copy.deepcopy(cfg); c["crop"]["edge_mode"] = "reflect"; c["seg"]["batch_size"] = 1
    seg = SegModel(c, device)          # first: loads the vendored training code (byte-identical to the original)
    sys.path.insert(0, str(PHASE2))
    import eval_v3 as E                # reuses those already-imported modules
    wdir = Path(c["seg"]["weights"]).parent
    ref_model = E.Model("ref", wdir, device, weights=c["seg"]["weights"])
    assert ref_model.variant == (seg.variant if seg.variant != "box" else ref_model.variant)
    recs = E.load_T1()[::11][:20]
    crops, boxes, ref_probs = [], [], []
    for r in recs:
        box = E.aug_v2.base_box(r["S"], r["cell"] if r["cell"] is not None and r["cell"].any() else None)
        crop, cbox, *_ = extract_crop(r["S"], box, c, seg)
        rc, _, rb = E.aug_v2.pipeline_crop(r["S"], [], box)
        assert np.array_equal(crop, rc) and np.allclose(cbox, rb)
        # network input: pipeline vs the evaluation/training preprocessing functions
        w, _ = E.pp.transform_source(ref_model.variant, rc, ref_model.ref)
        want = E.tinyseg.to_input(w, ref_model.size)
        if ref_model.in_ch == 4:
            import cv2
            hs = E.pp.h_channel(cv2.resize(w, (ref_model.size,) * 2, interpolation=cv2.INTER_AREA))
            want = np.concatenate([want, ((hs - ref_model.hmu) / ref_model.hsd)[..., None]], -1)
        assert np.array_equal(seg.make_input(crop, cbox), want)
        crops.append(crop); boxes.append(cbox); ref_probs.append(ref_model(rc))
    p = seg.predict(crops, boxes)
    ref = np.stack(ref_probs)
    diff = float(np.abs(p - ref).max())
    d = []
    for k in range(len(p)):
        for ch in (0, 1):
            a, b = p[k, ch] > 0.5, ref[k, ch] > 0.5
            d.append(1.0 if a.sum() + b.sum() == 0 else 2 * (a & b).sum() / (a.sum() + b.sum()))
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/parity_v3.json").write_text(json.dumps({"model": c["seg"]["weights"], "variant": seg.variant,
                                                             "cells": len(p), "max_abs_prob_diff": diff, "min_dice": min(d)}, indent=1))
    assert diff < 1e-5 and min(d) == 1.0
