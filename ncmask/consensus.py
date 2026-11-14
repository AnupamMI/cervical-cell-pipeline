"""Cross-check of two detectors: one-to-one matching of A and B boxes (Hungarian on IoU).

match_boxes(a, b, thr) -> pairs (i, j) with IoU >= thr, maximising the total IoU over all one-to-one assignments.
Tags for A boxes: consensus (matched) | s_only (A only). B boxes without a partner: l_only.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import geometry


def match_boxes(a, b, iou_thr):
    """a (N,4), b (M,4) xyxy -> (pairs list of (i, j, iou) sorted by i, a_matched bool (N,), b_matched bool (M,)).
    Pairs below iou_thr are never matched: the assignment is solved on IoU with sub-threshold entries set to 0
    and those pairs discarded, so a sub-threshold pair can't block a valid one."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    am, bm = np.zeros(len(a), bool), np.zeros(len(b), bool)
    if not len(a) or not len(b):
        return [], am, bm
    iou = geometry.iou_matrix(a, b)
    w = np.where(iou >= iou_thr, iou, 0.0)
    ri, ci = linear_sum_assignment(-w)
    pairs = [(int(i), int(j), float(iou[i, j])) for i, j in zip(ri, ci) if iou[i, j] >= iou_thr]
    for i, j, _ in pairs:
        am[i] = bm[j] = True
    return sorted(pairs), am, bm
