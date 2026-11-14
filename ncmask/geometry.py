"""Pure coordinate math: crop windows, crop <-> FOV mapping, tiles, NMS. No model code here."""
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CropWindow:
    """Crop of a FOV around a detector box. x0,y0,x1,y1: the crop actually cut (clamped to the FOV), FOV px,
    half-open. ux0..uy1: the unclamped window (box + margin). box: the detector box in CROP coordinates."""
    x0: int
    y0: int
    x1: int
    y1: int
    ux0: int
    uy0: int
    ux1: int
    uy1: int
    box: tuple
    truncated: bool          # the unclamped window extends past the FOV edge
    edge_sides: tuple        # which crop sides lie on the FOV edge because of clamping: subset of "ltrb"

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def h(self):
        return self.y1 - self.y0


def crop_window(box, W, H, margin):
    """box (x1,y1,x2,y2) float FOV px -> CropWindow. Window = box + margin*max(w,h) per side, with the same
    floor/ceil rounding as aug_v2.pipeline_crop, then clamped to [0,W) x [0,H)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    m = margin * max(x2 - x1, y2 - y1)
    ux0, uy0 = int(np.floor(x1 - m)), int(np.floor(y1 - m))
    ux1, uy1 = int(np.ceil(x2 + m)), int(np.ceil(y2 + m))
    x0, y0, cx1, cy1 = max(0, ux0), max(0, uy0), min(W, ux1), min(H, uy1)
    if cx1 <= x0 or cy1 <= y0:
        raise ValueError(f"box {box} does not intersect the {W}x{H} image")
    sides = "".join(s for s, hit in zip("ltrb", (ux0 < 0, uy0 < 0, ux1 > W, uy1 > H)) if hit)
    return CropWindow(x0, y0, cx1, cy1, ux0, uy0, ux1, uy1,
                      (x1 - x0, y1 - y0, x2 - x0, y2 - y0), bool(sides), tuple(sides))


def crop_to_fov(xy, win):
    """Points (N,2) in crop coordinates -> FOV coordinates."""
    return np.asarray(xy, float) + (win.x0, win.y0)


def fov_to_crop(xy, win):
    return np.asarray(xy, float) - (win.x0, win.y0)


def paste(fov_shape, mask, win):
    """Crop-sized mask -> full FOV-sized bool mask."""
    out = np.zeros(fov_shape, bool)
    out[win.y0:win.y1, win.x0:win.x1] = mask.astype(bool)
    return out


def clip_rect(box_crop, crop_w, crop_h, tol):
    """Integer half-open rectangle (x0,y0,x1,y1) = box + tol px per side, in crop coords, clipped to the crop."""
    bx1, by1, bx2, by2 = box_crop
    return (max(0, int(math.floor(bx1 - tol))), max(0, int(math.floor(by1 - tol))),
            min(crop_w, int(math.ceil(bx2 + tol))), min(crop_h, int(math.ceil(by2 + tol))))


def tiles(W, H, tile, overlap):
    """Tile origins (x0,y0,x1,y1) covering the image; the last tile in each direction is aligned to the edge."""
    def starts(n):
        if n <= tile:
            return [0]
        step = tile - overlap
        s = list(range(0, n - tile + 1, step))
        if s[-1] + tile < n:
            s.append(n - tile)
        return s
    return [(x, y, min(W, x + tile), min(H, y + tile)) for y in starts(H) for x in starts(W)]


def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def containment_suppress(boxes, scores, thr, min_area_ratio=0.0):
    """Greedy, in descending score (ties -> lower index): drop box i if >= thr of ITS area lies inside an already
    kept higher-scoring box j (and, if min_area_ratio > 0, area_i >= min_area_ratio * area_j, so that a small cell
    lying on a big one can be kept). Returns (keep bool array, max inside-fraction per box over kept boxes)."""
    b = np.asarray(boxes, float).reshape(-1, 4)
    s = np.asarray(scores, float)
    n = len(b)
    area = np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1e-9)
    iw = np.clip(np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1]), 0, None)
    inside = iw * ih / area[:, None]                     # inside[i, j] = share of box i inside box j
    keep, frac, kept = np.zeros(n, bool), np.zeros(n), []
    for i in sorted(range(n), key=lambda k: (-s[k], k)):
        cand = [j for j in kept if min_area_ratio <= 0 or area[i] >= min_area_ratio * area[j]]
        frac[i] = max((inside[i, j] for j in kept), default=0.0)
        if cand and max(inside[i, j] for j in cand) >= thr:
            continue
        keep[i] = True; kept.append(i)
    return keep, frac


def nms(boxes, scores, iou_thr):
    """Class-agnostic greedy NMS, deterministic: ties in score are broken by original index. Returns kept indices
    in descending score order."""
    boxes, scores = np.asarray(boxes, float).reshape(-1, 4), np.asarray(scores, float)
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    keep, sup = [], np.zeros(len(scores), bool)
    iou = iou_matrix(boxes, boxes) if len(boxes) else np.zeros((0, 0))
    for i in order:
        if sup[i]:
            continue
        keep.append(i)
        sup |= iou[i] > iou_thr
    return np.array(keep, int)
