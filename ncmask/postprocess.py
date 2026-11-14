"""Per-crop post-processing of TinyUNet probabilities into one cell mask and its nucleus mask, plus flags.

  probabilities are resized to the crop with cv2.resize (bilinear), as in tinyseg.postprocess / run_nc.postprocess
  cell     = thresholded cell map -> connected component (8-conn) containing the box centre; if none, the component
             with the largest overlap with the box; clipped to box + clip_tolerance_px; if clipping split it, the
             part containing the centre (else the largest part); holes filled
  nucleus  = thresholded nucleus map AND cell; blobs < nucleus_min_component_px removed
  flags    = box_clipped    the clip to box + clip_tol removed pixels of the cell component (see clipped_frac)
             box_too_tight  the cell component (before clipping) reaches a crop side that is not the FOV edge
             no_nucleus     cell present, no nucleus blob >= nucleus_min_component_px
             multi_nucleus  2+ nucleus blobs >= nucleus_min_component_px
             empty_mask     no cell pixels
"""
import cv2
import numpy as np
from scipy import ndimage as ndi

from . import geometry


def components(mask):
    """8-connected labelling -> (n_labels incl. background, labels int32, areas (n,))."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return n, lab, st[:, cv2.CC_STAT_AREA]


def select_component(mask, centre, box_rect):
    """The component containing centre (x, y); else the one with the largest overlap with box_rect
    (x0, y0, x1, y1, half-open); ties -> lowest label. Returns a bool mask (all False if nothing qualifies)."""
    n, lab, _ = components(mask)
    if n <= 1:
        return np.zeros(mask.shape, bool)
    h, w = mask.shape
    cx, cy = min(max(int(centre[0]), 0), w - 1), min(max(int(centre[1]), 0), h - 1)
    k = int(lab[cy, cx])
    if k == 0:
        x0, y0, x1, y1 = box_rect
        ov = np.bincount(lab[y0:y1, x0:x1].ravel(), minlength=n)[1:]
        if ov.size == 0 or ov.max() == 0:
            return np.zeros(mask.shape, bool)
        k = 1 + int(np.argmax(ov))
    return lab == k


def touches_sides(mask, sides):
    """Does the mask have a pixel on any of the given crop sides ('l','t','r','b')?"""
    return any({"l": mask[:, 0], "t": mask[0, :], "r": mask[:, -1], "b": mask[-1, :]}[s].any() for s in sides)


def postprocess(prob, crop_h, crop_w, box, thr_n, thr_c, clip_tol, nuc_min, fov_edge_sides=(), valid=None):
    """prob (2, S, S); box = detector box in crop coords; fov_edge_sides = crop sides that are the FOV edge;
    valid = (x0, y0, x1, y1) part of the crop inside the FOV (reflect mode), default the whole crop.
    Returns dict(cell bool HxW, nucleus bool HxW, flags list, n_nuc int, clip_removed_px int,
    clipped_frac = share of the cell component's in-FOV pixels removed by the clip to box + clip_tol)."""
    pn = cv2.resize(prob[0], (crop_w, crop_h))
    pc = cv2.resize(prob[1], (crop_w, crop_h))
    bx1, by1, bx2, by2 = box
    centre = ((bx1 + bx2) / 2, (by1 + by2) / 2)
    box_rect = geometry.clip_rect(box, crop_w, crop_h, 0)
    flags = []
    comp = select_component(pc > thr_c, centre, box_rect)
    if comp.any() and touches_sides(comp, [s for s in "ltrb" if s not in fov_edge_sides]):
        flags.append("box_too_tight")
    x0, y0, x1, y1 = geometry.clip_rect(box, crop_w, crop_h, clip_tol)
    keep = np.zeros_like(comp); keep[y0:y1, x0:x1] = True
    cell = comp & keep
    vx0, vy0, vx1, vy1 = valid if valid is not None else (0, 0, crop_w, crop_h)
    comp_v = comp[vy0:vy1, vx0:vx1]
    clip_removed = int(comp_v.sum() - cell[vy0:vy1, vx0:vx1].sum())
    clipped_frac = clip_removed / int(comp_v.sum()) if comp_v.any() else 0.0
    if clip_removed:
        flags.append("box_clipped")
    if cell.any():
        n, _, _ = components(cell)
        if n > 2:  # clipping split the cell: the part at the centre, else the largest part
            part = select_component(cell, centre, (x0, y0, x1, y1))
            cell = part if part.any() else cell
        cell = ndi.binary_fill_holes(cell)
    nuc = (pn > thr_n) & cell
    n_nuc = 0
    if nuc.any():
        n, lab, areas = components(nuc)
        big = np.zeros(n, bool); big[1:] = areas[1:] >= nuc_min
        nuc = big[lab]
        n_nuc = int(big.sum())
    if not cell.any():
        flags.append("empty_mask")
    elif n_nuc == 0:
        flags.append("no_nucleus")
    if n_nuc > 1:
        flags.append("multi_nucleus")
    return {"cell": cell, "nucleus": nuc, "flags": flags, "n_nuc": n_nuc, "clip_removed_px": clip_removed,
            "clipped_frac": clipped_frac}
