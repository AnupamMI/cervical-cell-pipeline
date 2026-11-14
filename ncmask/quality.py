"""Stage 0 (image QC, per FOV) and Stage 5 (per-cell gate). Pure functions; thresholds come from config.

Stage 0 flags a FOV, it never skips it:
  blurry            variance of the Laplacian of the grey image at half resolution < qc.blur_lapvar_min
  too_dark/too_bright  mean grey outside [qc.mean_grey_min, qc.mean_grey_max]
  dark_regions      share of pixels with grey <= 20 > qc.dark_frac_max
  low_stain/heavy_stain  share of pixels with optical density > 0.15 in all channels (Io = 240) outside
                    [qc.stained_frac_min, qc.stained_frac_max]

Stage 5 gate -> status measurable | review | excluded, with every reason listed:
  excluded: cell_at_edge, no_nucleus, empty_mask, box_too_small, implausible_mask (cell area outside the implausible
            limits, nucleus area > limit, or N:C > gate.nc_implausible_max)
  review:   multi_nucleus, occluded (occluded_frac > gate.occluded_review), s_only_low_conf (s_only and conf <
            gate.s_only_review_conf), nucleus_touches_cell_boundary, likely_inflammatory (cell area < size.cell_area_
            review_min), likely_cluster (cell area > size.cell_area_review_max)
"""
import cv2
import numpy as np

EXCLUDE = ("cell_at_edge", "no_nucleus", "empty_mask", "box_too_small")


def image_qc(img, q):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (max(1, g.shape[1] // 2), max(1, g.shape[0] // 2)), interpolation=cv2.INTER_AREA)
    od = -np.log((img.astype(np.float32) + 1) / 240.0)
    m = {"lap_var": float(cv2.Laplacian(small, cv2.CV_64F).var()), "mean_grey": float(g.mean()),
         "dark_frac": float((g <= 20).mean()), "stained_frac": float((od > 0.15).all(-1).mean())}
    flags = []
    if m["lap_var"] < q["blur_lapvar_min"]: flags.append("blurry")
    if m["mean_grey"] < q["mean_grey_min"]: flags.append("too_dark")
    if m["mean_grey"] > q["mean_grey_max"]: flags.append("too_bright")
    if m["dark_frac"] > q["dark_frac_max"]: flags.append("dark_regions")
    if m["stained_frac"] < q["stained_frac_min"]: flags.append("low_stain")
    if m["stained_frac"] > q["stained_frac_max"]: flags.append("heavy_stain")
    return flags, {k: round(v, 5) for k, v in m.items()}


def nucleus_touches_cell_boundary(cell, nuc):
    """Any nucleus pixel on the cell mask's inner boundary (4-connected)."""
    if not cell.any() or not nuc.any():
        return False
    c = cell.astype(np.uint8)
    er = cv2.erode(c, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    inner = c.astype(bool) & ~er.astype(bool)
    return bool((inner & nuc.astype(bool)).any())


def gate(flags, cell_area, nuc_area, nc, occluded_frac, det_tag, conf, nuc_touch, size, g):
    """-> (status, reasons list). flags: the per-cell pipeline flags."""
    exc, rev = [f for f in EXCLUDE if f in flags], []
    if cell_area > 0 and (cell_area < size["cell_area_implausible_min"] or cell_area > size["cell_area_implausible_max"]
                          or nuc_area > size["nucleus_area_implausible_max"]
                          or (nc == nc and nc is not None and nc > g["nc_implausible_max"])):
        exc.append("implausible_mask")
    if "multi_nucleus" in flags: rev.append("multi_nucleus")
    if occluded_frac > g["occluded_review"]: rev.append("occluded")
    if det_tag == "s_only" and conf < g["s_only_review_conf"]: rev.append("s_only_low_conf")
    if nuc_touch: rev.append("nucleus_touches_cell_boundary")
    if 0 < cell_area < size["cell_area_review_min"]: rev.append("likely_inflammatory")
    if cell_area > size["cell_area_review_max"]: rev.append("likely_cluster")
    status = "excluded" if exc else "review" if rev else "measurable"
    return status, exc + rev
