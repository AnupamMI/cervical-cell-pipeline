"""Input variants for TinyUNet v3 (applied to the BGR uint8 source crop, before cropping/augmentation).

V0  raw RGB (identity)
V1  Macenko stain normalisation to a reference fitted on TRAIN-split crops only
V2  V1, then CLAHE on the L channel of LAB (clipLimit 2.0, tileGridSize 8x8)
V3  raw RGB + haematoxylin channel (skimage rgb2hed, H only), standardised with train-split mean/std;
    computed on the network-sized (224 px, INTER_AREA) crop, appended as a 4th input channel

Macenko (Macenko et al. 2009): OD = -log((I + 1) / Io), Io = 240; pixels with OD < 0.15 in any channel are dropped;
plane of the two largest eigenvectors of the OD covariance; stain vectors at the 1st / 99th percentile angle; H = the
vector with the larger red OD; concentrations by least squares; max concentration = 99th percentile.
Fallback (counted, returned flag): < 200 stained pixels or a degenerate fit -> image left unchanged.
Note: Pap stain is not H&E; Macenko's two-stain model is an approximation here (as requested).
"""
import json
from pathlib import Path

import cv2
import numpy as np

IO, BETA, ALPHA = 240.0, 0.15, 1.0


def _od(rgb):
    return -np.log((rgb.astype(np.float64) + 1) / IO)


def macenko_stains(rgb):
    """rgb uint8 (N,3) or image -> (HE 3x2 unit columns, maxC (2,)) or None."""
    od = _od(rgb.reshape(-1, 3))
    od = od[(od > BETA).all(1)]
    if len(od) < 200:
        return None
    _, v = np.linalg.eigh(np.cov(od.T))
    v = v[:, [2, 1]]
    if v[0, 0] < 0: v[:, 0] *= -1
    if v[0, 1] < 0: v[:, 1] *= -1
    t = od @ v
    phi = np.arctan2(t[:, 1], t[:, 0])
    lo, hi = np.percentile(phi, ALPHA), np.percentile(phi, 100 - ALPHA)
    a = v @ np.array([np.cos(lo), np.sin(lo)]); b = v @ np.array([np.cos(hi), np.sin(hi)])
    he = np.array([a, b]).T if a[0] > b[0] else np.array([b, a]).T
    he /= np.linalg.norm(he, axis=0, keepdims=True)
    if not np.isfinite(he).all() or abs(np.linalg.det(he.T @ he)) < 1e-6:
        return None
    c = np.linalg.lstsq(he, od.T, rcond=None)[0]
    maxc = np.percentile(c, 99, axis=1)
    if (maxc <= 1e-6).any():
        return None
    return he, maxc


def fit_reference(bgr_images):
    """Pool the pixels of the given (train) crops into one reference fit."""
    px = np.concatenate([im[..., ::-1].reshape(-1, 3) for im in bgr_images])
    r = macenko_stains(px)
    if r is None:
        raise RuntimeError("reference fit failed")
    return {"HE": r[0].tolist(), "maxC": r[1].tolist(), "Io": IO}


def macenko_normalize(bgr, ref):
    """-> (bgr uint8, ok). ok False = fallback (unchanged)."""
    rgb = bgr[..., ::-1]
    r = macenko_stains(rgb)
    if r is None:
        return bgr, False
    he, maxc = r
    od = _od(rgb.reshape(-1, 3))
    c = np.linalg.lstsq(he, od.T, rcond=None)[0]
    c *= (np.array(ref["maxC"]) / maxc)[:, None]
    out = IO * np.exp(-np.array(ref["HE"]) @ c)
    out = np.clip(out, 0, 255).T.reshape(rgb.shape).astype(np.uint8)
    return np.ascontiguousarray(out[..., ::-1]), True


def clahe_L(bgr, clip=2.0, tile=8):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def h_channel(bgr_sized):
    """bgr uint8 at network size -> float32 haematoxylin optical density (skimage rgb2hed, channel 0)."""
    from skimage.color import rgb2hed
    return rgb2hed(bgr_sized[..., ::-1].astype(np.float32) / 255.0)[..., 0].astype(np.float32)


def transform_source(variant, bgr, ref=None):
    """Source-crop transform for V0-V3 -> (bgr, ok)."""
    if variant in ("V0", "V3"):
        return bgr, True
    out, ok = macenko_normalize(bgr, ref)
    if variant == "V2":
        out = clahe_L(out)
    return out, ok


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1))
