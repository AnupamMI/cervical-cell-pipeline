"""Pipeline-like training samples from single-cell crops (fine-tune v2).

From one annotated crop (only the centre cell is labelled):
  1. optional copy-paste overlap: part of another crop's cell is laid over the target cell's edge
     (multiplicative blend, like overlapping cytoplasm); labels are unchanged (the target continues underneath)
  2. a detector-like box: the cell polygon's bounding box, jittered (centre +-10% of w/h, scale 0.85-1.2);
     for crops without a cell polygon, the crop shrunk by the pipeline padding
  3. the pipeline crop around that box (0.17 * max side padding); outside the crop: reflect for the image,
     0 for labels
  4. resize to SIZE; box mask as an extra channel
  5. geometric (flips, rot90, small affine, mild elastic) and photometric augmentation (HSV, HED stain,
     gamma, CLAHE, greyscale, Gaussian / defocus blur, noise)
  6. boundary weight maps (W_EDGE within EDGE_PX of the nucleus / cell GT edge)
"""
import cv2
import numpy as np
from skimage.color import hed2rgb, rgb2hed

import data

PAD = 0.17
SIZE = 224
EDGE_PX, W_EDGE = 3, 3.0


def load_item(it):
    img = cv2.imread(it["path"])
    nuc = data.rasterize(it, data.NUC) if data.has(it, data.NUC) else None
    cell = data.rasterize(it, data.CELL) if data.has(it, data.CELL) else None
    return img, nuc, cell


def base_box(img, cell):
    H, W = img.shape[:2]
    if cell is not None and cell.any():
        ys, xs = np.nonzero(cell)
        return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)
    bw, bh = W / (1 + 2 * PAD), H / (1 + 2 * PAD)  # crop = box + pad on each side
    return (W - bw) / 2, (H - bh) / 2, (W + bw) / 2, (H + bh) / 2


def centred_box(img):
    """The box the pipeline would have used if this crop were its padded crop."""
    return base_box(img, None)


def jitter_box(box, rng, shift=0.10, scale=(0.85, 1.2)):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2 + rng.uniform(-shift, shift) * w
    cy = (y1 + y2) / 2 + rng.uniform(-shift, shift) * h
    s = rng.uniform(*scale); w, h = w * s, h * s
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def pipeline_crop(img, labels, box, border="reflect"):
    """Crop box + PAD*max(w,h) from img (outside the crop: reflect, or border='bg' = the median colour of the
    crop's outer frame) and label maps (0 outside). Returns the crop, the label crops and the box in crop coordinates."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    m = PAD * max(x2 - x1, y2 - y1)
    X0, Y0 = int(np.floor(x1 - m)), int(np.floor(y1 - m))
    X1, Y1 = int(np.ceil(x2 + m)), int(np.ceil(y2 + m))
    p = max(0, -X0, -Y0, X1 - W, Y1 - H)
    if p:
        if border == "bg":
            frame = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
            img = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=[int(v) for v in np.median(frame, 0)])
        else:
            img = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_REFLECT_101)
        labels = [cv2.copyMakeBorder(l, p, p, p, p, cv2.BORDER_CONSTANT, value=0) for l in labels]
    sl = (slice(Y0 + p, Y1 + p), slice(X0 + p, X1 + p))
    return img[sl], [l[sl] for l in labels], (x1 - X0, y1 - Y0, x2 - X0, y2 - Y0)


def paste_overlap(img, nuc, cell, src_img, src_cell, rng):
    """Lay part of the source cell over the target cell's edge; target labels stay as they are."""
    if cell is None or not cell.any() or src_cell is None or not src_cell.any():
        return img
    ys, xs = np.nonzero(src_cell)
    patch = src_img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    pm = src_cell[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    k = rng.integers(4); patch, pm = np.rot90(patch, k), np.rot90(pm, k)
    if rng.random() < 0.5: patch, pm = patch[:, ::-1], pm[:, ::-1]
    tys, txs = np.nonzero(cell)
    tsize = max(tys.max() - tys.min(), txs.max() - txs.min())
    s = rng.uniform(0.5, 1.0) * tsize / max(patch.shape[:2])
    patch = cv2.resize(np.ascontiguousarray(patch), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    pm = cv2.resize(np.ascontiguousarray(pm), (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_LINEAR)
    pm = cv2.GaussianBlur(pm, (0, 0), 2)
    ph, pw = pm.shape
    # centre the patch just outside a random point of the target cell's boundary
    cs, _ = cv2.findContours(cell.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)[:, 0]
    bx, by = c[rng.integers(len(c))]
    tcx, tcy = txs.mean(), tys.mean()
    v = np.array([bx - tcx, by - tcy]); v /= np.linalg.norm(v) + 1e-6
    d = rng.uniform(0.15, 0.45) * max(ph, pw)
    px, py = int(bx + v[0] * d - pw / 2), int(by + v[1] * d - ph / 2)
    H, W = img.shape[:2]
    a0x, a0y, a1x, a1y = max(0, px), max(0, py), min(W, px + pw), min(H, py + ph)
    if a1x <= a0x or a1y <= a0y:
        return img
    alpha = pm[a0y - py:a1y - py, a0x - px:a1x - px] * rng.uniform(0.6, 1.0)
    if nuc is not None:  # never cover the target nucleus
        guard = cv2.dilate(nuc, np.ones((15, 15), np.uint8))[a0y:a1y, a0x:a1x]
        alpha = alpha * (guard == 0)
    region = img[a0y:a1y, a0x:a1x].astype(np.float32)
    src = patch[a0y - py:a1y - py, a0x - px:a1x - px].astype(np.float32)
    blend = region * (src / 255.0)  # overlapping cytoplasm absorbs light: multiplicative
    out = img.copy()
    out[a0y:a1y, a0x:a1x] = np.clip(region * (1 - alpha[..., None]) + blend * alpha[..., None], 0, 255).astype(np.uint8)
    return out


def geometric(img, lab, rng):
    """img (S,S,3) uint8, lab (S,S,C) uint8 (nucleus, cell, box, real). Labels use a 0 border, so pixels
    the warp brings in from outside get real = 0."""
    if rng.random() < 0.5: img, lab = img[:, ::-1], lab[:, ::-1]
    if rng.random() < 0.5: img, lab = img[::-1], lab[::-1]
    k = rng.integers(4); img, lab = np.rot90(img, k), np.rot90(lab, k)
    img, lab = np.ascontiguousarray(img), np.ascontiguousarray(lab)
    S = img.shape[0]
    s = rng.uniform(0.9, 1.1); tx, ty = rng.uniform(-0.04, 0.04, 2) * S
    A = cv2.getRotationMatrix2D((S / 2, S / 2), rng.uniform(-15, 15), s); A[:, 2] += (tx, ty)
    img = cv2.warpAffine(img, A, (S, S), borderMode=cv2.BORDER_REFLECT)
    lab = cv2.warpAffine(lab, A, (S, S), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if rng.random() < 0.3:  # mild elastic deformation
        dx = cv2.GaussianBlur(rng.uniform(-1, 1, (S, S)).astype(np.float32), (0, 0), 8) * 80
        dy = cv2.GaussianBlur(rng.uniform(-1, 1, (S, S)).astype(np.float32), (0, 0), 8) * 80
        gx, gy = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32))
        mx, my = gx + dx, gy + dy
        img = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        lab = cv2.remap(lab, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, lab.reshape(S, S, -1)


def photometric(img, rng):
    # notebook HSV jitter
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 180
    hsv[..., 1] *= rng.uniform(0.7, 1.3); hsv[..., 2] *= rng.uniform(0.85, 1.15)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.5:  # HED stain jitter, +-0.04
        hed = rgb2hed(img[..., ::-1].astype(np.float32) / 255)
        hed = hed * rng.uniform(0.96, 1.04, 3) + rng.uniform(-0.04, 0.04, 3) * np.abs(hed).max(axis=(0, 1))
        img = (np.clip(hed2rgb(hed), 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()
    if rng.random() < 0.5:  # gamma
        g = rng.uniform(0.8, 1.2)
        img = (255 * (img / 255.0) ** g).astype(np.uint8)
    if rng.random() < 0.2:  # CLAHE on lightness
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[..., 0] = cv2.createCLAHE(2.0, (4, 4)).apply(lab[..., 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if rng.random() < 0.1:  # greyscale
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    r = rng.random()
    if r < 0.3:  # notebook Gaussian blur
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
    elif r < 0.45:  # defocus (disk) blur
        rad = int(rng.integers(1, 4))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1,) * 2).astype(np.float32)
        img = cv2.filter2D(img, -1, k / k.sum())
    if rng.random() < 0.3:  # mild Gaussian noise
        img = np.clip(img + rng.normal(0, rng.uniform(2, 6), img.shape), 0, 255).astype(np.uint8)
    return img


def edge_weight(mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * EDGE_PX + 1,) * 2)
    band = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, k) > 0
    return np.where(band, W_EDGE, 1.0).astype(np.float32)


def make_sample(img, nuc, cell, box, size=SIZE):
    """Pipeline crop for a given box, resized; returns img (S,S,3),
    labels (S,S,4: nuc, cell, box, real = 1 where the pixel comes from the crop, 0 where reflect-padded), valid (2,)."""
    H, W = img.shape[:2]
    n = nuc if nuc is not None else np.zeros((H, W), np.uint8)
    c = cell if cell is not None else np.zeros((H, W), np.uint8)
    crop, (n, c, real), b = pipeline_crop(img, [n, c, np.ones((H, W), np.uint8)], box)
    bm = np.zeros(crop.shape[:2], np.uint8)
    x1, y1, x2, y2 = (int(round(v)) for v in b)
    bm[max(0, y1):y2, max(0, x1):x2] = 1
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    lab = np.stack([cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST) for m in (n, c, bm, real)], -1)
    valid = np.array([nuc is not None, cell is not None], np.float32)
    return crop, lab, valid


def train_sample(items, i, rng, size=SIZE, p_paste=0.3):
    img, nuc, cell = load_item(items[i])
    if rng.random() < p_paste:
        j = int(rng.integers(len(items)))
        if j != i and data.has(items[j], data.CELL):
            simg, _, scell = load_item(items[j])
            img = paste_overlap(img, nuc, cell, simg, scell, rng)
    box = jitter_box(base_box(img, cell), rng)
    crop, lab, valid = make_sample(img, nuc, cell, box, size)
    crop, lab = geometric(crop, lab, rng)
    crop = photometric(crop, rng)
    w = np.stack([edge_weight(lab[..., 0]), edge_weight(lab[..., 1])], -1)
    return crop, lab, valid, w
