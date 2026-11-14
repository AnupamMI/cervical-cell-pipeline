"""Track B TinyUNet: model, preprocessing, postprocess, augmentation and loss copied from the training notebook."""
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F

SIZE = 160


def cbr(i, o, k=3, s=1, g=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, groups=g, bias=False), nn.BatchNorm2d(o), nn.ReLU6(inplace=True))


class DS(nn.Module):  # depthwise-separable block x2
    def __init__(self, i, o, s=1):
        super().__init__()
        self.b = nn.Sequential(cbr(i, i, 3, s, g=i), cbr(i, o, 1), cbr(o, o, 3, 1, g=o), cbr(o, o, 1))

    def forward(self, x):
        return self.b(x)


class TinyUNet(nn.Module):
    def __init__(self, ch=(16, 32, 64, 96, 128), in_ch=3):
        super().__init__()
        self.stem = cbr(in_ch, ch[0], 3, 1)
        self.enc = nn.ModuleList([DS(ch[i], ch[i + 1], 2) for i in range(len(ch) - 1)])
        self.dec = nn.ModuleList([DS(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(len(ch) - 1))])
        self.head = nn.Conv2d(ch[0], 2, 1)

    def forward(self, x):
        skips = [self.stem(x)]
        for e in self.enc:
            skips.append(e(skips[-1]))
        y = skips.pop()
        for d in self.dec:
            s = skips.pop()
            y = d(torch.cat([F.interpolate(y, size=s.shape[-2:], mode="nearest"), s], 1))
        return self.head(y)


def load(path, device, in_ch=3):
    m = TinyUNet(in_ch=in_ch)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return m.to(device).eval()


def expand_stem(model, in_ch, std=1e-3, seed=0):
    """Return a copy of a 3-channel TinyUNet with in_ch inputs: the RGB weights are kept, the extra
    input channels of the first conv get small random weights."""
    new = TinyUNet(in_ch=in_ch)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    w = sd["stem.0.weight"]
    g = torch.Generator().manual_seed(seed)
    extra = torch.randn(w.shape[0], in_ch - w.shape[1], *w.shape[2:], generator=g) * std
    sd["stem.0.weight"] = torch.cat([w.cpu(), extra], 1)
    new.load_state_dict(sd, strict=True)
    return new


def to_input(bgr, size=SIZE):
    x = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32) / 255.0
    return (x - 0.5) / 0.25


def prep(x):
    """(N,S,S,3) BGR uint8 at SIZE -> normalised NCHW tensor (same as to_input)."""
    return torch.from_numpy(((x[..., ::-1].astype(np.float32) / 255 - 0.5) / 0.25).transpose(0, 3, 1, 2).copy())


def postprocess(prob, H, W, thr=(0.5, 0.5)):
    """prob: (2,S,S) -> full-res uint8 masks. Keep the cell blob covering the centre, nucleus blob inside it."""
    pn = cv2.resize(prob[0], (W, H)); pc = cv2.resize(prob[1], (W, H))
    cell = (pc > thr[1]).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cell)
    if n > 1:
        k = lab[H // 2, W // 2] or 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        cell = (lab == k).astype(np.uint8)
        h, w = cell.shape; ff = np.pad(cell, 1).copy(); cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
        cell = cell | (1 - ff[1:-1, 1:-1])
    nuc = (pn > thr[0]).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(nuc)
    if n > 2:  # several blobs: keep the one with the most probability mass (inside the cell)
        mass = [(pn * (lab == i) * (cell if cell.any() else 1)).sum() for i in range(1, n)]
        nuc = (lab == 1 + int(np.argmax(mass))).astype(np.uint8)
    cell = cell | nuc
    return nuc, cell, float(pn[nuc > 0].mean()) if nuc.any() else 0.0


def augment(img, m, rng):
    if rng.random() < 0.5: img, m = img[:, ::-1], m[:, ::-1]
    if rng.random() < 0.5: img, m = img[::-1], m[::-1]
    k = rng.integers(4); img, m = np.rot90(img, k), np.rot90(m, k)
    img = np.ascontiguousarray(img); m = np.ascontiguousarray(m)
    # small scale/shift (keeps the target cell centred-ish, like real crops)
    s = rng.uniform(0.9, 1.1); tx, ty = rng.uniform(-0.04, 0.04, 2) * SIZE
    A = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), rng.uniform(-15, 15), s); A[:, 2] += (tx, ty)
    img = cv2.warpAffine(img, A, (SIZE, SIZE), borderMode=cv2.BORDER_REFLECT)
    m = cv2.warpAffine(m, A, (SIZE, SIZE), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    # stain / illumination jitter in HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 180
    hsv[..., 1] *= rng.uniform(0.7, 1.3); hsv[..., 2] *= rng.uniform(0.85, 1.15)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.3: img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
    return img, m


def loss_fn(logit, m, weight=None):
    """Masked BCE + soft Dice. weight (N,2,S,S), optional: per-pixel BCE weight (e.g. higher near the GT edge)."""
    y, valid = m[:, :2], m[:, 2:4]
    wv = valid if weight is None else valid * weight
    bce = (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * wv).sum() / wv.sum().clamp(min=1)
    p = torch.sigmoid(logit) * valid
    inter = (p * y).sum((2, 3)); den = p.sum((2, 3)) + (y * valid).sum((2, 3))
    has = valid.amax((2, 3))  # channel labelled anywhere in the image (pixels may be ignored)
    dice = 1 - ((2 * inter + 1) / (den + 1))
    dice = (dice * has).sum() / has.sum().clamp(min=1)
    return bce + dice


@torch.no_grad()
def predict(model, imgs, bs=64):
    model.eval(); outs = []; dev = next(model.parameters()).device
    for b in range(0, len(imgs), bs):
        x = torch.from_numpy(np.stack([to_input(i) for i in imgs[b:b + bs]]).transpose(0, 3, 1, 2).copy())
        outs.append(torch.sigmoid(model(x.to(dev))).cpu().numpy())
    return np.concatenate(outs)
