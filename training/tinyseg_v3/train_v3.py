#!/usr/bin/env python3
"""TinyUNet v3: the v2_1 recipe trained FROM SCRATCH on the Phase 2 split, one input variant per run.

Recipe (as tinyseg_retrain/finetune_v2.py --in-ch 3 --ignore-pad, i.e. v2_1), using the training code's own functions:
  sample: aug_v2.load_item -> paste_overlap (p 0.3) -> jitter_box(base_box) -> make_sample (crop + 0.17, mirror pad,
          224 px) -> geometric (flips, rot90, affine, elastic) -> edge_weight
          The ONLY change: aug_v2.photometric is NOT applied (no colour augmentation; it also held blur/noise).
  loss:   tinyseg.loss_fn (masked BCE + soft Dice, boundary weight 3 within 3 px), padded pixels ignored
  model selection: mean over two fixed val sets (plain: centred box; jitter: seeded jittered box, seed 123)
          of 0.5 * nucleus Dice + 0.5 * cell Dice, padded pixels ignored (finetune_v2.val_score rule)
  model:  tinyseg.TinyUNet(in_ch=3, or 4 for V3), random init (torch.manual_seed(seed)), not from v1
Variants: V0 raw, V1 Macenko, V2 Macenko + LAB-CLAHE (source crops read from cache/V1, cache/V2), V3 RGB + H channel.

    python train_v3.py --variant V0 --epochs 60 --out runs/V0
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
TR = Path.home() / "Downloads/ANUPAM KA HOOD/seg_audit/extracted/tinyseg_retrain/tinyseg_retrain"
sys.path.insert(0, str(TR)); sys.path.insert(0, str(HERE))
import aug_v2, data, tinyseg  # noqa: E402  (original training code, read-only)
import preprocess as pp  # noqa: E402

SIZE = 224


def items_for(names, variant):
    by = {i["file"]: i for i in data.load_coco(TR / "data")}
    out = []
    for n in names:
        it = dict(by[n])
        if variant in ("V1", "V2"):
            it["path"] = str(HERE / "cache" / variant / n)
        out.append(it)
    return out


class HChan:
    def __init__(self):
        s = json.load(open(HERE / "h_stats.json")); self.mu, self.sd = s["mean"], s["std"]

    def __call__(self, crops):  # (N,S,S,3) BGR uint8 at network size -> (N,1,S,S) float32
        return torch.from_numpy(np.stack([(pp.h_channel(c) - self.mu) / self.sd for c in crops])[:, None].astype(np.float32))


def to_tensor(crops, in_ch, hch):
    x = tinyseg.prep(np.stack(crops))
    return torch.cat([x, hch(crops)], 1) if in_ch == 4 else x


def _load(it):
    return aug_v2.load_item(it)


def preload(items, workers):
    """Decoded crops + rasterised masks for every item, computed by aug_v2.load_item itself (in parallel), kept in RAM."""
    from multiprocessing import Pool
    with Pool(workers) as pool:
        return pool.map(_load, items, chunksize=64)


class TrainSet(Dataset):
    def __init__(self, items, seed, cache, photometric=False):
        self.items, self.seed, self.epoch, self.cache, self.photometric = items, seed, 0, cache, photometric

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, self.epoch, i))
        img, nuc, cell = self.cache[i]
        if rng.random() < 0.3:                                    # same order of RNG draws as aug_v2.train_sample
            j = int(rng.integers(len(self.items)))
            if j != i and data.has(self.items[j], data.CELL):
                simg, _, scell = self.cache[j]
                img = aug_v2.paste_overlap(img, nuc, cell, simg, scell, rng)
        box = aug_v2.jitter_box(aug_v2.base_box(img, cell), rng)
        crop, lab, valid = aug_v2.make_sample(img, nuc, cell, box, SIZE)
        crop, lab = aug_v2.geometric(crop, lab, rng)               # V0-V3: aug_v2.photometric deliberately skipped
        if self.photometric:                                      # V4 (--photometric): exactly aug_v2.train_sample's order
            crop = aug_v2.photometric(crop, rng)
        w = np.stack([aug_v2.edge_weight(lab[..., 0]), aug_v2.edge_weight(lab[..., 1])], -1)
        return crop, lab, valid, w


def make_collate(in_ch, hch):
    def collate(batch):
        crops, labs, valid, w = zip(*batch)
        x = to_tensor(crops, in_ch, hch)
        y = np.stack([l[..., :2] for l in labs]).astype(np.float32).transpose(0, 3, 1, 2)
        v = np.broadcast_to(np.stack(valid)[:, :, None, None], y.shape) * np.stack([l[..., 3] for l in labs]).astype(np.float32)[:, None]
        return x, torch.from_numpy(np.concatenate([y, v], 1).copy()), torch.from_numpy(np.stack(w).transpose(0, 3, 1, 2).copy())
    return collate


def val_set(items, jitter, seed=123):
    crops, labs, valid = [], [], []
    for k, it in enumerate(items):
        img, nuc, cell = aug_v2.load_item(it)
        box = aug_v2.jitter_box(aug_v2.base_box(img, cell), np.random.default_rng((seed, k))) if jitter else aug_v2.centred_box(img)
        c, l, v = aug_v2.make_sample(img, nuc, cell, box, SIZE)
        crops.append(c); labs.append(l); valid.append(v)
    return crops, labs, np.stack(valid)


@torch.no_grad()
def val_score(model, vs, in_ch, hch, device):
    crops, labs, valid = vs
    model.eval(); pr = []
    for b in range(0, len(crops), 128):
        pr.append(torch.sigmoid(model(to_tensor(crops[b:b + 128], in_ch, hch).to(device))).cpu().numpy())
    pr = np.concatenate(pr) > 0.5
    dn, dc = [], []
    for k in range(len(crops)):
        real = labs[k][..., 3] > 0
        for c, lst in ((0, dn), (1, dc)):
            if valid[k, c]:
                g = (labs[k][..., c] > 0) & real; p = pr[k, c] & real
                lst.append(2 * (p & g).sum() / max(p.sum() + g.sum(), 1))
    return float(np.mean(dn)), float(np.mean(dc))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=("V0", "V1", "V2", "V3"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3, help="peak OneCycle LR (the notebook's from-scratch value)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N train crops")
    ap.add_argument("--photometric", action="store_true", help="V4: aug_v2.photometric ON (HSV, HED, gamma, CLAHE, grey, blur, noise)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    in_ch = 4 if a.variant == "V3" else 3
    (out / "config.json").write_text(json.dumps({"in_ch": in_ch, "size": SIZE, "variant": a.variant, **({"photometric": True} if a.photometric else {})}))
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    split = json.load(open(HERE / "split_v3.json"))
    tr, va = items_for(split["train"], a.variant), items_for(split["val"], a.variant)
    if a.limit:
        tr = tr[:a.limit]
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = tinyseg.TinyUNet(in_ch=in_ch).to(device)
    hch = HChan() if in_ch == 4 else None
    t0 = time.time()
    vsets = {"plain": val_set(va, False), "jitter": val_set(va, True)}
    print(f"[{a.variant}] train {len(tr)} val {len(va)} x2 ({time.time() - t0:.0f}s), params {sum(p.numel() for p in model.parameters())}", flush=True)

    def score():
        r = {k: val_score(model, v, in_ch, hch, device) for k, v in vsets.items()}
        return float(np.mean([0.5 * n + 0.5 * c for n, c in r.values()])), r

    t2 = time.time()
    cache = preload(tr, a.workers)
    for k in (0, len(tr) // 2, len(tr) - 1):                     # cached arrays == aug_v2.load_item output
        ref = aug_v2.load_item(tr[k])
        assert all((x is None and y is None) or np.array_equal(x, y) for x, y in zip(cache[k], ref))
    print(f"[{a.variant}] preloaded {len(cache)} crops in {time.time() - t2:.0f}s", flush=True)
    ds = TrainSet(tr, a.seed, cache, a.photometric)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, collate_fn=make_collate(in_ch, hch),
                    drop_last=True, generator=torch.Generator().manual_seed(a.seed), persistent_workers=a.workers > 0,
                    worker_init_fn=lambda w: np.random.seed(a.seed + w))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(dl), pct_start=0.1)
    best, log = -1.0, open(out / "log.txt", "w")
    for ep in range(a.epochs):
        ds.epoch = ep
        model.train(); t1 = time.time(); tl = n = 0
        for x, m, w in dl:
            x, m, w = x.to(device, non_blocking=True), m.to(device, non_blocking=True), w.to(device, non_blocking=True)
            loss = tinyseg.loss_fn(model(x), m, w)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tl += loss.item() * len(x); n += len(x)
        if not np.isfinite(tl / max(n, 1)):                    # NaN / inf loss: stop, mark the run failed
            (out / "failed.json").write_text(json.dumps({"reason": "non-finite loss", "epoch": ep}))
            print(f"[{a.variant}] non-finite loss at epoch {ep}: abort", flush=True)
            sys.exit(3)
        s, r = score()
        star = ""
        if s > best:
            best, star = s, "*"; torch.save(model.state_dict(), out / "best.pt")
        line = (f"ep {ep:02d} loss {tl / n:.4f} " + " ".join(f"{k}: nuc {v[0]:.4f} cell {v[1]:.4f}" for k, v in r.items())
                + f"  score {s:.4f} {star} {time.time() - t1:.0f}s")
        print(line, flush=True); log.write(line + "\n"); log.flush()
    torch.save(model.state_dict(), out / "last.pt")
    (out / "done.json").write_text(json.dumps({"best_score": best, "epochs": a.epochs, "wall_s": round(time.time() - t0)}))
    print(f"best {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
