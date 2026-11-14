"""COCO loading, grouped split and a cached 160x160 copy of the dataset.

Split: grouped by parent FOV (e.g. b1_cervical_0074) so no field leaks across train/val/test,
70/15/15 inside each batch (b1/b2/b3) so every batch is represented in every split.
"""
import json, os, collections, random
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from tinyseg import SIZE

NUC, CELL = 1, 2
ROOT = Path("/home/nvidia/tinyseg_retrain/data")
CACHE = Path("/home/nvidia/tinyseg_retrain/cache")


def load_coco(root=ROOT):
    d = json.load(open(root / "annotations" / "instances_default.json"))
    anns = collections.defaultdict(list)
    for a in d["annotations"]:
        anns[a["image_id"]].append(a)
    return [dict(id=im["id"], file=im["file_name"], w=im["width"], h=im["height"],
                 path=str(root / "images" / "default" / im["file_name"]), anns=anns.get(im["id"], []))
            for im in d["images"]]


def rasterize(item, cat):
    m = np.zeros((item["h"], item["w"]), np.uint8)
    for a in item["anns"]:
        if a["category_id"] != cat:
            continue
        for poly in a["segmentation"]:
            pts = np.round(np.array(poly, np.float32).reshape(-1, 2)).astype(np.int32)
            cv2.fillPoly(m, [pts], 1)
    return m


def has(item, cat):
    return any(a["category_id"] == cat for a in item["anns"])


def batch_of(fname):
    return fname.split("_")[0]


def fov_of(fname):
    # b1_cervical_0074_cell_6.png -> b1_cervical_0074
    return fname.split("_cell")[0]


def make_split(items, seed=42):
    groups = collections.defaultdict(list)
    for it in items:
        groups[fov_of(it["file"])].append(it["file"])
    split = {"train": [], "val": [], "test": []}
    for b in sorted({batch_of(k) for k in groups}):
        keys = sorted(k for k in groups if batch_of(k) == b)
        random.Random(seed).shuffle(keys)
        c1, c2 = int(0.70 * len(keys)), int(0.85 * len(keys))
        for i, k in enumerate(keys):
            split["train" if i < c1 else "val" if i < c2 else "test"] += groups[k]
    return {s: sorted(v) for s, v in split.items()}


def _small(it):
    img = cv2.resize(cv2.imread(it["path"]), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    m = np.zeros((SIZE, SIZE, 4), np.uint8)  # nuc, cell, nuc_valid, cell_valid
    if has(it, NUC): m[..., 0] = cv2.resize(rasterize(it, NUC), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST); m[..., 2] = 1
    if has(it, CELL): m[..., 1] = cv2.resize(rasterize(it, CELL), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST); m[..., 3] = 1
    return img, m


def build_cache(workers=12):
    """Writes cache/split.json and cache/{train,val,test}.npz (X: images at SIZE, M: masks + valid flags)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    items = [i for i in load_coco() if has(i, NUC) or has(i, CELL)]  # drop fully empty crops
    split = make_split(items)
    json.dump(split, open(CACHE / "split.json", "w"), indent=1)
    by = {i["file"]: i for i in items}
    with Pool(workers) as pool:
        for s, files in split.items():
            X, M = zip(*pool.map(_small, [by[f] for f in files], chunksize=32))
            np.savez(CACHE / f"{s}.npz", X=np.stack(X), M=np.stack(M), files=np.array(files))
            cnt = collections.Counter(batch_of(f) for f in files)
            print(f"{s}: {len(files)} crops, {len({fov_of(f) for f in files})} FOVs, {dict(sorted(cnt.items()))}")
    return split


def load_cache(split):
    z = np.load(CACHE / f"{split}.npz")
    return z["X"], z["M"], z["files"]


if __name__ == "__main__":
    build_cache()
