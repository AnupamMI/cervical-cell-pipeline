"""Fit train-only references and cache V1/V2 source crops (lossless PNG) for every crop in split_v3.json.
    python build_cache.py   ->  stain_ref.json, h_stats.json, cache/{V1,V2}/<name>.png, cache/fallbacks.json
"""
import json, random
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

import preprocess as pp

HERE = Path(__file__).resolve().parent
IMG = Path.home() / "Downloads/ANUPAM KA HOOD/seg_audit/extracted/tinyseg_retrain/tinyseg_retrain/data/images/default"
REF = None


def init(ref):
    global REF
    REF = ref


def job(name):
    bgr = cv2.imread(str(IMG / name))
    out = {}
    for v in ("V1", "V2"):
        im, ok = pp.transform_source(v, bgr, REF)
        assert cv2.imwrite(str(HERE / "cache" / v / name), im)
        out[v] = ok
    return name, out


def main():
    split = json.load(open(HERE / "split_v3.json"))
    sample = random.Random(42).sample(split["train"], 300)
    imgs = [cv2.imread(str(IMG / n)) for n in sample]
    ref = pp.fit_reference(imgs)
    pp.save_json(ref, HERE / "stain_ref.json")
    hs = np.concatenate([pp.h_channel(cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA)).ravel() for im in imgs])
    pp.save_json({"mean": float(hs.mean()), "std": float(hs.std()), "n_crops": len(imgs), "size": 224}, HERE / "h_stats.json")
    names = sorted({n for v in split.values() for n in v})
    for v in ("V1", "V2"):
        (HERE / "cache" / v).mkdir(parents=True, exist_ok=True)
    fb = {"V1": [], "V2": []}
    with Pool(10, initializer=init, initargs=(ref,)) as pool:
        for k, (n, ok) in enumerate(pool.imap_unordered(job, names, chunksize=32), 1):
            for v, o in ok.items():
                if not o:
                    fb[v].append(n)
            if k % 2000 == 0:
                print(k, len(names), flush=True)
    pp.save_json({"n": len(names), "fallback_counts": {v: len(x) for v, x in fb.items()}, "fallbacks": fb}, HERE / "cache/fallbacks.json")
    print("ref", ref, "\nH stats", json.load(open(HERE / "h_stats.json")), "\nfallbacks", {v: len(x) for v, x in fb.items()}, "of", len(names))


if __name__ == "__main__":
    main()
