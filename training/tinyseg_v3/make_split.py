"""Phase 2 split (leak-free), from the Phase 1 exploration outputs. Read-only on all sources.

train/val: tinyseg_retrain only. Unit = same-cell-merged FOV group (explore/fov_groups.csv).
Held out (tests, never in train/val): T1 = clean 212, T1b = the 50 ZIP01 cells of "check the segmentation".
Every group that contains a T1 or T1b crop is removed from the train/val pool (the other crops of those groups are
not used at all). Remaining groups -> 85/15 by crop count, within each batch (b1/b2/b3), random.Random(42).
Crops with neither a nucleus nor a cell polygon are dropped. Checks: no group in two sets; no confirmed same-cell
pair (explore/near_verified.csv + near_within_lab.csv) across sets.
    python make_split.py  ->  split_v3.json, split_v3_manifest.json
"""
import json, random
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EX = HERE.parent / "explore"
CHK = Path.home() / "Downloads/Anupam_AI/check the segmentation/nc_ratio_ground_truth/ground_truth_annotations/cervical_segmentation_01"
SEED, FRAC = 42, 0.85


def main():
    g = pd.read_csv(EX / "fov_groups.csv"); g = g[g.source == "tinyseg_retrain"].copy()
    ann = pd.read_csv(EX / "ann_per_image.csv"); ann = ann[ann.source == "tinyseg_retrain"][["name", "n_nuc", "n_cell"]]
    g = g.merge(ann, on="name", how="left")
    t1 = sorted((EX / "clean_212.txt").read_text().split())
    t1b = sorted(p.stem + ".png" for p in CHK.glob("*.txt"))
    assert len(t1) == 212 and len(t1b) == 50 and set(t1 + t1b) <= set(g.name)
    g["batch"] = g.name.str.split("_").str[0]
    assert g.groupby("group").batch.nunique().max() == 1, "a merged group spans batches"
    held = set(g.group[g.name.isin(t1)]) | set(g.group[g.name.isin(t1b)])
    held_extra = g[g.group.isin(held) & ~g.name.isin(t1 + t1b)]
    pool = g[~g.group.isin(held) & ((g.n_nuc > 0) | (g.n_cell > 0))]
    unlabelled = g[~g.group.isin(held) & (g.n_nuc == 0) & (g.n_cell == 0)]
    train, val = [], []
    for b in sorted(pool.batch.unique()):
        p = pool[pool.batch == b]
        sizes = p.groupby("group").size()
        keys = sorted(sizes.index); random.Random(SEED).shuffle(keys)
        tot, acc = sizes.sum(), 0
        for k in keys:
            (train if acc < FRAC * tot else val).append(k)
            acc += sizes[k]
    tr, va = pool[pool.group.isin(train)], pool[pool.group.isin(val)]
    split = {"train": sorted(tr.name), "val": sorted(va.name), "t1_clean212": t1, "t1b_check_zip01": t1b}
    # leak checks
    where = {}
    for s, names in split.items():
        for n in names:
            where.setdefault(n, set()).add(s)
    grp = dict(zip(g.name, g.group))
    gsets = {}
    for s, names in split.items():
        for n in names:
            gsets.setdefault(grp[n], set()).add(s)
    bad_groups = {k: v for k, v in gsets.items() if len(v - {"t1_clean212", "t1b_check_zip01"}) and len(v) > 1}
    pairs = pd.concat([pd.read_csv(EX / "near_within_lab.csv")[["name_a", "name_b"]],
                       pd.read_csv(EX / "near_verified.csv").query("confirmed")[["name_a", "name_b"]]])
    leak = [(a, b) for a, b in zip(pairs.name_a, pairs.name_b)
            if a in where and b in where and not (where[a] & where[b]) and
            ({"train", "val"} & (where[a] | where[b]))]
    assert not bad_groups and not leak, (bad_groups, leak[:5])
    (HERE / "split_v3.json").write_text(json.dumps(split, indent=1))
    man = {"seed": SEED, "train_frac_by_crops": FRAC, "unit": "same-cell-merged FOV group (explore/fov_groups.csv)",
           "counts": {k: len(v) for k, v in split.items()},
           "groups": {"train": len(train), "val": len(val), "held_out": len(held)},
           "t1_and_t1b_overlap": sorted(set(t1) & set(t1b)),
           "held_out_groups_other_crops_not_used": len(held_extra),
           "unlabelled_dropped": len(unlabelled),
           "train_crops_with_nuc_and_cell": int(((tr.n_nuc > 0) & (tr.n_cell > 0)).sum()),
           "val_crops_with_nuc_and_cell": int(((va.n_nuc > 0) & (va.n_cell > 0)).sum()),
           "by_batch": {s: d.batch.value_counts().to_dict() for s, d in (("train", tr), ("val", va))},
           "leak_checks": {"groups_in_two_sets": 0, "same_cell_pairs_across_sets": 0, "pairs_checked": len(pairs)}}
    (HERE / "split_v3_manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
