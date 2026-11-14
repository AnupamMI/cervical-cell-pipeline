#!/usr/bin/env python3
"""Step 3: pick the winning segmentation model (rules approved by the user):

1. Primary = nucleus Dice with DETECTOR boxes, averaged over T1 and T2 (corrected+independent)
   (T1 only if the T2 evaluation failed). Its 95 % CI: cluster bootstrap (2,000, seed 0) resampling FOV groups
   within each set independently, statistic = mean of the two set means.
2. Highest primary wins among V0-V3; ties (equal to 3 decimals) broken by N:C MAE (lower), then cell Dice (higher).
3. A variant beats V0 only if its CI does not overlap V0's CI; otherwise V0 is chosen.
4. If that model is clearly worse than the old v2_1 on T1 or on T2 (corrected+independent) — nucleus Dice with
   detector boxes, non-overlapping CIs, lower — the old v2_1 is kept.
Writes results/winner.json and appends '## Winner' to results/RESULTS.md.
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

import os
ROOT = Path(__file__).resolve().parents[1]; RES = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
P2 = ROOT / "phase2"
NCM = Path.home() / "Downloads/ANUPAM KA HOOD/nc_mask_pipeline"


def boot_avg(parts, n=2000, seed=0):
    """parts: list of DataFrames (group, value) -> (point, lo, hi) of the mean of per-part means."""
    rng = np.random.default_rng(seed)
    point = float(np.mean([p.v.mean() for p in parts]))
    by = [{g: d.v.values for g, d in p.groupby("group")} for p in parts]
    stats = []
    for _ in range(n):
        ms = []
        for b in by:
            keys = list(b); pick = rng.choice(len(keys), len(keys))
            ms.append(np.concatenate([b[keys[i]] for i in pick]).mean())
        stats.append(np.mean(ms))
    return point, float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def overlap(a, b):
    return a[2] >= b[1] and a[1] <= b[2]


def run_dir(v):
    if v == "v2_1":
        return str(NCM / "models/seg/v2_1")
    for d in (P2 / "runs" / v, P2 / "runs" / f"{v}_r1", P2 / "runs" / f"{v}_r2"):
        if (d / "done.json").exists():
            return str(d)
    return None


def main():
    per = pd.read_csv(RES / "per_cell.csv")
    per = per[(per.status == "scored") & (per.protocol == "det")]
    t1 = per[(per.set == "T1") & (per.subset == "all")]
    t2 = per[(per.set == "T2") & (per.subset == "corrected+independent")]
    use_t2 = len(t2) > 0
    models = [m for m in ("V0", "V1", "V2", "V3", "v2_1") if (per.model == m).any()]
    notes = []
    if not use_t2:
        notes.append("T2 evaluation missing: decision on T1 only.")
    stats = {}
    for m in models:
        parts = [t1[t1.model == m][["group", "nuc_dice"]].rename(columns={"nuc_dice": "v"})]
        if use_t2:
            parts.append(t2[t2.model == m][["group", "nuc_dice"]].rename(columns={"nuc_dice": "v"}))
        pt, lo, hi = boot_avg([p.dropna() for p in parts])
        sub = pd.concat([t1[t1.model == m]] + ([t2[t2.model == m]] if use_t2 else []))
        stats[m] = {"primary": pt, "lo": lo, "hi": hi, "nc_mae": float(sub.nc_abs_err.mean()), "cell_dice": float(sub.cell_dice.mean()),
                    "t1": [float(x) for x in boot_avg([parts[0].dropna()])],
                    "t2": [float(x) for x in boot_avg([parts[1].dropna()])] if use_t2 else None}
    cands = [m for m in ("V0", "V1", "V2", "V3") if m in stats]
    if "V0" not in cands:
        sys.exit("V0 not evaluated: no decision")
    order = sorted(cands, key=lambda m: (-round(stats[m]["primary"], 3), stats[m]["nc_mae"], -stats[m]["cell_dice"]))
    top = order[0]
    v0 = (stats["V0"]["primary"], stats["V0"]["lo"], stats["V0"]["hi"])
    tt = (stats[top]["primary"], stats[top]["lo"], stats[top]["hi"])
    if top == "V0":
        winner = "V0"; notes.append("V0 has the highest primary score.")
    elif overlap(tt, v0):
        winner = "V0"; notes.append(f"{top} has the highest primary score ({tt[0]:.4f}) but its CI [{tt[1]:.4f}, {tt[2]:.4f}] overlaps "
                                    f"V0's [{v0[1]:.4f}, {v0[2]:.4f}]: not established as better, so V0 is chosen.")
    else:
        winner = top; notes.append(f"{top} beats V0: primary {tt[0]:.4f} [{tt[1]:.4f}, {tt[2]:.4f}] vs V0 {v0[0]:.4f} [{v0[1]:.4f}, {v0[2]:.4f}], no overlap.")
    final = winner
    if "v2_1" in stats:
        worse = []
        for key, name in (("t1", "T1"), ("t2", "T2 corrected+independent")):
            w, o = stats[winner][key], stats["v2_1"][key]
            if w and o and w[0] < o[0] and not overlap(w, o):
                worse.append(f"{name}: {winner} {w[0]:.4f} [{w[1]:.4f}, {w[2]:.4f}] < v2_1 {o[0]:.4f} [{o[1]:.4f}, {o[2]:.4f}]")
        if worse:
            final = "v2_1"; notes.append(f"{winner} is clearly worse than the old v2_1 ({'; '.join(worse)}), so the old v2_1 is kept.")
        else:
            notes.append(f"{winner} is not clearly worse than the old v2_1 on T1 or T2 (CIs overlap or {winner} is higher).")
    out = {"winner": final, "best_new_variant": winner, "run_dir": run_dir(final), "decided_on": "T1 + T2 corrected+independent" if use_t2 else "T1 only",
           "stats": stats, "order": order, "notes": notes}
    (RES / "winner.json").write_text(json.dumps(out, indent=1))
    L = ["", "## Winner", "",
         f"**{final}**" + (" (old model kept)" if final == "v2_1" else "") + f" — decided on {out['decided_on']}.", "",
         "Primary = nucleus Dice with YOLO26s boxes, mean of the set means; CI = cluster bootstrap within each set.", "",
         "| model | primary | 95 % CI | T1 | T2 corr+indep | N:C MAE | cell Dice |", "|---|---|---|---|---|---|---|"]
    for m in models:
        s = stats[m]; t2s = f"{s['t2'][0]:.4f} [{s['t2'][1]:.4f}, {s['t2'][2]:.4f}]" if s["t2"] else "n/a"
        L.append(f"| {m}{' (old)' if m == 'v2_1' else ''} | {s['primary']:.4f} | [{s['lo']:.4f}, {s['hi']:.4f}] | "
                 f"{s['t1'][0]:.4f} [{s['t1'][1]:.4f}, {s['t1'][2]:.4f}] | {t2s} | {s['nc_mae']:.4f} | {s['cell_dice']:.4f} |")
    L += ["", "Reasoning:", ""] + [f"- {n}" for n in notes] + [""]
    with open(RES / "RESULTS.md", "a") as f:
        f.write("\n".join(L) + "\n")
    print(json.dumps({k: out[k] for k in ("winner", "best_new_variant", "decided_on", "notes")}, indent=1))


if __name__ == "__main__":
    main()
