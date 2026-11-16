#!/usr/bin/env python3
"""V4 vs v2_1 decision (rule set by the user, 2026-11-16):
  primary = nucleus Dice with YOLO26s boxes, mean of T1 and T2 (corrected+independent); 95 % CI = cluster bootstrap
  (2,000, seed 0; FOV groups resampled within each set) -- exactly Step 3's statistic (pick_winner.boot_avg).
  V4 replaces v2_1 only if its primary is HIGHER and its CI does not overlap v2_1's; otherwise v2_1 stays.
Reads results_v4/{per_cell.csv, summary.csv, RESULTS.md, eval.json}; writes results_v4/decision.json,
results/RESULTS_V4.md and (only if V4 wins) results_v4/README_HEAD_v04.md for the v0.4 release.
"""
import argparse, json, re, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "steps"))
from pick_winner import boot_avg, overlap  # noqa: E402

import os
R4 = Path(os.environ.get("R4_DIR", ROOT / "results_v4")); RES = Path(os.environ.get("RES_OUT", ROOT / "results"))


def train_summary(run):
    eps = []
    for l in (run / "log.txt").read_text().splitlines():
        m = re.match(r"ep (\d+) loss ([\d.na]+) .*score ([\d.]+)\s*\*?\s*(\d+)s", l)
        if m:
            eps.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4))))
    best = max(eps, key=lambda e: e[2])
    return {"run": str(run), "epochs": len(eps), "best_epoch": best[0], "best_val": best[2], "last_val": eps[-1][2],
            "s_per_epoch": round(sum(e[3] for e in eps[1:]) / max(1, len(eps) - 1)), "train_min": round(sum(e[3] for e in eps) / 60, 1),
            "still_rising": best[0] >= len(eps) - 5}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); a = ap.parse_args()
    run = Path(a.run) if Path(a.run).is_absolute() else ROOT / a.run
    ts = train_summary(run)
    tr = json.loads((RES / "train_report.json").read_text()) if (RES / "train_report.json").exists() else {}
    per = pd.read_csv(R4 / "per_cell.csv")
    per = per[(per.status == "scored") & (per.protocol == "det")]
    t1 = per[(per.set == "T1") & (per.subset == "all")]
    t2 = per[(per.set == "T2") & (per.subset == "corrected+independent")]
    st = {}
    for m in ("V4", "v2_1"):
        parts = [d[d.model == m][["group", "nuc_dice"]].rename(columns={"nuc_dice": "v"}).dropna() for d in (t1, t2)]
        p, lo, hi = boot_avg(parts)
        st[m] = {"primary": p, "lo": lo, "hi": hi, "t1": list(boot_avg([parts[0]])), "t2": list(boot_avg([parts[1]]))}
    v4, old = st["V4"], st["v2_1"]
    a4, ao = (v4["primary"], v4["lo"], v4["hi"]), (old["primary"], old["lo"], old["hi"])
    higher, ov = a4[0] > ao[0], overlap(a4, ao)
    if higher and not ov:
        winner = "V4"; why = (f"V4 replaces v2_1: primary {a4[0]:.4f} [{a4[1]:.4f}, {a4[2]:.4f}] > v2_1 {ao[0]:.4f} "
                              f"[{ao[1]:.4f}, {ao[2]:.4f}] and the CIs do not overlap.")
    else:
        winner = "v2_1"
        why = (f"v2_1 stays: V4 primary {a4[0]:.4f} [{a4[1]:.4f}, {a4[2]:.4f}] vs v2_1 {ao[0]:.4f} [{ao[1]:.4f}, {ao[2]:.4f}] — "
               + ("V4 is not higher" if not higher else "V4 is higher but the CIs overlap (difference not established)")
               + ("; and the CIs overlap" if (not higher and ov) else "") + ".")
    dec = {"winner": winner, "reason": why, "higher": bool(higher), "ci_overlap": bool(ov), "stats": st, "train": ts}
    (R4 / "decision.json").write_text(json.dumps(dec, indent=1))
    S = pd.read_csv(R4 / "summary.csv")
    v0 = tr.get("V0", {})
    L = ["# TinyUNet V4 vs v2_1", "",
         "**V4** = the V0 recipe (raw RGB, 224 px, crop +0.17, mirror padding, from scratch, 40 epochs, seed 42, the leak-free "
         "v3 split) **with `aug_v2.photometric` fully on** (HSV, HED stain jitter, gamma, CLAHE, greyscale, Gaussian/defocus "
         "blur, noise; applied after the geometric augmentation, exactly as in `aug_v2.train_sample`). Only change vs V0: "
         "`train_v3.py --photometric`. Evaluated exactly as Step 2 (`eval_final.py --models V4,v2_1`).", "",
         "## Decision", "", f"**{winner}** — {why}", "",
         "Rule: V4 replaces v2_1 only if its primary score (nucleus Dice with YOLO26s boxes, mean of T1 and T2 corrected+"
         "independent) is higher AND its 95 % CI does not overlap v2_1's.", "",
         "| model | primary | 95 % CI | T1 | T2 corr+indep |", "|---|---|---|---|---|"]
    for m in ("V4", "v2_1"):
        s = st[m]
        L.append(f"| {m} | {s['primary']:.4f} | [{s['lo']:.4f}, {s['hi']:.4f}] | {s['t1'][0]:.4f} [{s['t1'][1]:.4f}, {s['t1'][2]:.4f}] | "
                 f"{s['t2'][0]:.4f} [{s['t2'][1]:.4f}, {s['t2'][2]:.4f}] |")
    L += ["", "Context (Step 2, same evaluation): V0 (no photometric aug) primary 0.8538 [0.8311, 0.8743]; T2 corr+indep 0.7819.", "",
          "## V4 training", "",
          "| run | epochs | best epoch | best val score | last | mean s/epoch | train min | still rising at end? |", "|---|---|---|---|---|---|---|---|",
          f"| `{Path(ts['run']).name}` | {ts['epochs']} | {ts['best_epoch']} | {ts['best_val']:.4f} | {ts['last_val']:.4f} | {ts['s_per_epoch']} | "
          f"{ts['train_min']} | {'yes' if ts['still_rising'] else 'no'} |", "",
          (f"V0 for comparison: best val {v0['best_score']:.4f} at epoch {v0['best_epoch']}, {v0['median_s_per_epoch']} s/epoch (median). "
           if v0.get("best_score") is not None else "")
          + "The validation set is the v3 val split *without* photometric changes, so V4's val score is not expected to rise "
          "vs V0; the test sets decide.", "",
          "## Full tables (from `results_v4/RESULTS.md`)", ""]
    body = (R4 / "RESULTS.md").read_text()
    body = body.split("\n", 1)[1] if body.startswith("# ") else body
    body = re.sub(r"## Training\n\n\(train report missing\)\n\n", "", body)
    L += [body.replace("tinyseg_v3/results/overlays/", "tinyseg_v3/results_v4/overlays/"), ""]
    (RES / "RESULTS_V4.md").write_text("\n".join(L) + "\n")
    if winner == "V4":
        ms = S[(S.protocol == "det") & (S.subset == "all") & (S.set == "T2")].set_index("model").ms_median.to_dict()
        def row(m, t):
            return {x: S[(S.protocol == "det") & (S.set == t[0]) & (S.subset == t[1]) & (S.model == m)].iloc[0] for x in [0]}[0]
        h = (ROOT / "release_v03/README_HEAD_v03.md").read_text()
        h = h.replace("# cervical-cell-pipeline (v0.3)", "# cervical-cell-pipeline (v0.4)")
        first_end = h.find("## Architecture")
        intro = ("Masking pipeline for Pap-smear cervical cytology FOVs (2028x1520, ~0.155 µm/px). **v0.4** = pipeline v3 (two-layer\n"
                 "masks, image QC, measurable/review/excluded gate) with the **TinyUNet V4** segmenter (`models/seg/V4_v3/best.pt`:\n"
                 "raw RGB, photometric augmentation, trained on the leak-free v3 split). v2_1 and V1 stay available as config options.\n"
                 "Older versions: `git checkout v0.3` (v2_1 default), `v0.2` (V1), `v0.1` (pipeline v1).\n\n")
        h = h[:h.find("\n") + 2] + intro + h[first_end:]
        cells = []
        for t in (("T1", "all"), ("T1b", "all"), ("T2", "all"), ("T2", "corrected+independent")):
            r = row("V4", t); cells.append(f"{r.nuc_dice:.3f} [{r.nuc_dice_lo:.3f}, {r.nuc_dice_hi:.3f}]")
        r2 = row("V4", ("T2", "corrected+independent"))
        v4row = (f"| **V4 (default)** | raw RGB, photometric aug, leak-free split | **{v4['primary']:.3f}** [{v4['lo']:.3f}, {v4['hi']:.3f}] | "
                 + " | ".join(cells) + f" | {r2.nc_abs_err:.4f} | {ms.get('V4', float('nan')):.0f} |\n")
        h = h.replace("| **v2_1 (default)** |", v4row + "| v2_1 (option) |")
        h = h.replace("**Why v2_1 is the default (v0.3).**", f"**Why V4 is the default (v0.4).** {why} V4 is trained on the leak-free "
                      "split, so unlike v2_1 none of its T1/T1b cells (or same-cell twins) were seen in training. See "
                      "`reports/RESULTS_V4.md`.\n\n**Why v2_1 was the default in v0.3.**")
        h = h.replace("V4 (V0 recipe on the leak-free split\nwith photometric augmentation fully on) is the follow-up experiment: `reports/RESULTS_V4.md` once available.",
                      f"V4 (the V0 recipe with photometric augmentation fully on): T2 corr+indep {v4['t2'][0]:.3f} vs V0 0.782 without it.")
        h = h.replace("Production sheets for v0.3: `reports/sheets_v0.3_1fov/`, `reports/sheets_v0.3_5fov/`",
                      "Production sheets for v0.4: `reports/sheets_v0.4_1fov/`, `reports/sheets_v0.4_5fov/`; v0.3 (v2_1): `reports/sheets_v0.3_*`")
        h = h.replace("**Use V1 instead:**", "**Use v2_1 instead:** `seg.weights: models/seg/v2_1/best.pt`, `seg.variant: auto`.\n**Use V1 instead:**")
        (R4 / "README_HEAD_v04.md").write_text(h)
    print(json.dumps({k: dec[k] for k in ("winner", "reason")}, indent=1))


if __name__ == "__main__":
    main()
