#!/usr/bin/env python3
"""Step 6: assemble ~/Downloads/ANUPAM KA HOOD/MORNING_REPORT.md from whatever the night produced (never assumes a
step succeeded) and copy it to the repo as reports/MORNING_REPORT.md (the shell wrapper commits + pushes it)."""
import json, re, subprocess
from datetime import datetime
from pathlib import Path

K = Path.home() / "Downloads/ANUPAM KA HOOD"
ROOT = K / "tinyseg_v3"; RES = ROOT / "results"; ST = ROOT / "overnight_status"
REPO = Path.home() / "Downloads/cervical-cell-pipeline"
V3 = K / "nc_mask_pipeline_v3"
OUT = K / "MORNING_REPORT.md"


def read(p, default=""):
    p = Path(p)
    return p.read_text() if p.exists() else default


def status(step):
    p = ST / f"{step}.json"
    if not p.exists():
        return {"state": "not run", "reason": ""}
    return json.loads(p.read_text())


def section(md, start, stop_prefix="## "):
    """Text of RESULTS.md from heading `start` to the next heading of the same level."""
    i = md.find(start)
    if i < 0:
        return ""
    j = md.find("\n" + stop_prefix, i + len(start))
    return md[i:j if j > 0 else None]


def git(*a):
    r = subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True)
    return (r.stdout or r.stderr).strip()


def step_times(log):
    t = {}
    for m in re.finditer(r"^\[(\S+ \S+)\] (START|STATUS) (step\w+)", log, re.M):
        t.setdefault(m.group(3), {})[m.group(2)] = m.group(1)
    return t


def main():
    log = read(ROOT / "overnight.log")
    steps = ["step0_push_v01", "step1_train", "step2_eval", "step3_winner", "step4_pipeline_v3", "step5_push_v02", "step6_morning_report"]
    times = step_times(log)
    first = re.search(r"^\[(\S+ \S+)\]", log, re.M)
    L = [f"# Morning report — {datetime.now():%Y-%m-%d %H:%M}", "",
         f"Overnight run started {first.group(1) if first else '?'}; log: `tinyseg_v3/overnight.log`.", "", "## What ran", "",
         "| step | state | reason / note | started | finished |", "|---|---|---|---|---|"]
    for s in steps:
        st = status(s) if s != "step6_morning_report" else {"state": "ok (this report)", "reason": ""}
        L.append(f"| {s} | {st['state']} | {st.get('reason', '')} | {times.get(s, {}).get('START', '')} | {times.get(s, {}).get('STATUS', st.get('time', '')[:19].replace('T', ' '))} |")
    fails = [s for s in steps[:-1] if status(s)["state"] not in ("ok",)]
    L += ["", ("**Failed / skipped:** " + ", ".join(f"{s} ({status(s)['state']}: {status(s).get('reason', '')})" for s in fails)) if fails
          else "All steps succeeded.", ""]
    # training
    L += ["## Training (V0–V3, 40 epochs, seed 42, from scratch, v2_1 recipe, no colour/blur/noise augmentation)", "",
          read(RES / "train_report.md", "(no training report)"), ""]
    tr = json.loads(read(RES / "train_report.json", "{}") or "{}")
    for v, r in tr.items():
        if r.get("status") == "ok" and r.get("val_still_rising_at_end"):
            L.append(f"- {v}: validation still rising at the end (best epoch {r['best_epoch']} of {r['epochs']}): more epochs could help.")
    L += ["", "Training log lines with timings: `grep training tinyseg_v3/overnight.log`.", ""]
    # T2 independence
    t2 = json.loads(read(ROOT / "t2_verify/t2_verify_summary.json", "{}") or "{}")
    if t2:
        L += ["## T2 independence check (your 92 cells vs NCNet V3's saved masks; NCNet never run)", "",
              f"- classes by your Dice rule: **{t2['class_counts']}**",
              f"- masks == polygons (rasterisation check): {t2['raster_check']}",
              f"- median share of polygon vertices on NCNet's mask boundary: cell {t2['cell_vtx_on_ncnet_raw']['median']:.1%}, "
              f"nucleus {t2['nucleus_vtx_on_ncnet_raw']['median']:.1%} (hand-drawn ZIP01 control ≈ 28 % / 36 %)",
              "- report: `tinyseg_v3/t2_verify/T2_VERIFY_REPORT.md`", ""]
    # results
    md = read(RES / "RESULTS.md")
    if md:
        L += ["## Results (YOLO26s boxes; full tables incl. GT boxes in `tinyseg_v3/results/RESULTS.md`)", ""]
        for h in re.findall(r"^### .*YOLO26s boxes$", md, re.M):
            L += [section(md, h, "### ").strip(), ""]
        w = section(md, "## Winner")
        L += [w.strip() if w else "## Winner\n\n(no winner: step 3 did not run)", ""]
    else:
        L += ["## Results", "", "(no results: step 2 did not run)", ""]
    # pipeline v3
    L += ["## Pipeline v3 (`nc_mask_pipeline_v3`)", ""]
    L += [read(V3 / "reports/STEP4_SUMMARY.md", "(step 4 did not produce a summary)"), ""]
    # git
    L += ["## Git", "", "```", git("log", "--oneline", "--decorate", "-8"), "```", "",
          f"- tags on origin: {', '.join(x.split('/')[-1] for x in git('ls-remote', '--tags', 'origin').split() if x.startswith('refs/tags/') and not x.endswith('^{}'))}",
          "- release with YOLO26l: https://github.com/AnupamMI/cervical-cell-pipeline/releases/tag/v0.1-models",
          f"- v0.2 release (V0–V3 weights, if pushed): https://github.com/AnupamMI/cervical-cell-pipeline/releases/tag/v0.2-models",
          f"- working tree: {git('status', '--short') or 'clean'}", ""]
    v02 = status("step5_push_v02")["state"] == "ok"
    tag = "v0.2" if v02 else "v0.1"
    L += ["## Run it on Colab (T4) tomorrow", "",
          "1. Colab → open notebook → GitHub → `AnupamMI/cervical-cell-pipeline` (private: sign in), file `notebooks/colab_quickstart.ipynb`.",
          "2. Runtime → Change runtime type → **T4 GPU**.",
          "3. Key icon (Secrets) → add `GH_TOKEN` = a GitHub token with read access to the repo → enable notebook access.",
          "4. Runtime → Run all. It clones the repo, installs `requirements-colab.txt` (keeps Colab's torch), fetches the YOLO26l "
          "weights from the release, mounts Drive (outputs: `MyDrive/cervical-cell-pipeline/out/`), runs the pipeline on `samples/`, "
          "shows a 3-panel sheet and runs the match check (expect PASS).",
          f"5. The repo's default branch is the latest version (`{tag}` tagged). To use exactly v0.1: `git checkout v0.1` in the clone.",
          "6. Your own FOVs: upload to Drive, then `!python run.py --input /content/drive/MyDrive/<your folder> --output "
          "/content/drive/MyDrive/cervical-cell-pipeline/out/<run>`.", "",
          "## Open questions", "",
          "1. T2 is crop-based (no source FOVs): if the FOVs ever turn up, the T2 numbers should be redone from FOVs.",
          "2. T2's vertex analysis says most outlines sit exactly on NCNet's boundary; read T2 results as agreement with those outlines.",
          "3. Macenko is a two-stain (H&E) model applied to Pap stain; 2 % of crops fall back to unchanged.",
          "4. Anything flagged as failed above.", ""]
    OUT.write_text("\n".join(L) + "\n")
    if REPO.exists():
        (REPO / "reports").mkdir(exist_ok=True)
        (REPO / "reports/MORNING_REPORT.md").write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
