#!/usr/bin/env python3
"""Step 5: v0.2 on top of v0.1 in ~/Downloads/cervical-cell-pipeline, pushed only if everything passes.

All work happens on a local branch v0.2-wip (main and the v0.1 tag are not touched until the end):
  1. pipeline v3 code (ncmask, run.py, config.yaml with the winner, tests, tools, vendor incl. tinyseg_v3 preprocess)
  2. training/tinyseg_v3: train/eval/preprocess/split scripts + overnight orchestration; splits/ (names only)
  3. models/: the winner (as configured) + V0-V3 bests with config.json, preprocessing files, sha256.json; WEIGHTS.md
  4. samples/expected regenerated with v0.2 (the reference for the match check); README, notebook, reports/
     (RESULTS.md, STEP4_SUMMARY.md, 1-FOV/5-FOV sheets as JPEG)
  5. fresh clone of v0.2-wip (local) -> fetch release weights -> run.py --input samples/ --limit 2 with CUDA and with
     CUDA_VISIBLE_DEVICES="" -> check_against_samples (the notebook's match check) must PASS for both
  6. fast-forward main, push main, tag v0.2, push tag.  Any failure before 6: main/v0.1 untouched, wip branch kept.
"""
import json, os, re, shutil, subprocess, sys, time
from pathlib import Path

K = Path.home() / "Downloads/ANUPAM KA HOOD"
REPO = Path.home() / "Downloads/cervical-cell-pipeline"
V3 = K / "nc_mask_pipeline_v3"; T = K / "tinyseg_v3"
RES = Path(os.environ.get("RESULTS_DIR", T / "results"))
PREFIX = os.environ.get("STEP4_PREFIX", "")        # dry runs use the dry-run step-4 outputs
DRY = os.environ.get("STEP5_DRY") == "1"           # dry run: everything except push/tag (main untouched)
PY = os.environ.get("PY", str(K / "seg_audit/venv/bin/python"))
SMOKE = Path("/tmp/ccp_smoke_v02")


def sh(cmd, cwd=REPO, check=True, env=None, timeout=3600):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env, timeout=timeout)
    out = (r.stdout + r.stderr).strip()
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))[:200]} -> rc {r.returncode}: {out[-1500:]}")
    return out


def git(*a, **kw):
    return sh(["git", *a], **kw)


def copytree(src, dst):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))


def gh_token():
    out = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n", capture_output=True, text=True).stdout
    return dict(l.split("=", 1) for l in out.splitlines() if "=" in l).get("password", "")


def main():
    w = json.loads((RES / "winner.json").read_text())
    if git("status", "--porcelain"):
        raise RuntimeError("repo working tree not clean")
    git("fetch", "-q", "origin")
    base = git("rev-parse", "origin/main")
    git("checkout", "-q", "-B", "v0.2-wip", base)
    # 1. pipeline v3 code
    for d in ("ncmask", "tests", "tools", "vendor"):
        copytree(V3 / d, REPO / d)
    for f in ("run.py", "config.yaml", "requirements.txt", "requirements-colab.txt", "requirements-cpu.txt"):
        shutil.copy2(V3 / f, REPO / f)
    # cross-device check for v3 references (status must match; per-cell floor on kept cells) - decision 2026-11-15 00:4x
    shutil.copy2(V3 / "scripts/check_against_samples.py", REPO / "scripts/check_against_samples.py")
    # 3. models: as configured in v3 (seg.weights) + V0-V3
    cfg = (V3 / "config.yaml").read_text()
    seg_rel = re.search(r"^  weights: (models/seg/\S+)", cfg, re.M).group(1)
    for d in (V3 / "models/seg").iterdir():
        if d.is_dir():
            copytree(d, REPO / "models/seg" / d.name)
    tv = REPO / "models/seg/tinyseg_v3"; tv.mkdir(parents=True, exist_ok=True)
    tr = json.loads((RES / "train_report.json").read_text())
    for v, r in tr.items():
        if r.get("status") != "ok":
            continue
        src = Path(r["run_dir"]); dst = tv / v; dst.mkdir(exist_ok=True)
        for f in ["best.pt", "config.json", "sha256.json", "args.json"] + r.get("preprocessing_files", []):
            if (src / f).exists():
                shutil.copy2(src / f, dst / f)
    # 2. training code + splits (names only)
    tdst = REPO / "training/tinyseg_v3"; tdst.mkdir(parents=True, exist_ok=True)
    for f in ("train_v3.py", "preprocess.py", "eval_v3.py", "eval_final.py", "make_split.py", "build_cache.py", "run_all.sh"):
        shutil.copy2(T / "phase2" / f, tdst / f)
    copytree(T / "steps", tdst / "overnight_steps")
    shutil.copy2(T / "overnight.sh", tdst / "overnight.sh")
    (tdst / "splits").mkdir(exist_ok=True)
    for f in ("split_v3.json", "split_v3_manifest.json", "stain_ref.json", "h_stats.json"):
        shutil.copy2(T / "phase2" / f, tdst / "splits" / f)
    for f in ("verify_t2.py", "control.py", "T2_VERIFY_REPORT.md"):
        shutil.copy2(T / "t2_verify" / f, tdst / f)
    # WEIGHTS.md
    rows = []
    for p in sorted(REPO.glob("models/**/*.pt")) + sorted((REPO / "models").glob("**/*.json")):
        if "detector_b" in p.parts:
            continue
        rel = p.relative_to(REPO)
        rows.append(f"| `{rel}` | {p.stat().st_size} | `{sh(['sha256sum', str(p)]).split()[0]}` |")
    (REPO / "models/WEIGHTS.md").write_text(
        "# Model weights and preprocessing files\n\n"
        f"Configured segmenter (config.yaml `seg.weights`): `{seg_rel}` — winner **{w['winner']}**.\n\n"
        "YOLO26l (`models/detector_b/yolo26l_1280_best.pt`, 105,729,697 bytes, SHA256 "
        "`703f3e716a5116b98d581497eed6bab62b49181577aeb2062f209cbb230fb72b`) is a GitHub Release asset "
        "(`v0.1-models`): `scripts/fetch_release_weights.py`.\n\n| file | bytes | SHA256 |\n|---|---|---|\n" + "\n".join(rows) + "\n")
    # 4. reports + regenerated sample references
    rep = REPO / "reports"; rep.mkdir(exist_ok=True)
    for f in ("RESULTS.md", "train_report.md"):
        if (RES / f).exists():
            shutil.copy2(RES / f, rep / f)
    shutil.copy2(V3 / f"reports/{PREFIX}STEP4_SUMMARY.md", rep / "STEP4_SUMMARY.md")
    import cv2
    for sd in sorted((V3 / "reports").glob("sheets_*")):
        if sd.name.startswith("sheets_dryrun") != bool(PREFIX):
            continue
        out = rep / sd.name; out.mkdir(exist_ok=True)
        for png in sd.glob("*.png"):
            cv2.imwrite(str(out / (png.stem + ".jpg")), cv2.imread(str(png)), [cv2.IMWRITE_JPEG_QUALITY, 92])
        shutil.copy2(sd / "summary.md", out / "summary.md")
    exp = REPO / "samples/expected"
    old = REPO / f"samples/expected_v0.1_tmp_{int(time.time())}"
    exp.rename(old)
    try:
        sh([PY, "run.py", "--input", "samples", "--output", "samples/expected"], timeout=3600)
    except Exception:
        old.rename(exp); raise
    shutil.rmtree(old)                                       # v0.1's reference stays in git history (tag v0.1)
    write_readme(w, seg_rel)
    update_notebook()
    git("add", "-A")
    msg = (f"v0.2: pipeline v3 (two layers, gate, image QC) + TinyUNet v3 {w['winner']}\n\n"
           f"- segmenter: {seg_rel} ({w['winner']}); TinyUNet v3 V0-V3 trained from scratch (40 epochs, seed 42)\n"
           "- results: reports/RESULTS.md; sheets: reports/sheets_1fov, reports/sheets_5fov\n"
           "- training/tinyseg_v3: train/eval/preprocess/split scripts, splits (names only), overnight orchestration\n\n"
           "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
    subprocess.run(["git", "commit", "-q", "-F", "-"], cwd=REPO, input=msg, text=True, check=True)
    # 5. fresh-clone smoke test from the wip branch
    if SMOKE.exists():
        shutil.rmtree(SMOKE)
    sh(["git", "clone", "-q", "--branch", "v0.2-wip", str(REPO), str(SMOKE)], cwd=Path.home())
    env = dict(os.environ, GH_TOKEN=gh_token())
    print("[step5]", sh([PY, "scripts/fetch_release_weights.py"], cwd=SMOKE, env=env))
    env.pop("GH_TOKEN")
    res = {}
    for name, extra in (("cuda", {}), ("cpu", {"CUDA_VISIBLE_DEVICES": ""})):
        e = dict(os.environ, **extra)
        log = sh([PY, "run.py", "--input", "samples/", "--output", f"out_{name}/", "--limit", "2"], cwd=SMOKE, env=e)
        dev = re.search(r"device=(\w+)", log)
        chk = sh([PY, "scripts/check_against_samples.py", "--output", f"out_{name}/", "--limit", "2"], cwd=SMOKE, env=e, check=False)
        res[name] = {"device": dev.group(1) if dev else "?", "check": chk.splitlines()[-1]}
        print(f"[step5] smoke {name}: device {res[name]['device']}, {res[name]['check']}", flush=True)
        if not chk.splitlines()[-1].startswith("PASS"):
            raise RuntimeError(f"smoke test {name} failed: {chk[-800:]}")
    # 6. publish
    if DRY:
        git("checkout", "-q", "main")
        print("[step5] DRY RUN: built + smoke-tested v0.2-wip; not merged, not pushed, not tagged", flush=True)
        return
    git("checkout", "-q", "main")
    git("merge", "-q", "--ff-only", "v0.2-wip")
    git("tag", "-a", "v0.2", "-m", f"v0.2: pipeline v3 + TinyUNet v3 {w['winner']}; fresh-clone smoke CUDA/CPU PASS")
    git("push", "-q", "origin", "main", timeout=1800)
    git("push", "-q", "origin", "v0.2", timeout=600)
    (RES / "step5.json").write_text(json.dumps({"commit": git("rev-parse", "--short", "HEAD"), "tag": "v0.2", "smoke": res}, indent=1))
    print("[step5] pushed main + tag v0.2", flush=True)


def write_readme(w, seg_rel):
    src = (REPO / "README.md").read_text()
    res = (RES / "RESULTS.md").read_text() if (RES / "RESULTS.md").exists() else ""
    win = res[res.find("## Winner"):] if "## Winner" in res else "(see reports/RESULTS.md)"
    head = f"""# cervical-cell-pipeline (v0.2)

Masking pipeline for Pap-smear cervical cytology FOVs (2028x1520, ~0.155 µm/px). **v0.2** = pipeline v3 with the
TinyUNet v3 segmenter `{seg_rel}` (winner: **{w['winner']}**). v0.1 (pipeline v1 + TinyUNet v2_1) stays available:
`git checkout v0.1`.

## Architecture

```
Stage 0  image QC (blur, exposure, stain range)            -> FOV flags only, never skipped
Stage 1  YOLO26s cell boxes (conf 0.15)                     -> the only source of candidate cells
         YOLO26l @1280 cross-check (consensus_flag)         -> tag consensus / s_only, never filters (auto-off without weights)
Stage 2  containment suppression (0.8, min area ratio 0.4), cell_at_edge (box <= 2 px from the FOV edge),
         crop_clamped (info), box size vs the training distribution
Stage 3  crop = box + 0.17 margin, mirror padding -> input variant (as trained) -> TinyUNet -> centre component,
         nucleus within the cell
Stage 4  two layers: FULL per-cell masks (measurement; overlaps allowed) + EXCLUSIVE label map (display);
         occluded_frac; no cell dropped for overlap
Stage 5  gate: measurable / review / excluded + reason codes (thresholds in config.yaml)
outputs  cells.csv, cells_full.coco.json + cells.coco.json (exclusive), label maps, overlay, dropped.csv (Stage 2),
         l_only.csv, run_manifest.json (config, model + preprocessing SHA256s, versions)
```

## Results summary

{win.strip()}

Full tables (T1 clean 212, T1b 50 ZIP01, T2 92 NC003 crop-based; GT and YOLO26s boxes; 95 % CIs): `reports/RESULTS.md`.
Production sheets: `reports/sheets_1fov/`, `reports/sheets_5fov/` (left YOLO26s boxes, middle masks, right status).

## Retrain TinyUNet v3

`training/tinyseg_v3/`: `make_split.py` (leak-free split: same-cell-merged FOV groups; name lists in `splits/`),
`build_cache.py` (Macenko reference / H stats from train crops only), `train_v3.py --variant V0|V1|V2|V3 --epochs 40`
(v2_1 recipe from scratch, no colour augmentation), `eval_final.py` (all test sets), `overnight.sh` (the unattended
orchestration used for this release). The training images themselves are not in the repo (tinyseg_retrain data).

"""
    rest = src[src.find("## Quick start"):] if "## Quick start" in src else src
    rest = rest.replace("v0.1-models`)", "v0.1-models`; used by v0.2 too)")
    (REPO / "README.md").write_text(head + rest)


def update_notebook():
    p = REPO / "notebooks/colab_quickstart.ipynb"
    nb = json.loads(p.read_text())
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "markdown" and "Colab quick start" in src:
            src = src.replace("Runs the masking pipeline (YOLO26s cell detector + YOLO26l cross-check + TinyUNet)",
                              "Runs pipeline v0.2 (YOLO26s cell detector + YOLO26l cross-check + TinyUNet v3, two-layer masks, "
                              "measurable/review/excluded gate)")
        if c["cell_type"] == "code" and src.startswith("# 7."):
            src = ("# 7. One 3-panel sheet (YOLO26s boxes, dashed = s_only | full masks | status green/orange/red + reason)\n"
                   "!python tools/make_sheets.py --run \"{OUT}\" --out /content/sheets\n"
                   "from IPython.display import Image, display\n"
                   "display(Image(\"/content/sheets/cervical_0001.png\", width=1800))\n")
        c["source"] = src.splitlines(True)
    p.write_text(json.dumps(nb, indent=1))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[step5] FAILED: {type(e).__name__}: {e}", flush=True)
        cur = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        subprocess.run(["git", "stash", "-q", "--include-untracked"], cwd=REPO)          # keep work, never delete
        subprocess.run(["git", "checkout", "-q", "main"], cwd=REPO)
        print(f"[step5] main and tag v0.1 untouched; work kept on branch v0.2-wip / git stash (was on {cur})", flush=True)
        sys.exit(1)
