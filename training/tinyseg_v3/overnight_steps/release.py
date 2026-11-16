#!/usr/bin/env python3
"""Release vX.Y of cervical-cell-pipeline from nc_mask_pipeline_v3 (generalises step5_push_v02.py; used for v0.3, v0.4).

  python steps/release.py --tag v0.3 --seg-weights models/seg/v2_1/best.pt --variant auto --readme-head HEAD.md
  python steps/release.py --tag v0.4 --install-run phase2/runs/V4 --name V4 --readme-head HEAD.md

 1. nc_mask_pipeline_v3/config.yaml: seg.weights / seg.variant (with --install-run: the run is copied first to
    models/seg/<name>_v3/ with config.json, sha256.json, env, log; existing folders are never overwritten)
 2. full pytest suite (incl. seg parity of the configured model vs the evaluation path) -> must pass
 3. 1 typical + 5 FOVs (dense / edge-heavy / multi_nucleus-heavy / sparse / random, same picks as v0.2) -> runs/<tag>_prod_*,
    3-panel sheets -> reports/sheets_<tag>_1fov|5fov
 4. repo: branch <tag>-wip from origin/main; v3 code, config, models, reports (sheets as JPEG), samples/expected
    regenerated, README head replaced (--readme-head), notebook text, WEIGHTS.md; commit
 5. fresh clone of the wip branch -> fetch release weights -> run.py --limit 2 with CUDA and CPU -> check_against_samples
    must PASS for both
 6. fast-forward main, tag <tag>, push.  Any failure before 6: main and older tags untouched, wip branch kept.
The GitHub token is read only via `git credential fill`, passed by env to the fetch script, never printed or stored.
"""
import argparse, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

K = Path.home() / "Downloads/ANUPAM KA HOOD"
REPO = Path.home() / "Downloads/cervical-cell-pipeline"
V3 = K / "nc_mask_pipeline_v3"; T = K / "tinyseg_v3"
PY = os.environ.get("PY", str(K / "seg_audit/venv/bin/python"))
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a); LOG.append(s); print(f"[release]", s, flush=True)


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


def install_run(run, name):
    src = (T / run).resolve() if not Path(run).is_absolute() else Path(run)
    cfgj = json.loads((src / "config.json").read_text())
    dest = V3 / "models/seg" / f"{name}_v3"
    files = [f for f in ("best.pt", "config.json", "args.json", "log.txt", "sha256.json", "env.txt") if (src / f).exists()]
    if dest.exists() and not all((dest / f).exists() and (dest / f).read_bytes() == (src / f).read_bytes() for f in files):
        raise RuntimeError(f"{dest} exists with different content: not overwritten")
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        if not (dest / f).exists():
            shutil.copy2(src / f, dest / f)
    if not (dest / "sha256.json").exists():
        (dest / "sha256.json").write_text(json.dumps({"best.pt": sh(["sha256sum", str(dest / "best.pt")]).split()[0]}, indent=1))
    say(f"installed {src} -> {dest} (variant {cfgj['variant']})")
    return f"models/seg/{dest.name}/best.pt", cfgj["variant"], src


def set_config(rel, variant):
    p = V3 / "config.yaml"; s = p.read_text()
    s2 = re.sub(r"^(  weights: )models/seg/\S+(.*)$", lambda m: f"{m.group(1)}{rel}{m.group(2)}", s, count=1, flags=re.M)
    s2 = re.sub(r"^(  variant: )\S+(.*)$", lambda m: f"{m.group(1)}{variant}{m.group(2)}", s2, count=1, flags=re.M)
    if f"  weights: {rel}" not in s2 or f"  variant: {variant}" not in s2:
        raise RuntimeError("could not update config.yaml")
    p.write_text(s2)
    say(f"config.yaml: seg.weights = {rel}, seg.variant = {variant}")


def tests():
    env = dict(os.environ, NCMASK_TINYSEG_DIR=str(K / "seg_audit/extracted/tinyseg_retrain/tinyseg_retrain"),
               NCMASK_NC_PIPELINE_DIR=str(K / "nc_pipeline"))
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/"], cwd=V3, capture_output=True, text=True, env=env, timeout=5400)
    out = r.stdout + r.stderr
    last = ([l for l in out.splitlines() if "passed" in l or "failed" in l or "error" in l.lower()] or ["?"])[-1]
    return r.returncode, out, last


def prod(tag):
    os.environ["STEP4_PREFIX"] = f"{tag}_"
    sys.path.insert(0, str(T / "steps"))
    import step4_pipeline_v3 as S4
    S4.PREFIX = f"{tag}_"
    typical, med, picks = S4.pick_fovs()
    o1, s1, t1, l1 = S4.prod_run(f"{tag}_prod_1fov", [typical], {typical: f"typical (median {med} cells in v0.1)"})
    o5, s5, t5, l5 = S4.prod_run(f"{tag}_prod_5fov", list(picks), picks)
    return [(o1, s1, t1, l1), (o5, s5, t5, l5)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seg-weights", default=None, help="path relative to the pipeline root")
    ap.add_argument("--variant", default="auto")
    ap.add_argument("--install-run", default=None, help="training run dir (relative to tinyseg_v3/) to install as the segmenter")
    ap.add_argument("--name", default=None, help="model name for --install-run (folder models/seg/<name>_v3)")
    ap.add_argument("--readme-head", required=True, help="markdown replacing the README up to '## Quick start'")
    ap.add_argument("--dry", action="store_true", help="everything except merge/tag/push")
    a = ap.parse_args()
    tag = a.tag
    head = Path(a.readme_head).read_text()
    # 1-3: pipeline_v3
    if a.install_run:
        rel, variant, run_src = install_run(a.install_run, a.name)
    else:
        rel, variant, run_src = a.seg_weights, a.variant, None
    if not (V3 / rel).is_file():
        raise FileNotFoundError(V3 / rel)
    set_config(rel, variant)
    rc, out, last = tests()
    (V3 / f"reports/tests_{tag}.txt").write_text(out)
    say("pytest:", last, f"(rc {rc})")
    if rc != 0:
        raise RuntimeError(f"test suite failed: nc_mask_pipeline_v3/reports/tests_{tag}.txt")
    parity = json.loads((V3 / "reports/parity_v3.json").read_text())
    say("seg parity (configured model vs evaluation path):", {k: parity[k] for k in ("model", "variant", "cells", "max_abs_prob_diff", "min_dice")})
    runs = prod(tag)
    for o, s, *_ in runs:
        say(f"prod run {o.name} OK; sheets {s}")
    summ = [f"# {tag}: pipeline v3 with `{rel}` (seg.variant {variant})", "",
            f"Tests: `{last}` (`nc_mask_pipeline_v3/reports/tests_{tag}.txt`). Seg parity: {parity}", ""]
    for (o, s, t, l), n in zip(runs, ("1 FOV", "5 FOVs")):
        summ += [f"### {n} (`{o}`)", "", t, "", f"Sheets: `{s}`", "", "```", *l, "```", ""]
    (V3 / f"reports/{tag}_SUMMARY.md").write_text("\n".join(summ) + "\n")
    # 4: repo
    if git("status", "--porcelain"):
        raise RuntimeError("repo working tree not clean")
    git("fetch", "-q", "origin")
    git("checkout", "-q", "-B", f"{tag}-wip", git("rev-parse", "origin/main"))
    for d in ("ncmask", "tests", "tools", "vendor"):
        copytree(V3 / d, REPO / d)
    for f in ("run.py", "config.yaml", "requirements.txt", "requirements-colab.txt", "requirements-cpu.txt",
              "scripts/check_against_samples.py", "scripts/fetch_release_weights.py"):
        shutil.copy2(V3 / f, REPO / f)
    for d in (V3 / "models/seg").iterdir():
        if d.is_dir() and not (REPO / "models/seg" / d.name).exists():
            copytree(d, REPO / "models/seg" / d.name); say(f"repo: added models/seg/{d.name}")
    if run_src is not None:
        dst = REPO / "models/seg/tinyseg_v3" / a.name; dst.mkdir(parents=True, exist_ok=True)
        for f in ("best.pt", "config.json", "sha256.json", "args.json"):
            if (run_src / f).exists():
                shutil.copy2(run_src / f, dst / f)
    tdst = REPO / "training/tinyseg_v3"
    for f in ("train_v3.py", "preprocess.py", "eval_v3.py", "eval_final.py", "make_split.py", "build_cache.py", "run_all.sh"):
        shutil.copy2(T / "phase2" / f, tdst / f)
    copytree(T / "steps", tdst / "overnight_steps")
    for f in ("overnight.sh", "v4.sh"):
        if (T / f).exists():
            shutil.copy2(T / f, tdst / f)
    for p in list((tdst / "overnight_steps").glob(".*.bak")):
        p.unlink()
    rows = []
    for p in sorted(REPO.glob("models/**/*.pt")) + sorted((REPO / "models").glob("**/*.json")):
        if "detector_b" in p.parts:
            continue
        rows.append(f"| `{p.relative_to(REPO)}` | {p.stat().st_size} | `{sh(['sha256sum', str(p)]).split()[0]}` |")
    (REPO / "models/WEIGHTS.md").write_text(
        "# Model weights and preprocessing files\n\n"
        f"Configured segmenter (config.yaml `seg.weights`, {tag}): `{rel}` (seg.variant `{variant}`).\n\n"
        "YOLO26l (`models/detector_b/yolo26l_1280_best.pt`, 105,729,697 bytes, SHA256 "
        "`703f3e716a5116b98d581497eed6bab62b49181577aeb2062f209cbb230fb72b`) is a GitHub Release asset "
        "(`v0.1-models`): `scripts/fetch_release_weights.py`.\n\n| file | bytes | SHA256 |\n|---|---|---|\n" + "\n".join(rows) + "\n")
    rep = REPO / "reports"
    import cv2
    for (o, s, *_ ) in runs:
        out = rep / s.name; out.mkdir(exist_ok=True)
        for png in s.glob("*.png"):
            cv2.imwrite(str(out / (png.stem + ".jpg")), cv2.imread(str(png)), [cv2.IMWRITE_JPEG_QUALITY, 92])
        shutil.copy2(s / "summary.md", out / "summary.md")
    shutil.copy2(V3 / f"reports/{tag}_SUMMARY.md", rep / f"{tag}_SUMMARY.md")
    for f in ("RESULTS.md", "RESULTS_V4.md", "train_report.md"):
        if (T / "results" / f).exists():
            shutil.copy2(T / "results" / f, rep / f)
    exp = REPO / "samples/expected"; old = REPO / f"samples/expected_prev_tmp_{int(time.time())}"
    exp.rename(old)
    try:
        sh([PY, "run.py", "--input", "samples", "--output", "samples/expected"], timeout=3600)
    except Exception:
        old.rename(exp); raise
    shutil.rmtree(old)                      # the previous reference stays in git history (previous tag)
    src = (REPO / "README.md").read_text()
    (REPO / "README.md").write_text(head.rstrip() + "\n\n" + src[src.find("## Quick start"):])
    nb_p = REPO / "notebooks/colab_quickstart.ipynb"; nb = json.loads(nb_p.read_text())
    for c in nb["cells"]:
        s = "".join(c["source"])
        s = re.sub(r"Runs pipeline v0\.\d+ \(YOLO26s cell detector \+ YOLO26l cross-check \+ TinyUNet[^,]*,",
                   f"Runs pipeline {tag} (YOLO26s cell detector + YOLO26l cross-check + TinyUNet `{rel}`,", s)
        c["source"] = s.splitlines(True)
    nb_p.write_text(json.dumps(nb, indent=1))
    git("add", "-A")
    msg = (f"{tag}: segmenter {rel} (seg.variant {variant})\n\n"
           f"- tests: {last}; seg parity max |prob diff| {parity['max_abs_prob_diff']}\n"
           f"- samples/expected regenerated; sheets reports/sheets_{tag}_1fov, reports/sheets_{tag}_5fov\n\n"
           "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
    subprocess.run(["git", "commit", "-q", "-F", "-"], cwd=REPO, input=msg, text=True, check=True)
    say("committed", git("rev-parse", "--short", "HEAD"), f"on {tag}-wip")
    # 5: fresh-clone smoke
    smoke = Path(f"/tmp/ccp_smoke_{tag.replace('.', '')}")
    if smoke.exists():
        shutil.rmtree(smoke)
    sh(["git", "clone", "-q", "--branch", f"{tag}-wip", str(REPO), str(smoke)], cwd=Path.home())
    env = dict(os.environ, GH_TOKEN=gh_token())
    say(sh([PY, "scripts/fetch_release_weights.py"], cwd=smoke, env=env).splitlines()[-1])
    del env
    res = {}
    for name, extra in (("cuda", {}), ("cpu", {"CUDA_VISIBLE_DEVICES": ""})):
        e = dict(os.environ, **extra)
        lg = sh([PY, "run.py", "--input", "samples/", "--output", f"out_{name}/", "--limit", "2"], cwd=smoke, env=e)
        dev = re.search(r"device=(\w+)", lg)
        chk = sh([PY, "scripts/check_against_samples.py", "--output", f"out_{name}/", "--limit", "2"], cwd=smoke, env=e, check=False)
        res[name] = {"device": dev.group(1) if dev else "?", "check": chk.splitlines()[-1], "full": chk}
        say(f"smoke {name}: device {res[name]['device']}, {res[name]['check']}")
        if not res[name]["check"].startswith("PASS"):
            raise RuntimeError(f"smoke test {name} failed: {chk[-800:]}")
    if a.dry:
        git("checkout", "-q", "main"); say("DRY: not merged / tagged / pushed"); return
    git("checkout", "-q", "main")
    git("merge", "-q", "--ff-only", f"{tag}-wip")
    git("tag", "-a", tag, "-m", f"{tag}: segmenter {rel}; tests {last}; fresh-clone smoke CUDA/CPU PASS")
    git("push", "-q", "origin", "main", timeout=1800)
    git("push", "-q", "origin", tag, timeout=600)
    (T / "results" / f"release_{tag}.json").write_text(json.dumps(
        {"tag": tag, "commit": git("rev-parse", "--short", "HEAD"), "seg_weights": rel, "variant": variant, "tests": last,
         "parity": parity, "smoke": res, "log": LOG}, indent=1))
    say(f"pushed main + tag {tag}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[release] FAILED: {type(e).__name__}: {e}", flush=True)
        cur = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        if cur != "main":
            subprocess.run(["git", "stash", "-q", "--include-untracked"], cwd=REPO)    # keep work, never delete
            subprocess.run(["git", "checkout", "-q", "main"], cwd=REPO)
        print(f"[release] main and existing tags untouched (was on {cur})", flush=True)
        sys.exit(1)
