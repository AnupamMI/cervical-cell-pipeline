#!/usr/bin/env python3
"""Step 4: plug the winner into nc_mask_pipeline_v3, run the test suite, run 1 + 5 production FOVs with sheets.

nc_mask_pipeline_v3 (built and tested before the night; never the original nc_mask_pipeline) gets:
  models/seg/<winner>_v3/ : best.pt, config.json (in_ch, size, variant), its preprocessing file(s), sha256.json, log
  config.yaml             : seg.weights -> that folder, seg.variant -> the winner's variant (explicit, checked at load)
Then: pytest (all must pass; incl. parity of the configured model vs the evaluation code path), FOV picks from the
v0.1 full run (runs/full of nc_mask_pipeline), runs/prod_1fov + runs/prod_5fov (new folders), sheets + summaries,
reports/STEP4_SUMMARY.md. Exit 0 only if everything succeeded.
"""
import csv, json, os, random, re, shutil, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path

K = Path.home() / "Downloads/ANUPAM KA HOOD"
V3 = K / "nc_mask_pipeline_v3"; T = K / "tinyseg_v3"
RES = Path(os.environ.get("RESULTS_DIR", T / "results"))
PREFIX = os.environ.get("STEP4_PREFIX", "")          # dry runs: "dryrun_" so real output names stay free
DATA = K / "nc_mask_pipeline/data/images"; FULL = K / "nc_mask_pipeline/runs/full"
PY = os.environ.get("PY", str(K / "seg_audit/venv/bin/python"))
NEED = {"V0": [], "V1": ["stain_ref.json"], "V2": ["stain_ref.json"], "V3": ["h_stats.json"]}
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a); LOG.append(s); print("[step4]", s, flush=True)


def new_path(p):
    p = Path(p)
    if not p.exists():
        return p
    stamp = time.strftime("%Y%m%d_%H%M")
    return p.with_name(f"{p.name}_{stamp}")


def install_winner():
    w = json.loads((RES / "winner.json").read_text())
    name = w["winner"]
    if name == "v2_1":
        say("winner is the old v2_1: config stays on models/seg/v2_1 (variant V0)")
        return name, "V0", "models/seg/v2_1/best.pt", w
    src = Path(w["run_dir"]); cfgj = json.loads((src / "config.json").read_text()); variant = cfgj["variant"]
    dest = V3 / "models/seg" / f"{name}_v3"
    files = ["best.pt", "config.json", "args.json", "log.txt", "sha256.json", "env.txt"] + NEED[variant]
    if dest.exists():
        same = all((dest / f).exists() and (dest / f).read_bytes() == (src / f).read_bytes() for f in files if (src / f).exists())
        if not same:
            dest = new_path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        s = src / f if (src / f).exists() else (T / "phase2" / f)
        if not s.exists():
            raise FileNotFoundError(f"winner file missing: {f}")
        if not (dest / f).exists():
            shutil.copy2(s, dest / f)
    rel = f"models/seg/{dest.name}/best.pt"
    say(f"installed {name} ({variant}) from {src} -> {dest}")
    return name, variant, rel, w


def set_config(rel, variant):
    p = V3 / "config.yaml"; s = p.read_text()
    s2 = re.sub(r"^(  weights: )models/seg/\S+(.*)$", lambda m: f"{m.group(1)}{rel}{m.group(2)}", s, count=1, flags=re.M)
    s2 = re.sub(r"^(  variant: )\S+(.*)$", lambda m: f"{m.group(1)}{variant}{m.group(2)}", s2, count=1, flags=re.M)
    if s2 == s and f"weights: {rel}" not in s:
        raise RuntimeError("could not update config.yaml")
    p.write_text(s2)
    say(f"config.yaml: seg.weights = {rel}, seg.variant = {variant}")


def run(cmd, **kw):
    r = subprocess.run(cmd, cwd=V3, capture_output=True, text=True, **kw)
    return r.returncode, r.stdout + r.stderr


def pick_fovs():
    rows = list(csv.DictReader(open(FULL / "cells.csv")))
    n, edge, multi = Counter(), Counter(), Counter()
    for r in rows:
        n[r["fov"]] += 1
        edge[r["fov"]] += "cell_at_edge" in r["flags"]
        multi[r["fov"]] += "multi_nucleus" in r["flags"]
    fovs = sorted(n)
    med = sorted(n.values())[len(n) // 2]
    typical = sorted(f for f in fovs if n[f] == med)[0]
    picks = {}
    def take(f, why):
        if f not in picks:
            picks[f] = why
    take(max(fovs, key=lambda f: (n[f], f)), f"dense ({n[max(fovs, key=lambda f: (n[f], f))]} cells in v0.1)")
    take(max((f for f in fovs if f not in picks), key=lambda f: (edge[f], f)), "edge-heavy (most cell_at_edge)")
    take(max((f for f in fovs if f not in picks), key=lambda f: (multi[f], f)), "multi_nucleus-heavy")
    take(min((f for f in fovs if f not in picks), key=lambda f: (n[f], f)), "sparse (fewest cells)")
    rest = [f for f in fovs if f not in picks and f != typical]
    take(random.Random(42).choice(rest), "random (seed 42)")
    return typical, med, picks


def prod_run(name, fovs, why):
    inp = new_path(V3 / "runs" / f"{name}_in"); inp.mkdir(parents=True)
    for f in fovs:
        src = DATA / f"{f}.tiff"
        if not src.exists():
            raise FileNotFoundError(src)
        (inp / src.name).symlink_to(src)
    out = new_path(V3 / "runs" / name)
    rc, txt = run([PY, "run.py", "--input", str(inp), "--output", str(out)], timeout=3600)
    if rc != 0:
        raise RuntimeError(f"{name} run failed: {txt[-1500:]}")
    why_f = out / "why.json"; why_f.write_text(json.dumps(why))
    sheets = new_path(V3 / "reports" / f"sheets_{name.replace('prod_', '')}")
    rc, txt = run([PY, "tools/make_sheets.py", "--run", str(out), "--out", str(sheets), "--why", str(why_f)], timeout=1800)
    if rc != 0:
        raise RuntimeError(f"sheets for {name} failed: {txt[-1500:]}")
    summ = [l for l in (out / "run.log").read_text().splitlines() if "SUMMARY" in l or "status:" in l or "Macenko" in l or "image QC" in l]
    say(f"{name}: {out.name} OK; sheets in {sheets}")
    return out, sheets, (sheets / "summary.md").read_text(), summ


def main():
    (V3 / "reports").mkdir(exist_ok=True)
    name, variant, rel, w = install_winner()
    set_config(rel, variant)
    env = dict(os.environ, NCMASK_TINYSEG_DIR=str(K / "seg_audit/extracted/tinyseg_retrain/tinyseg_retrain"),
               NCMASK_NC_PIPELINE_DIR=str(K / "nc_pipeline"))
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/"], cwd=V3, capture_output=True, text=True, env=env, timeout=5400)
    test_out = (r.stdout + r.stderr)
    (V3 / "reports/tests_step4.txt").write_text(test_out)
    last = [l for l in test_out.splitlines() if "passed" in l or "failed" in l or "error" in l.lower()][-1:]
    say("pytest:", last[0] if last else "?", "(rc", r.returncode, ")")
    parity = json.loads((V3 / "reports/parity_v3.json").read_text()) if (V3 / "reports/parity_v3.json").exists() else {}
    if r.returncode != 0:
        raise SystemExit("test suite failed: see nc_mask_pipeline_v3/reports/tests_step4.txt")
    typical, med, picks = pick_fovs()
    o1, s1, t1, l1 = prod_run(f"{PREFIX}prod_1fov", [typical], {typical: f"typical (median {med} cells in v0.1)"})
    o5, s5, t5, l5 = prod_run(f"{PREFIX}prod_5fov", list(picks), picks)
    md = [f"Winner installed: **{name}** (variant {variant}) -> `{rel}`; `config.yaml` seg.weights / seg.variant set. "
          f"Decision: {'; '.join(w.get('notes', []))}", "",
          f"Tests: `{last[0] if last else '?'}` (log `nc_mask_pipeline_v3/reports/tests_step4.txt`). "
          f"Seg parity of the configured model vs the evaluation path: {parity}", "",
          f"### 1 FOV (`{o1}`)", "", t1, "", f"Sheets: `{s1}`", "", "```", *l1, "```", "",
          f"### 5 FOVs (`{o5}`)", "", t5, "", f"Sheets: `{s5}`", "", "```", *l5, "```", "", "Step 4 log:", "", *[f"- {x}" for x in LOG]]
    (V3 / f"reports/{PREFIX}STEP4_SUMMARY.md").write_text("\n".join(md) + "\n")
    say("STEP4_SUMMARY.md written")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        (V3 / "reports").mkdir(exist_ok=True)
        (V3 / "reports/STEP4_SUMMARY.md").write_text(f"Step 4 FAILED: {type(e).__name__}: {e}\n\n" + "\n".join(f"- {x}" for x in LOG) + "\n")
        print("[step4] FAILED:", type(e).__name__, e, flush=True)
        sys.exit(1)
