#!/usr/bin/env python3
"""Step 1 report: per variant, the finished run, epochs, best/last validation score, s/epoch, whether validation was
still rising at the end; saves env (pip freeze), SHA256s and the preprocessing files each variant needs into the run
folder (new files only, never overwrites). Writes tinyseg_v3/results/train_report.json and .md."""
import hashlib, json, re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; P2 = ROOT / "phase2"; RES = ROOT / "results"
PY = sys.executable
NEEDS = {"V0": [], "V1": ["stain_ref.json"], "V2": ["stain_ref.json"], "V3": ["h_stats.json"]}


def sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()


def write_new(p, text):
    p = Path(p)
    if not p.exists():
        p.write_text(text)


def run_dir(v):
    for d in (P2 / "runs" / v, P2 / "runs" / f"{v}_r1", P2 / "runs" / f"{v}_r2"):
        if (d / "done.json").exists():
            return d
    return None


def main():
    RES.mkdir(exist_ok=True)
    freeze = subprocess.run([PY, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    out = {}
    for v in ("V0", "V1", "V2", "V3"):
        d = run_dir(v)
        if d is None:
            tried = [x.name for x in (P2 / "runs").glob(f"{v}*") if x.is_dir()]
            out[v] = {"status": "failed", "tried": tried,
                      "failed_json": [json.loads((P2 / "runs" / t / "failed.json").read_text()) for t in tried if (P2 / "runs" / t / "failed.json").exists()]}
            continue
        lines = (d / "log.txt").read_text().splitlines()
        sc = [float(re.search(r"score ([0-9.]+)", l).group(1)) for l in lines]
        secs = [int(re.search(r"(\d+)s$", l.strip()).group(1)) for l in lines if re.search(r"(\d+)s$", l.strip())]
        best_ep = int(max(range(len(sc)), key=lambda i: sc[i]))
        for f in NEEDS[v]:
            if not (d / f).exists():
                shutil.copy2(P2 / f, d / f)
        write_new(d / "env.txt", freeze)
        files = ["best.pt", "last.pt", "config.json", "args.json", "log.txt"] + NEEDS[v]
        shas = {f: sha(d / f) for f in files if (d / f).exists()}
        write_new(d / "sha256.json", json.dumps(shas, indent=1))
        last5 = sc[-5:]
        out[v] = {"status": "ok", "run_dir": str(d), "epochs": len(sc), "best_epoch": best_ep, "best_score": sc[best_ep],
                  "last_score": sc[-1], "median_s_per_epoch": sorted(secs)[len(secs) // 2] if secs else None,
                  "total_train_min": round(sum(secs) / 60, 1),
                  "val_still_rising_at_end": bool(best_ep >= len(sc) - 5),
                  "last5_scores": last5, "preprocessing_files": NEEDS[v], "sha256": shas,
                  "wall_s": json.loads((d / "done.json").read_text()).get("wall_s")}
    (RES / "train_report.json").write_text(json.dumps(out, indent=1))
    md = ["| variant | status | run | epochs | best epoch | best val score | last | s/epoch | train min | still rising at end? |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for v, r in out.items():
        if r["status"] != "ok":
            md.append(f"| {v} | **FAILED** | {', '.join(r['tried'])} | | | | | | | |"); continue
        md.append(f"| {v} | ok | {Path(r['run_dir']).name} | {r['epochs']} | {r['best_epoch']} | {r['best_score']:.4f} | "
                  f"{r['last_score']:.4f} | {r['median_s_per_epoch']} | {r['total_train_min']} | "
                  f"{'yes (best in the last 5 epochs)' if r['val_still_rising_at_end'] else 'no'} |")
    (RES / "train_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    ok = out["V0"]["status"] == "ok"
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
