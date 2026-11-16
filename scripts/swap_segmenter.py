#!/usr/bin/env python3
"""Swap the segmenter of the web app in one command: (optionally point config.yaml at new weights) ->
re-export seg to ONNX -> parity test on samples/ -> redeploy the Space (+ wait + smoke test).

    python scripts/swap_segmenter.py                                   # re-export whatever config.yaml names
    python scripts/swap_segmenter.py --weights models/seg/V4/best.pt   # e.g. once V4 lands (variant from its config.json)
    python scripts/swap_segmenter.py --weights models/seg/V1_v3/best.pt --variant V1
    ... [--no-deploy]   stop after the parity test

Stops at the first failing step. If the parity test fails after --weights changed config.yaml, config.yaml is
restored (the Space is never deployed with a segmenter that failed parity).
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
CONFIG = HERE / "config.yaml"


def set_seg(text, weights, variant):
    """Replace seg.weights / seg.variant values inside the `seg:` block, keeping comments and layout."""
    m = re.search(r"^seg:\s*\n((?:[ \t]+.*\n|\s*\n)*)", text, re.M)
    if not m:
        raise SystemExit("config.yaml: no seg: section")
    block = m.group(1)
    new = re.sub(r"^(\s+weights:\s*)\S+", lambda g: g.group(1) + weights, block, count=1, flags=re.M)
    if variant:
        new = re.sub(r"^(\s+variant:\s*)\S+", lambda g: g.group(1) + variant, new, count=1, flags=re.M)
    return text[:m.start(1)] + new + text[m.end(1):]


def run(*cmd):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.call([str(c) for c in cmd], cwd=HERE)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", help="new seg weights, relative to the repo (config.yaml is updated)")
    ap.add_argument("--variant", help="seg.variant (default: auto = from the model's config.json)")
    ap.add_argument("--no-deploy", action="store_true")
    a = ap.parse_args(argv)
    py = sys.executable
    original = CONFIG.read_text(encoding="utf-8")
    if a.weights:
        if not (HERE / a.weights).is_file():
            raise SystemExit(f"not found: {a.weights}")
        CONFIG.write_text(set_seg(original, Path(a.weights).as_posix(), a.variant or "auto"), encoding="utf-8")
        print(f"config.yaml: seg.weights -> {a.weights}, seg.variant -> {a.variant or 'auto'}")
    try:
        if run(py, "scripts/export_onnx.py", "--only", "seg"):
            raise RuntimeError("export failed")
        if run(py, "scripts/check_onnx_parity.py"):
            raise RuntimeError("parity test failed")
    except RuntimeError as e:
        if a.weights:
            CONFIG.write_text(original, encoding="utf-8")
            print("config.yaml restored; re-run scripts/export_onnx.py --only seg to restore the previous bundle")
        print(f"STOP: {e}")
        return 1
    if a.no_deploy:
        print("\nparity passed; not deploying (--no-deploy). Commit config.yaml, then run scripts/deploy_space.*")
        return 0
    return run(py, "scripts/deploy_space.py", "--wait", "--smoke")


if __name__ == "__main__":
    sys.exit(main())
