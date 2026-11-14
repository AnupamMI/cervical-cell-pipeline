#!/usr/bin/env python3
"""Masking-only cell pipeline: FOV -> YOLO26 cell boxes -> TinyUNet cell + nucleus masks -> label maps, COCO, CSV.

    python run.py --input <folder of FOV images> --output <new folder> [--config config.yaml] [--limit N] [--overwrite]

Exit code 0 when the run finished (even if some FOVs failed; see run_manifest.json), 2 on a usage/config error.
"""
import argparse
import os
import sys
from pathlib import Path

# deterministic cuBLAS; must be set before CUDA is initialised
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ncmask import config as cfgmod  # noqa: E402
from ncmask.runner import RunError, run  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="folder with FOV images (never written to)")
    ap.add_argument("--output", required=True, help="output folder (created; must differ from --input)")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (sorted by name); 0 = all")
    ap.add_argument("--overwrite", action="store_true", help="re-process FOVs already completed in --output")
    a = ap.parse_args(argv)
    if a.limit < 0:
        ap.error("--limit must be >= 0")
    try:
        m = run(a.input, a.output, a.config, a.limit, a.overwrite)
    except (RunError, cfgmod.ConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0 if m["status"] == "finished" else 1


if __name__ == "__main__":
    sys.exit(main())
