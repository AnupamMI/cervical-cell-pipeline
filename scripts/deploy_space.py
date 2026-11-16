#!/usr/bin/env python3
"""Sync the web app + the current ONNX bundle to the Hugging Face Space (Docker SDK, CPU basic, private).

    python scripts/deploy_space.py [--space AnupamZZZZ/cervical-cell-demo] [--wait] [--smoke] [--dry-run]

Token: the HF_TOKEN environment variable (never printed). The Space holds everything it needs: code, the ONNX
weights (git LFS) and the sample FOVs; no runtime dependency on GitHub. Files on the Space that are no longer part
of the staged set are deleted (sync, not append).

  --wait    poll the Space until it is RUNNING (prints the build log tail on a build / runtime error)
  --smoke   then run scripts/space_smoke.py (health + every sample with and without consensus, timings)
  --dry-run stage only, print the file list, upload nothing
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
DEFAULT_SPACE = "AnupamZZZZ/cervical-cell-demo"

# what the Docker image needs (see Dockerfile); paths relative to the repo root
INCLUDE = ["Dockerfile", ".dockerignore", "requirements-app.txt", "config.yaml", "ncmask", "vendor", "app/__init__.py",
           "app/backend", "app/frontend/package.json", "app/frontend/package-lock.json", "app/frontend/index.html",
           "app/frontend/tsconfig.json", "app/frontend/vite.config.ts", "app/frontend/public", "app/frontend/src",
           "models/onnx"]
SKIP_PARTS = {"__pycache__", "node_modules", "dist", ".pytest_cache"}

SPACE_README = """---
title: Cervical Cell Pipeline
emoji: 🔬
colorFrom: pink
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: FOV cell detection, segmentation and quality gate (research demo)
---

# Cervical Cell Pipeline · FOV demo

**Research demo, not for diagnosis.** Upload one Pap-smear field of view (2028 x 1520, TIFF/PNG/JPEG, up to 25 MB):
YOLO26s detects cells, YOLO26l @1280 cross-checks them (tag only, never filters), a TinyUNet segments each cell and
its nucleus, and a quality gate marks every mask measurable / review / excluded.

Uploads are deleted right after decoding; results (overlay, cells.csv, COCO) are deleted after 15 minutes.
API docs: `/api/docs`. Source of truth: the `app-fov` branch of the cervical-cell-pipeline repository; this Space is
synced by `scripts/deploy_space.py`.

Models in this build: see `/api/health` (ONNX SHA256s and the PyTorch checkpoints they were exported from).
"""

GITATTRIBUTES = "*.onnx filter=lfs diff=lfs merge=lfs -text\n*.tiff filter=lfs diff=lfs merge=lfs -text\n" \
                "*.tif filter=lfs diff=lfs merge=lfs -text\n*.webp filter=lfs diff=lfs merge=lfs -text\n"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def check_bundle():
    """The bundle must be exported from exactly the weights config.yaml names now."""
    from ncmask import config as cfgmod
    man_p = HERE / "models" / "onnx" / "manifest.json"
    if not man_p.is_file():
        raise SystemExit("no ONNX bundle: run python scripts/export_onnx.py first")
    man = json.loads(man_p.read_text())
    cfg = cfgmod.load(HERE / "config.yaml")
    secs = ["detector", "seg"] + (["detector_b"] if cfg["consensus"]["detection_mode"] != "single" else [])
    for sec in secs:
        m = man["models"].get(sec)
        if m is None:
            raise SystemExit(f"bundle has no {sec}: re-run scripts/export_onnx.py")
        if sha256(cfg[sec]["weights"]) != m["source_sha256"]:
            raise SystemExit(f"bundle {sec} is stale (exported from {m['source']}, config.yaml now names "
                             f"{cfg[sec]['weights']}): re-run scripts/export_onnx.py and the parity check")
        if sha256(HERE / "models" / "onnx" / m["onnx"]) != m["onnx_sha256"]:
            raise SystemExit(f"{m['onnx']} does not match its manifest SHA256")
    return man


def stage(dst):
    files = []
    for item in INCLUDE:
        src = HERE / item
        if not src.exists():
            raise SystemExit(f"missing {item}")
        paths = [src] if src.is_file() else [p for p in src.rglob("*") if p.is_file()]
        for p in paths:
            rel = p.relative_to(HERE)
            if SKIP_PARTS & set(rel.parts) or p.suffix in (".pyc", ".tsbuildinfo"):
                continue
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst / rel)
            files.append(rel.as_posix())
    for p in sorted((HERE / "samples").glob("*.tiff")):
        (dst / "samples").mkdir(exist_ok=True)
        shutil.copy2(p, dst / "samples" / p.name)
        files.append(f"samples/{p.name}")
    (dst / "README.md").write_text(SPACE_README, encoding="utf-8")
    (dst / ".gitattributes").write_text(GITATTRIBUTES)
    return sorted(set(files) | {"README.md", ".gitattributes"})


def build_log_tail(space, n=60):
    import requests
    tok = os.environ["HF_TOKEN"]
    try:
        r = requests.get(f"https://huggingface.co/api/spaces/{space}/logs/build",
                         headers={"Authorization": f"Bearer {tok}"}, stream=True, timeout=20)
        lines = []
        for raw in r.iter_lines(decode_unicode=True):
            if raw and raw.startswith("data:"):
                try:
                    lines.append(json.loads(raw[5:]).get("data", "").rstrip())
                except json.JSONDecodeError:
                    pass
        return "\n".join(lines[-n:])
    except Exception as e:  # pragma: no cover - network
        return f"(could not read build logs: {type(e).__name__})"


def wait_running(api, space, timeout_s=1800):
    t0, last = time.time(), None
    while time.time() - t0 < timeout_s:
        rt = api.get_space_runtime(space)
        if rt.stage != last:
            print(f"  [{time.time() - t0:5.0f}s] {rt.stage}", flush=True)
            last = rt.stage
        if rt.stage == "RUNNING":
            return True
        if rt.stage in ("BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"):
            print(build_log_tail(space))
            return False
        time.sleep(15)
    print("timed out waiting for the Space")
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", default=DEFAULT_SPACE)
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if not os.environ.get("HF_TOKEN") and not a.dry_run:
        raise SystemExit("set HF_TOKEN (a Hugging Face token with write access)")
    man = check_bundle()
    print("bundle:", ", ".join(f"{k} <- {v['source']} ({v['onnx_sha256'][:12]})" for k, v in man["models"].items()))
    tmp = Path(tempfile.mkdtemp(prefix="space-stage-"))
    try:
        files = stage(tmp)
        size = sum((tmp / f).stat().st_size for f in files)
        print(f"staged {len(files)} files, {size / 1e6:.1f} MB")
        if a.dry_run:
            print("\n".join(sorted(files)))
            return 0
        from huggingface_hub import HfApi
        from huggingface_hub.errors import HfHubHTTPError
        api = HfApi(token=os.environ["HF_TOKEN"])
        try:
            url = api.create_repo(a.space, repo_type="space", space_sdk="docker", private=True, exist_ok=True)
        except HfHubHTTPError as e:
            if getattr(e.response, "status_code", None) == 402:
                raise SystemExit("Hugging Face refused to create the Space (402 Payment Required): Docker Spaces on "
                                 "the free cpu-basic hardware need a PRO subscription on this account "
                                 "(https://huggingface.co/pro). Nothing was uploaded.")
            raise
        info = api.space_info(a.space)
        if not info.private:
            raise SystemExit(f"{a.space} exists and is PUBLIC; refusing to deploy (make it private first)")
        print(f"space: {url} (private)")
        commit = api.upload_folder(folder_path=str(tmp), repo_id=a.space, repo_type="space",
                                   commit_message=f"sync app + weights ({', '.join(m['source'] for m in man['models'].values())})",
                                   delete_patterns=["*", "**/*"])
        print(f"pushed: {commit.oid[:12] if hasattr(commit, 'oid') else commit}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ok = True
    if a.wait or a.smoke:
        print("waiting for the Space build ...")
        ok = wait_running(api, a.space)
    if ok and a.smoke:
        import subprocess
        ok = subprocess.call([sys.executable, str(HERE / "scripts" / "space_smoke.py"), "--space", a.space]) == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
