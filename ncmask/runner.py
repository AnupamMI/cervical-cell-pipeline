"""Run the pipeline over a folder: listing, safety checks, resume, per-FOV isolation of failures, manifest, summary."""
import csv
import datetime as dt
import hashlib
import json
import logging
import platform
import shutil
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from . import config as cfgmod
from .fov import CELL_COLS, L_ONLY_COLS, process_fov, write_fov_outputs
from .imageio import ImageLoadError, load_bgr8

log = logging.getLogger("ncmask")


DROP_COLS = ["fov", "reason", "cell_id", "conf", "box_x1", "box_y1", "box_x2", "box_y2", "detail"]


class RunError(RuntimeError):
    pass


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(cfg):
    """Hash of everything that changes results (not logging)."""
    c = {k: v for k, v in cfg.items() if k not in ("logging", "_config_path")}
    return hashlib.sha256(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()[:16]


def setup_logging(out_dir, level):
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    fh = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    for h in (fh, sh):
        h.setFormatter(fmt); root.addHandler(h)
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    for noisy in ("ultralytics", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def check_paths(inp, out, recursive):
    inp, out = Path(inp).expanduser().resolve(), Path(out).expanduser().resolve()
    if not inp.is_dir():
        raise RunError(f"input folder not found: {inp}")
    if out == inp:
        raise RunError("output folder must differ from the input folder")
    if inp in out.parents and recursive:
        raise RunError("output folder is inside the input folder and input.recursive is true: outputs would be re-read")
    if out in inp.parents:
        raise RunError("input folder is inside the output folder; choose a separate output folder")
    return inp, out


def list_fovs(inp, exts, recursive):
    """-> (list of (fov_id, path)), ignored file names). fov_id = relative path without extension ('/' -> '__');
    if two files share a stem, the extension is appended to keep ids unique."""
    it = inp.rglob("*") if recursive else inp.iterdir()
    files, ignored = [], []
    for p in sorted(it):
        if not p.is_file() or p.name.startswith("."):
            continue
        (files if p.suffix.lower() in exts else ignored).append(p)
    stems = Counter(str(p.relative_to(inp).with_suffix("")) for p in files)
    out = []
    for p in files:
        rel = str(p.relative_to(inp).with_suffix(""))
        fid = rel if stems[rel] == 1 else rel + "_" + p.suffix.lower().lstrip(".")
        out.append((fid.replace("/", "__"), p))
    return out, [str(p.relative_to(inp)) for p in ignored]


def versions():
    import cv2, numpy, torch
    v = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
         "cuda_available": torch.cuda.is_available(), "opencv": cv2.__version__, "numpy": numpy.__version__}
    try:
        import ultralytics; v["ultralytics"] = ultralytics.__version__
    except Exception:  # pragma: no cover
        pass
    import scipy, skimage
    v["scipy"], v["scikit_image"] = scipy.__version__, skimage.__version__
    if torch.cuda.is_available():
        v["cuda"] = torch.version.cuda; v["gpu"] = torch.cuda.get_device_name(0)
    return v


def set_determinism(seed, device):
    import random
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def pick_device(want):
    import torch
    if want == "cuda" and not torch.cuda.is_available():
        raise RunError("device: cuda requested but CUDA is not available")
    return "cuda" if (want == "cuda" or (want == "auto" and torch.cuda.is_available())) else "cpu"


def run(input_dir, output_dir, config_path, limit=0, overwrite=False):
    cfg = cfgmod.load(config_path)
    inp, out = check_paths(input_dir, output_dir, cfg["input"]["recursive"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "fovs").mkdir(exist_ok=True)
    setup_logging(out, cfg["logging"]["level"])
    t_start, started = time.time(), now()
    chash = config_hash(cfg)
    log.info("nc_mask_pipeline: input=%s output=%s config=%s (hash %s)", inp, out, cfg["_config_path"], chash)

    fovs, ignored = list_fovs(inp, cfg["input"]["extensions"], cfg["input"]["recursive"])
    if ignored:
        log.info("ignoring %d non-image file(s) (extension not in input.extensions)", len(ignored))
    if limit:
        fovs = fovs[:limit]
    log.info("%d FOV image(s) to consider", len(fovs))

    # resume safety: a previous run with a different config must not be silently mixed in
    prev = out / "run_manifest.json"
    if prev.exists() and not overwrite:
        try:
            ph = json.loads(prev.read_text()).get("config_hash")
        except Exception:
            ph = None
        if ph and ph != chash:
            raise RunError(f"{out} holds results of a different config (hash {ph} != {chash}); "
                           "use --overwrite or a new output folder")

    device = pick_device(cfg["device"])
    set_determinism(int(cfg["seed"]), device)
    from .detect import Detector
    from .segment import SegModel
    detector = Detector(cfg, device)
    mode = cfg["consensus"]["detection_mode"]
    detector_b = Detector(cfg, device, "detector_b") if mode != "single" else None
    seg = SegModel(cfg, device)
    log.info("device=%s  detector imgsz=%d conf=%s tiling=%s  seg in_ch=%d size=%d  crop margin=%s edge_mode=%s",
             device, detector.imgsz, cfg["detector"]["conf"], cfg["detector"]["tiling"]["mode"], seg.in_ch, seg.size,
             cfg["crop"]["margin"], cfg["crop"]["edge_mode"])
    if cfg["consensus"].get("auto_off"):
        log.warning("consensus auto-off: %s", cfg["consensus"]["auto_off"])
    log.info("detection_mode=%s%s", mode, "" if detector_b is None else
             f"  B={cfg['detector_b']['weights']} imgsz={detector_b.imgsz} conf={cfg['detector_b']['conf']} "
             f"match_iou={cfg['consensus']['match_iou']}")

    models = {"detector": {"path": cfg["detector"]["weights"], "sha256": sha256(cfg["detector"]["weights"]),
                           "names": detector.names, "imgsz": detector.imgsz},
              "seg": {"path": cfg["seg"]["weights"], "sha256": sha256(cfg["seg"]["weights"]),
                      "model_config": seg.model_config_file,
                      "model_config_sha256": sha256(seg.model_config_file) if seg.model_config_file else None,
                      "in_ch": seg.in_ch, "size": seg.size},
              "seg_code": {p.name: sha256(p) for p in sorted(Path(cfg["seg"]["code_dir"]).glob("*.py"))},
              "seg_variant": seg.variant,
              "seg_preprocess_code": {"path": cfg["seg"]["preprocess_code"], "sha256": sha256(cfg["seg"]["preprocess_code"])},
              "seg_preprocessing_file": ({"path": seg.preprocessing_file, "sha256": sha256(seg.preprocessing_file)}
                                         if seg.preprocessing_file else None)}
    if detector_b is not None:
        models["detector_b"] = {"path": cfg["detector_b"]["weights"], "sha256": sha256(cfg["detector_b"]["weights"]),
                                "names": detector_b.names, "imgsz": detector_b.imgsz}
    manifest = {"status": "running", "started_utc": started, "input": str(inp), "output": str(out),
                "config_hash": chash, "config": cfg, "models": models, "versions": versions(), "device": device}
    prev.write_text(json.dumps(manifest, indent=1, default=str))

    ok, failed, skipped, times, flag_counts, n_cells = [], [], [], [], Counter(), 0
    drop_counts, grey_fovs, tag_counts, n_l_only = Counter(), [], Counter(), 0
    status_counts, reason_counts, qc_counts = Counter(), Counter(), Counter()
    for k, (fid, path) in enumerate(fovs, 1):
        d = out / "fovs" / fid
        if (d / "done.json").exists() and not overwrite:
            skipped.append(fid)
            continue
        tmp = out / "fovs" / (fid + ".tmp")
        try:
            t0 = time.perf_counter()
            img, iinfo = load_bgr8(path, cfg["input"]["sixteen_bit"])
            if iinfo["grey"]:
                grey_fovs.append(fid)
                log.warning("[%d/%d] %s: grayscale image (%s); the detector and seg models are colour-trained, "
                            "expect fewer / poorer cells", k, len(fovs), fid,
                            f"{iinfo['src_channels']}-channel file" if iinfo["src_channels"] < 3 else "R=G=B")
            rows, coco, cl, nl, ovl, info = process_fov(img, fid, path.name, detector, seg, cfg, detector_b)
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
            meta = {"fov": fid, "source": str(path), "source_sha256": sha256(path), "image": iinfo,
                    "width": img.shape[1], "height": img.shape[0], "config_hash": chash, "finished_utc": now(), **info}
            write_fov_outputs(tmp, rows, coco, cl, nl, ovl, meta)
            if d.exists():
                shutil.rmtree(d)
            tmp.rename(d)
            dt_s = time.perf_counter() - t0
            times.append(dt_s); ok.append(fid); n_cells += len(rows)
            for r in rows:
                flag_counts.update(r["flags"].split(";") if r["flags"] else ["(none)"])
            drop_counts.update(d["reason"] for d in info["dropped"])
            tag_counts.update(r["det_tag"] for r in rows)
            status_counts.update(r["status"] for r in rows)
            reason_counts.update(x for r in rows for x in r["reasons"].split(";") if x)
            qc_counts.update(info["fov_qc"] or ["(none)"])
            if info["fov_qc"]:
                log.warning("[%d/%d] %s: image QC flags %s (FOV processed anyway) %s", k, len(fovs), fid,
                            ",".join(info["fov_qc"]), info["fov_qc_metrics"])
            n_l_only += len(info["l_only"])
            if not rows:
                log.warning("[%d/%d] %s: zero detections", k, len(fovs), fid)
            log.info("[%d/%d] %s: %d cells (%s), %.2fs", k, len(fovs), fid, len(rows),
                     "tiled" if info["detector"]["tiled"] else "full-frame", dt_s)
        except ImageLoadError as e:
            failed.append({"fov": fid, "source": str(path), "stage": "load", "error": str(e)})
            log.error("[%d/%d] %s: skipped, cannot load: %s", k, len(fovs), fid, e)
        except Exception as e:  # one bad FOV must not stop the run
            failed.append({"fov": fid, "source": str(path), "stage": "process", "error": f"{type(e).__name__}: {e}"})
            log.exception("[%d/%d] %s: failed", k, len(fovs), fid)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)

    # run-level cells.csv from every completed FOV of this listing (including ones done in earlier runs)
    all_rows = 0
    with open(out / "cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CELL_COLS); w.writeheader()
        for fid, _ in fovs:
            p = out / "fovs" / fid / "cells.csv"
            if (out / "fovs" / fid / "done.json").exists() and p.exists():
                with open(p, newline="") as g:
                    for r in csv.DictReader(g):
                        w.writerow(r); all_rows += 1
    # run-level l_only.csv (detector-B boxes with no detector-A partner; never segmented)
    n_l_csv = 0
    if mode != "single":
        with open(out / "l_only.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=L_ONLY_COLS); w.writeheader()
            for fid, _ in fovs:
                p = out / "fovs" / fid / "done.json"
                if p.exists():
                    for r in json.loads(p.read_text()).get("l_only", []):
                        w.writerow(r); n_l_csv += 1
    # run-level dropped.csv (contained boxes + cells that lost too much to overlaps), from the done.json files
    n_dropped = 0
    with open(out / "dropped.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DROP_COLS); w.writeheader()
        for fid, _ in fovs:
            p = out / "fovs" / fid / "done.json"
            if p.exists():
                for d in json.loads(p.read_text()).get("dropped", []):
                    b = d["box"]
                    w.writerow({"fov": fid, "reason": d["reason"], "cell_id": d["cell_id"], "conf": d["conf"],
                                "box_x1": b[0], "box_y1": b[1], "box_x2": b[2], "box_y2": b[3], "detail": d["detail"]})
                    n_dropped += 1

    tsum = {}
    if times:
        tsum = {"mean_s": round(statistics.mean(times), 3), "median_s": round(statistics.median(times), 3),
                "max_s": round(max(times), 3)}
    manifest.update({"status": "finished", "finished_utc": now(), "wall_time_s": round(time.time() - t_start, 2),
                     "counts": {"fovs_listed": len(fovs), "fovs_ok": len(ok), "fovs_failed": len(failed),
                                "fovs_skipped_already_done": len(skipped), "cells_this_run": n_cells,
                                "cells_in_cells_csv": all_rows, "ignored_files": len(ignored),
                                "dropped_this_run": dict(drop_counts), "dropped_in_dropped_csv": n_dropped,
                                "grayscale_fovs_this_run": len(grey_fovs),
                                "det_tags_this_run": dict(tag_counts), "l_only_this_run": n_l_only,
                                "status_this_run": dict(status_counts), "reasons_this_run": dict(reason_counts.most_common()),
                                "fov_qc_this_run": dict(qc_counts), "macenko_fallbacks_this_run": seg.fallbacks,
                                "l_only_in_l_only_csv": n_l_csv},
                     "flag_counts_this_run": dict(flag_counts.most_common()), "time_per_fov": tsum,
                     "grayscale_fovs": grey_fovs,
                     "failures": failed, "skipped_already_done": skipped, "ignored_files": ignored})
    prev.write_text(json.dumps(manifest, indent=1, default=str))

    log.info("=" * 70)
    log.info("SUMMARY  FOVs: %d ok, %d failed, %d skipped (already done)  |  cells this run: %d  (cells.csv: %d)",
             len(ok), len(failed), len(skipped), n_cells, all_rows)
    for fl, n in flag_counts.most_common():
        log.info("  flag %-16s %6d", fl, n)
    if mode != "single":
        log.info("  detection_mode %s: %s   l_only (B only, not segmented): %d -> l_only.csv", mode,
                 ", ".join(f"{k} {v}" for k, v in tag_counts.most_common()), n_l_only)
    log.info("  status: %s", ", ".join(f"{k} {status_counts.get(k, 0)}" for k in ("measurable", "review", "excluded")))
    for rs, n in reason_counts.most_common():
        log.info("  reason %-32s %6d", rs, n)
    log.info("  FOV image QC: %s", dict(qc_counts))
    if seg.variant in ("V1", "V2"):
        log.info("  Macenko fallbacks (crop left unchanged, training rule): %d", seg.fallbacks)
    for rs, n in drop_counts.most_common():
        log.info("  dropped (%s) %6d   -> dropped.csv", rs, n)
    if grey_fovs:
        log.warning("  %d grayscale FOV(s): models are colour-trained (%s)", len(grey_fovs), ", ".join(grey_fovs[:10]))
    if tsum:
        log.info("  time per FOV: mean %.2fs  median %.2fs  max %.2fs", tsum["mean_s"], tsum["median_s"], tsum["max_s"])
    for f in failed:
        log.info("  FAILED %s (%s): %s", f["fov"], f["stage"], f["error"])
    return manifest
