"""Load and validate config.yaml. Relative paths are resolved against the config file's folder."""
import copy
from pathlib import Path

import yaml


class ConfigError(ValueError):
    pass


REQUIRED = {
    "input": ["extensions", "recursive", "sixteen_bit"],
    "detector": ["weights", "imgsz", "conf", "nms_iou", "max_det", "min_box_px", "tiling", "containment_thr",
                 "containment_min_area_ratio"],
    "crop": ["margin", "edge_mode"],
    "seg": ["code_dir", "weights", "model_config", "batch_size", "thr_nucleus", "thr_cell"],
    "postprocess": ["clip_tolerance_px", "nucleus_min_component_px", "cell_at_edge_px", "drop_lost_frac"],
    "output": ["overlay", "overlay_max_side", "coco_polygon_epsilon_px"],
    "logging": ["level"],
}
PATH_KEYS = [("detector", "weights"), ("seg", "code_dir"), ("seg", "weights"), ("seg", "model_config"),
             ("detector_b", "weights"), ("detector_c", "weights"), ("seg", "preprocess_code"), ("seg", "preprocessing_dir")]
V3_SECTIONS = {"qc": ["blur_lapvar_min", "mean_grey_min", "mean_grey_max", "dark_frac_max", "stained_frac_min", "stained_frac_max"],
               "size": ["box_area_min", "box_area_max", "cell_area_review_min", "cell_area_review_max",
                        "cell_area_implausible_min", "cell_area_implausible_max", "nucleus_area_implausible_max"],
               "gate": ["occluded_review", "s_only_review_conf", "nc_implausible_max"]}
C_CLASSES = ("cluster", "debris")   # detector C classes the pipeline may use (never `cell`)
MODES = ("single", "consensus_flag", "consensus_filter")


def load(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: not a YAML mapping")
    cfg = copy.deepcopy(cfg)
    cfg["_config_path"] = str(path)
    for sec, keys in REQUIRED.items():
        if sec not in cfg or not isinstance(cfg[sec], dict):
            raise ConfigError(f"config: missing section '{sec}'")
        for k in keys:
            if k not in cfg[sec]:
                raise ConfigError(f"config: missing key '{sec}.{k}'")
    for sec, keys in V3_SECTIONS.items():
        if sec not in cfg or not isinstance(cfg[sec], dict):
            raise ConfigError(f"config: missing section '{sec}'")
        for k in keys:
            if k not in cfg[sec] or not isinstance(cfg[sec][k], (int, float)) or isinstance(cfg[sec][k], bool):
                raise ConfigError(f"config: '{sec}.{k}' missing or not a number")
    cfg["seg"].setdefault("variant", "auto"); cfg["seg"].setdefault("preprocessing_dir", None)
    if cfg["seg"]["variant"] not in ("auto", "V0", "V1", "V2", "V3", "box"):
        raise ConfigError("seg.variant must be auto | V0 | V1 | V2 | V3 | box")
    if "preprocess_code" not in cfg["seg"]:
        raise ConfigError("config: missing key 'seg.preprocess_code'")
    for k in ("device", "seed"):
        if k not in cfg:
            raise ConfigError(f"config: missing key '{k}'")
    base = path.parent
    # cross-check settings (optional sections: absent = single-detector mode)
    cfg.setdefault("consensus", {"detection_mode": "single", "match_iou": 0.5})
    if cfg["consensus"].get("detection_mode") not in MODES:
        raise ConfigError(f"consensus.detection_mode must be one of {' | '.join(MODES)}")
    if not 0 < float(cfg["consensus"].get("match_iou", -1)) <= 1:
        raise ConfigError("consensus.match_iou must be in (0, 1]")
    if cfg["consensus"]["detection_mode"] != "single":
        db = cfg.get("detector_b")
        if not isinstance(db, dict) or any(k not in db for k in ("weights", "imgsz", "conf", "nms_iou", "max_det")):
            raise ConfigError("consensus modes need a detector_b section with weights, imgsz, conf, nms_iou, max_det")
        if not 0 < db["conf"] < 1 or not 0 < db["nms_iou"] <= 1:
            raise ConfigError("detector_b.conf and detector_b.nms_iou must be in (0, 1]")
    dcc = cfg.get("detector_c")
    if dcc is not None:
        if not isinstance(dcc, dict) or any(k not in dcc for k in ("enabled", "weights", "imgsz", "conf", "nms_iou",
                                                                    "max_det", "classes", "near_frac")):
            raise ConfigError("detector_c needs enabled, weights, imgsz, conf, nms_iou, max_det, classes, near_frac")
        if not 0 < dcc["conf"] < 1 or not 0 < dcc["nms_iou"] <= 1 or not 0 < dcc["near_frac"] <= 1:
            raise ConfigError("detector_c.conf, nms_iou and near_frac must be in (0, 1]")
        if not dcc["classes"] or not all(c in C_CLASSES for c in dcc["classes"]):
            raise ConfigError(f"detector_c.classes: a non-empty subset of {list(C_CLASSES)} (its cell class is only a vote)")
        dcc.setdefault("cell_conf", 0.25)
        if not 0 < dcc["cell_conf"] < 1:
            raise ConfigError("detector_c.cell_conf must be in (0, 1)")
        # one C run: cluster / debris at conf, its cell class (votes only) at cell_conf; per-class thresholds
        dcc["class_conf"] = {**{c: float(dcc["conf"]) for c in dcc["classes"]}, "cell": float(dcc["cell_conf"])}
    v = cfg.setdefault("voting", {"enabled": False})
    v.setdefault("enabled", False); v.setdefault("min_votes", 2); v.setdefault("match_iou", 0.5)
    v.setdefault("skip_in_cluster_debris", True)
    if not isinstance(v["enabled"], bool) or v["min_votes"] not in (1, 2, 3) or not 0 < float(v["match_iou"]) <= 1:
        raise ConfigError("voting: enabled true|false, min_votes 1|2|3, match_iou in (0, 1]")
    for sec, k in PATH_KEYS:
        if sec not in cfg:
            continue
        v = cfg[sec][k]
        if v is not None:
            p = Path(str(v)).expanduser()
            cfg[sec][k] = str(p if p.is_absolute() else (base / p).resolve())
    # YOLO26l (>100 MB) is distributed as a GitHub Release asset, not in git. Without it the consensus stage
    # switches itself off (detection_mode -> single) instead of failing; the runner logs this.
    if cfg["consensus"]["detection_mode"] != "single" and not Path(cfg["detector_b"]["weights"]).is_file():
        cfg["consensus"]["auto_off"] = (f"detector_b weights not found ({cfg['detector_b']['weights']}); "
                                        f"{cfg['consensus']['detection_mode']} -> single. "
                                        "Fetch them with scripts/fetch_release_weights.py")
        cfg["consensus"]["detection_mode"] = "single"
    if dcc is not None and dcc["enabled"] and not Path(dcc["weights"]).is_file():
        dcc["auto_off"] = f"detector_c weights not found ({dcc['weights']}); detector C off"
        dcc["enabled"] = False
    validate(cfg)
    return cfg


def validate(c):
    d, t, cr, s, pp = c["detector"], c["detector"]["tiling"], c["crop"], c["seg"], c["postprocess"]
    if c["device"] not in ("auto", "cuda", "cpu"):
        raise ConfigError("device must be auto | cuda | cpu")
    if c["input"]["sixteen_bit"] not in ("cv2", "scale", "max"):
        raise ConfigError("input.sixteen_bit must be cv2 | scale | max")
    if not 0 < d["conf"] < 1 or not 0 < d["nms_iou"] <= 1:
        raise ConfigError("detector.conf and detector.nms_iou must be in (0, 1]")
    if int(d["imgsz"]) < 0 or int(d["max_det"]) < 1 or d["min_box_px"] < 0:
        raise ConfigError("detector.imgsz >= 0, max_det >= 1, min_box_px >= 0")
    for k in ("mode", "trigger_factor", "tile_px", "overlap_px", "edge_px"):
        if k not in t:
            raise ConfigError(f"config: missing key 'detector.tiling.{k}'")
    # YAML 1.1 reads a bare `off` / `on` as a boolean
    t["mode"] = {False: "off", True: "on"}.get(t["mode"], t["mode"])
    if t["mode"] not in ("off", "on", "auto"):
        raise ConfigError("detector.tiling.mode must be off | on | auto")
    if not 0 <= t["overlap_px"] < t["tile_px"]:
        raise ConfigError("detector.tiling: need 0 <= overlap_px < tile_px")
    if not 0 <= cr["margin"] <= 2:
        raise ConfigError("crop.margin must be in [0, 2]")
    if cr["edge_mode"] not in ("clamp", "reflect"):
        raise ConfigError("crop.edge_mode must be clamp | reflect")
    if int(s["batch_size"]) < 1:
        raise ConfigError("seg.batch_size must be >= 1")
    for k in ("thr_nucleus", "thr_cell"):
        if not 0 < s[k] < 1:
            raise ConfigError(f"seg.{k} must be in (0, 1)")
    ct = pp["clip_tolerance_px"]
    if not (ct == "margin" or (isinstance(ct, (int, float)) and not isinstance(ct, bool) and ct >= 0)):
        raise ConfigError("postprocess.clip_tolerance_px must be 'margin' or a number >= 0")
    if pp["nucleus_min_component_px"] < 0 or pp["cell_at_edge_px"] < 0:
        raise ConfigError("postprocess values must be >= 0")
    if pp["drop_lost_frac"] is not None and not 0 <= pp["drop_lost_frac"] < 1:
        raise ConfigError("postprocess.drop_lost_frac must be null or in [0, 1)")
    if d["containment_thr"] is not None and not 0 <= d["containment_thr"] <= 1:
        raise ConfigError("detector.containment_thr must be null or in [0, 1]")
    if not 0 <= d["containment_min_area_ratio"] <= 1:
        raise ConfigError("detector.containment_min_area_ratio must be in [0, 1]")
    files = [("detector", "weights"), ("seg", "weights")]
    if c["consensus"]["detection_mode"] != "single":
        files.append(("detector_b", "weights"))
    for sec, k in files:
        if not Path(c[sec][k]).is_file():
            raise ConfigError(f"{sec}.{k}: file not found: {c[sec][k]}")
    if not Path(s["preprocess_code"]).is_file():
        raise ConfigError(f"seg.preprocess_code: file not found: {s['preprocess_code']}")
    if s["preprocessing_dir"] is not None and not Path(s["preprocessing_dir"]).is_dir():
        raise ConfigError(f"seg.preprocessing_dir: folder not found: {s['preprocessing_dir']}")
    if not (Path(s["code_dir"]) / "tinyseg.py").is_file():
        raise ConfigError(f"seg.code_dir: no tinyseg.py in {s['code_dir']}")
    if s["model_config"] is not None and not Path(s["model_config"]).is_file():
        raise ConfigError(f"seg.model_config: file not found: {s['model_config']}")
    c["input"]["extensions"] = [e.lower() if e.startswith(".") else "." + e.lower() for e in c["input"]["extensions"]]
