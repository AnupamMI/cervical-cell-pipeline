# FOV web app (research demo, not for diagnosis)

Single-FOV inference in the browser: upload one Pap-smear field of view (or pick a sample), get every detected cell
with its cell + nucleus outline, a measurable / review / excluded status with reasons, cluster / debris regions from
an experimental third detector, and downloads (overlay PNG, cells.csv, COCO). Scope: single FOV images only, no
stitched slides, no job queue.

- **Hugging Face Space (private, not deployed yet):** `AnupamZZZZ/cervical-cell-demo`. Creating it was refused
  with *402 Payment Required*: Hugging Face now requires a PRO subscription for Docker (and Gradio) Spaces on the
  free cpu-basic hardware. Everything is ready: after subscribing, `scripts/deploy_space.ps1` creates the private
  Space, pushes code + ONNX weights (LFS), waits for the build and smoke-tests it. The deployed app will be at
  `https://anupamzzzz-cervical-cell-demo.hf.space` (sign-in required while private).
- **API docs (OpenAPI):** `/api/docs` on any running instance

## Architecture

```
browser (React + TS + Vite)                          FastAPI (app/backend, one uvicorn worker, gzip)
  landing: drop zone + samples (static first paint)    POST /api/infer ─┐   one request at a time (busy -> 503)
  viewer:  OpenSeadragon + SVG overlay                                  ▼
  side panel / downloads           ◀── JSON ──   ncmask.fov.process_fov   (the pipeline itself)
                                                   ├ OnnxDetector  A  YOLO26s        detector_a.onnx  (the only cell source)
                                                   ├ OnnxDetector  B  YOLO26l @1280  detector_b.onnx  (consensus tag only)
                                                   ├ OnnxDetector  C  YOLO26s @1024  detector_c.onnx  (cluster / debris, info only)
                                                   └ OnnxSegModel     TinyUNet       seg/seg.onnx
                                                 ResultStore: temp dir, 15 min TTL (overlay, CSV, COCO, display JPEG)
```

- **Same pipeline, different runtime.** The app calls `ncmask.fov.process_fov`, the function `run.py` uses. Only
  the model calls are swapped for ONNX Runtime (`ncmask/onnx_backend.py`): `OnnxDetector` subclasses
  `Detector` (only the forward pass is new: Ultralytics' rect letterbox, one-to-many head and agnostic NMS in numpy),
  `OnnxSegModel` subclasses `SegModel` (crop, input variants V0-V3/box, Macenko etc. unchanged). Image QC,
  containment suppression, post-processing, overlaps, the gate and the COCO/CSV outputs are the pipeline's own code.
- **No torch at runtime.** The image installs `requirements-app.txt` only (onnxruntime CPU, FastAPI, OpenCV
  headless, scikit-image, SciPy). The vendored training code imports torch at module level; `torch_shim()` lets it
  import without torch and is removed from `sys.modules` straight after. CUDA: install `onnxruntime-gpu`; providers
  are chosen automatically (`NCMASK_DEVICE=auto|cuda|cpu`).
- **Consensus off** = detection mode `single` (YOLO26l never runs; `det_tag` = `single`). On = `consensus_flag`:
  every YOLO26s box is segmented and tagged `consensus` / `s_only`; YOLO26l never removes a cell.
- **Clusters off** = detector C never runs (`clusters=false`); on (default) it adds boxes and info tags only.

### Voting: which cells get segmented (app default)

With `vote=true` (the app's default; a "Vote" switch turns it off), a box is segmented only if **at least 2 of the
3 detectors** have it (one-to-one match, IoU >= 0.5): A = YOLO26s, B = YOLO26l @1280, C = detector C's own `cell`
class (conf >= 0.25; a vote only, never a box of its own). Cells YOLO26s missed but YOLO26l **and** C both found are
candidates too (`det_tag` `bc_only`, YOLO26l's box). A voted cell with >= 50 % of its box inside a cluster or debris
box is **not** segmented either. Cluster and debris boxes are independent detections and are never segmented.
Everything that is not segmented stays visible: status **`not_segmented`**, reason `not_voted` / `inside_cluster` /
`inside_debris`, its votes (`votes`, `voters` such as `AC`), no mask. If a detector is switched off, the requirement
is min(2, detectors that ran): A + B or A + C must both agree; A alone segments everything.

On the 10 samples (318 boxes): 275 segmented, 43 not segmented (29 found by one detector only, 14 inside a cluster /
debris box), 2 `bc_only` cells added; measurable 130 -> 123. Agreement pattern of the segmented cells: A+B+C 135,
A+C 138, A+B 1, B+C 1 (detector C's cell class confirms many YOLO26s cells that YOLO26l misses). With `vote=false`
the app behaves exactly like pipeline v3 (every YOLO26s box segmented, C information only). The batch pipeline
(`run.py`) keeps `voting.enabled: false` in `config.yaml` so its outputs still match the `samples/expected`
references; set it to `true` to vote there too.

### Detector C (clusters & debris): what it does and doesn't do

`models/detector_c/yolo26s_1024_ccd_best.pt` (SHA256 `a72ae657…df23255`, also in `models/WEIGHTS.md` and
`/api/health`): YOLO26s trained at imgsz 1024 on classes `[cell, cluster, debris]`, 100 epochs (freeze 10, AdamW,
cos_lr); classes and imgsz verified from the checkpoint. **Test mAP50: cell 0.829, cluster 0.503, debris 0.539**,
with many background false positives at conf 0.25.

Below: detector C with voting **off** (`vote=false`). With voting on, its cell class is the third vote and its
cluster / debris boxes block segmentation (above).

- **Does:** run full-frame on the FOV as decoded (the same image detectors A and B see; letterboxed to 1024, i.e.
  2028x1520 -> 1024x768), keep only its `cluster` and `debris` boxes with conf >= **0.5** (`detector_c.conf`), per-class
  NMS. Each cell gets the info tags `near_cluster` / `near_debris` (the C box ids) when >= 50 % of its detector box
  (`detector_c.near_frac`) lies inside a C box of that class. C boxes are recorded in the JSON (`clusters.boxes`),
  `cells.csv` (rows with `object` = `cluster` / `debris`, `cell_id` = `C1`, `C2`, …, no status), COCO (categories 3
  `cluster` and 4 `debris`, `attributes.info_only: true`) and the overlay PNG (dotted, muted outlines).
- **Doesn't:** its `cell` class is never used (YOLO26s stays the only source of cells), and it never changes
  measurable / review / excluded, reasons or flags: the tags are computed after the gate and are not gate inputs.
  The parity check verifies this on every sample (statuses, reasons and flags identical with C on and off), and
  `tests/test_app.py::test_clusters_off_same_statuses` checks it through the API. Reason: cluster/debris mAP50 of
  ~0.5 is not validated enough to exclude or review cells. The UI says so: "Experimental: shown for information,
  doesn't change cell status."

### ONNX parity (`scripts/check_onnx_parity.py`, `tests/test_onnx_parity.py`)

Full pipeline, PyTorch vs ONNX, on all 10 `samples/` FOVs (CPU). Thresholds: box IoU >= 0.99 with the same count
(detectors A, B and C separately; C also the same class per matched pair), mask Dice >= 0.999 (per FOV and per
cell), identical gate status for every cell, identical `near_cluster` / `near_debris`. Result
(`reports/onnx_parity.json`): **10/10 pass**. Worst box IoU 0.999996 (YOLO26s) / 0.999998 (YOLO26l) / 0.999992
(detector C, 1-13 boxes per FOV, same counts), worst per-cell mask Dice 0.99999, every FOV-level Dice 1.000000,
0 status differences, and detector C changes no status / reason / flag on any FOV.

Two details that matter for parity: (1) with Ultralytics 8.4 these YOLO26 checkpoints (detector C included) run in
PyTorch with `head.end2end = False` (one-to-many head + NMS), so the export keeps that head (`nms` left at `None`;
`nms=False` would force the one-to-one end-to-end head, which gives different boxes); `export_onnx.py` now refuses
an export whose metadata says `end2end`; (2) the export has dynamic H/W so the rect letterbox (2028x1520 -> 640x480 /
1280x960 / 1024x768) matches PyTorch exactly.

## API

| method | path | |
|---|---|---|
| POST | `/api/infer?consensus=true\|false&clusters=true\|false&vote=true\|false` | all default true. multipart: `file` (.tif/.tiff/.png/.jpg/.jpeg, <= 25 MB) **or** `sample` (a name from `/api/samples`). Synchronous. |
| GET | `/api/infer/{id}/overlay.png` | full-resolution overlay (status-coloured boxes, outlines, nuclei) |
| GET | `/api/infer/{id}/cells.csv` | the pipeline's `cells.csv` columns (+ `near_cluster`, `near_debris`, `object`); detector-C boxes as extra rows |
| GET | `/api/infer/{id}/coco.json[?layer=full]` | COCO, exclusive (display) layer by default; `layer=full` = measurement layer (overlaps allowed) |
| GET | `/api/infer/{id}/image.jpg` | the decoded FOV as JPEG (what the viewer displays; browsers cannot show TIFF) |
| GET | `/api/samples` | sample FOVs (+ `/api/samples/{name}/thumb.jpg`) |
| GET | `/api/health` | models (ONNX SHA256 + source checkpoint SHA256; detector C: classes, conf, test mAP50), providers, versions, bundle check, typical stage times |

`POST /api/infer` response (abridged):

```json
{
  "id": "…", "expires_utc": "…",
  "size": {"width": 2028, "height": 1520, "expected": [2028, 1520], "is_expected": true},
  "image_qc": {"ok": true, "flags": [], "metrics": {"lap_var": 18.4, "mean_grey": 201.2, …}},
  "consensus": {"requested": true, "used": true, "l_only": 1},
  "clusters": {"requested": true, "used": true, "info_only": true, "conf": 0.5, "near_frac": 0.5,
               "boxes": [{"id": "C1", "class": "cluster", "conf": 0.89, "box": [x1, y1, x2, y2]}, …]},
  "voting": {"used": true, "detectors_run": 3, "votes_needed": 2, "skip_in_cluster_debris": true, "bc_only": 0},
  "cells": [{"id": 1, "box": [x1, y1, x2, y2], "conf": 0.9, "det_tag": "consensus", "b_conf": 0.88, "ab_iou": 0.93,
             "status": "review", "reasons": ["multi_nucleus"], "flags": ["multi_nucleus"],
             "cell_area_px": 63405, "nucleus_area_px": 1210, "nc_ratio": 0.0195, "occluded_frac": 0.0,
             "votes": 3, "voters": "ABC", "near_cluster": [], "near_debris": [],
             "cell_polygons": [[[x, y], …]], "nucleus_polygons": [[[x, y], …]]}],
  "counts": {"total": 38, "status": {"measurable": 6, "review": 6, "excluded": 13, "not_segmented": 13},
             "votes": {"3/3": 14, "2/3": 16, "1/3": 8},
             "reasons": {"cell_at_edge": 17, …}, "det_tag": {…}, "dropped": {…},
             "clusters": {"cluster": 4, "debris": 9}, "near": {"cluster": 10, "debris": 0}},
  "timings_ms": {"detect": …, "consensus": …, "clusters": …, "segment": …, "gate": …, "total": …, "decode": …, "request": …},
  "warnings": ["Image is 1014x760; the models were built for 2028x1520 FOVs …"],
  "downloads": {"overlay.png": "/api/infer/{id}/overlay.png", …}
}
```

Polygons in the JSON are the exclusive (display) layer, simplified by 1 px for the viewer; the COCO download keeps
exact outlines. `gate` time = everything except the models (QC, crops, post-processing, overlaps, gate,
polygons, overlay).

Errors: `{"error": code, "detail": message}`. 400 `bad_request` / `unreadable_image`, 404 `not_found` / `expired`,
413 `too_large`, 415 `unsupported_type`, 422 (invalid `consensus` / `clusters` / `vote`), **503 `busy`** with `Retry-After: 5` when another
image is being analysed (one at a time; the UI shows "busy, retry"), 500 `internal_error`.

**Privacy.** An upload is written to a private temp file only to decode it and is deleted immediately after. Results
live in a private temp folder for 15 minutes (swept every 30 s, also checked on every access) and are deleted on
shutdown. Logs contain the result id prefix, image size, cell count and timings, never file names or pixels.
Result ids are 24 random URL-safe characters.

## Frontend

React 19 + TypeScript + Vite, OpenSeadragon. Calm, light and minimal: off-white `#F5F5F7` background, white cards
(radius 22 px, very soft shadows), near-black text, **one accent** (deep indigo `#3D35A8`) for primary actions,
switches, selection and focus, system font stack (`-apple-system, "SF Pro", "Inter", system-ui`), type scale
12 / 14 / 15 / 17 / 22 / 44 px. No decorative effects: the 3D cell, ShaderGradient and glass layers were removed
together with their dependencies (`three`, `@react-three/*`, `@shadergradient/react`, `camera-controls`,
`three-stdlib`); the only runtime dependencies are React and OpenSeadragon.

- **Landing:** headline, one-line explanation, "Research demo, not for diagnosis." in small muted text, a drop zone,
  the two switches (Consensus check, Clusters & debris (detector C), both on by default), five sample thumbnails.
  The header + intro are static HTML in `index.html` (same markup React renders), so the first paint needs no JS.
- **Analysis:** progress bar + the stage names (Detect, Consensus, Clusters & debris, Vote, Segment, Quality gate; stages
  that are off are hidden). The call is synchronous, so the current stage is an estimate from the warm-up timings.
- **Viewer:** the image fills its card edge to edge (4:3 on phones). A slim segmented control toggles layers
  (Detections · Masks · Status · Clusters & debris · Flagged only); Consensus, Clusters and Vote are switches (they
  re-run the image). Not-segmented boxes are thin dotted grey; zoom − / + / fit. Detector-C boxes are dotted, muted slate-blue (cluster) / umber (debris) outlines
  with a light tint and a label ("Cluster 0.89") that keeps its on-screen size at any zoom.
- **Side panel:** status counts (Measurable / Review / Excluded / Not segmented), a Votes column (e.g. 2/3) in the
  table and "Found by" in the cell details, Clusters & debris counts (+ cells near each) with the note "Experimental: shown for
  information, doesn't change cell status.", image-check warnings, top reasons, selected-cell details, the sortable
  cell table (select a row to zoom), downloads as quiet secondary buttons, stage timings.
- **Status colours** stay distinct from the accent and from detector C: text green `#1B7A36` / orange `#A34A00` /
  red `#C0261C` on light tints (all >= 5:1); on the image, brighter strokes `#2FBF4F` / `#FF9500` / `#FF3B30`. Every
  status also has a text label.
- **Motion:** short fades / slides (150-350 ms) only; all animation and transition is off under
  `prefers-reduced-motion`, and OpenSeadragon then zooms without animation.
- **Keyboard:** every control is a button / switch / pressed-toggle with a visible focus ring; the viewer canvas
  takes OpenSeadragon's keyboard navigation (arrows, +/−); table rows have a button, Enter zooms to the cell.

### UI checks (Playwright, Chromium)

Screens in `reports/app_screens/`: landing, analysis and viewer with results (clusters / debris shown) at 1440x900
and 390x844, plus a selected cell (desktop) and the side panel (phone). Issues found and fixed during the check:
cluster/debris swatches too faint, C labels outside the image at the top edge, overlay spilling over the legend at
high zoom, wrapped values in the selected-cell grid, "Flagged only" wrapping alone on phones, grey bands around the
phone image, duplicated "not for diagnosis" note on the landing, clipped table column.

- **Lighthouse 12.8.2 (landing):** mobile performance **99**, accessibility **100**, best practices 100;
  desktop **100 / 100 / 100** (FCP 1.5 s / LCP 1.8 s / TBT 10 ms / CLS 0 on the mobile profile). Getting there
  needed gzip on the FastAPI app (the 244 KB JS bundle was served uncompressed: mobile 92 before) and the static
  first paint.
- **Contrast:** computed for every visible text element against its composited background on landing, analysis and
  viewer (340 elements): all >= 4.5:1.

## Run locally

```powershell
# backend + models (Python 3.12)
python -m venv .venv; .\.venv\Scripts\python -m pip install -r requirements-app.txt   # runtime only (no torch)
# ONNX bundle: export once from the PyTorch weights (needs the training env: torch + ultralytics + onnx), or copy
# models/onnx/ from the Space
.\.venv\Scripts\python scripts\export_onnx.py
# frontend
cd app\frontend; npm ci; npm run build; cd ..\..
# serve (app + API on http://127.0.0.1:7860)
.\.venv\Scripts\python -m uvicorn app.backend.main:app --port 7860
```

Frontend dev with hot reload: run uvicorn as above, then `cd app/frontend && npm run dev` (Vite proxies `/api`).
Tests: `python -m pytest tests/test_app.py` (API) and `tests/test_onnx_parity.py` (needs torch + ultralytics).
Settings (env): `NCMASK_CONFIG`, `NCMASK_BUNDLE`, `NCMASK_DEVICE`, `NCMASK_THREADS`, `NCMASK_TTL_S`,
`NCMASK_WARMUP=0` (skip the start-up warm-up run).

## Deploy (Hugging Face Space)

```powershell
$env:HF_TOKEN = "<write token>"          # never printed or logged by the scripts
.\scripts\deploy_space.ps1                 # or: scripts/deploy_space.sh
```

`scripts/deploy_space.py` checks that the ONNX bundle was exported from exactly the weights `config.yaml` names
(SHA256), creates the Space if needed (**private**, Docker SDK, CPU basic; refuses to push to a public Space),
stages the code + `models/onnx/` + `samples/*.tiff` + a Space README + LFS `.gitattributes`, syncs it (files no
longer staged are deleted), waits for the remote Docker build (prints the build log on failure) and runs
`scripts/space_smoke.py` (health incl. detector C loaded, frontend, every sample with and without consensus,
downloads, 25 MB limit;
writes `reports/space_smoke.json`). The Space contains its own weights: no runtime dependency on GitHub.

## Swap the segmenter (one command)

```powershell
.\scripts\swap_segmenter.ps1 -Weights models/seg/V4/best.pt          # variant from its config.json (auto)
.\scripts\swap_segmenter.ps1 -Weights models/seg/V1_v3/best.pt -Variant V1
.\scripts\swap_segmenter.ps1                                          # re-export whatever config.yaml names
```

(`scripts/swap_segmenter.sh` takes `--weights/--variant/--no-deploy`.) Steps: update `seg.weights` /
`seg.variant` in `config.yaml` (comments kept) → `export_onnx.py --only seg` (copies `config.json` and
`stain_ref.json` / `h_stats.json` next to `seg.onnx`) → full parity check on `samples/` → deploy + wait + smoke test.
It stops at the first failure; if parity fails, `config.yaml` is restored and nothing is deployed. Commit the
`config.yaml` change afterwards.

## Limitations

- **Research demo, not for diagnosis.** No clinical validation; see the pipeline's limitations in the main README.
- **Single FOV only**: no stitched slides or tiling UI, no batch, no job queue. One request at a time per instance;
  concurrent users get "busy, retry" (503).
- **Synchronous**: the analysis screen's stage progress is an estimate from the warm-up timings in `/api/health`;
  the real per-stage times are shown with the result.
- **CPU time per FOV** (`tools/bench_app.py`, 10 samples, engine call only, Ryzen 5 5600H, ONNX Runtime CPU;
  `reports/app_timing_cpu_*.json`):

  | detectors | median, all 12 threads | median, 2 threads (Space-like) | detector C stage |
  |---|---|---|---|
  | all three (A + B consensus + C clusters) | 3.67 s | 5.26 s | 287 / 485 ms |
  | consensus off (A + C) | 2.47 s | 2.37 s | 361 / 479 ms |
  | clusters off (A + B) | 3.77 s | 4.60 s | 0 / 0 ms |
  | both off (A only) | 1.78 s | 1.96 s | 0 / 0 ms |

  YOLO26l @1280 is the slowest stage (~1.6-2.8 s); detector C adds ~0.3-0.5 s. Medians of single runs on a laptop
  that was also serving the app, so differences under ~0.3 s are noise (e.g. "clusters off" vs "all three" with all
  threads). Not yet measured on the Space (not deployed, see above); its shared vCPUs are likely slower.
- **Detector C is experimental**: mAP50 ~0.5 for cluster / debris; boxes are information only (see above).
- **Image size**: the models were built for 2028x1520 FOVs at ~0.155 µm/px; other sizes run with a warning (and
  pixel areas are then not comparable). Images above 40 MP are refused (decompression-bomb guard).
- **Results expire after 15 minutes** and are lost when the Space restarts (temp storage, by design).
