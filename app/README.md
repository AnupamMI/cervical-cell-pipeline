# FOV web app (research demo, not for diagnosis)

Single-FOV inference in the browser: upload one Pap-smear field of view (or pick a sample), get every detected cell
with its cell + nucleus outline, a measurable / review / excluded status with reasons, and downloads (overlay PNG,
cells.csv, COCO). Scope: single FOV images only, no stitched slides, no job queue.

- **Hugging Face Space (private, not deployed yet):** `AnupamZZZZ/cervical-cell-demo`. Creating it was refused
  with *402 Payment Required*: Hugging Face now requires a PRO subscription for Docker (and Gradio) Spaces on the
  free cpu-basic hardware. Everything is ready: after subscribing, `scripts/deploy_space.ps1` creates the private
  Space, pushes code + ONNX weights (LFS), waits for the build and smoke-tests it. The deployed app will be at
  `https://anupamzzzz-cervical-cell-demo.hf.space` (sign-in required while private).
- **API docs (OpenAPI):** `/api/docs` on any running instance

## Architecture

```
browser (React + TS + Vite)                          FastAPI (app/backend, one uvicorn worker)
  landing: ShaderGradient + R3F cell (lazy)            POST /api/infer ─┐   one request at a time (busy -> 503)
  viewer:  OpenSeadragon + SVG overlay                                  ▼
  side panel / downloads           ◀── JSON ──   ncmask.fov.process_fov   (the pipeline itself, unchanged)
                                                   ├ OnnxDetector  YOLO26s        models/onnx/detector_a.onnx
                                                   ├ OnnxDetector  YOLO26l @1280  models/onnx/detector_b.onnx  (tag only)
                                                   └ OnnxSegModel  TinyUNet       models/onnx/seg/seg.onnx
                                                 ResultStore: temp dir, 15 min TTL (overlay, CSV, COCO, display JPEG)
```

- **Same pipeline, different runtime.** The app calls `ncmask.fov.process_fov`, the function `run.py` uses. Only
  the three model calls are swapped for ONNX Runtime (`ncmask/onnx_backend.py`): `OnnxDetector` subclasses
  `Detector` (only the forward pass is new: Ultralytics' rect letterbox, one-to-many head and agnostic NMS in numpy),
  `OnnxSegModel` subclasses `SegModel` (crop, input variants V0-V3/box, Macenko etc. unchanged). Image QC,
  containment suppression, post-processing, overlaps, the gate and the COCO/CSV outputs are the pipeline's own code.
- **No torch at runtime.** The image installs `requirements-app.txt` only (onnxruntime CPU, FastAPI, OpenCV
  headless, scikit-image, SciPy). The vendored training code imports torch at module level; `torch_shim()` lets it
  import without torch and is removed from `sys.modules` straight after. CUDA: install `onnxruntime-gpu`; providers
  are chosen automatically (`NCMASK_DEVICE=auto|cuda|cpu`).
- **Consensus off** = detection mode `single` (YOLO26l never runs; `det_tag` = `single`). On = `consensus_flag`:
  every YOLO26s box is segmented and tagged `consensus` / `s_only`; YOLO26l never removes a cell.

### ONNX parity (`scripts/check_onnx_parity.py`, `tests/test_onnx_parity.py`)

Full pipeline, PyTorch vs ONNX, on all 10 `samples/` FOVs (CPU). Thresholds: box IoU >= 0.99 with the same count,
mask Dice >= 0.999 (per FOV and per cell), identical gate status for every cell. Result
(`reports/onnx_parity.json`): **10/10 pass**. Worst box IoU 0.999996 (YOLO26s) / 0.999998 (YOLO26l), worst
per-cell mask Dice 0.99999, every FOV-level Dice 1.000000, 0 status differences.

Two details that matter for parity: (1) with Ultralytics 8.4.183 these YOLO26 checkpoints run in PyTorch with
`head.end2end = False` (one-to-many head + NMS), so the export keeps that head (`nms` left at `None`; `nms=False`
would force the one-to-one end-to-end head, which gives different boxes); (2) the export has dynamic H/W so the
rect letterbox (2028x1520 -> 640x480 / 1280x960) matches PyTorch exactly.

## API

| method | path | |
|---|---|---|
| POST | `/api/infer?consensus=true\|false` | multipart: `file` (.tif/.tiff/.png/.jpg/.jpeg, <= 25 MB) **or** `sample` (a name from `/api/samples`). Synchronous. |
| GET | `/api/infer/{id}/overlay.png` | full-resolution overlay (status-coloured boxes, outlines, nuclei) |
| GET | `/api/infer/{id}/cells.csv` | the pipeline's `cells.csv` columns |
| GET | `/api/infer/{id}/coco.json[?layer=full]` | COCO, exclusive (display) layer by default; `layer=full` = measurement layer (overlaps allowed) |
| GET | `/api/infer/{id}/image.jpg` | the decoded FOV as JPEG (what the viewer displays; browsers cannot show TIFF) |
| GET | `/api/samples` | sample FOVs (+ `/api/samples/{name}/thumb.jpg`) |
| GET | `/api/health` | models (ONNX SHA256 + source checkpoint SHA256), providers, versions, bundle check, typical stage times |

`POST /api/infer` response (abridged):

```json
{
  "id": "…", "expires_utc": "…",
  "size": {"width": 2028, "height": 1520, "expected": [2028, 1520], "is_expected": true},
  "image_qc": {"ok": true, "flags": [], "metrics": {"lap_var": 18.4, "mean_grey": 201.2, …}},
  "consensus": {"requested": true, "used": true, "l_only": 1},
  "cells": [{"id": 1, "box": [x1, y1, x2, y2], "conf": 0.9, "det_tag": "consensus", "b_conf": 0.88, "ab_iou": 0.93,
             "status": "review", "reasons": ["multi_nucleus"], "flags": ["multi_nucleus"],
             "cell_area_px": 63405, "nucleus_area_px": 1210, "nc_ratio": 0.0195, "occluded_frac": 0.0,
             "cell_polygons": [[[x, y], …]], "nucleus_polygons": [[[x, y], …]]}],
  "counts": {"total": 38, "status": {"measurable": 9, "review": 11, "excluded": 18},
             "reasons": {"cell_at_edge": 17, …}, "det_tag": {…}, "dropped": {…}},
  "timings_ms": {"detect": …, "consensus": …, "segment": …, "gate": …, "total": …, "decode": …, "request": …},
  "warnings": ["Image is 1014x760; the models were built for 2028x1520 FOVs …"],
  "downloads": {"overlay.png": "/api/infer/{id}/overlay.png", …}
}
```

Polygons in the JSON are the exclusive (display) layer, simplified by 1 px for the viewer; the COCO download keeps
exact outlines. `gate` time = everything except the three models (QC, crops, post-processing, overlaps, gate,
polygons, overlay).

Errors: `{"error": code, "detail": message}`. 400 `bad_request` / `unreadable_image`, 404 `not_found` / `expired`,
413 `too_large`, 415 `unsupported_type`, 422 (invalid `consensus`), **503 `busy`** with `Retry-After: 5` when another
image is being analysed (one at a time; the UI shows "busy, retry"), 500 `internal_error`.

**Privacy.** An upload is written to a private temp file only to decode it and is deleted immediately after. Results
live in a private temp folder for 15 minutes (swept every 30 s, also checked on every access) and are deleted on
shutdown. Logs contain the result id prefix, image size, cell count and timings, never file names or pixels.
Result ids are 24 random URL-safe characters.

## Frontend

React 19 + TypeScript + Vite. Landing (static HTML until React loads) → analysis (stage list) → workspace
(viewer + side panel). Keyboard: every control is a button / switch with a visible focus ring; the viewer canvas
takes OpenSeadragon's keyboard navigation (arrows, +/−); table rows have a button, Enter zooms to the cell.

- **ShaderGradient** (MIT) background, landing only, hosted in our own R3F `<Canvas>` with `ShaderGradientCanvas`'s
  settings so the frame loop can be controlled. `lightType="3d"` (the `env` presets would fetch HDR files from a
  third-party site). Frozen = the current frame is copied to an image and the WebGL canvas unmounted.
- **React Three Fiber + drei** (MIT) 3D cell, landing only: flattened noisy icosphere (cytoplasm), darker nucleus,
  granules, pink/indigo point lights, tilts toward the pointer. `frameloop="never"` when the tab is hidden; not
  mounted on phones (static illustration: the "lighter effects" choice), under reduced motion, or outside the landing.
- **Start-up:** first paint shows a still frame of the gradient (`public/gradient-still.webp`, 7 KB) and the static
  cell; the live WebGL scenes start on the first interaction (pointer move, touch, scroll, key) and cross-fade in.
  Shaders are compiled before the first frame (`effects/useCompiled.ts`). Reason: starting WebGL costs a one-off
  ~250 ms of main-thread work on a laptop CPU (context, programs, ShaderGradient set-up) and ~1.7 s under
  Lighthouse's 4x-throttled mobile profile, which on page load alone pushed mobile performance to 61-63.
- **Liquid Glass JS** (MIT, (c) Armagan Amcalar): its layer model is ported to CSS (`components/Glass.tsx`,
  `.glass` in `styles.css`): backdrop blur + tint, rim/edge band (`::before`), base sheen + corner boost (`::after`).
  The library itself is not used at runtime: it refracts a one-off `html2canvas` snapshot of the page, which cannot
  capture the WebGL background (it would be blank/frozen), costs a full-page capture on load, and has no npm
  package; CSS `backdrop-filter` refracts the live gradient for free. Glass is on the header, upload card, layer
  toolbar and side panel, never over the image viewer; text surfaces get a darker fill.
- **Liquid Logo** (paper-design) was not used: it is licensed under PolyForm Shield 1.0.0 (not open source) and is
  a Next.js application rather than a library, so not "light".
- **OpenSeadragon** (BSD-3) viewer, lazy-loaded chunk (prefetched when an analysis starts). Overlay = one SVG in
  image pixels, transform synced on every viewport frame, `vector-effect: non-scaling-stroke`.
- **Colours.** Brand: black base, pink `#F6C1D9`, indigo `#4B3FB3`. Status: green `#3ED68C` measurable, orange
  `#FF9F45` review, red `#FF6262` excluded. Status never uses brand hues, and every status mark also has a text label
  and a distinct marker shape (circle / diamond / square). With "Status colours" off, overlays are neutral white.

### UI checks (Playwright, Chromium)

Screens in `reports/app_screens/` (landing, analysis, viewer at 1440x900 and 390x844; reduced motion; empty + QC
warning; upload error). Flow tested: upload (file + sample) → analysis → viewer → each layer toggle → Flagged only
→ consensus off/on re-run → table row click (zoom + highlight) → keyboard selection → click on the image → all
three downloads; busy (second concurrent request → 503 + Retry-After) and empty / QC-warning states.

- **Lighthouse 12.8.2 (landing):** mobile performance **95**, accessibility **100**, best practices 100;
  desktop **100 / 100 / 100**. Lighthouse does not interact with the page, so this measures the landing without
  the WebGL start-up (see "Start-up" above).
- **Contrast:** measured, not assumed: each text element's colour against the rendered pixels behind it (text
  hidden, worst 2 % of background pixels) on landing (several gradient frames, scrolled), analysis and viewer at
  both sizes: 255 elements, all >= 4.5:1 (lowest 5.48:1).
- Reduced motion: no canvas at all. Hidden tab: 3D cell 244 → 0 draw calls/s. Viewer / analysis: no WebGL effect
  mounted (only OpenSeadragon's canvas).

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
`scripts/space_smoke.py` (health, frontend, every sample with and without consensus, downloads, 25 MB limit;
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
- **CPU**: measured locally (Ryzen 5 5600H, ONNX Runtime limited to 2 threads like the Space's 2 vCPUs, the exact
  Docker file set, no torch): **~5.9 s per FOV with consensus, ~2.5 s without** (YOLO26l @1280 is the slowest
  stage). Not yet measured on the Space (not deployed, see above); its shared vCPUs are likely slower.
- **Image size**: the models were built for 2028x1520 FOVs at ~0.155 µm/px; other sizes run with a warning (and
  pixel areas are then not comparable). Images above 40 MP are refused (decompression-bomb guard).
- **Results expire after 15 minutes** and are lost when the Space restarts (temp storage, by design).
- The 3D cell does not render on phones (static illustration) and the live effects wait for the first interaction.
