# cervical-cell-pipeline

Masking-only cell pipeline for Pap-smear cervical cytology FOVs (2028x1520, ~0.155 µm/px):

```
FOV image ─► YOLO26s cell boxes (conf 0.15) ─► [YOLO26l @1280 cross-check: tags consensus / s_only, never filters]
          ─► crop (box + 17 % margin, mirror padding) ─► TinyUNet v2_1 (cell + nucleus)
          ─► post-process per box ─► paste into FOV (overlaps → higher confidence) ─► label maps, COCO, CSV, overlay
```

It produces masks only, with no N:C ratio or classification. Version **v0.1** = the validated pipeline
(YOLO26s + YOLO26l consensus_flag + TinyUNet v2_1).

## Quick start

### Google Colab (T4)
Open `notebooks/colab_quickstart.ipynb` in Colab and set a Colab Secret `GH_TOKEN` (a GitHub token with read access
to this private repo). The notebook clones the repo, installs `requirements-colab.txt` (keeps Colab's torch), fetches
the YOLO26l weights from the release, mounts Google Drive (outputs in `MyDrive/cervical-cell-pipeline/out/`), runs
the pipeline on `samples/fovs` with CUDA, shows a 3-panel sheet and checks the outputs against `samples/expected`.

### Locally
```bash
git clone https://github.com/AnupamMI/cervical-cell-pipeline.git && cd cervical-cell-pipeline
python3 -m venv venv && . venv/bin/activate
# Jetson (JetPack 6, CUDA 12.6):      pip install -r requirements.txt
# x86 + CUDA:  install torch/torchvision for your CUDA, then    pip install -r requirements-colab.txt
# CPU only:    see requirements-cpu.txt
GH_TOKEN=<token> python scripts/fetch_release_weights.py      # optional: YOLO26l for the consensus tag
python run.py --input samples/fovs --output out/
python scripts/check_against_samples.py --output out/          # compare with the committed reference outputs
```
The device is chosen automatically (CUDA if available, else CPU). All paths in `config.yaml` are relative to the
config file. Without the YOLO26l weights the consensus stage switches itself off (`detection_mode -> single`) and the
run log says so; masks and cells are unchanged, only the `det_tag` column is `single`.

### Weights
In git: YOLO26s detector and the TinyUNet weights. YOLO26l (106 MB) is a GitHub Release asset (`v0.1-models`),
fetched by `scripts/fetch_release_weights.py` (token from the `GH_TOKEN` environment variable, SHA256 verified).
Sizes and SHA256s: [`models/WEIGHTS.md`](models/WEIGHTS.md).

### Samples
`samples/fovs/`: 10 FOVs of slide MAX06 (cervical_0000-0009). `samples/expected/`: the outputs of this version on
the Jetson (cells.csv, label maps, COCO, overlays), the reference for `scripts/check_against_samples.py`.

### Tests
`python -m pytest tests/`. Tests that need the original training data (seg parity, provenance) or the YOLO26l
weights are skipped when those are absent.

## Run

```bash
venv/bin/python run.py --input /path/to/fovs --output /path/to/results [--config config.yaml] [--limit N] [--overwrite]
```

- `--input` is only read, never written. The run refuses if `--output` equals `--input` or contains it. It also refuses
  if `--output` is inside `--input` while `input.recursive: true`.
- **Resumable.** A FOV is complete when `fovs/<id>/done.json` exists. A re-run skips completed FOVs and retries failed ones.
  `--overwrite` re-processes everything. A re-run into a folder made with a **different config** is refused (config
  hash in `run_manifest.json`). Use `--overwrite` or a new folder.
- A FOV that can't be read, or that fails for any reason, is logged and recorded in the manifest. The run continues.
- Exit code: 0 = run finished (check `counts.fovs_failed`), 2 = usage, config or path error.
- Tests: `venv/bin/python -m pytest tests/` (51 tests, about 50 s on the Orin). The parity and provenance tests need the
  original `../seg_audit/extracted/tinyseg_retrain/tinyseg_retrain` and `../nc_pipeline/run_nc.py`. Override the
  locations with `NCMASK_TINYSEG_DIR` / `NCMASK_NC_PIPELINE_DIR`. Those tests are skipped if the originals are missing.

## Input

- `.jpg .jpeg .png .tif .tiff .bmp` (configurable), sorted by name, non-recursive by default. Other files are ignored and listed in the manifest.
- 8- or 16-bit; grayscale, RGB, RGBA (alpha dropped), grey+alpha; EXIF orientation applied; first page of multi-page TIFFs.
- Grayscale content (1/2-channel files, or colour files with R=G=B) is processed with a logged warning: the models are colour-trained.
- Images are converted to 8-bit **BGR**, the order the training code used (`cv2.imread`). For 8-bit files the loader's
  output is byte-identical to `cv2.imread(path)` (tested). 16-bit: see `input.sixteen_bit`.
- FOV id = file name without extension (`sub__dir__name` when recursive). If two files share a stem, the extension is appended.

## Configuration (`config.yaml`)

All paths and parameters live here. Relative paths resolve against the config file's folder.

| key | default | meaning |
|---|---|---|
| `input.extensions` | jpg…bmp | files to process |
| `input.recursive` | false | scan sub-folders |
| `input.sixteen_bit` | `cv2` | `cv2` = exactly `cv2.imread(path)` (OpenCV uses round(x/257) for TIFF, x>>8 for PNG); `scale` = round(x/257) for every format; `max` = scale by the image max (10/12-bit data in 16-bit files) |
| `device` | auto | auto / cuda / cpu |
| `seed` | 0 | seeds + deterministic cuDNN |
| `consensus.detection_mode` | consensus_flag | `single` = detector A only; `consensus_flag` = A and B run, every A box segmented and tagged (`det_tag` consensus / s_only); `consensus_filter` = s_only boxes not segmented (dropped.csv reason `s_only_filtered`). B-only boxes go to `l_only.csv` in both consensus modes |
| `consensus.match_iou` | 0.5 | A↔B one-to-one Hungarian matching on IoU; matched cells keep A's box |
| `detector_b.*` | models/detector_b/yolo26l_1280_best.pt, imgsz 1280, conf 0.15, nms_iou 0.7, max_det 300 | cross-check detector (YOLO26l), consensus modes only |
| `detector.weights` | models/detector/… | YOLO checkpoint |
| `detector.imgsz` | 640 | inference size (= training imgsz; 0 = read from the checkpoint) |
| `detector.conf` | 0.15 | confidence threshold (0.25 missed visible cells; see REPORT.md) |
| `detector.nms_iou` | 0.7 | class-agnostic NMS IoU (the model is end-to-end/NMS-free; NMS also merges tiles) |
| `detector.max_det` | 300 | per image / tile |
| `detector.min_box_px` | 16 | boxes narrower or shorter than this are not segmented (`box_too_small`) |
| `detector.containment_thr` | 0.8 | before seg, drop a box with ≥ this share of its area inside a kept higher-conf box (null/0 = off) |
| `detector.containment_min_area_ratio` | 0.4 | only drop it if it is also at least this share of the containing box's area (keeps small cells lying on big ones); 0 = containment alone |
| `detector.tiling.mode` | off | off / on / auto (tile when max side > `trigger_factor` × imgsz) |
| `detector.tiling.trigger_factor` | 4.0 | with 640: auto tiles only FOVs > 2560 px |
| `detector.tiling.tile_px` / `overlap_px` | 1280 / 640 | tile size in FOV px; overlap must exceed the largest cell |
| `detector.tiling.edge_px` | 2 | boxes within this of an inner tile edge are dropped (cut cells; the neighbouring tile sees them whole) |
| `crop.margin` | 0.17 | crop = box + margin × max(box w, h) per side (= `aug_v2.PAD`, the training/eval value) |
| `crop.edge_mode` | reflect | `reflect` = mirror-pad outside the FOV exactly like training (`aug_v2.pipeline_crop`; needs margin = 0.17); masks are cut back to the FOV. `clamp` = cut the crop at the FOV edge |
| `seg.code_dir` | vendor/tinyseg_retrain | where `tinyseg.py` / `aug_v2.py` are imported from |
| `seg.weights` | models/seg/v2_1/best.pt | `models/seg/v2/best.pt` = 4-channel variant (RGB + detector-box mask) |
| `seg.model_config` | null | null = `config.json` next to the weights; without one: 3-channel, 160 px (v1) |
| `seg.batch_size` | 64 | crops per forward pass |
| `seg.thr_nucleus` / `thr_cell` | 0.5 / 0.5 | probability thresholds |
| `postprocess.clip_tolerance_px` | margin | cell mask clipped to box + this per side; `margin` = crop.margin × max(box w, h) = the crop window, i.e. no clipping inside the crop; a number = px |
| `postprocess.nucleus_min_component_px` | 20 | nucleus blobs below this are removed, and don't count for `multi_nucleus` / `no_nucleus` |
| `postprocess.cell_at_edge_px` | 2 | detector box within this of the FOV border → `cell_at_edge` |
| `postprocess.drop_lost_frac` | 0.5 | after overlap resolution, drop cells that lost more than this share of their pixels (null = keep all) |
| `output.overlay` | true | write `overlay.png` |
| `output.overlay_max_side` | 0 | downscale the overlay (0 = full size) |
| `output.coco_polygon_epsilon_px` | 0 | polygon simplification (0 = exact outline) |
| `logging.level` | INFO | |

## What happens per box

1. **Crop.** Box + `margin` per side, rounded exactly as `aug_v2.pipeline_crop`. Outside the FOV the crop is
   mirror-padded by `aug_v2.pipeline_crop` itself (default), or cut at the edge (`clamp`). The in-FOV part of the
   window is stored (`crop_x0..crop_y1`).
2. **Segment.** `tinyseg.to_input` (INTER_AREA resize to 224, BGR→RGB, normalise) → TinyUNet → sigmoid. For v2, a 4th
   channel holds the box mask (1 inside the detector box).
3. **Post-process (in the crop).** Probabilities are resized to the crop (bilinear) and thresholded.
   - **Cell:** the 8-connected component containing the box centre. If none does, the component with the largest overlap
     with the box. It is clipped to box + `clip_tolerance_px` (by default the crop window, so nothing is removed; any removal sets `box_clipped` / `clipped_frac`). If clipping split it, the part at the centre is kept (else
     the largest part). Holes are filled.
   - **Nucleus:** nucleus pixels inside that cell, with blobs < `nucleus_min_component_px` removed. All remaining blobs are
     kept; 2+ blobs raises `multi_nucleus`.
4. **Map back.** Crop masks are pasted at the crop offset. Cells are painted in order of confidence (ties → lower id). A
   pixel already owned by a higher-confidence cell stays there; the loser records `overlap_lost` and `overlap_lost_px`.
   A nucleus pixel belongs to cell *i* only where the cell label is *i*. A cell losing more than `drop_lost_frac` is
   dropped (see Flags). Before step 1, boxes mostly inside a higher-confidence box are dropped (`containment_thr`).

## Outputs

```
<output>/
  run_manifest.json   config (resolved), config hash, model + code SHA256s, versions, UTC start/end, counts,
                      flag counts, time per FOV, failures, skipped, ignored files
  cells.csv           all cells of all completed FOVs in this input listing
  l_only.csv          consensus modes: detector-B boxes with no A partner (fov, b_conf, box, max_iou_with_A); never segmented
  dropped.csv         boxes/cells removed by containment suppression or the overlap-loss rule
                      (fov, reason, cell_id, conf, box_x1..box_y2, detail)
  run.log             full log (also printed to the console), with an end-of-run summary
  fovs/<fov_id>/
    overlay.png         box (cyan = not flagged, orange = flagged), cell outline (green; dashed = s_only), nucleus fill (red), "id conf[*]"
    cell_labels.png     uint16 PNG, 0 = background, value = cell_id
    nucleus_labels.png  uint16 PNG, value = cell_id of the cell the nucleus belongs to
    cells.coco.json     COCO; category 2 "cell-area" (one per detection), 1 "nucleus-area"; polygons in FOV px;
                        attributes: cell_id, conf, flags, det_box_xyxy, overlap_lost_px, clipped_frac (nucleus: cell_id)
    cells.csv           this FOV's rows
    done.json           completion marker: source path + SHA256, image info, timings, detector info, dropped boxes/cells
```

`cells.csv` columns: `fov, cell_id, box_x1, box_y1, box_x2, box_y2, conf, cell_area_px, nucleus_area_px, flags,
crop_x0, crop_y0, crop_x1, crop_y1, overlap_lost_px, clipped_frac, det_tag, b_conf, ab_iou`. `det_tag` = single |
consensus | s_only; `b_conf` / `ab_iou` = the matched B box's confidence and IoU (empty if none). `clipped_frac` = share of the cell component's
in-FOV pixels removed by the clip (0 with the default). Areas are the final pixel counts in the label maps (after overlap
resolution). `flags` are `;`-separated. Cell ids are 1..N per FOV, ordered by confidence (then y, x).

Each FOV's files are written to `fovs/<id>.tmp/` and renamed into place when complete, so a crash never leaves a
half-written FOV that looks done.

## Flags

| flag | meaning |
|---|---|
| `cell_at_edge` | the detector box itself is within `cell_at_edge_px` (2) of the FOV border: the cell is really cut off by the image edge |
| `crop_clamped` | info only: the padded crop window (box + margin) extends past the FOV edge (mirror-padded in `reflect` mode, cut in `clamp` mode) |
| `box_clipped` | the clip to box + `clip_tolerance_px` removed pixels of the cell (`clipped_frac` > 0); never with the default `margin` |
| `box_too_tight` | the cell component, before clipping, reaches a crop side that is not the FOV edge: the cell is probably larger than the box + margin |
| `no_nucleus` | the cell has no nucleus blob ≥ `nucleus_min_component_px` (also set if overlap resolution took the whole nucleus) |
| `multi_nucleus` | 2+ nucleus blobs inside the cell |
| `empty_mask` | no cell pixels: nothing above threshold at/overlapping the box, or every pixel went to higher-confidence cells |
| `box_too_small` | box width or height < `min_box_px`; not segmented (area 0, no crop) |
| `overlap_lost` | some of this cell's pixels went to a higher-confidence cell (`overlap_lost_px`) |
| `overlap_won` | info only: this cell kept pixels that a lower-confidence cell also claimed |

In the overlay, a cell counts as flagged (orange box, `*` after its confidence) if it has any flag other than the
info-only `crop_clamped` / `overlap_won`.

**Dropped, not in `cells.csv` / COCO / label maps** (listed in `dropped.csv` and each FOV's `done.json`, and logged):
- `contained`: before segmentation, a box with ≥ `detector.containment_thr` (0.8) of its area inside a kept
  higher-confidence box. Cell ids are assigned after this step, so they stay 1..N.
- `overlap_lost`: after overlap resolution, a cell that lost more than `postprocess.drop_lost_frac` (0.5) of its pixels
  to higher-confidence cells. Its id is not reused, so ids can have gaps. Its pixels go to the next cells in confidence order.

## Known limitations

- **The detector's training data is not available** (`/content/cell_dataset_v2`, Colab), so its training scale and
  domain are unknown. Full-frame at 640 beat tiling on local data (see REPORT.md), so tiling is off by default.
- **The detector misses some cells.** On the real-data overlays, a few clearly visible cells have no box at conf 0.25.
  Nothing downstream can recover them.
- **Straight or jagged outline segments** come from the overlap boundary between neighbours (ownership goes to the
  higher-confidence cell pixel by pixel). Without clipping, a cell that is larger than its crop is cut by the crop border (`box_too_tight`).
- **Containment suppression** drops a box ≥ 80% inside a higher-confidence box only if it is also ≥ 0.4 × that box's
  area, so small cells lying on big ones are kept (set the ratio to 0 to drop on containment alone).
- **Grayscale input** runs with a warning, but the models were trained on colour. On one FOV, the grey version gave 16 cells vs 29.
- **No accuracy check of FOV-level masks.** No FOV with GT masks was available. Mask quality is verified at crop level
  (parity with the eval code) and visually on overlays.
- COCO polygons are external outlines (holes can't be represented). `area` is the exact pixel count.
- Determinism holds for the same inputs, config and device. GPU and CPU outputs may differ slightly (not tested).
- Label PNGs are uint16: at most 65,535 cells per FOV (checked).
