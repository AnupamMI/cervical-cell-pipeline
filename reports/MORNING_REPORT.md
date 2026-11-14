# Morning report — 2026-11-15 00:35

Overnight run started 2026-11-14 19:38:17; log: `tinyseg_v3/overnight.log`.

## What ran

| step | state | reason / note | started | finished |
|---|---|---|---|---|
| step0_push_v01 | ok |  |  | 2026-11-14 19:49:42 |
| step1_train | ok |  |  | 2026-11-15 00:22:28 |
| step2_eval | ok |  | 2026-11-15 00:22:28 | 2026-11-15 00:27:01 |
| step3_winner | ok |  | 2026-11-15 00:27:01 | 2026-11-15 00:27:04 |
| step4_pipeline_v3 | ok |  | 2026-11-15 00:27:04 | 2026-11-15 00:28:30 |
| step5_push_v02 | ok |  | 2026-11-15 00:33:06 | 2026-11-15 00:35:16 |
| step6_morning_report | ok (this report) |  | 2026-11-15 00:35:26 | 2026-11-15 00:30:20 |

All steps succeeded.

## Training (V0–V3, 40 epochs, seed 42, from scratch, v2_1 recipe, no colour/blur/noise augmentation)

| variant | status | run | epochs | best epoch | best val score | last | s/epoch | train min | still rising at end? |
|---|---|---|---|---|---|---|---|---|---|
| V0 | ok | V0 | 40 | 29 | 0.9448 | 0.9423 | 119 | 92.0 | no |
| V1 | ok | V1 | 40 | 29 | 0.9303 | 0.9268 | 107 | 71.1 | no |
| V2 | ok | V2 | 40 | 27 | 0.9299 | 0.9275 | 107 | 71.5 | no |
| V3 | ok | V3 | 40 | 29 | 0.9450 | 0.9430 | 116 | 77.2 | no |



Training log lines with timings: `grep training tinyseg_v3/overnight.log`.

## T2 independence check (your 92 cells vs NCNet V3's saved masks; NCNet never run)

- classes by your Dice rule: **{'corrected': 89, 'identical': 3}**
- masks == polygons (rasterisation check): {'nucleus': {'all_equal': True, 'cells_differing': 0, 'max_diff_px': 0}, 'cell': {'all_equal': True, 'cells_differing': 0, 'max_diff_px': 0}}
- median share of polygon vertices on NCNet's mask boundary: cell 94.5%, nucleus 100.0% (hand-drawn ZIP01 control ≈ 28 % / 36 %)
- report: `tinyseg_v3/t2_verify/T2_VERIFY_REPORT.md`

## Results (YOLO26s boxes; full tables incl. GT boxes in `tinyseg_v3/results/RESULTS.md`)

### T1 – all – YOLO26s boxes

cells 212, scored 186, unmatched GT 2, no GT cell 24

| model | nucleus Dice | nucleus IoU | nucleus BF@2px | cell Dice | cell IoU | cell BF@2px | N:C MAE | no nucleus | multi-nucleus | N:C>1 | params | ms/cell |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 0.926 [0.905, 0.940] † | 0.878 [0.858, 0.894] † | 0.848 [0.818, 0.874] † | 0.982 [0.979, 0.984] † | 0.965 [0.960, 0.969] † | 0.746 [0.719, 0.769] † | 0.0034 [0.0029, 0.0039] † | 2 | 7 | 0 | 118,402 | 27.0 |
| V1 | 0.912 [0.892, 0.928] † | 0.857 [0.835, 0.874] † | 0.781 [0.742, 0.816] | 0.979 [0.976, 0.982] † | 0.960 [0.953, 0.965] † | 0.697 [0.663, 0.727] † | 0.0041 [0.0033, 0.0051] | 2 | 9 | 0 | 118,402 | 83.5 |
| V2 | 0.906 [0.885, 0.923] † | 0.845 [0.822, 0.864] | 0.748 [0.709, 0.784] | 0.977 [0.973, 0.981] † | 0.957 [0.949, 0.963] † | 0.687 [0.649, 0.717] | 0.0046 [0.0037, 0.0058] | 3 | 10 | 0 | 118,402 | 89.0 |
| V3 | 0.938 [0.923, 0.948] † | 0.891 [0.873, 0.904] † | 0.867 [0.833, 0.894] † | **0.982** [0.980, 0.984] | **0.965** [0.961, 0.970] | **0.749** [0.721, 0.773] | 0.0028 [0.0025, 0.0032] † | 1 | 3 | 0 | 118,546 | 31.1 |
| v2_1 | **0.939** [0.920, 0.950] | **0.894** [0.873, 0.908] | **0.880** [0.846, 0.907] | 0.981 [0.979, 0.983] † | 0.963 [0.959, 0.968] † | 0.728 [0.702, 0.754] † | **0.0027** [0.0023, 0.0032] | 1 | 6 | 0 | 118,402 | 26.9 |

### T1b – all – YOLO26s boxes

cells 50, scored 48, unmatched GT 2, no GT cell 0

| model | nucleus Dice | nucleus IoU | nucleus BF@2px | cell Dice | cell IoU | cell BF@2px | N:C MAE | no nucleus | multi-nucleus | N:C>1 | params | ms/cell |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 0.920 [0.876, 0.947] † | 0.870 [0.824, 0.900] † | 0.818 [0.763, 0.865] † | 0.982 [0.978, 0.985] † | 0.965 [0.958, 0.971] † | 0.694 [0.654, 0.733] † | 0.0034 [0.0025, 0.0045] † | 0 | 1 | 0 | 118,402 | 27.5 |
| V1 | 0.900 [0.849, 0.936] † | 0.842 [0.789, 0.883] † | 0.773 [0.713, 0.827] † | **0.983** [0.980, 0.986] | **0.967** [0.962, 0.972] | **0.715** [0.676, 0.752] | 0.0038 [0.0028, 0.0050] † | 1 | 2 | 0 | 118,402 | 94.5 |
| V2 | 0.905 [0.856, 0.936] † | 0.845 [0.796, 0.882] † | 0.751 [0.682, 0.812] † | 0.981 [0.977, 0.985] † | 0.964 [0.956, 0.971] † | 0.704 [0.662, 0.747] † | 0.0039 [0.0030, 0.0050] † | 0 | 2 | 0 | 118,402 | 96.1 |
| V3 | **0.923** [0.877, 0.949] | **0.874** [0.827, 0.904] | **0.835** [0.774, 0.883] | 0.982 [0.978, 0.985] † | 0.964 [0.958, 0.971] † | 0.703 [0.656, 0.747] † | **0.0029** [0.0021, 0.0038] | 0 | 1 | 0 | 118,546 | 31.7 |
| v2_1 (in-sample, not ranked) | 0.930 [0.890, 0.953] | 0.883 [0.840, 0.911] | 0.863 [0.812, 0.905] | 0.984 [0.981, 0.987] | 0.969 [0.964, 0.974] | 0.726 [0.687, 0.764] | 0.0024 [0.0018, 0.0030] | 0 | 3 | 0 | 118,402 | 27.1 |

### T2 crop-based – all – YOLO26s boxes

cells 92, scored 89, unmatched GT 3, no GT cell 0

| model | nucleus Dice | nucleus IoU | nucleus BF@2px | cell Dice | cell IoU | cell BF@2px | N:C MAE | no nucleus | multi-nucleus | N:C>1 | params | ms/cell |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 0.780 [0.737, 0.820] | 0.685 [0.640, 0.726] | 0.462 [0.400, 0.520] | 0.945 [0.935, 0.954] † | 0.899 [0.883, 0.914] † | 0.378 [0.338, 0.417] † | 0.0104 [0.0073, 0.0139] | 2 | 5 | 0 | 118,402 | 29.2 |
| V1 | 0.908 [0.887, 0.926] † | 0.842 [0.814, 0.866] † | 0.749 [0.698, 0.794] † | 0.953 [0.943, 0.962] † | 0.913 [0.897, 0.928] † | 0.404 [0.361, 0.442] † | 0.0037 [0.0027, 0.0049] † | 0 | 6 | 0 | 118,402 | 63.9 |
| V2 | 0.911 [0.899, 0.923] † | 0.843 [0.825, 0.863] † | 0.735 [0.690, 0.779] † | 0.947 [0.935, 0.958] † | 0.903 [0.882, 0.921] † | 0.407 [0.362, 0.446] † | 0.0044 [0.0032, 0.0058] † | 0 | 5 | 0 | 118,402 | 67.9 |
| V3 | 0.783 [0.735, 0.827] | 0.698 [0.649, 0.743] | 0.514 [0.436, 0.587] | 0.950 [0.941, 0.959] † | 0.908 [0.892, 0.923] † | 0.415 [0.374, 0.455] † | 0.0106 [0.0076, 0.0141] | 7 | 4 | 0 | 118,546 | 33.5 |
| v2_1 | **0.926** [0.911, 0.938] | **0.868** [0.847, 0.887] | **0.803** [0.757, 0.841] | **0.959** [0.951, 0.966] | **0.924** [0.910, 0.936] | **0.452** [0.414, 0.484] | **0.0032** [0.0024, 0.0040] | 0 | 7 | 0 | 118,402 | 29.3 |

### T2 crop-based – corrected+independent – YOLO26s boxes

cells 89, scored 88, unmatched GT 1, no GT cell 0

| model | nucleus Dice | nucleus IoU | nucleus BF@2px | cell Dice | cell IoU | cell BF@2px | N:C MAE | no nucleus | multi-nucleus | N:C>1 | params | ms/cell |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 0.782 [0.740, 0.823] | 0.688 [0.647, 0.731] | 0.467 [0.407, 0.530] | 0.945 [0.936, 0.953] † | 0.898 [0.883, 0.913] † | 0.379 [0.338, 0.420] † | 0.0100 [0.0071, 0.0133] | 2 | 5 | 0 | 118,402 | 29.1 |
| V1 | 0.907 [0.886, 0.924] † | 0.840 [0.812, 0.864] † | 0.749 [0.698, 0.795] † | 0.953 [0.943, 0.962] † | 0.913 [0.896, 0.928] † | 0.404 [0.361, 0.445] † | 0.0037 [0.0027, 0.0050] † | 0 | 6 | 0 | 118,402 | 64.0 |
| V2 | 0.910 [0.898, 0.923] † | 0.842 [0.824, 0.861] † | 0.735 [0.685, 0.777] † | 0.947 [0.934, 0.958] † | 0.903 [0.881, 0.921] † | 0.408 [0.363, 0.448] † | 0.0045 [0.0032, 0.0059] † | 0 | 5 | 0 | 118,402 | 68.0 |
| V3 | 0.783 [0.735, 0.828] | 0.699 [0.650, 0.744] | 0.520 [0.447, 0.593] | 0.950 [0.940, 0.959] † | 0.907 [0.890, 0.923] † | 0.415 [0.372, 0.457] † | 0.0104 [0.0073, 0.0137] | 7 | 4 | 0 | 118,546 | 33.4 |
| v2_1 | **0.925** [0.911, 0.938] | **0.868** [0.845, 0.886] | **0.807** [0.762, 0.845] | **0.959** [0.951, 0.966] | **0.924** [0.909, 0.936] | **0.452** [0.415, 0.487] | **0.0031** [0.0024, 0.0040] | 0 | 7 | 0 | 118,402 | 29.3 |

## Winner

**V1** — decided on T1 + T2 corrected+independent.

Primary = nucleus Dice with YOLO26s boxes, mean of the set means; CI = cluster bootstrap within each set.

| model | primary | 95 % CI | T1 | T2 corr+indep | N:C MAE | cell Dice |
|---|---|---|---|---|---|---|
| V0 | 0.8538 | [0.8311, 0.8743] | 0.9257 [0.9058, 0.9401] | 0.7819 [0.7402, 0.8229] | 0.0055 | 0.9699 |
| V1 | 0.9096 | [0.8952, 0.9222] | 0.9120 [0.8914, 0.9282] | 0.9072 [0.8862, 0.9244] | 0.0039 | 0.9707 |
| V2 | 0.9084 | [0.8962, 0.9186] | 0.9063 [0.8845, 0.9233] | 0.9104 [0.8978, 0.9227] | 0.0046 | 0.9677 |
| V3 | 0.8607 | [0.8351, 0.8842] | 0.9383 [0.9227, 0.9481] | 0.7830 [0.7351, 0.8280] | 0.0053 | 0.9719 |
| v2_1 (old) | 0.9321 | [0.9212, 0.9406] | 0.9388 [0.9210, 0.9502] | 0.9255 [0.9105, 0.9377] | 0.0029 | 0.9741 |

Reasoning:

- V1 beats V0: primary 0.9096 [0.8952, 0.9222] vs V0 0.8538 [0.8311, 0.8743], no overlap.
- V1 is not clearly worse than the old v2_1 on T1 or T2 (CIs overlap or V1 is higher).

## Pipeline v3 (`nc_mask_pipeline_v3`)

Winner installed: **V1** (variant V1) -> `models/seg/V1_v3/best.pt`; `config.yaml` seg.weights / seg.variant set. Decision: V1 beats V0: primary 0.9096 [0.8952, 0.9222] vs V0 0.8538 [0.8311, 0.8743], no overlap.; V1 is not clearly worse than the old v2_1 on T1 or T2 (CIs overlap or V1 is higher).

Tests: `60 passed in 41.87s` (log `nc_mask_pipeline_v3/reports/tests_step4.txt`). Seg parity of the configured model vs the evaluation path: {'model': '/home/jetson/Downloads/ANUPAM KA HOOD/nc_mask_pipeline_v3/models/seg/V1_v3/best.pt', 'variant': 'V1', 'cells': 20, 'max_abs_prob_diff': 0.0, 'min_dice': 1.0}

### 1 FOV (`/home/jetson/Downloads/ANUPAM KA HOOD/nc_mask_pipeline_v3/runs/prod_1fov`)

| FOV | why picked | cells | measurable | review | excluded | top reasons | image QC | s/FOV |
|---|---|---|---|---|---|---|---|---|
| cervical_0013 | typical (median 29 cells in v0.1) | 34 | 12 | 8 | 14 | cell_at_edge 14, multi_nucleus 7, occluded 6, nucleus_touches_cell_boundary 5 | ok | 3.76 |


Sheets: `/home/jetson/Downloads/ANUPAM KA HOOD/nc_mask_pipeline_v3/reports/sheets_1fov`

```
2026-11-15 00:27:57,805 INFO    ncmask: SUMMARY  FOVs: 1 ok, 0 failed, 0 skipped (already done)  |  cells this run: 34  (cells.csv: 34)
2026-11-15 00:27:57,806 INFO    ncmask:   status: measurable 12, review 8, excluded 14
2026-11-15 00:27:57,808 INFO    ncmask:   FOV image QC: {'(none)': 1}
2026-11-15 00:27:57,808 INFO    ncmask:   Macenko fallbacks (crop left unchanged, training rule): 7
```

### 5 FOVs (`/home/jetson/Downloads/ANUPAM KA HOOD/nc_mask_pipeline_v3/runs/prod_5fov`)

| FOV | why picked | cells | measurable | review | excluded | top reasons | image QC | s/FOV |
|---|---|---|---|---|---|---|---|---|
| cervical_0229 | random (seed 42) | 36 | 18 | 2 | 16 | cell_at_edge 16, occluded 5, nucleus_touches_cell_boundary 4, s_only_low_conf 4 | ok | 3.8 |
| cervical_0373 | sparse (fewest cells) | 21 | 9 | 3 | 9 | cell_at_edge 9, multi_nucleus 5, no_nucleus 3, likely_cluster 2 | ok | 1.76 |
| cervical_0558 | dense (41 cells in v0.1) | 43 | 19 | 4 | 20 | cell_at_edge 19, multi_nucleus 7, occluded 6, no_nucleus 4 | ok | 2.61 |
| cervical_0980 | edge-heavy (most cell_at_edge) | 40 | 13 | 3 | 24 | cell_at_edge 24, multi_nucleus 11, nucleus_touches_cell_boundary 7, no_nucleus 4 | ok | 2.41 |
| cervical_1083 | multi_nucleus-heavy | 31 | 11 | 2 | 18 | cell_at_edge 17, no_nucleus 6, likely_inflammatory 6, occluded 6 | ok | 2.13 |


Sheets: `/home/jetson/Downloads/ANUPAM KA HOOD/nc_mask_pipeline_v3/reports/sheets_5fov`

```
2026-11-15 00:28:20,126 INFO    ncmask: SUMMARY  FOVs: 5 ok, 0 failed, 0 skipped (already done)  |  cells this run: 171  (cells.csv: 171)
2026-11-15 00:28:20,127 INFO    ncmask:   status: measurable 70, review 14, excluded 87
2026-11-15 00:28:20,127 INFO    ncmask:   FOV image QC: {'(none)': 5}
2026-11-15 00:28:20,127 INFO    ncmask:   Macenko fallbacks (crop left unchanged, training rule): 18
```

Step 4 log:

- installed V1 (V1) from /home/jetson/Downloads/ANUPAM KA HOOD/tinyseg_v3/phase2/runs/V1 -> /home/jetson/Downloads/ANUPAM KA HOOD/nc_mask_pipeline_v3/models/seg/V1_v3
- config.yaml: seg.weights = models/seg/V1_v3/best.pt, seg.variant = V1
- pytest: 60 passed in 41.87s (rc 0 )
- prod_1fov: prod_1fov OK; sheets in /home/jetson/Downloads/ANUPAM KA HOOD/nc_mask_pipeline_v3/reports/sheets_1fov
- prod_5fov: prod_5fov OK; sheets in /home/jetson/Downloads/ANUPAM KA HOOD/nc_mask_pipeline_v3/reports/sheets_5fov


## Git

```
e78c375 (HEAD -> main, tag: v0.2, origin/main, v0.2-wip) v0.2: pipeline v3 (two layers, gate, image QC) + TinyUNet v3 V1
6a76e13 reports: morning report 2026-11-15
14801ac (tag: v0.1) check_against_samples: pooled Dice >= 0.99 per FOV + per-cell floor 0.95 (cross-device tolerance; CPU vs GPU measured)
d9c586f samples: FOVs at samples/ top level so 'run.py --input samples/' works; docs/notebook updated
20bd0fd (tag: v0.1-models) v0.1: cervical cell masking pipeline (YOLO26s + YOLO26l consensus_flag + TinyUNet v2_1)
```

- tags on origin: v0.1, v0.1-models, v0.2
- release with YOLO26l: https://github.com/AnupamMI/cervical-cell-pipeline/releases/tag/v0.1-models
- v0.2 release (V0–V3 weights, if pushed): https://github.com/AnupamMI/cervical-cell-pipeline/releases/tag/v0.2-models
- working tree: clean

## Run it on Colab (T4) tomorrow

1. Colab → open notebook → GitHub → `AnupamMI/cervical-cell-pipeline` (private: sign in), file `notebooks/colab_quickstart.ipynb`.
2. Runtime → Change runtime type → **T4 GPU**.
3. Key icon (Secrets) → add `GH_TOKEN` = a GitHub token with read access to the repo → enable notebook access.
4. Runtime → Run all. It clones the repo, installs `requirements-colab.txt` (keeps Colab's torch), fetches the YOLO26l weights from the release, mounts Drive (outputs: `MyDrive/cervical-cell-pipeline/out/`), runs the pipeline on `samples/`, shows a 3-panel sheet and runs the match check (expect PASS).
5. The repo's default branch is the latest version (`v0.2` tagged). To use exactly v0.1: `git checkout v0.1` in the clone.
6. Your own FOVs: upload to Drive, then `!python run.py --input /content/drive/MyDrive/<your folder> --output /content/drive/MyDrive/cervical-cell-pipeline/out/<run>`.

## Open questions

1. T2 is crop-based (no source FOVs): if the FOVs ever turn up, the T2 numbers should be redone from FOVs.
2. T2's vertex analysis says most outlines sit exactly on NCNet's boundary; read T2 results as agreement with those outlines.
3. Macenko is a two-stain (H&E) model applied to Pap stain; 2 % of crops fall back to unchanged.
4. Anything flagged as failed above.

