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
