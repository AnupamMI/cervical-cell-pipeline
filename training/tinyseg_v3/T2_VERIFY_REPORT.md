# T2 (NC003, 92 cells): independence check against NCNet V3 (2026-11-14)

Inputs (read-only): `~/Downloads/nc_ratio_new_evaluation/{ground_truth_annotations, ground_truth_masks, results/ground_truth_nc_ratios.csv}`
and NCNet V3's **saved** predictions in the same folder: `predicted_masks` (centre component, used for NC) and
`predicted_masks_raw` (thresholded, no post-processing). NCNet was not run.

Control (same metrics, same NCNet checkpoint's saved predictions, `~/Downloads/nc_ratio_model_predictions`):
- 50 **ZIP01** cells: known hand-drawn (they are NCNet's own training labels);
- 50 **ZIP02** cells.

Scripts: `verify_t2.py`, `control.py`. Per-cell data: `t2_verify.csv`, `control.csv`. Overlays: `overlays/`.

## 4. Rasterisation check
Re-filling every `.txt` polygon with the protocol's rule (Pillow `polygon(fill=1, outline=1)`, original size) reproduces
the saved GT masks **exactly: 92/92 cells, nucleus and cell, 0 differing pixels.**

## 1. Dice, your GT vs NCNet V3 (post-processed; raw is within 0.001)

| | min | p5 | median | p95 | max |
|---|---|---|---|---|---|
| cell | 0.966 | 0.993 | **0.997** | 0.998 | 0.998 |
| nucleus | 0.952 | 0.962 | **0.980** | 0.990 | 0.992 |

## 2. Vertices vs NCNet's mask boundary, and spacing

| metric (median over cells) | **T2 NC003** | ZIP01 hand-drawn (control) | ZIP02 |
|---|---|---|---|
| cell vertices ≤ 0.5 px from NCNet's raw boundary | **94.5%** (min 84%) | 28.4% (min 10%) | 97.2% |
| nucleus vertices ≤ 0.5 px | **100%** (min 91%) | 35.7% (min 8%) | 100% |
| median vertex distance to NCNet's boundary | **0.0002 px** | 1.0 px | 0.0001 px |
| cells with any cell vertex > 3 px off NCNet | **1 / 92** (NC003_0013, max 19.1 px) | 39 / 50 | 2 / 50 |
| cells with any nucleus vertex > 3 px off NCNet | **0 / 92** | 6 / 50 | 0 / 50 |
| integer vertex coordinates | 100% | 100% | 100% |
| step length median (cell / nucleus) | 5.8 / 3.8 px | — | — |
| step CV, cell / nucleus | 0.95 / 0.58 | 1.19 / 0.98 | 0.85 / 0.58 |

How to read this:
- **Integer coordinates don't discriminate.** All three sets, including the hand-drawn one, have 100% integer vertices (CVAT-side snapping or export).
- **Vertex position does discriminate.** Hand-drawn polygons sit about 1 px from NCNet's edge, with many vertices several px away. T2's vertices coincide with the centres of NCNet's boundary pixels to within 0.0002 px, the same signature as ZIP02.
- **Spacing:** T2's step CV is close to ZIP02's and lower than the hand-drawn set's. That is consistent with a mask-to-polygon conversion followed by point removal.

## 3. Classification (your rule: min Dice over nucleus and cell, vs NCNet post-processed)

| class | rule | cells |
|---|---|---|
| identical to NCNet | min Dice ≥ 0.99 | **3** (NC003_0007, NC003_0060, NC003_0080) |
| NCNet-based but corrected | 0.90 ≤ min Dice < 0.99 | **89** |
| independent | min Dice < 0.90, or freehand-like (< 50% vertices on NCNet's edge and < 50% integer vertices) | **0** |

**Caveat on the "corrected" class.** For 87 of the 89, only the *nucleus* Dice is below 0.99. A ~50 px nucleus outlined
with a vertex every ~3.8 px loses about 2% Dice to corner-cutting even when every vertex lies on NCNet's edge. So by
Dice the set is 89 "corrected", but by vertex evidence **91 of 92 cells carry no edit larger than 1–3 px**, and
**1 cell (NC003_0013) has a clear local edit** (one region moved up to 19 px).

My reading: the 92 outlines were most likely created in CVAT from NCNet V3 pre-annotations and then reviewed. Only
NC003_0013 shows a visible correction. "Annotated by hand" in the sense of reviewed and accepted is compatible with
this; "drawn from blank" is not.

Overlays (`overlays/`; GT green cell / red nucleus, thick; NCNet magenta / cyan, thin):
1. `1_identical_NC003_0060.png`, `2_identical_NC003_0007.png`
2. `3_corrected_NC003_0052.png` (lowest min Dice), `4_corrected_NC003_0024.png` (median)
3. `5_T2_local_edit_cervical_0161_a37_NILM.png` (NC003_0013, the one cell with a real local edit)
4. `6_ZIP01_handdrawn_control_b2_cervical_0231_cell_2.png` (a hand-drawn control for comparison)

There is no "independent" cell to show.

## T2 reporting subsets (as you asked: all 92 used as T2)

- **T2-all:** 92 cells.
- **T2 "independent + corrected"** (your Dice rule): 89 cells.
- **T2 "identical"**: 3 cells.

With only 3 cells, the "identical" subset is too small for statistics. I'd add a vertex-based split:
- **T2-edited:** NC003_0013, 1 cell.
- **T2-traced:** 91 cells.

Either way, T2 numbers mainly measure agreement with NCNet V3's outlines.

## Source FOVs for the pipeline crops: not found

ZIP003 contains only crops (tight: 56/92 NC003 cells touch the crop border), no FOV images. I searched every file named
like the source FOVs (e.g. `cervical_0099.*`, 90 candidates across MAX01–06, SJH*, the `Run_Inference-CVAT` slides,
the CVAT task export, and this pipeline's `data/images`) and template-matched NC003 crops into them:
- best `TM_CCOEFF_NORMED` 0.45–0.70 (a true source gives ≈ 0.99);
- no pixel-exact placement.

The FOV numbers repeat across slides, but **the slide these crops came from is not on this machine**, as far as I can find.
