"""Example FOVs for the consensus evaluation (reads runs/consensus_eval/{boxes.json, per_fov.csv}; images; GT xml).

    python tools/consensus_examples.py --eval runs/consensus_eval --images data/images --gt data/annotations.xml

Writes <eval>/examples/{drops_gt_*, removes_fp_*}.png: 3 FOVs where consensus (A∩B) drops the most A boxes that match a
GT cell, and 3 where it drops the most A boxes that match no GT box. Colours:
  green thin       GT box (CVAT annotations.xml)
  cyan thin        A box kept by consensus (matched to a B box)
  red thick        A-only box that matches a GT cell        -> consensus would LOSE a labelled cell
  orange thick     A-only box that matches no GT box        -> consensus would remove it (a FP only if GT is complete)
  magenta dashed   B-only box (l_only)
"""
import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ncmask.consensus import match_boxes  # noqa: E402
from ncmask.fov import draw_dashed  # noqa: E402
from ncmask.imageio import load_bgr8  # noqa: E402
from tools.eval_consensus import gt_match, read_cvat  # noqa: E402


def rect(img, b, col, th):
    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]) - 1, int(b[3]) - 1), col, th, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True); ap.add_argument("--images", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--a-conf", type=float, default=0.15); ap.add_argument("--b-conf", type=float, default=0.15)
    ap.add_argument("--match-iou", type=float, default=0.5); ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    ev = Path(a.eval); out = ev / "examples"; out.mkdir(exist_ok=True)
    boxes = json.loads((ev / "boxes.json").read_text()); gt = read_cvat(a.gt)
    pf = pd.read_csv(ev / "per_fov.csv")
    picks = [("drops_gt", f) for f in pf.sort_values(["s_only_tp", "fov"], ascending=[False, True]).fov[:a.n]] + \
            [("removes_fp", f) for f in pf.sort_values(["s_only_fp", "fov"], ascending=[False, True]).fov[:a.n]]
    summary = []
    for k, (kind, fov) in enumerate(picks, 1):
        name = fov + ".tiff"
        get = lambda n, t: (np.array(boxes[name][n]["box"], float).reshape(-1, 4)[np.array(boxes[name][n]["conf"]) >= t],
                            np.array(boxes[name][n]["conf"], float)[np.array(boxes[name][n]["conf"]) >= t])
        A, Ac = get("A", a.a_conf); B, Bc = get("B", a.b_conf); g = gt[name]
        _, am, bm = match_boxes(A, B, a.match_iou)
        tpA, _ = gt_match(A, Ac, g)
        img, _ = load_bgr8(Path(a.images) / name)
        for b in g: rect(img, b, (0, 200, 0), 2)
        for b in A[am]: rect(img, b, (255, 255, 0), 1)
        for b in A[~am & tpA]: rect(img, b, (0, 0, 255), 5)
        for b in A[~am & ~tpA]: rect(img, b, (0, 140, 255), 5)
        for b in B[~bm]:
            x1, y1, x2, y2 = (int(v) for v in b)
            draw_dashed(img, np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]), (255, 0, 255), 3)
        r = {"kind": kind, "fov": fov, "gt": int(len(g)), "A": int(len(A)), "B": int(len(B)), "consensus": int(am.sum()),
             "s_only_matching_gt": int((~am & tpA).sum()), "s_only_no_gt": int((~am & ~tpA).sum()), "l_only": int((~bm).sum())}
        head = np.full((70, img.shape[1], 3), 255, np.uint8)
        cv2.putText(head, f"{fov}  [{kind}]  GT {r['gt']}  A {r['A']}  B {r['B']}  consensus {r['consensus']}  "
                    f"A-only&GT (red) {r['s_only_matching_gt']}  A-only no-GT (orange) {r['s_only_no_gt']}  B-only (magenta) {r['l_only']}",
                    (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imwrite(str(out / f"{k}_{kind}_{fov}.png"), np.vstack([head, img]))
        summary.append(r); print(r)
    (out / "examples.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
