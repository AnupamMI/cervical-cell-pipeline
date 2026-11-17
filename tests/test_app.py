"""Web app API (app/backend): endpoints, limits, busy handling, expiry, output schema.
Needs fastapi + onnxruntime and the ONNX bundle (scripts/export_onnx.py); skipped otherwise."""
import csv
import importlib.util
import io
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
need = [m for m in ("fastapi", "onnxruntime", "httpx", "multipart") if importlib.util.find_spec(m) is None]
pytestmark = [pytest.mark.skipif(bool(need), reason=f"needs {need}"),
              pytest.mark.skipif(not (ROOT / "models" / "onnx" / "manifest.json").is_file(), reason="no ONNX bundle")]

SAMPLE = "cervical_0001"
CELL_KEYS = {"id", "box", "conf", "det_tag", "b_conf", "ab_iou", "status", "reasons", "flags", "cell_area_px",
             "nucleus_area_px", "nc_ratio", "occluded_frac", "cell_polygons", "nucleus_polygons", "near_cluster",
             "near_debris", "votes", "voters"}


@pytest.fixture(scope="module")
def client():
    os.environ["NCMASK_WARMUP"] = "0"
    from fastapi.testclient import TestClient
    from app.backend.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def unvoted(client):
    """vote=false: every YOLO26s box segmented (pipeline v3 behaviour, as the samples/expected references)."""
    r = client.post("/api/infer?consensus=true&vote=false", data={"sample": SAMPLE})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(scope="module")
def result(client):
    r = client.post("/api/infer?consensus=true", data={"sample": SAMPLE})
    assert r.status_code == 200, r.text
    return r.json()


def png_bytes(scale=0.5):
    img = cv2.imread(str(ROOT / "samples" / f"{SAMPLE}.tiff"))
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".png", img)[1].tobytes()


def test_health(client):
    h = client.get("/api/health").json()
    man = json.loads((ROOT / "models" / "onnx" / "manifest.json").read_text())
    assert h["status"] in ("ok", "warming_up")
    assert {"detector", "seg"} <= set(h["models"])
    for k, m in h["models"].items():
        assert m["onnx_sha256"] == man["models"][k]["onnx_sha256"] == m["onnx_sha256_manifest"]
        assert len(m["source_sha256"]) == 64
    assert h["versions"]["onnxruntime"] and h["providers"]
    assert h["bundle_check"] == []
    assert h["max_upload_bytes"] == 25 * 1024 * 1024 and h["result_ttl_s"] == 900
    c = h["models"]["detector_c"]
    assert h["clusters_available"] is True and c["imgsz"] == 1024 and c["conf"] == 0.5
    assert c["names"] == {"0": "cell", "1": "cluster", "2": "debris"} and c["classes_used"] == ["cluster", "debris"]
    assert c["source_sha256"] == "a72ae657980c941447a10430cd399478ae8318c18c4cdc53f35ba6b99df23255"


def test_samples(client):
    s = client.get("/api/samples").json()["samples"]
    assert SAMPLE in [x["name"] for x in s]
    t = client.get(s[0]["thumb"])
    assert t.status_code == 200 and t.content[:3] == b"\xff\xd8\xff"


def test_infer_schema(result):
    r = result
    assert {"id", "size", "image_qc", "cells", "counts", "timings_ms", "consensus", "clusters", "warnings",
            "downloads"} <= set(r)
    assert r["size"] == {"width": 2028, "height": 1520, "expected": [2028, 1520], "is_expected": True}
    assert r["warnings"] == []
    assert set(r["image_qc"]) == {"ok", "flags", "metrics"}
    assert len(r["cells"]) > 10
    for c in r["cells"]:
        assert set(c) == CELL_KEYS
        assert c["status"] in ("measurable", "review", "excluded", "not_segmented")
        assert c["det_tag"] in ("consensus", "s_only", "bc_only")
        assert c["votes"] == len(c["voters"]) and set(c["voters"]) <= set("ABC")
        x1, y1, x2, y2 = c["box"]
        assert 0 <= x1 < x2 <= 2028 and 0 <= y1 < y2 <= 1520
        for poly in c["cell_polygons"] + c["nucleus_polygons"]:
            assert len(poly) >= 3 and all(len(p) == 2 for p in poly)
        if c["status"] == "measurable":
            assert c["cell_polygons"] and c["reasons"] == []
    st = r["counts"]["status"]
    assert sum(st.values()) == r["counts"]["total"] == len(r["cells"])
    assert r["counts"]["reasons"] == {k: v for k, v in r["counts"]["reasons"].items() if v > 0}
    assert set(r["timings_ms"]) >= {"detect", "consensus", "segment", "gate", "total", "decode", "request"}
    assert r["timings_ms"]["consensus"] > 0 and r["consensus"]["used"] is True
    # detector C: information only
    cl = r["clusters"]
    assert cl["used"] is True and cl["info_only"] is True and r["timings_ms"]["clusters"] > 0
    assert cl["boxes"] and {b["class"] for b in cl["boxes"]} <= {"cluster", "debris"}
    assert all(b["conf"] >= 0.5 and b["id"].startswith("C") for b in cl["boxes"])
    assert r["counts"]["clusters"] == {k: sum(b["class"] == k for b in cl["boxes"]) for k in ("cluster", "debris")}
    ids = {b["id"] for b in cl["boxes"]}
    assert all(set(c["near_cluster"]) | set(c["near_debris"]) <= ids for c in r["cells"])
    for c in r["cells"]:   # near_* = >= 50 % of the cell box inside a C box of that class
        x1, y1, x2, y2 = c["box"]
        for k in ("cluster", "debris"):
            want = [b["id"] for b in cl["boxes"] if b["class"] == k and
                    max(0, min(x2, b["box"][2]) - max(x1, b["box"][0])) *
                    max(0, min(y2, b["box"][3]) - max(y1, b["box"][1])) / ((x2 - x1) * (y2 - y1)) >= 0.5]
            assert c["near_" + k] == want


def test_voting(result):
    """Default vote=true: segmented only with >= 2 of 3 detectors and not inside a cluster / debris box."""
    v = result["voting"]
    assert v["used"] and v["detectors_run"] == 3 and v["votes_needed"] == 2
    cl = {b["id"]: b for b in result["clusters"]["boxes"]}
    for c in result["cells"]:
        if c["status"] == "not_segmented":
            assert c["cell_polygons"] == [] and c["cell_area_px"] == 0
            assert c["reasons"] in (["not_voted"], ["inside_cluster"], ["inside_debris"], ["inside_cluster", "inside_debris"])
            assert (c["reasons"] == ["not_voted"]) == (c["votes"] < 2)
        else:
            assert c["votes"] >= 2 and not c["near_cluster"] and not c["near_debris"]
    assert result["counts"]["status"]["not_segmented"] > 0
    assert sum(result["counts"]["votes"].values()) == len(result["cells"])


def test_matches_pipeline_reference(unvoted):
    result = unvoted
    """Same statuses as the committed reference run (samples/expected, Jetson GPU, PyTorch) for matched cells."""
    from ncmask.consensus import match_boxes
    with open(ROOT / "samples" / "expected" / "fovs" / SAMPLE / "cells.csv") as f:
        ref = list(csv.DictReader(f))
    a = np.array([[float(x[k]) for k in ("box_x1", "box_y1", "box_x2", "box_y2")] for x in ref])
    b = np.array([c["box"] for c in result["cells"]])
    pairs, am, bm = match_boxes(a, b, 0.9)
    assert len(ref) == len(result["cells"]) and am.all() and bm.all()
    diff = [(ref[i]["cell_id"], ref[i]["status"], result["cells"][j]["status"]) for i, j, _ in pairs
            if ref[i]["status"] != result["cells"][j]["status"]]
    assert not diff, diff


def test_downloads(client, result):
    d = result["downloads"]
    ov = client.get(d["overlay.png"])
    assert ov.status_code == 200 and ov.content[:8] == b"\x89PNG\r\n\x1a\n" and "attachment" in ov.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(client.get(d["cells.csv"]).text)))
    from ncmask.fov import CELL_COLS
    assert list(rows[0]) == CELL_COLS
    assert sum(r["object"] == "cell" for r in rows) == len(result["cells"])
    crow = [r for r in rows if r["object"] != "cell"]
    assert [r["cell_id"] for r in crow] == [b["id"] for b in result["clusters"]["boxes"]]
    assert all(r["status"] == "" for r in crow)
    coco = client.get(d["coco.json"]).json()
    assert {c["name"] for c in coco["categories"]} == {"cell-area", "nucleus-area", "cluster", "debris"}
    cann = [a for a in coco["annotations"] if a["category_id"] in (3, 4)]
    assert len(cann) == len(result["clusters"]["boxes"]) and all(a["attributes"]["info_only"] for a in cann)
    assert coco["images"][0]["width"] == 2028
    full = client.get(d["coco.json"] + "?layer=full").json()
    assert "FULL" in full["info"]["description"]
    img = client.get(d["image.jpg"])
    assert img.content[:3] == b"\xff\xd8\xff"
    assert client.get(d["coco.json"] + "?layer=bogus").status_code == 422


def test_consensus_off(client):
    r = client.post("/api/infer?consensus=false", data={"sample": SAMPLE}).json()
    assert {c["det_tag"] for c in r["cells"]} == {"single"}
    assert r["timings_ms"]["consensus"] == 0 and r["consensus"]["used"] is False


def test_clusters_off_same_statuses(client, unvoted):
    """Voting off: detector C never changes measurable / review / excluded (or reasons, flags, det_tag)."""
    result = unvoted
    r = client.post("/api/infer?consensus=true&clusters=false&vote=false", data={"sample": SAMPLE}).json()
    assert r["clusters"]["used"] is False and r["clusters"]["boxes"] == [] and r["timings_ms"]["clusters"] == 0
    key = lambda cells: [(c["id"], c["box"], c["status"], c["reasons"], c["flags"], c["det_tag"]) for c in cells]
    assert key(r["cells"]) == key(result["cells"])
    assert all(c["near_cluster"] == [] == c["near_debris"] for c in r["cells"])
    assert r["counts"]["clusters"] == {"cluster": 0, "debris": 0}
    assert client.post("/api/infer?clusters=maybe", data={"sample": SAMPLE}).status_code == 422
    assert client.post("/api/infer?vote=maybe", data={"sample": SAMPLE}).status_code == 422


def test_upload_png_other_size_warns_and_is_not_kept(client):
    before = set(Path(tempfile.gettempdir()).glob("ncmask-upload-*"))
    r = client.post("/api/infer", files={"file": ("my_fov.png", png_bytes(0.5), "image/png")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["size"]["width"] == 1014 and not j["size"]["is_expected"]
    assert any("1014x760" in w for w in j["warnings"])
    assert set(Path(tempfile.gettempdir()).glob("ncmask-upload-*")) == before   # upload deleted after decoding


@pytest.mark.parametrize("name,data,status,code", [
    ("x.gif", b"GIF89a", 415, "unsupported_type"),
    ("x.png", b"not an image at all", 400, "unreadable_image"),
    ("x.png", b"\0" * (25 * 1024 * 1024 + 1), 413, "too_large"),
])
def test_upload_errors(client, name, data, status, code):
    r = client.post("/api/infer", files={"file": (name, data, "application/octet-stream")})
    assert r.status_code == status and r.json()["error"] == code, r.text


def test_request_errors(client):
    assert client.post("/api/infer").status_code == 400                                       # neither
    assert client.post("/api/infer", data={"sample": SAMPLE},
                       files={"file": ("a.png", png_bytes(0.1), "image/png")}).status_code == 400   # both
    assert client.post("/api/infer", data={"sample": "nope"}).status_code == 404
    assert client.post("/api/infer?consensus=maybe", data={"sample": SAMPLE}).status_code == 422
    assert client.get("/api/infer/short/cells.csv").status_code == 404
    assert client.get("/api/infer/" + "a" * 24 + "/cells.csv").json()["error"] == "expired"
    assert client.get("/api/infer/" + "a" * 24 + "/secret.txt").status_code == 404


def test_busy(client):
    from app.backend.main import S
    assert S.busy.acquire(blocking=False)
    try:
        r = client.post("/api/infer", data={"sample": SAMPLE})
        assert r.status_code == 503 and r.json()["error"] == "busy" and r.headers["retry-after"] == "5"
    finally:
        S.busy.release()


def test_expiry(client, result):
    from app.backend.main import S
    rid = result["id"]
    assert client.get(f"/api/infer/{rid}/cells.csv").status_code == 200
    S.store.created[rid] -= 15 * 60 + 1
    assert S.store.sweep() >= 1
    assert not (S.store.root / rid).exists()
    assert client.get(f"/api/infer/{rid}/cells.csv").json()["error"] == "expired"
