"""FastAPI app: single-FOV inference (ONNX Runtime) + the built React frontend.

    uvicorn app.backend.main:app --port 7860

Privacy: an upload is written to a private temp file only for decoding and deleted right after. Results (display
JPEG, overlay, CSV, COCO) live in a private temp folder for 15 minutes, then a sweeper deletes them. Logs carry
the result id, image size, counts and timings: never file names or pixel data.
"""
import logging
import os
import re
import secrets
import shutil
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.backend import engine as eng  # noqa: E402
from ncmask.imageio import ImageLoadError  # noqa: E402

log = logging.getLogger("app")
APP_VERSION = "0.1.0"
MAX_UPLOAD = 25 * 1024 * 1024
TTL_S = int(os.environ.get("NCMASK_TTL_S", 15 * 60))
EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
RID = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
FILES = {"overlay.png": "image/png", "cells.csv": "text/csv", "coco.json": "application/json",
         "image.jpg": "image/jpeg"}
SAMPLES = Path(os.environ.get("NCMASK_SAMPLES", ROOT / "samples"))
DIST = Path(os.environ.get("NCMASK_FRONTEND", ROOT / "app" / "frontend" / "dist"))


def err(status, code, detail, headers=None):
    return JSONResponse({"error": code, "detail": detail}, status_code=status, headers=headers)


class ResultStore:
    """root/<id>/{files}; an id expires TTL_S after creation. Swept every 30 s and checked on every access."""

    def __init__(self, ttl):
        self.root = Path(tempfile.mkdtemp(prefix="ncmask-results-"))
        self.ttl = ttl
        self.created = {}
        self.lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._sweeper, daemon=True).start()

    def put(self, files):
        rid = secrets.token_urlsafe(18)
        d = self.root / rid
        d.mkdir(mode=0o700)
        for name, data in files.items():
            (d / name).write_bytes(data)
        with self.lock:
            self.created[rid] = time.time()
        return rid, datetime.fromtimestamp(time.time() + self.ttl, timezone.utc).isoformat(timespec="seconds")

    def path(self, rid, name):
        with self.lock:
            t = self.created.get(rid)
        if t is None:
            return None
        if time.time() - t > self.ttl:
            self.delete(rid)
            return None
        p = self.root / rid / name
        return p if p.is_file() else None

    def delete(self, rid):
        with self.lock:
            self.created.pop(rid, None)
        shutil.rmtree(self.root / rid, ignore_errors=True)

    def sweep(self):
        now = time.time()
        with self.lock:
            old = [r for r, t in self.created.items() if now - t > self.ttl]
        for r in old:
            self.delete(r)
        return len(old)

    def _sweeper(self):
        while not self._stop.wait(30):
            try:
                n = self.sweep()
                if n:
                    log.info("deleted %d expired result(s)", n)
            except Exception:  # pragma: no cover
                log.exception("sweep failed")

    def close(self):
        self._stop.set()
        shutil.rmtree(self.root, ignore_errors=True)


class State:
    engine: "eng.Engine" = None
    store: ResultStore = None
    busy = threading.Lock()
    warming = False
    started = time.time()
    thumbs = {}


S = State()


def _warmup():
    files = sample_files()
    if not files:
        return
    with S.busy:
        S.warming = True
        try:
            img, _ = S.engine.decode(files[0])
            t = S.engine.warmup(img)
            log.info("warm-up done: %s", t)
        except Exception:
            log.exception("warm-up failed")
        finally:
            S.warming = False


@asynccontextmanager
async def lifespan(app):
    logging.basicConfig(level=os.environ.get("NCMASK_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    S.engine = eng.Engine()
    S.store = ResultStore(TTL_S)
    log.info("models loaded (providers %s, consensus %s)", S.engine.providers, S.engine.consensus_available)
    if os.environ.get("NCMASK_WARMUP", "1") == "1":
        threading.Thread(target=_warmup, daemon=True).start()
    yield
    S.store.close()


app = FastAPI(title="Cervical Cell Pipeline (FOV demo)", version=APP_VERSION, lifespan=lifespan,
              docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)


@app.middleware("http")
async def headers(request: Request, call_next):
    r = await call_next(request)
    r.headers.setdefault("X-Content-Type-Options", "nosniff")
    r.headers.setdefault("Referrer-Policy", "no-referrer")
    if request.url.path.startswith("/api/"):
        r.headers.setdefault("Cache-Control", "no-store")
    return r


def sample_files():
    if not SAMPLES.is_dir():
        return []
    return sorted(p for p in SAMPLES.iterdir() if p.is_file() and p.suffix.lower() in EXTS)


# ---------------------------------------------------------------------------------------------------------- API
@app.get("/api/health")
def health():
    h = S.engine.health
    return {"status": "warming_up" if S.warming else "ok", "app_version": APP_VERSION, "busy": S.busy.locked(),
            "uptime_s": round(time.time() - S.started), "result_ttl_s": TTL_S, "max_upload_bytes": MAX_UPLOAD,
            "typical_ms": S.engine.typical_ms, **h}


@app.get("/api/samples")
def samples():
    return {"samples": [{"name": p.stem, "file": p.name, "bytes": p.stat().st_size,
                         "thumb": f"/api/samples/{p.stem}/thumb.jpg"} for p in sample_files()]}


@app.get("/api/samples/{name}/thumb.jpg")
def sample_thumb(name: str):
    p = next((f for f in sample_files() if f.stem == name), None)
    if p is None:
        return err(404, "not_found", "no such sample")
    if name not in S.thumbs:
        import cv2
        img, _ = S.engine.decode(p)
        s = 360 / max(img.shape[:2])
        S.thumbs[name] = eng.encode_jpg(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), 82)
    return Response(S.thumbs[name], media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})


def _save_upload(file: UploadFile, dst: Path):
    """Stream to dst, refusing more than MAX_UPLOAD bytes. Returns the size, or None if too large."""
    n = 0
    with open(dst, "wb") as f:
        while True:
            chunk = file.file.read(1 << 20)
            if not chunk:
                return n
            n += len(chunk)
            if n > MAX_UPLOAD:
                return None
            f.write(chunk)


@app.post("/api/infer")
def infer(file: UploadFile | None = File(None, description="FOV image: .tif/.tiff/.png/.jpg/.jpeg, at most 25 MB"),
          sample: str | None = Form(None, description="instead of a file: the name of a sample from /api/samples"),
          consensus: bool = Query(True, description="run the YOLO26l consensus check (tag only, never filters)")):
    if (file is None) == (sample is None):
        return err(400, "bad_request", "send exactly one of: a file, or a sample name")
    if file is not None and Path(file.filename or "").suffix.lower() not in EXTS:
        return err(415, "unsupported_type", "Unsupported file type. Use .tif, .tiff, .png, .jpg or .jpeg.")
    if sample is not None and not any(p.stem == sample for p in sample_files()):
        return err(404, "not_found", "no such sample")
    if not S.busy.acquire(blocking=False):
        return err(503, "busy", "Another image is being analysed right now (one at a time on this demo). "
                   "Please retry in a few seconds.", headers={"Retry-After": "5"})
    tmpdir = None
    try:
        t0 = time.perf_counter()
        if file is not None:
            tmpdir = Path(tempfile.mkdtemp(prefix="ncmask-upload-"))
            src = tmpdir / ("upload" + Path(file.filename).suffix.lower())
            if _save_upload(file, src) is None:
                return err(413, "too_large", f"File is larger than {MAX_UPLOAD // (1024 * 1024)} MB.")
        else:
            src = next(p for p in sample_files() if p.stem == sample)
        try:
            img, iinfo = S.engine.decode(src)
        except ImageLoadError as e:
            return err(400, "unreadable_image", f"Could not read this image: {e}")
        finally:
            if tmpdir is not None:          # the upload is gone as soon as it is decoded
                shutil.rmtree(tmpdir, ignore_errors=True)
                tmpdir = None
        t_decode = time.perf_counter() - t0
        res = S.engine.infer(img, consensus, name=(sample + ".tiff") if sample else "upload")
        files = {"overlay.png": eng.encode_png(res["overlay"]), "cells.csv": eng.cells_csv(res["rows"]),
                 "coco.json": eng.coco_bytes(res["coco"]), "coco_full.json": eng.coco_bytes(res["coco_full"]),
                 "image.jpg": eng.encode_jpg(img, 90)}
        rid, exp = S.store.put(files)
        out = eng.to_json(res, img.shape, iinfo, rid, exp, consensus, S.engine)
        out["timings_ms"]["decode"] = round(1000 * t_decode, 1)
        out["timings_ms"]["request"] = round(1000 * (time.perf_counter() - t0), 1)
        out["source"] = {"sample": sample} if sample else {"upload": True}
        log.info("result %s: %dx%d, %d cells, consensus %s, %.0f ms", rid[:6], img.shape[1], img.shape[0],
                 len(out["cells"]), res["consensus_used"], out["timings_ms"]["request"])
        return out
    except Exception:
        log.exception("inference failed")
        return err(500, "internal_error", "Inference failed on this image.")
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)
        S.busy.release()


@app.get("/api/infer/{rid}/{name}")
def result_file(rid: str, name: str, layer: str = Query("exclusive", pattern="^(exclusive|full)$")):
    if not RID.match(rid) or name not in FILES:
        return err(404, "not_found", "unknown result file")
    stored = "coco_full.json" if (name == "coco.json" and layer == "full") else name
    p = S.store.path(rid, stored)
    if p is None:
        return err(404, "expired", "This result has expired (results are kept for 15 minutes) or never existed.")
    dl = {"overlay.png": "overlay.png", "cells.csv": "cells.csv",
          "coco.json": "cells_full.coco.json" if layer == "full" else "cells.coco.json"}.get(name)
    return FileResponse(p, media_type=FILES[name], filename=dl,
                        content_disposition_type="attachment" if dl else "inline")


# ------------------------------------------------------------------------------------------------- frontend
if DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            return err(404, "not_found", "unknown API route")
        f = (DIST / path).resolve()
        if path and f.is_file() and DIST.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(DIST / "index.html", headers={"Cache-Control": "no-cache"})
