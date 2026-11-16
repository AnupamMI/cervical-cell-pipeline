# FOV demo web app: FastAPI + ONNX Runtime (CPU) serving the built React frontend.
# Built remotely by the Hugging Face Space (scripts/deploy_space.*); also builds locally after
# `python scripts/export_onnx.py` (needs models/onnx/). No torch, no ultralytics in the image.

# ---- frontend -----------------------------------------------------------------------------------------------------
FROM node:24-slim AS web
WORKDIR /web
COPY app/frontend/package.json app/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY app/frontend/ ./
RUN npm run build

# ---- runtime ------------------------------------------------------------------------------------------------------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NCMASK_DEVICE=cpu \
    NCMASK_THREADS=2 \
    OMP_NUM_THREADS=2
# libgomp: OpenMP runtime used by onnxruntime / scikit-image wheels on slim images
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
# Hugging Face Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt
COPY --chown=user ncmask/ ncmask/
COPY --chown=user vendor/ vendor/
COPY --chown=user app/__init__.py app/
COPY --chown=user app/backend/ app/backend/
COPY --chown=user config.yaml .
COPY --chown=user models/onnx/ models/onnx/
COPY --chown=user samples/*.tiff samples/
COPY --from=web --chown=user /web/dist app/frontend/dist
USER user
ENV HOME=/home/user
EXPOSE 7860
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--no-server-header"]
