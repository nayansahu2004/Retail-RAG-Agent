# syntax=docker/dockerfile:1.7
# Multi-stage build: compiler toolchain and pip caches stay in the builder stage.

FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM python:3.12-slim
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

RUN useradd --create-home --uid 1000 appuser && chown appuser:appuser /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser scripts ./scripts
COPY --chown=appuser:appuser eval ./eval
COPY --chown=appuser:appuser data ./data
COPY --chown=appuser:appuser pyproject.toml ./
USER appuser

# Bake catalog, retrieval artifacts, and model weights into the image.
RUN python -m scripts.generate_catalog \
 && python -m scripts.ingest \
 && python -m scripts.build_eval_set

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

# Single worker: scale out with more containers (and Redis-backed memory), not more workers.
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
