# ---- Stage 1: build -----------------------------------------------------------
# Builds wheels for all deps so the runtime image carries no build toolchain.
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ---- Stage 2: runtime ---------------------------------------------------------
FROM python:3.11-slim AS runtime

# Non-root user: required posture for Container Apps.
RUN useradd --create-home --uid 1001 appuser
WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt && \
    rm -rf /wheels

# Application code + the versioned model artifact.
COPY main.py .
COPY model_v1.0.0.pkl .

ENV MODEL_PATH=/app/model_v1.0.0.pkl \
    API_VERSION=1.0.0 \
    PORT=8000

USER appuser
EXPOSE 8000

# Liveness probe target is /health (configured on Container Apps too).
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
