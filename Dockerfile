# ──────────────────────────────────────────────────────────────────
# Deep Learning Image Classifier — Production Docker Image
# Base: python:3.11-slim  |  Backend: CPU-only PyTorch
# ──────────────────────────────────────────────────────────────────

# ────────── Stage 1 : dependency installer ──────────────────────
FROM python:3.11-slim AS builder

WORKDIR /install

# System build dependencies (needed to compile some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements so Docker can cache this layer
COPY requirements-prod.txt .

# Install into a local directory (--prefix) so we can COPY it cleanly
RUN pip install --no-cache-dir --prefix=/install/pkg -r requirements-prod.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu


# ────────── Stage 2 : runtime image ─────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="MLOps Team" \
      version="1.0.0" \
      description="ResNet18 image classifier API"

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# System runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /install/pkg /usr/local

# Copy application source
COPY app/          ./app/
COPY models/       ./models/

# Create outputs dir (writable by app)
RUN mkdir -p outputs && chown -R appuser:appuser /app

USER appuser

# Environment defaults (overridable via k8s env vars)
ENV MODEL_PATH=models/best_model.pth \
    CLASS_MAP_PATH=models/class_to_idx.json \
    MODEL_VERSION=1.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Healthcheck — used by Docker and k8s liveness probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 2 workers: enough for CPU-bound inference; tune via env var
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
