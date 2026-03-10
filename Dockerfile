# syntax=docker/dockerfile:1
# =============================================================================
# Final Expense Voice Bot — main application container
#
# Base: CUDA 12.1 runtime so torch / Parakeet TDT can use the GPU.
# The vLLM and CosyVoice servers are separate containers (see docker-compose).
#
# Build:  docker build -t voicebot .
# Run:    docker compose up voicebot
# =============================================================================

ARG CUDA_VERSION=12.1.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04

# ---------------------------------------------------------------------------
# Non-privileged user (best-practice adopted from ShayneP/local-voice-ai)
# ---------------------------------------------------------------------------
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    curl \
    libsndfile1 \
    ffmpeg \
    sox \
  && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python / pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# ---------------------------------------------------------------------------
# Python dependencies
# Install before copying source for better layer caching.
# ---------------------------------------------------------------------------
WORKDIR /app

COPY requirements.txt .

# torch with CUDA must be installed from the NVIDIA index
RUN pip install --no-cache-dir setuptools<70 numpy==1.26.4 \
 && pip install --no-cache-dir \
      torch torchaudio \
      --index-url https://download.pytorch.org/whl/cu121 \
 && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Application source
# ---------------------------------------------------------------------------
COPY . .
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 9000

HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=60s \
  CMD curl -fsS http://localhost:9000/health || exit 1

ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "9000"]
