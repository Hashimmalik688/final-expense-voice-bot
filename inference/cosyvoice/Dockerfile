# syntax=docker/dockerfile:1
# =============================================================================
# CosyVoice 2 TTS inference server
#
# Clones the FunAudioLLM/CosyVoice repo, installs its deps, downloads
# CosyVoice2-0.5B weights on first boot, and exposes a FastAPI REST API at
# port 8001 compatible with src/tts/cosyvoice_handler.py.
# =============================================================================

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    git curl ffmpeg sox libsndfile1 \
  && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /cosyvoice-root

# Clone CosyVoice (shallow clone for speed)
RUN git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git .

# Pin setuptools before building deps that use pkg_resources
RUN pip install --no-cache-dir "setuptools<70" numpy==1.26.4
RUN pip install --no-cache-dir \
      torch torchaudio \
      --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt || true

# Add repo root to sys.path via a .pth file (no setup.py in CosyVoice)
RUN python -c \
    "import site, pathlib; \
     (pathlib.Path(site.getsitepackages()[0]) / 'cosyvoice.pth').write_text('/cosyvoice-root\n')"

# Copy our FastAPI wrapper
COPY server.py /cosyvoice-root/server.py

EXPOSE 8001

HEALTHCHECK --interval=10s --timeout=5s --retries=30 --start-period=60s \
  CMD curl -fsS http://localhost:8001/health || exit 1

CMD ["python", "server.py"]
