#!/usr/bin/env bash
# =============================================================================
# scripts/deploy_gpu.sh
# =============================================================================
# One-command bare-metal deployment for the Final Expense Voice Bot on
# Ubuntu 22.04 / 24.04 with an NVIDIA A100 40 GB GPU (Vast.ai or bare metal).
#
# USAGE
#   sudo bash scripts/deploy_gpu.sh [OPTIONS]
#
# OPTIONS
#   --hf-token TOKEN       HuggingFace access token (use for gated / large models)
#   --project-dir  DIR     Source directory to deploy from (default: repo root)
#   --deploy-dir   DIR     Install destination            (default: /opt/voicebot)
#   --model-dir    DIR     Model weight storage           (default: /opt/models)
#   --skip-apt             Skip apt update + package install (faster re-deploy)
#   --skip-cuda            Skip CUDA toolkit install check
#   --skip-models          Skip model weight downloads (already downloaded)
#   --skip-services        Skip systemd unit file generation
#   -y / --yes             Non-interactive; skip all prompts
#
# DISK REQUIREMENTS  (recommend at least 100 GB)
#   Mimo v2 Flash 7B          ~15 GB
#   Parakeet TDT 0.6B          ~1.2 GB
#   CosyVoice2 0.5B            ~1.0 GB
#   sentence-transformers MiniLM ~0.1 GB
#   Python envs (torch/vLLM/NeMo) ~30 GB
#   ─────────────────────────────────────
#   Total                      ~47 GB
#
# GPU MEMORY BUDGET  (A100 40 GB)
#   vLLM  — Mimo 7B (bfloat16)    ~16 GB   (gpu_memory_utilization=0.42)
#   CosyVoice 2 (0.5B)             ~2 GB
#   Parakeet TDT (0.6B, in-proc)   ~2 GB
#   CUDA context + overhead        ~2 GB
#   ───────────────────────────────────
#   Total                         ~22 GB   (18 GB headroom remaining)
#
# SERVICE ARCHITECTURE AFTER DEPLOYMENT
#
#   ┌─────────────────────────────────────────────────────────────┐
#   │  voicebot-vllm.service                                       │
#   │  vLLM OpenAI-compatible server  localhost:8000               │
#   │  Model: XiaomiMiMo/MiMo-7B-RL (Mimo v2 Flash)               │
#   └─────────────────────────────────────────────────────────────┘
#
#   ┌─────────────────────────────────────────────────────────────┐
#   │  voicebot-tts.service                                        │
#   │  CosyVoice 2 streaming TTS server  localhost:8001            │
#   │  Model: FunAudioLLM/CosyVoice2-0.5B                         │
#   └─────────────────────────────────────────────────────────────┘
#
#   ┌─────────────────────────────────────────────────────────────┐
#   │  voicebot.service  (main application)                        │
#   │  FastAPI management API  0.0.0.0:9000                        │
#   │  SIP listener  0.0.0.0:5060                                  │
#   │  Parakeet TDT STT  (in-process, CUDA)                        │
#   └─────────────────────────────────────────────────────────────┘
#
#   redis.service  (system package, localhost:6379)
#
# =============================================================================
set -Eeuo pipefail

# ─── Error handler ────────────────────────────────────────────────────────────
_err_handler() {
    local code=$1 line=$2
    echo ""
    echo "  ✗  Script failed at line $line (exit code $code)."
    echo "     Check the output above for details."
    echo ""
    exit "$code"
}
trap '_err_handler $? $LINENO' ERR

# ─── Colour helpers ───────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "  ${CYAN}▶${RESET} $*"; }
ok()      { echo -e "  ${GREEN}✓${RESET} $*"; }
warn()    { echo -e "  ${YELLOW}⚠${RESET} $*"; }
die()     { echo -e "  ${RED}✗ ERROR:${RESET} $*" >&2; exit 1; }
section() { echo -e "\n${BOLD}${CYAN}══ $* ══${RESET}"; }

# ─── Defaults ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_SRC="$(dirname "$SCRIPT_DIR")"   # repo root (where this script lives)
DEPLOY_DIR="/opt/voicebot"
MODEL_DIR="/opt/models"
COSYVOICE_SRC="/opt/cosyvoice-src"

# Separate virtual envs per service (prevents dependency conflicts)
VENV_BOT="${DEPLOY_DIR}/.venv"
VENV_VLLM="/opt/vllm-env"
VENV_TTS="/opt/cosyvoice-env"

LOG_DIR="/var/log/voicebot"
PYTHON_BIN=""                            # resolved after Python install

# ─── Detect whether systemd is usable (containers often lack it) ──────────────
HAS_SYSTEMD=false
if command -v systemctl &>/dev/null && systemctl is-system-running &>/dev/null 2>&1; then
    HAS_SYSTEMD=true
fi
TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu121"
VLLM_VERSION="0.6.6"
NEMO_VERSION="2.1.0"
CUDA_REQUIRED_MAJOR=12
HF_TOKEN=""
SKIP_APT=false
SKIP_CUDA=false
SKIP_MODELS=false
SKIP_SERVICES=false
SKIP_TTS_VENV=false
YES=false

# ─── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf-token)     HF_TOKEN="$2";           shift 2 ;;
        --project-dir)  PROJECT_SRC="$2";        shift 2 ;;
        --deploy-dir)   DEPLOY_DIR="$2";         shift 2 ;;
        --model-dir)    MODEL_DIR="$2";          shift 2 ;;
        --skip-apt)      SKIP_APT=true;          shift   ;;
        --skip-cuda)     SKIP_CUDA=true;         shift   ;;
        --skip-models)   SKIP_MODELS=true;       shift   ;;
        --skip-services) SKIP_SERVICES=true;     shift   ;;
        --skip-tts-venv) SKIP_TTS_VENV=true;     shift   ;;
        -y|--yes)        YES=true;               shift   ;;
        *) die "Unknown option: $1  (run with --help to see options)" ;;
    esac
done

# ─── Banner ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}"
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║   Final Expense Voice Bot — GPU Deployment               ║"
echo "  ║   Ubuntu 22.04 / 24.04  ·  NVIDIA A100 40 GB            ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo -e "${RESET}"
echo "  Project source : $PROJECT_SRC"
echo "  Deploy target  : $DEPLOY_DIR"
echo "  Model storage  : $MODEL_DIR"
echo ""

# ─── Confirm ──────────────────────────────────────────────────────────────────
if [[ "$YES" != true ]]; then
    read -rp "  Continue? [y/N] " _confirm
    [[ "${_confirm,,}" == y ]] || { echo "Aborted."; exit 0; }
    echo ""
fi

START_TS=$(date +%s)


# ═══════════════════════════════════════════════════════════════════════════════
section "STEP 1/11  Pre-flight checks"
# ═══════════════════════════════════════════════════════════════════════════════

# Must run as root (needed for apt, systemd, ufw)
[[ $EUID -eq 0 ]] || die "Run with sudo or as root."
ok "Running as root"

# Ubuntu 22.04 / 24.04
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    if [[ "$ID" != "ubuntu" ]] || [[ "$VERSION_ID" != "22.04" && "$VERSION_ID" != "24.04" ]]; then
        warn "Expected Ubuntu 22.04 or 24.04; detected ${PRETTY_NAME:-unknown}. Proceeding anyway."
    else
        ok "OS: $PRETTY_NAME"
    fi
fi

# NVIDIA GPU
if ! command -v nvidia-smi &>/dev/null; then
    die "nvidia-smi not found. Install NVIDIA drivers (run-file or DKMS) first."
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
ok "GPU: ${GPU_NAME} (${GPU_VRAM})"

# Disk space check — require at least 80 GB free
FREE_GB=$(df "$MODEL_DIR" 2>/dev/null | tail -1 | awk '{print int($4/1024/1024)}' || df / | tail -1 | awk '{print int($4/1024/1024)}')
if [[ "$FREE_GB" -lt 80 ]]; then
    warn "Free disk space: ${FREE_GB} GB. Recommend at least 80 GB. Proceeding anyway."
else
    ok "Free disk space: ${FREE_GB} GB"
fi

# HuggingFace token warning
if [[ -z "$HF_TOKEN" ]]; then
    warn "No --hf-token supplied. Downloads may be rate-limited or fail for gated models."
    warn "Get a token at https://huggingface.co/settings/tokens and pass --hf-token TOKEN"
fi


# ═══════════════════════════════════════════════════════════════════════════════
section "STEP 2/11  System packages"
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$SKIP_APT" == false ]]; then
    info "Running apt-get update …"
    apt-get update -qq

    info "Installing system packages …"
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        build-essential \
        git git-lfs \
        curl wget \
        software-properties-common \
        python3-pip \
        python3-venv \
        python3-dev \
        libssl-dev \
        libffi-dev \
        libsndfile1 libsndfile1-dev \
        libasound2-dev \
        ffmpeg \
        sox \
        portaudio19-dev \
        libopus0 libopus-dev \
        redis-server \
        ufw \
        jq \
        rsync \
        netcat-openbsd
    ok "System packages installed"

    # Activate git-lfs
    git lfs install --system --skip-smudge
    ok "git-lfs configured"
else
    warn "SKIP_APT=true — skipping apt update"
fi

# ─── Resolve Python 3.10 ──────────────────────────────────────────────────────
# Ubuntu 22.04 ships Python 3.10. Ubuntu 24.04 ships Python 3.12 — install 3.10.
if python3.10 --version &>/dev/null 2>&1; then
    PYTHON_BIN="python3.10"
    ok "Python 3.10 found: $(python3.10 --version)"
elif [[ "$SKIP_APT" == false ]]; then
    info "Python 3.10 not found — installing via deadsnakes PPA …"
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -qq
    apt-get install -y -qq python3.10 python3.10-venv python3.10-dev
    PYTHON_BIN="python3.10"
    ok "Python 3.10 installed"
else
    # Fall back to whatever python3 is available
    PYTHON_BIN="python3"
    warn "Python 3.10 not found, using $(python3 --version)"
fi


# ═══════════════════════════════════════════════════════════════════════════════
section "STEP 3/11  CUDA toolkit"
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$SKIP_CUDA" == true ]]; then
    warn "SKIP_CUDA=true — skipping CUDA check"
else
    # Read installed CUDA version from nvcc or nvidia-smi
    CUDA_VER=""
    if command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    else
        # nvidia-smi reports the driver-side CUDA version (sufficient for runtime)
        CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || true)
    fi

    if [[ -n "$CUDA_VER" ]]; then
        CUDA_MAJOR=${CUDA_VER%%.*}
        ok "CUDA detected: $CUDA_VER"
        if [[ "$CUDA_MAJOR" -lt "$CUDA_REQUIRED_MAJOR" ]]; then
            warn "CUDA $CUDA_VER < required $CUDA_REQUIRED_MAJOR.x. Attempting upgrade …"
            _need_cuda_install=true
        fi
    else
        warn "nvcc not found — CUDA toolkit may be missing (drivers only)."
        info "Installing CUDA 12.4 toolkit from NVIDIA apt repository …"
        _need_cuda_install=true
    fi

    if [[ "${_need_cuda_install:-false}" == true ]]; then
        # NVIDIA apt repository for Ubuntu 22.04 / 24.04
        _OS_VER=${VERSION_ID//./}   # "2204" or "2404"
        _KEYRING="cuda-keyring_1.1-1_all.deb"
        wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${_OS_VER}/x86_64/${_KEYRING}" -O /tmp/${_KEYRING}
        dpkg -i /tmp/${_KEYRING}
        apt-get update -qq
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
            cuda-toolkit-12-4 \
            libcudnn9-cuda-12
        export PATH="/usr/local/cuda/bin:$PATH"
        ok "CUDA 12.4 toolkit installed"
    fi

    # Export CUDA paths for all subsequent pip installs
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi


# ═══════════════════════════════════════════════════════════════════════════════
section "STEP 4/11  Redis"
# ═══════════════════════════════════════════════════════════════════════════════

# Harden Redis — bind to localhost only
if [[ -f /etc/redis/redis.conf ]]; then
    sed -i 's/^bind .*/bind 127.0.0.1 -::1/' /etc/redis/redis.conf
    sed -i 's/^protected-mode .*/protected-mode yes/' /etc/redis/redis.conf
fi
# Vast.ai containers don't have systemd — start Redis directly as a daemon
if command -v systemctl &>/dev/null && systemctl is-system-running &>/dev/null 2>&1; then
    systemctl enable redis-server
    systemctl start redis-server
else
    # No systemd (Docker/container environment like Vast.ai)
    redis-server --daemonize yes --bind 127.0.0.1 --protected-mode yes
    sleep 1
fi
redis-cli ping | grep -q PONG && ok "Redis running on localhost:6379"


# ═══════════════════════════════════════════════════════════════════════════════
section "STEP 5/11  Project sync + directories"
# ═══════════════════════════════════════════════════════════════════════════════

mkdir -p "$DEPLOY_DIR" "$MODEL_DIR" "$LOG_DIR"
mkdir -p "${MODEL_DIR}/mimo" "${MODEL_DIR}/parakeet" "${MODEL_DIR}/cosyvoice" "${MODEL_DIR}/minilm"

# Sync source code to deploy directory (preserves git history if src is a repo)
info "Syncing project: $PROJECT_SRC → $DEPLOY_DIR"
rsync -a --delete \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='logs/' \
    "$PROJECT_SRC/" "$DEPLOY_DIR/"
ok "Project synced to $DEPLOY_DIR"


# ═══════════════════════════════════════════════════════════════════════════════
section "STEP 6/11  Python virtual environments + packages"
# ═══════════════════════════════════════════════════════════════════════════════
# Three isolated envs to avoid dependency conflicts:
#   VENV_BOT   — main bot (FastAPI, NeMo/Parakeet, RAG, SIP, …)
#   VENV_VLLM  — vLLM OpenAI server only
#   VENV_TTS   — CosyVoice 2 TTS server only

_pip_upgrade() {
    "$1/bin/pip" install --upgrade pip wheel setuptools -q
}

# Like _pip_upgrade but caps setuptools at <70 so legacy setup.py files that
# import pkg_resources (e.g. openai-whisper) still build correctly.
_pip_upgrade_compat() {
    "$1/bin/pip" install --upgrade pip wheel "setuptools<70" -q
}

# ─── 6a. Bot venv ─────────────────────────────────────────────────────────────
info "Creating bot venv: $VENV_BOT"
[[ -d "$VENV_BOT" ]] || "$PYTHON_BIN" -m venv "$VENV_BOT"
_pip_upgrade "$VENV_BOT"

info "Installing PyTorch (CUDA 12.1) into bot venv …"
"${VENV_BOT}/bin/pip" install -q \
    "torch>=2.5.0" "torchaudio>=2.5.0" \
    --index-url "$TORCH_CUDA_INDEX"

info "Installing NeMo ASR toolkit (Parakeet TDT) …"
# NeMo has many heavy deps; install with retries
"${VENV_BOT}/bin/pip" install -q \
    "nemo_toolkit[asr]==${NEMO_VERSION}" \
    --extra-index-url https://pypi.nvidia.com || \
"${VENV_BOT}/bin/pip" install -q "nemo_toolkit[asr]"

info "Installing bot application dependencies …"
"${VENV_BOT}/bin/pip" install -q \
    -r "${DEPLOY_DIR}/requirements.txt"

"${VENV_BOT}/bin/pip" install -q \
    "sentence-transformers>=3.0.0" \
    "pyVoIP>=1.6.8" \
    "aiohttp>=3.10.0" \
    "scikit-learn>=1.4.0" \
    "uvicorn[standard]>=0.30.0" \
    "huggingface_hub[cli]>=0.23.0,<1.0"

ok "Bot venv ready: $VENV_BOT"

# ─── 6b. vLLM venv ────────────────────────────────────────────────────────────
info "Creating vLLM venv: $VENV_VLLM"
[[ -d "$VENV_VLLM" ]] || "$PYTHON_BIN" -m venv "$VENV_VLLM"
_pip_upgrade "$VENV_VLLM"

info "Installing vLLM ${VLLM_VERSION} …"
# vLLM bundles its own torch; install it first then vLLM to avoid conflicts
"${VENV_VLLM}/bin/pip" install -q \
    "torch>=2.5.0" --index-url "$TORCH_CUDA_INDEX"
"${VENV_VLLM}/bin/pip" install -q \
    "vllm==${VLLM_VERSION}" \
    --extra-index-url https://download.pytorch.org/whl/cu121
ok "vLLM venv ready: $VENV_VLLM"

# ─── 6c. CosyVoice venv ───────────────────────────────────────────────────────
if [[ "$SKIP_TTS_VENV" == true ]]; then
    warn "SKIP_TTS_VENV=true — skipping CosyVoice venv setup"
else
info "Cloning CosyVoice 2 source …"
if [[ ! -d "$COSYVOICE_SRC/.git" ]]; then
    git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git "$COSYVOICE_SRC"
    cd "$COSYVOICE_SRC"
    git submodule update --init --recursive
else
    info "CosyVoice source already cloned at $COSYVOICE_SRC — pulling latest …"
    cd "$COSYVOICE_SRC" && git pull --ff-only
fi
cd /

info "Creating CosyVoice venv: $VENV_TTS"
[[ -d "$VENV_TTS" ]] || "$PYTHON_BIN" -m venv "$VENV_TTS"
# Use compat upgrader: setuptools<70 keeps pkg_resources importable, which is
# required by openai-whisper's legacy setup.py (line 5 imports pkg_resources).
_pip_upgrade_compat "$VENV_TTS"

info "Installing CosyVoice 2 dependencies …"
"${VENV_TTS}/bin/pip" install -q \
    "torch>=2.5.0" "torchaudio>=2.5.0" \
    --index-url "$TORCH_CUDA_INDEX"

# Pre-install numpy before the full requirements file: pyworld's setup.py
# imports numpy at build time and will fail if it isn't present yet.
"${VENV_TTS}/bin/pip" install -q "numpy==1.26.4"

# tensorrt-cu12 packages use a stub build backend (wheel_stub.buildapi) that
# pip cannot resolve from PyPI.  Install them up-front from NVIDIA's index,
# falling back gracefully if unavailable (CosyVoice works without TensorRT;
# it falls back to direct PyTorch inference).
"${VENV_TTS}/bin/pip" install -q \
    "tensorrt-cu12==10.13.3.9" \
    "tensorrt-cu12-bindings==10.13.3.9" \
    "tensorrt-cu12-libs==10.13.3.9" \
    --extra-index-url https://pypi.nvidia.com \
    || warn "tensorrt-cu12 unavailable — CosyVoice will use PyTorch inference fallback"

# --no-build-isolation: uses the venv's already-installed setuptools/numpy so
# that pkg_resources (openai-whisper) and numpy (pyworld) are visible to each
# package's build backend subprocess.
# Exclude tensorrt lines (already handled above) to avoid redundant failures.
grep -v '^tensorrt' "${COSYVOICE_SRC}/requirements.txt" | \
    "${VENV_TTS}/bin/pip" install -q -r /dev/stdin --no-build-isolation

# CosyVoice has no setup.py / pyproject.toml → editable install is not
# possible. Instead, drop a .pth file so the source tree is on sys.path.
SITE_PKGS="$("${VENV_TTS}/bin/python" -c \
    'import sysconfig; print(sysconfig.get_path("purelib"))')"
echo "$COSYVOICE_SRC" > "${SITE_PKGS}/cosyvoice-src.pth"
# Also add the Matcha-TTS submodule that CosyVoice imports from third_party/
echo "${COSYVOICE_SRC}/third_party/Matcha-TTS" >> "${SITE_PKGS}/cosyvoice-src.pth"
info "CosyVoice source added to sys.path via ${SITE_PKGS}/cosyvoice-src.pth"

"${VENV_TTS}/bin/pip" install -q \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0"
ok "CosyVoice venv ready: $VENV_TTS"
fi  # end SKIP_TTS_VENV


# ═══════════════════════════════════════════════════════════════════════════════
section "STEP 7/11  CosyVoice TTS server script"
# ═══════════════════════════════════════════════════════════════════════════════

# Write the standalone TTS HTTP server that voicebot-tts.service runs.
# It exposes:
#   GET  /health
#   POST /v1/tts/stream   — streaming raw 16-bit PCM @ 22 050 Hz

cat > "${DEPLOY_DIR}/scripts/cosyvoice_server.py" << 'PYEOF'
#!/usr/bin/env python3
"""
CosyVoice 2 TTS HTTP server.

Exposes the API consumed by src/tts/cosyvoice_handler.py:

  GET  /health
       Returns {"status":"ok","model_loaded":true}

  POST /v1/tts/stream
       Body: {"text":"...", "voice_id":"friendly_female",
               "sample_rate":22050, "speed":1.0, "stream":true}
       Returns: streaming application/octet-stream (raw 16-bit LE PCM)

VOICE IDs supported (maps to CosyVoice2 built-in SFT speakers):
  friendly_female  →  英文女  (English female,  clear and warm)
  friendly_male    →  英文男  (English male)
  default          →  英文女

Environment variables:
  COSYVOICE_MODEL_DIR   Path to downloaded CosyVoice2-0.5B weights
                        (default: /opt/models/cosyvoice)
  TTS_HOST              Bind host  (default: 127.0.0.1)
  TTS_PORT              Bind port  (default: 8001)
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import queue as _queue

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("cosyvoice-server")

MODEL_DIR = os.getenv("COSYVOICE_MODEL_DIR", "/opt/models/cosyvoice")
HOST      = os.getenv("TTS_HOST", "127.0.0.1")
PORT      = int(os.getenv("TTS_PORT", "8001"))

# Map voice_id → CosyVoice2 SFT speaker name
VOICE_MAP: dict[str, str] = {
    "friendly_female": "英文女",
    "friendly_male":   "英文男",
    "default":         "英文女",
}

app = FastAPI(title="CosyVoice 2 TTS Server")
_model = None          # CosyVoice2 instance
_model_lock = threading.Lock()


@app.on_event("startup")
def _load_model() -> None:
    global _model
    logger.info("Loading CosyVoice2 from %s …", MODEL_DIR)
    try:
        # CosyVoice2 init is CPU-heavy; do it once at startup
        from cosyvoice.cli.cosyvoice import CosyVoice2
        with _model_lock:
            _model = CosyVoice2(
                MODEL_DIR,
                load_jit=False,   # JIT compile slows cold start, skip for now
                load_trt=False,   # TensorRT opt — enable manually for perf
            )
        logger.info(
            "CosyVoice2 ready.  Available speakers: %s",
            _model.list_avail_spks(),
        )
    except Exception as exc:
        logger.error("CosyVoice2 load failed: %s", exc, exc_info=True)
        # Server stays up — /health will report model_loaded=false


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


class TTSRequest(BaseModel):
    text:        str
    voice_id:    str   = "friendly_female"
    sample_rate: int   = 22050
    speed:       float = 1.0
    stream:      bool  = True


@app.post("/v1/tts/stream")
def tts_stream(req: TTSRequest) -> StreamingResponse:
    if _model is None:
        raise HTTPException(503, "Model not loaded")

    spk_id  = VOICE_MAP.get(req.voice_id, "英文女")
    text    = req.text.strip()
    speed   = max(0.5, min(2.0, req.speed))   # clamp to safe range

    if not text:
        raise HTTPException(400, "Empty text")

    # CosyVoice2 inference is synchronous; run it in a thread so we don't
    # block the event loop.  We push chunks into a regular queue and yield
    # them in the streaming response generator.
    chunk_queue: _queue.Queue[bytes | None] = _queue.Queue(maxsize=200)

    def _infer() -> None:
        try:
            with _model_lock:
                gen = _model.inference_sft(text, spk_id, stream=True, speed=speed)
                for chunk in gen:
                    # Each chunk: {"tts_speech": Tensor(float32, shape [1, T])}
                    audio_f32 = chunk["tts_speech"].squeeze().numpy()  # float32
                    # CosyVoice outputs 22050 Hz float32 [-1, 1]
                    # Convert to int16 PCM for the SIP/RTP pipeline
                    audio_i16 = (
                        np.clip(audio_f32, -1.0, 1.0) * 32767
                    ).astype(np.int16)
                    chunk_queue.put(audio_i16.tobytes())
        except Exception as exc:
            logger.error("TTS inference error: %s", exc, exc_info=True)
        finally:
            chunk_queue.put(None)   # sentinel: end of stream

    threading.Thread(target=_infer, daemon=True).start()

    def _generate():
        while True:
            data = chunk_queue.get()
            if data is None:
                break
            yield data

    return StreamingResponse(
        _generate(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate":  str(req.sample_rate),
            "X-Sample-Width": "2",     # 16-bit
            "X-Channels":     "1",     # mono
        },
    )


if __name__ == "__main__":
    logger.info("Starting CosyVoice TTS server on %s:%d", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
PYEOF

chmod +x "${DEPLOY_DIR}/scripts/cosyvoice_server.py"
ok "TTS server script written: ${DEPLOY_DIR}/scripts/cosyvoice_server.py"


# ═══════════════════════════════════════════════════════════════════════════════
section "STEP 8/11  Download model weights"
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$SKIP_MODELS" == true ]]; then
    warn "SKIP_MODELS=true — skipping model downloads"
else
    # Configure HuggingFace token if provided
    # Resolve huggingface-cli: prefer the bin script, fall back to python -m
    if [[ -x "${VENV_BOT}/bin/huggingface-cli" ]]; then
        HF_CLI=("${VENV_BOT}/bin/huggingface-cli")
    else
        HF_CLI=("${VENV_BOT}/bin/python" -m huggingface_hub.commands.huggingface_cli)
    fi
    # Export HF_TOKEN so all child processes inherit it
    if [[ -n "$HF_TOKEN" ]]; then
        export HF_TOKEN
        "${HF_CLI[@]}" login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
    fi
    _hf_dl() { "${HF_CLI[@]}" download "$@"; }

    # ── Mimo v2 Flash (7B) ────────────────────────────────────────────────────
    info "Downloading Mimo v2 Flash (XiaomiMiMo/MiMo-7B-RL) → ${MODEL_DIR}/mimo"
    info "  This is ~15 GB — expect 5–20 minutes depending on bandwidth"
    _hf_dl XiaomiMiMo/MiMo-7B-RL \
        --local-dir "${MODEL_DIR}/mimo" \
        --local-dir-use-symlinks False
    ok "Mimo v2 Flash downloaded"

    # ── Parakeet TDT 0.6B ─────────────────────────────────────────────────────
    info "Downloading Parakeet TDT (nvidia/parakeet-tdt-0.6b-v2) → ${MODEL_DIR}/parakeet"
    _hf_dl nvidia/parakeet-tdt-0.6b-v2 \
        --local-dir "${MODEL_DIR}/parakeet" \
        --local-dir-use-symlinks False
    ok "Parakeet TDT downloaded"

    # ── CosyVoice2 0.5B ───────────────────────────────────────────────────────
    info "Downloading CosyVoice2-0.5B (FunAudioLLM/CosyVoice2-0.5B) → ${MODEL_DIR}/cosyvoice"
    _hf_dl FunAudioLLM/CosyVoice2-0.5B \
        --local-dir "${MODEL_DIR}/cosyvoice" \
        --local-dir-use-symlinks False
    ok "CosyVoice2 downloaded"

    # ── sentence-transformers MiniLM (RAG engine) ─────────────────────────────
    info "Downloading all-MiniLM-L6-v2 (RAG semantic search) → ${MODEL_DIR}/minilm"
    _hf_dl sentence-transformers/all-MiniLM-L6-v2 \
        --local-dir "${MODEL_DIR}/minilm" \
        --local-dir-use-symlinks False
    ok "MiniLM downloaded"
fi


# ═══════════════════════════════════════════════════════════════════════════════
section "STEP 9/11  Environment file"
# ═══════════════════════════════════════════════════════════════════════════════

_ENV="${DEPLOY_DIR}/.env"
if [[ ! -f "$_ENV" ]]; then
    info "Generating .env from .env.example …"
    cp "${DEPLOY_DIR}/.env.example" "$_ENV"

    # Point all model paths to the downloaded locations
    _BOT_IP=$(hostname -I | awk '{print $1}')

    # Use sed to patch the template values
    sed -i "s|^SIP_LOCAL_IP=.*|SIP_LOCAL_IP=${_BOT_IP}|"                   "$_ENV"
    sed -i "s|^LLM_MODEL=.*|LLM_MODEL=${MODEL_DIR}/mimo|"                  "$_ENV"
    sed -i "s|^VLLM_API_URL=.*|VLLM_API_URL=http://127.0.0.1:8000|"       "$_ENV"
    sed -i "s|^STT_MODEL=.*|STT_MODEL=${MODEL_DIR}/parakeet|"              "$_ENV"
    sed -i "s|^TTS_API_URL=.*|TTS_API_URL=http://127.0.0.1:8001|"         "$_ENV"
    sed -i "s|^TTS_MODEL=.*|TTS_MODEL=${MODEL_DIR}/cosyvoice|"             "$_ENV"
    sed -i "s|^RAG_EMBEDDING_MODEL=.*|RAG_EMBEDDING_MODEL=${MODEL_DIR}/minilm|" "$_ENV"
    [[ -n "$HF_TOKEN" ]] && sed -i "s|^HF_TOKEN=.*|HF_TOKEN=${HF_TOKEN}|" "$_ENV"

    chmod 600 "$_ENV"
    ok ".env generated at $_ENV"
    warn "Edit $_ENV to set VICIdial credentials and SIP password before going live."
else
    ok ".env already exists at $_ENV — skipping (delete to regenerate)"
fi


# ═══════════════════════════════════════════════════════════════════════════════
section "STEP 10/11  Systemd services"
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$SKIP_SERVICES" == true ]]; then
    warn "SKIP_SERVICES=true — skipping systemd configuration"
else

# ─── voicebot-vllm.service ────────────────────────────────────────────────────
tee /etc/systemd/system/voicebot-vllm.service > /dev/null << EOF
[Unit]
Description=Voice Bot — vLLM OpenAI Server (Mimo v2 Flash)
Documentation=https://github.com/vllm-project/vllm
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${DEPLOY_DIR}
EnvironmentFile=${DEPLOY_DIR}/.env

# GPU device — change CUDA_VISIBLE_DEVICES to use a specific GPU index
Environment=CUDA_VISIBLE_DEVICES=0
Environment=TRANSFORMERS_OFFLINE=1
Environment=HF_DATASETS_OFFLINE=1

# A100 40 GB budget:
#   gpu_memory_utilization=0.42 → vLLM uses ≤ 16.8 GB
#   Max model context: 8192 tokens (longer = more KV cache VRAM)
#   max-num-seqs=30 → 30 concurrent call legs maximum
ExecStart=${VENV_VLLM}/bin/python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_DIR}/mimo \
    --served-model-name MiMo-7B-RL \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.42 \
    --max-model-len 8192 \
    --max-num-seqs 30 \
    --disable-log-requests \
    --trust-remote-code

Restart=always
RestartSec=10
TimeoutStartSec=300
TimeoutStopSec=30
StandardOutput=append:${LOG_DIR}/vllm.log
StandardError=append:${LOG_DIR}/vllm.log

[Install]
WantedBy=multi-user.target
EOF

# ─── voicebot-tts.service ─────────────────────────────────────────────────────
tee /etc/systemd/system/voicebot-tts.service > /dev/null << EOF
[Unit]
Description=Voice Bot — CosyVoice 2 TTS Server
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${DEPLOY_DIR}
EnvironmentFile=${DEPLOY_DIR}/.env
Environment=CUDA_VISIBLE_DEVICES=0
Environment=COSYVOICE_MODEL_DIR=${MODEL_DIR}/cosyvoice
Environment=TTS_HOST=127.0.0.1
Environment=TTS_PORT=8001
Environment=PYTHONPATH=${COSYVOICE_SRC}

ExecStart=${VENV_TTS}/bin/python ${DEPLOY_DIR}/scripts/cosyvoice_server.py

Restart=always
RestartSec=15
TimeoutStartSec=120
TimeoutStopSec=30
StandardOutput=append:${LOG_DIR}/tts.log
StandardError=append:${LOG_DIR}/tts.log

[Install]
WantedBy=multi-user.target
EOF

# ─── voicebot.service ─────────────────────────────────────────────────────────
# The main bot loads Parakeet TDT in-process (no separate STT service needed).
# It waits for vLLM and TTS to be ready via the health-check loop in src/main.py.
tee /etc/systemd/system/voicebot.service > /dev/null << EOF
[Unit]
Description=Voice Bot — Final Expense (FastAPI + SIP + Parakeet STT)
After=network.target redis.service voicebot-vllm.service voicebot-tts.service
Wants=network-online.target redis.service
Requires=voicebot-vllm.service voicebot-tts.service

[Service]
Type=simple
User=root
WorkingDirectory=${DEPLOY_DIR}
EnvironmentFile=${DEPLOY_DIR}/.env
Environment=CUDA_VISIBLE_DEVICES=0
Environment=PYTHONPATH=${DEPLOY_DIR}
Environment=HF_HUB_OFFLINE=1
Environment=TRANSFORMERS_OFFLINE=1

ExecStartPre=/bin/bash -c '\\
    for i in \$(seq 1 60); do \\
        curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1 && \\
        curl -sf http://127.0.0.1:8001/health > /dev/null 2>&1 && break; \\
        echo "Waiting for vLLM and TTS (\$i/60)…"; sleep 5; \\
    done'

ExecStart=${VENV_BOT}/bin/uvicorn src.main:app \
    --host 0.0.0.0 \
    --port 9000 \
    --workers 1 \
    --loop asyncio \
    --log-level info \
    --access-log

Restart=always
RestartSec=10
TimeoutStartSec=600
TimeoutStopSec=30
StandardOutput=append:${LOG_DIR}/voicebot.log
StandardError=append:${LOG_DIR}/voicebot.log

[Install]
WantedBy=multi-user.target
EOF

# ─── voicebot-monitor.service + timer (restart check every 5 min) ────────────
tee /etc/systemd/system/voicebot-monitor.service > /dev/null << 'EOF'
[Unit]
Description=Voice Bot — Health Monitor

[Service]
Type=oneshot
ExecStart=/bin/bash -c '\
    _fail=0; \
    for svc in voicebot-vllm voicebot-tts voicebot; do \
        systemctl is-active --quiet "$svc" || { \
            echo "[monitor] $svc is down — restarting"; \
            systemctl restart "$svc"; \
            _fail=1; \
        }; \
    done; \
    [[ $_fail -eq 0 ]] && echo "[monitor] all services healthy"'
EOF

tee /etc/systemd/system/voicebot-monitor.timer > /dev/null << 'EOF'
[Unit]
Description=Voice Bot — Health Monitor (every 5 min)

[Timer]
OnBootSec=5min
OnUnitActiveSec=5min
Unit=voicebot-monitor.service

[Install]
WantedBy=timers.target
EOF

# ─── Reload systemd and enable services ──────────────────────────────────────
if [[ "$HAS_SYSTEMD" == true ]]; then
    systemctl daemon-reload
    systemctl enable voicebot-vllm voicebot-tts voicebot voicebot-monitor.timer
    ok "Systemd units created and enabled"
else
    warn "systemd not available (container/Vast.ai) — systemd units written but not enabled"
    info "Services will be started directly via shell commands instead"
fi

fi  # SKIP_SERVICES


# ═══════════════════════════════════════════════════════════════════════════════
section "STEP 11/11  Firewall"
# ═══════════════════════════════════════════════════════════════════════════════

# Only configure ufw if it's available and not already active in a conflicting way
if command -v ufw &>/dev/null; then
    # Allow SSH before enabling, to avoid locking ourselves out
    ufw allow ssh                       > /dev/null 2>&1 || true
    ufw allow 5060/udp comment "SIP"    > /dev/null 2>&1 || true
    ufw allow 5060/tcp comment "SIP/TCP" > /dev/null 2>&1 || true
    ufw allow 9000/tcp comment "VoiceBot API" > /dev/null 2>&1 || true
    # Internal ports: vLLM (8000) and TTS (8001) bound to 127.0.0.1 — no ufw rule needed
    if ufw --force enable > /dev/null 2>&1; then
        ok "ufw rules: SSH + 5060/udp + 5060/tcp + 9000/tcp"
        ufw status numbered | head -20
    else
        warn "ufw could not be enabled (likely container/Vast.ai) — ports are open by default"
    fi
else
    warn "ufw not found; configuring iptables directly …"
    iptables -I INPUT -p udp --dport 5060 -j ACCEPT -m comment --comment "SIP"
    iptables -I INPUT -p tcp --dport 5060 -j ACCEPT -m comment --comment "SIP/TCP"
    iptables -I INPUT -p tcp --dport 9000 -j ACCEPT -m comment --comment "VoiceBot API"
    # Persist rules if iptables-persistent is installed
    command -v netfilter-persistent &>/dev/null && netfilter-persistent save || true
    ok "iptables rules applied (SIP 5060, API 9000)"
fi


# ═══════════════════════════════════════════════════════════════════════════════
section "Start services + health checks"
# ═══════════════════════════════════════════════════════════════════════════════

_wait_http() {
    local name=$1 url=$2 timeout=${3:-300}
    local elapsed=0
    info "Waiting for ${name} (${url}) …"
    while [[ $elapsed -lt $timeout ]]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            ok "${name} is ready  (${elapsed}s)"
            return 0
        fi
        sleep 5; elapsed=$((elapsed + 5))
        printf "."
    done
    echo ""
    warn "${name} did not respond within ${timeout}s"
    return 1
}

if [[ "$SKIP_SERVICES" != true ]]; then
    if [[ "$HAS_SYSTEMD" == true ]]; then
        # ── systemd path ──────────────────────────────────────────────────────
        info "Starting voicebot-vllm …"
        systemctl start voicebot-vllm
        _wait_http "vLLM" "http://127.0.0.1:8000/health" 300

        info "Starting voicebot-tts …"
        systemctl start voicebot-tts
        _wait_http "CosyVoice TTS" "http://127.0.0.1:8001/health" 120

        info "Starting voicebot …"
        systemctl start voicebot
        _wait_http "Voice Bot API" "http://127.0.0.1:9000/health" 60

        info "Starting health monitor timer …"
        systemctl start voicebot-monitor.timer
    else
        # ── Container / no-systemd path ───────────────────────────────────────
        # Write a helper start script that can be re-used manually
        cat > "${DEPLOY_DIR}/scripts/start_all.sh" << 'STARTEOF'
#!/usr/bin/env bash
set -euo pipefail
DEPLOY_DIR="/opt/voicebot"
LOG_DIR="/var/log/voicebot"
source "${DEPLOY_DIR}/.env" 2>/dev/null || true
mkdir -p "$LOG_DIR"

echo "[start_all] Starting vLLM …"
nohup /opt/vllm-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model /opt/models/mimo \
    --served-model-name MiMo-7B-RL \
    --host 127.0.0.1 --port 8000 \
    --dtype bfloat16 --gpu-memory-utilization 0.42 \
    --max-model-len 8192 --max-num-seqs 30 \
    --disable-log-requests --trust-remote-code \
    > "${LOG_DIR}/vllm.log" 2>&1 &
echo $! > /tmp/voicebot-vllm.pid

echo "[start_all] Starting CosyVoice TTS …"
export COSYVOICE_MODEL_DIR=/opt/models/cosyvoice
export TTS_HOST=127.0.0.1 TTS_PORT=8001
export PYTHONPATH="/opt/cosyvoice-src:/opt/cosyvoice-src/third_party/Matcha-TTS"
nohup /opt/cosyvoice-env/bin/python "${DEPLOY_DIR}/scripts/cosyvoice_server.py" \
    > "${LOG_DIR}/tts.log" 2>&1 &
echo $! > /tmp/voicebot-tts.pid

# Wait for vLLM + TTS to be ready
for i in $(seq 1 60); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && \
    curl -sf http://127.0.0.1:8001/health >/dev/null 2>&1 && break
    echo "  Waiting for vLLM + TTS ($i/60)…"; sleep 5
done

echo "[start_all] Starting Voice Bot …"
export PYTHONPATH="${DEPLOY_DIR}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nohup /opt/voicebot/.venv/bin/uvicorn src.main:app \
    --host 0.0.0.0 --port 9000 --workers 1 \
    --loop asyncio --log-level info --access-log \
    > "${LOG_DIR}/voicebot.log" 2>&1 &
echo $! > /tmp/voicebot-main.pid

echo "[start_all] All services launched."
echo "  PIDs: vllm=$(cat /tmp/voicebot-vllm.pid) tts=$(cat /tmp/voicebot-tts.pid) bot=$(cat /tmp/voicebot-main.pid)"
STARTEOF
        chmod +x "${DEPLOY_DIR}/scripts/start_all.sh"

        # Also write a stop script
        cat > "${DEPLOY_DIR}/scripts/stop_all.sh" << 'STOPEOF'
#!/usr/bin/env bash
for pf in /tmp/voicebot-vllm.pid /tmp/voicebot-tts.pid /tmp/voicebot-main.pid; do
    [[ -f "$pf" ]] && kill "$(cat "$pf")" 2>/dev/null && rm -f "$pf" && echo "Stopped $(basename "$pf" .pid)"
done
STOPEOF
        chmod +x "${DEPLOY_DIR}/scripts/stop_all.sh"
        ok "Container launcher scripts written:"
        info "  ${DEPLOY_DIR}/scripts/start_all.sh"
        info "  ${DEPLOY_DIR}/scripts/stop_all.sh"

        # Now actually start everything
        info "Starting all services (no-systemd mode) …"
        bash "${DEPLOY_DIR}/scripts/start_all.sh"

        _wait_http "vLLM" "http://127.0.0.1:8000/health" 300
        _wait_http "CosyVoice TTS" "http://127.0.0.1:8001/health" 120
        _wait_http "Voice Bot API" "http://127.0.0.1:9000/health" 60
    fi
fi


# ─── Final summary ────────────────────────────────────────────────────────────
ELAPSED=$(( $(date +%s) - START_TS ))
_BOT_IP=$(hostname -I | awk '{print $1}')

echo ""
echo -e "${BOLD}${GREEN}"
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║   Deployment complete in ${ELAPSED}s                        "
echo "  ╚══════════════════════════════════════════════════════════╝"
echo -e "${RESET}"
cat << INFO
  ── Services ───────────────────────────────────────────────────────
  vLLM (Mimo v2 Flash)  :  http://127.0.0.1:8000/health
  CosyVoice 2 TTS       :  http://127.0.0.1:8001/health
  Voice Bot API         :  http://${_BOT_IP}:9000/health
  Redis                 :  127.0.0.1:6379
  SIP endpoint          :  sip:voicebot@${_BOT_IP}:5060

  ── Virtual environments ────────────────────────────────────────────
  Bot          : ${VENV_BOT}
  vLLM         : ${VENV_VLLM}
  CosyVoice    : ${VENV_TTS}

  ── Logs ─────────────────────────────────────────────────────────────
  tail -f ${LOG_DIR}/voicebot.log
  tail -f ${LOG_DIR}/vllm.log
  tail -f ${LOG_DIR}/tts.log
INFO

if [[ "$HAS_SYSTEMD" == true ]]; then
cat << INFO
  journalctl -u voicebot -f

  ── Service management ───────────────────────────────────────────────
  systemctl status voicebot voicebot-vllm voicebot-tts
  systemctl restart voicebot
  systemctl stop voicebot voicebot-vllm voicebot-tts
INFO
else
cat << INFO

  ── Service management (container mode) ──────────────────────────────
  Start all : bash ${DEPLOY_DIR}/scripts/start_all.sh
  Stop all  : bash ${DEPLOY_DIR}/scripts/stop_all.sh
  Check PIDs: cat /tmp/voicebot-*.pid
INFO
fi

cat << INFO

  ── Next steps ────────────────────────────────────────────────────────
  1. Edit ${DEPLOY_DIR}/.env — set VICIDIAL_* credentials and SIP_PASSWORD
  2. Configure Asterisk: point SIP/voicebot to ${_BOT_IP}:5060
  3. Test: curl http://${_BOT_IP}:9000/health
  4. Reload knowledge base (no restart): curl -X POST http://${_BOT_IP}:9000/reload/knowledge

INFO

