#!/usr/bin/env bash
# =============================================================================
# start.sh — Single-command startup for the Final Expense Voice Bot
#
# Launches (in order):
#   1. ngrok tunnel (tmux session "ngrok")
#   2. vLLM server  (tmux session "vllm")
#   3. Kokoro TTS   (tmux session "kokoro")
#   4. Filler audio generation (one-shot, waits for TTS to be healthy)
#   5. FastAPI app   (tmux session "voicebot")
#
# Usage:
#   chmod +x start.sh && ./start.sh
#
# Stop everything:
#   ./stop.sh
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${REPO_DIR}/.env"
VENV="/venv/main/bin/activate"
LOG_DIR="${REPO_DIR}/logs"

mkdir -p "${LOG_DIR}"

# ── Colours ────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[start]${NC} $*"; }
warn()  { echo -e "${YELLOW}[start]${NC} $*"; }
fail()  { echo -e "${RED}[start]${NC} $*"; exit 1; }

# ── Helper: wait for HTTP health endpoint ──────────────────────────────────
wait_for_health() {
    local url="$1" label="$2" timeout="${3:-60}"
    info "Waiting for ${label} at ${url} …"
    for i in $(seq 1 "${timeout}"); do
        if curl -sf "${url}" >/dev/null 2>&1; then
            info "${label} is healthy (${i}s)"
            return 0
        fi
        sleep 1
    done
    fail "${label} did not become healthy within ${timeout}s"
}

# ── 0. Activate venv ──────────────────────────────────────────────────────
if [[ -f "${VENV}" ]]; then
    # shellcheck disable=SC1090
    source "${VENV}"
    info "Activated venv"
else
    warn "No venv found at ${VENV} — using system Python"
fi

# ── 1. ngrok tunnel ───────────────────────────────────────────────────────
if tmux has-session -t ngrok 2>/dev/null; then
    info "ngrok session already running"
else
    info "Starting ngrok tunnel …"
    bash "${REPO_DIR}/scripts/start_ngrok.sh"
fi

# ── 2. vLLM server (port 8000) ───────────────────────────────────────────
if tmux has-session -t vllm 2>/dev/null; then
    info "vLLM session already running"
else
    info "Starting vLLM server …"
    tmux new-session -d -s vllm \
        "source ${VENV} && bash ${REPO_DIR}/scripts/start_vllm.sh 2>&1 | tee ${LOG_DIR}/vllm.log"
fi
wait_for_health "http://127.0.0.1:8000/health" "vLLM" 120

# ── 3. Kokoro TTS server (port 8001) ─────────────────────────────────────
if tmux has-session -t kokoro 2>/dev/null; then
    info "Kokoro session already running"
else
    info "Starting Kokoro TTS server …"
    tmux new-session -d -s kokoro \
        "source ${VENV} && python ${REPO_DIR}/scripts/kokoro_server.py 2>&1 | tee ${LOG_DIR}/kokoro.log"
fi
wait_for_health "http://127.0.0.1:8001/health" "Kokoro TTS" 90

# ── 4. Generate filler audio (if missing) ────────────────────────────────
FILLERS_DIR="${REPO_DIR}/src/tts/fillers"
if [[ -d "${FILLERS_DIR}" ]] && [[ $(find "${FILLERS_DIR}" -name "*.wav" 2>/dev/null | wc -l) -ge 3 ]]; then
    info "Filler audio already exists — skipping generation"
else
    info "Generating filler audio clips …"
    cd "${REPO_DIR}" && python scripts/generate_fillers.py
fi

# ── 5. FastAPI voice bot (port 9000) ─────────────────────────────────────
if tmux has-session -t voicebot 2>/dev/null; then
    info "Voice bot session already running"
else
    info "Starting voice bot …"
    tmux new-session -d -s voicebot \
        "source ${VENV} && cd ${REPO_DIR} && uvicorn src.main:app --host 0.0.0.0 --port 9000 2>&1 | tee ${LOG_DIR}/bot.log"
fi
wait_for_health "http://127.0.0.1:9000/health" "Voice Bot" 90

# ── Done ──────────────────────────────────────────────────────────────────
PUBLIC_URL=$(grep '^PUBLIC_URL=' "${ENV_FILE}" 2>/dev/null | cut -d= -f2-)
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Final Expense Voice Bot — All services running             ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  vLLM      : http://127.0.0.1:8000   (tmux: vllm)         ║"
echo "║  Kokoro TTS: http://127.0.0.1:8001   (tmux: kokoro)       ║"
echo "║  Voice Bot : http://127.0.0.1:9000   (tmux: voicebot)     ║"
echo "║  Public URL: ${PUBLIC_URL:-N/A}"
echo "║                                                             ║"
echo "║  Logs: ${LOG_DIR}/                         ║"
echo "║  Stop: ./stop.sh                                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
