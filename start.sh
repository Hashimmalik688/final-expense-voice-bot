#!/usr/bin/env bash
# =============================================================================
# start.sh — Final Expense Voice Bot (LiveKit architecture)
#
# Launch order:
#   1. vLLM server       (tmux: vllm)       port 8000
#   2. Qwen/XTTS server  (tmux: tts)        port 8002
#   3. LiveKit server    (tmux: livekit)    port 7880
#   4. Agent workers     (tmux: agent-N)    N = MAX_BOTS workers
#   5. Token/admin UI    (tmux: tokenserv)  port 9000
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

# ── Load .env so we can read MAX_BOTS etc. ─────────────────────────────────
if [[ -f "${ENV_FILE}" ]]; then
    set -o allexport
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +o allexport
fi

MAX_BOTS="${MAX_BOTS:-1}"

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

# ── Helper: start tmux session ──────────────────────────────────────────────
start_session() {
    local name="$1" cmd="$2" logfile="$3"
    if tmux has-session -t "${name}" 2>/dev/null; then
        info "${name} session already running"
    else
        tmux new-session -d -s "${name}" \
            "source ${VENV} 2>/dev/null || true; ${cmd} 2>&1 | tee ${logfile}"
        info "Started tmux session: ${name}"
    fi
}

# ── 0. Activate venv ──────────────────────────────────────────────────────
if [[ -f "${VENV}" ]]; then
    # shellcheck disable=SC1090
    source "${VENV}"
    info "Activated venv at ${VENV}"
else
    warn "No venv found at ${VENV} — using system Python"
fi

# ── 1. vLLM server (port 8000) ───────────────────────────────────────────
start_session vllm \
    "bash ${REPO_DIR}/scripts/start_vllm.sh" \
    "${LOG_DIR}/vllm.log"
wait_for_health "http://127.0.0.1:8000/health" "vLLM" 180

# ── 2. Qwen/XTTS TTS server (port 8002) ──────────────────────────────────
start_session tts \
    "cd ${REPO_DIR} && python src/tts/qwen_server.py" \
    "${LOG_DIR}/tts.log"
wait_for_health "http://127.0.0.1:8002/health" "TTS Server" 120

# ── 3. LiveKit server (port 7880) ────────────────────────────────────────
if ! command -v livekit-server >/dev/null 2>&1; then
    warn "livekit-server not found in PATH — skipping (set up separately if needed)"
else
    start_session livekit \
        "livekit-server --config ${REPO_DIR}/config/livekit.yaml" \
        "${LOG_DIR}/livekit.log"
    wait_for_health "http://127.0.0.1:7880" "LiveKit" 30
fi

# ── 4. LiveKit agent workers (N = MAX_BOTS) ───────────────────────────────
info "Starting ${MAX_BOTS} agent worker(s) …"
for i in $(seq 1 "${MAX_BOTS}"); do
    session="agent-${i}"
    start_session "${session}" \
        "cd ${REPO_DIR} && PYTHONPATH=${REPO_DIR} python src/agent.py start" \
        "${LOG_DIR}/agent-${i}.log"
done

# ── 5. Token / admin UI server (port 9000) ────────────────────────────────
start_session tokenserv \
    "cd ${REPO_DIR} && PYTHONPATH=${REPO_DIR} python -m uvicorn src.token_server:app --host 0.0.0.0 --port 9000" \
    "${LOG_DIR}/tokenserv.log"
wait_for_health "http://127.0.0.1:9000/health" "Token Server" 30

# ── Done ──────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Final Expense Voice Bot (LiveKit) — All services running   ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  vLLM        : http://127.0.0.1:8000   (tmux: vllm)       ║"
echo "║  TTS Server  : http://127.0.0.1:8002   (tmux: tts)        ║"
echo "║  LiveKit     : ws://127.0.0.1:7880     (tmux: livekit)    ║"
printf  "║  Agents      : %d worker(s)             (tmux: agent-N)   ║\n" "${MAX_BOTS}"
echo "║  Browser UI  : http://127.0.0.1:9000   (tmux: tokenserv)  ║"
echo "║  Admin       : http://127.0.0.1:9000/admin                 ║"
echo "║                                                              ║"
echo "║  Logs: ${LOG_DIR}/                              ║"
echo "║  Stop: ./stop.sh                                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
