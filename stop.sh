#!/usr/bin/env bash
# =============================================================================
# stop.sh — Graceful shutdown of all Final Expense Voice Bot services
#
# Stops (in reverse order):
#   1. Token/admin server  (tmux: tokenserv)
#   2. Agent workers       (tmux: agent-1 … agent-N)
#   3. LiveKit server      (tmux: livekit)
#   4. TTS server          (tmux: tts)
#   5. vLLM server         (tmux: vllm)
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${REPO_DIR}/.env"

# Read MAX_BOTS from .env so we know which agent sessions to stop
MAX_BOTS=1
if [[ -f "${ENV_FILE}" ]]; then
    _mb=$(grep '^MAX_BOTS=' "${ENV_FILE}" 2>/dev/null | cut -d= -f2- | tr -d ' ' || true)
    [[ -n "${_mb}" ]] && MAX_BOTS="${_mb}"
fi

GREEN='\033[0;32m'
NC='\033[0m'
info() { echo -e "${GREEN}[stop]${NC} $*"; }

stop_session() {
    local name="$1"
    if tmux has-session -t "${name}" 2>/dev/null; then
        tmux send-keys -t "${name}" C-c 2>/dev/null || true
        sleep 1
        tmux kill-session -t "${name}" 2>/dev/null || true
        info "Stopped ${name}"
    else
        info "${name} not running"
    fi
}

# Token / admin UI
stop_session tokenserv

# Agent workers
for i in $(seq 1 "${MAX_BOTS}"); do
    stop_session "agent-${i}"
done

# LiveKit server
stop_session livekit

# TTS server
stop_session tts

# vLLM server
stop_session vllm

# Kill any orphan processes on our ports
for port in 9000 8002 7880 8000; do
    pid=$(lsof -ti ":${port}" 2>/dev/null || true)
    if [[ -n "${pid}" ]]; then
        kill "${pid}" 2>/dev/null || true
        info "Killed orphan PID ${pid} on port ${port}"
    fi
done

echo ""
echo "All services stopped."
