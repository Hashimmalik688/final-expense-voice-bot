#!/usr/bin/env bash
# =============================================================================
# stop.sh — Graceful shutdown of all Final Expense Voice Bot services
#
# Stops (in reverse order):
#   1. FastAPI voice bot  (tmux: voicebot)
#   2. Kokoro TTS server  (tmux: kokoro)
#   3. vLLM server        (tmux: vllm)
#   4. ngrok tunnel       (tmux: ngrok)
# =============================================================================
set -euo pipefail

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

stop_session voicebot
stop_session kokoro
stop_session vllm
stop_session ngrok

# Kill any orphan processes on our ports
for port in 9000 8001 8000; do
    pid=$(lsof -ti ":${port}" 2>/dev/null || true)
    if [[ -n "${pid}" ]]; then
        kill "${pid}" 2>/dev/null || true
        info "Killed orphan PID ${pid} on port ${port}"
    fi
done

echo ""
echo "All services stopped."
