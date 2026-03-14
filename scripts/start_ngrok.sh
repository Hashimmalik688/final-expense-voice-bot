#!/usr/bin/env bash
# Run this once after every Vast.ai instance start.
# It launches ngrok in a detached tmux session, waits for the URL,
# and writes it to .env so the bot uses the correct webhook address.
set -euo pipefail

ENV_FILE="$(dirname "$0")/../.env"
LOG=/tmp/ngrok.log

# Kill any previous instance
pkill -f "ngrok http" 2>/dev/null || true
sleep 1

# Start in a detached tmux session so it outlives this shell
tmux new-session -d -s ngrok "ngrok http 9000 --log=stdout --log-format=json > ${LOG} 2>&1" 2>/dev/null \
  || ngrok http 9000 --log=stdout --log-format=json > "${LOG}" 2>&1 &

echo "Waiting for ngrok URL..."
for i in $(seq 1 20); do
    URL=$(grep -o '"url":"https://[^"]*"' "${LOG}" 2>/dev/null | tail -1 | sed 's/"url":"//;s/"//')
    [ -n "${URL}" ] && break
    sleep 0.5
done

if [ -z "${URL:-}" ]; then
    echo "ERROR: ngrok did not start. Check ${LOG}"; exit 1
fi

# Update .env
if grep -q "^PUBLIC_URL=" "${ENV_FILE}" 2>/dev/null; then
    sed -i "s|^PUBLIC_URL=.*|PUBLIC_URL=${URL}|" "${ENV_FILE}"
else
    echo "PUBLIC_URL=${URL}" >> "${ENV_FILE}"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "  ngrok tunnel active"
echo "  Public URL : ${URL}"
echo "  Webhook    : ${URL}/vicidial/webhook"
echo "  WS stream  : ${URL/https/wss}/ws/media/{call_id}"
echo "  .env       : PUBLIC_URL updated"
echo "╚══════════════════════════════════════════════════════════╝"
