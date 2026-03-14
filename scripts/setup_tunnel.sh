#!/usr/bin/env bash
# =============================================================================
# setup_tunnel.sh  –  Expose the voice-bot FastAPI server (port 9000) to the
# public internet so the VICIdial webhook has a reachable URL.
#
# TWO OPTIONS — pick ONE:
#
#   Option A:  ngrok  (installed, authtoken stored in ~/.config/ngrok/ngrok.yml)
#              Starts the tunnel, extracts the URL, and writes it to .env so
#              the bot always knows its own public address.
#              NOTE: free-tier URL changes on each restart — upgrade to ngrok
#              paid plan and use --domain= for a permanent URL.
#
#   Option B:  Cloudflare Tunnel  (permanent URL, requires domain on Cloudflare)
#
# Usage:
#   chmod +x scripts/setup_tunnel.sh
#   ./scripts/setup_tunnel.sh [a|b]
# =============================================================================
set -euo pipefail

BOT_PORT=9000
ENV_FILE="$(dirname "$0")/../.env"

# ─── Option A ────────────────────────────────────────────────────────────────
# ngrok  (authtoken already saved by: ngrok config add-authtoken <token>)
#
# To get a FREE PERMANENT domain (one per account):
#   1. Log in at https://dashboard.ngrok.com
#   2. Cloud Edge → Domains → "Claim a free static domain"
#   3. Use:  ngrok http --domain=your-static-name.ngrok-free.app 9000
# ─────────────────────────────────────────────────────────────────────────────
run_ngrok() {
    LOG=/tmp/ngrok.log

    # Optional: export NGROK_STATIC_DOMAIN=your-static-name.ngrok-free.app
    if [[ -n "${NGROK_STATIC_DOMAIN:-}" ]]; then
        DOMAIN_FLAG="--domain=${NGROK_STATIC_DOMAIN}"
    else
        DOMAIN_FLAG=""
    fi

    echo "Starting ngrok tunnel → localhost:${BOT_PORT} …"
    ngrok http ${DOMAIN_FLAG} --log=stdout --log-format=json "${BOT_PORT}" > "${LOG}" 2>&1 &
    NGROK_PID=$!
    echo "ngrok PID: ${NGROK_PID}"

    # Wait up to 10 s for the URL to appear in the log
    for i in $(seq 1 20); do
        URL=$(grep -o '"url":"https://[^"]*"' "${LOG}" 2>/dev/null | tail -1 | sed 's/"url":"//;s/"//')
        if [[ -n "${URL}" ]]; then
            break
        fi
        sleep 0.5
    done

    if [[ -z "${URL:-}" ]]; then
        echo "ERROR: ngrok did not report a URL within 10 s. Check ${LOG}"
        exit 1
    fi

    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "  ngrok tunnel active"
    echo "  Public URL : ${URL}"
    echo "  Webhook    : ${URL}/vicidial/webhook"
    echo "  WS stream  : ${URL/https/wss}/ws/media/{call_id}"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    # Write PUBLIC_URL into .env so the bot reads it on startup
    if grep -q "^PUBLIC_URL=" "${ENV_FILE}" 2>/dev/null; then
        sed -i "s|^PUBLIC_URL=.*|PUBLIC_URL=${URL}|" "${ENV_FILE}"
    else
        echo "PUBLIC_URL=${URL}" >> "${ENV_FILE}"
    fi
    echo "PUBLIC_URL written to ${ENV_FILE}"

    # Keep the script alive so the tunnel stays up; Ctrl-C to stop
    wait "${NGROK_PID}"
}

# ─── Option B ────────────────────────────────────────────────────────────────
# Cloudflare Tunnel  (cloudflared)
#
# Prerequisites (one-time setup on your domain):
#   1.  Your domain must be on Cloudflare (free plan is fine).
#   2.  Install cloudflared:
#         curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
#              -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared
#   3.  Authenticate once (opens browser — do this locally, copy cert.pem to server):
#         cloudflared tunnel login
#   4.  Create a named tunnel (one-time):
#         cloudflared tunnel create voicebot
#         # This writes a credentials file under ~/.cloudflared/<UUID>.json
#         # and prints the tunnel UUID — store it in CLOUDFLARE_TUNNEL_NAME below.
#   5.  Create a CNAME DNS record pointing your subdomain to the tunnel:
#         cloudflared tunnel route dns voicebot bot.yourdomain.com
#
# After setup, run:   ./setup_tunnel.sh b
# The URL will ALWAYS be:  https://bot.yourdomain.com  — it never changes.
# ─────────────────────────────────────────────────────────────────────────────
run_cloudflare_tunnel() {
    CLOUDFLARE_TUNNEL_NAME="${CLOUDFLARE_TUNNEL_NAME:-voicebot}"

    echo "Starting Cloudflare Tunnel: ${CLOUDFLARE_TUNNEL_NAME} → localhost:${BOT_PORT}"
    echo "VICIdial webhook URL will be: https://<your-subdomain>/vicidial/webhook"
    echo ""

    # The config file at ~/.cloudflared/config.yml (created during setup) maps
    # the ingress rule.  If you prefer an explicit config, point to it with:
    #   cloudflared tunnel --config /path/to/config.yml run
    exec cloudflared tunnel run "${CLOUDFLARE_TUNNEL_NAME}"
}

# ─── Dispatch ─────────────────────────────────────────────────────────────────
OPTION="${1:-a}"   # default to ngrok
case "${OPTION}" in
    a|A) run_ngrok ;;
    b|B) run_cloudflare_tunnel ;;
    *)
        echo "Usage: $0 [a|b]"
        echo "  a  =  ngrok (default)"
        echo "  b  =  Cloudflare Tunnel"
        exit 1
        ;;
esac
