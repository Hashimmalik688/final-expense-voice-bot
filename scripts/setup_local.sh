#!/usr/bin/env bash
# =============================================================================
# Local Development Setup
# =============================================================================
# Sets up a Python virtual environment and installs all dependencies for
# local development and testing (no GPU required for mock mode).
#
# Usage:
#   chmod +x scripts/setup_local.sh
#   ./scripts/setup_local.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"

echo "============================================"
echo "  Final Expense Voice Bot — Local Setup"
echo "============================================"

# --- Python version check ---
PYTHON=${PYTHON:-python3}
echo "[1/5] Checking Python version …"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: $PYTHON not found. Install Python 3.10+ first."
    exit 1
fi

PY_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "  Found Python $PY_VERSION"

# --- Virtual environment ---
echo "[2/5] Creating virtual environment …"
if [ ! -d "$VENV_DIR" ]; then
    $PYTHON -m venv "$VENV_DIR"
    echo "  Created: $VENV_DIR"
else
    echo "  Already exists: $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# --- Upgrade pip ---
echo "[3/5] Upgrading pip …"
pip install --quiet --upgrade pip setuptools wheel

# --- Install dependencies ---
echo "[4/5] Installing Python dependencies …"
pip install --quiet -r "$PROJECT_DIR/requirements.txt"

# --- Environment file ---
echo "[5/5] Setting up .env file …"
if [ ! -f "$PROJECT_DIR/.env" ]; then
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    echo "  Created .env from .env.example — edit it with your settings."
else
    echo "  .env already exists — skipping."
fi

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Run automated tests:"
echo "  python -m tests.test_local_conversation auto"
echo "  python -m tests.test_sip_connection"
echo ""
echo "Run interactive conversation test:"
echo "  python -m tests.test_local_conversation"
echo ""
echo "Start the bot:"
echo "  python -m src.main"
echo ""
