"""src/token_server.py — LiveKit token server + admin API.

Serves:
  GET  /           → static/index.html  (browser call test UI)
  GET  /token      → LiveKit JWT for browser caller
  GET  /health     → health check (compatible with start.sh polling)
  GET  /admin      → static/admin.html  (operations dashboard)
  GET  /admin/status   → JSON health of all services
  GET  /admin/calls    → active/recent calls JSON
  WS   /admin/ws       → live event stream (call events + log lines)
  POST /admin/script   → save sales_script.yaml
  POST /admin/settings → save tuning params to .env
  POST /admin/bots     → update MAX_BOTS in .env

Run:
  uvicorn src.token_server:app --host 0.0.0.0 --port 9000

Accessing the browser UI:
  http://localhost:9000

Admin panel:
  http://localhost:9000/admin

No authentication on the token endpoint intentionally — this is a local
development server. For production, add OAuth2 / JWT guard on /admin routes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Optional

import aiohttp
from dotenv import load_dotenv, set_key
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger(__name__)

LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "secret")
VLLM_URL = os.environ.get("VLLM_API_URL", "http://127.0.0.1:8000")
TTS_SERVER_URL = os.environ.get("TTS_SERVER_URL", "http://127.0.0.1:8002")

# MOCK_MODE=true  → works on Windows without LiveKit server, vLLM, or GPU.
# The /token endpoint returns a fake token, and /mock/ws provides a simple
# echo-style Sarah persona over WebSocket so the browser UI shows green.
MOCK_MODE = os.environ.get("MOCK_MODE", "false").lower() == "true"

# Scripted opening + fallback replies used in mock mode
_MOCK_OPENING = (
    "Hi, this is Sarah with American Beneficiary — "
    "I'm calling about the final expense coverage inquiry we received. "
    "Did I catch you at an okay time?"
)
_MOCK_REPLIES = [
    "That's great to hear. Just to confirm I have you in the right area — "
    "are you between the ages of 45 and 85?",
    "Perfect. And are you currently covered under any final expense or "
    "burial insurance policy?",
    "I understand. A lot of people in your situation find that even a small "
    "policy can take a huge burden off their family. "
    "Can I take just two more minutes to walk you through what's available?",
    "Based on what you've told me it sounds like you could qualify for "
    "coverage starting as low as $20 a month. Does that sound like "
    "something worth looking into?",
    "Wonderful — let me connect you with one of our senior benefit "
    "specialists who can lock in your rate today.",
]

REPO_ROOT = Path(__file__).parent.parent
ENV_FILE = REPO_ROOT / ".env"
SCRIPT_FILE = REPO_ROOT / "config" / "sales_script.yaml"
LOGS_DIR = REPO_ROOT / "logs"
CALLS_DIR = LOGS_DIR / "calls"
STATIC_DIR = REPO_ROOT / "static"

# Admin WebSocket manager
_admin_clients: list[WebSocket] = []

app = FastAPI(title="Sarah Voice Bot — Token & Admin Server", version="2.0.0")

# Serve static files (browser UI + admin UI)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Token endpoint — browser connects to LiveKit using this JWT
# ---------------------------------------------------------------------------

@app.get("/token")
async def get_token(room: Optional[str] = None):
    """Issue a LiveKit access token for a browser caller.

    In MOCK_MODE the server issues a real-looking JWT signed with the local
    key but points the browser at the mock WebSocket endpoint instead of a
    real LiveKit room, so the UI shows "Connected" without a LiveKit server.
    """
    room_name = room or f"call-{uuid.uuid4().hex[:8]}"

    if MOCK_MODE:
        # Build a minimal JWT-shaped token (signed locally) so the browser
        # doesn't error, then redirect it to our own mock WS endpoint.
        try:
            import jwt as _jwt  # PyJWT — installed with livekit-api
            import time
            payload = {
                "sub": f"caller-{uuid.uuid4().hex[:6]}",
                "iss": LIVEKIT_API_KEY,
                "nbf": int(time.time()),
                "exp": int(time.time()) + 3600,
                "room": room_name,
                "mock": True,
            }
            token = _jwt.encode(payload, LIVEKIT_API_SECRET, algorithm="HS256")
        except Exception:
            token = f"mock-token-{uuid.uuid4().hex}"
        # Point browser at our own mock WebSocket room endpoint
        mock_ws = f"ws://localhost:9000"
        return {
            "token": token,
            "room": room_name,
            "url": mock_ws,
            "livekit_http": "http://localhost:9000",
            "mock": True,
        }

    try:
        from livekit import api as lk_api  # type: ignore
        token = (
            lk_api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
            .with_identity(f"caller-{uuid.uuid4().hex[:6]}")
            .with_name("Test Caller")
            .with_grants(
                lk_api.VideoGrants(room_join=True, room=room_name)
            )
            .to_jwt()
        )
    except Exception as exc:
        logger.error("Token generation failed: %s", exc)
        raise HTTPException(500, f"Token generation failed: {exc}") from exc

    lk_http = LIVEKIT_URL.replace("ws://", "http://").replace("wss://", "https://")
    return {
        "token": token,
        "room": room_name,
        "url": LIVEKIT_URL,
        "livekit_http": lk_http,
    }


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    if MOCK_MODE:
        return {
            "status": "ok",
            "vllm": "mock",
            "tts": "mock",
            "livekit": "mock",
            "mock_mode": True,
        }
    checks = await asyncio.gather(
        _check_http(f"{VLLM_URL}/health", "vllm"),
        _check_http(f"{TTS_SERVER_URL}/health", "tts"),
        _check_http(
            LIVEKIT_URL.replace("ws://", "http://").replace("wss://", "https://"),
            "livekit",
        ),
        return_exceptions=True,
    )
    vllm_ok, tts_ok, lk_ok = [
        c if isinstance(c, bool) else False for c in checks
    ]
    return {
        "status": "ok" if (vllm_ok and tts_ok) else "degraded",
        "vllm": "ok" if vllm_ok else "degraded",
        "tts": "ok" if tts_ok else "degraded",
        "livekit": "ok" if lk_ok else "degraded",
    }


# ---------------------------------------------------------------------------
# Index / Admin HTML pages
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    p = STATIC_DIR / "index.html"
    if p.exists():
        return FileResponse(str(p))
    return JSONResponse({"error": "static/index.html not found"}, status_code=404)


@app.get("/admin")
async def admin():
    p = STATIC_DIR / "admin.html"
    if p.exists():
        return FileResponse(str(p))
    return JSONResponse({"error": "static/admin.html not found"}, status_code=404)


# ---------------------------------------------------------------------------
# Admin API endpoints
# ---------------------------------------------------------------------------

@app.get("/admin/status")
async def admin_status():
    """Health status of all system components."""
    return await health()


@app.get("/admin/calls")
async def admin_calls():
    """Load call history from JSONL log files."""
    calls = []
    if CALLS_DIR.exists():
        for jfile in sorted(CALLS_DIR.glob("*.jsonl"), reverse=True)[:100]:
            events = []
            try:
                for line in jfile.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        events.append(json.loads(line))
            except Exception:
                pass
            if events:
                summary = _summarise_call(jfile.stem, events)
                calls.append(summary)
    return {"calls": calls}


@app.get("/admin/calls/{call_id}")
async def admin_call_detail(call_id: str):
    """Full event log for one call."""
    # Sanitise call_id to prevent path traversal
    safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "", call_id)
    jfile = CALLS_DIR / f"{safe_id}.jsonl"
    if not jfile.exists():
        raise HTTPException(404, "Call not found")
    events = []
    for line in jfile.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                events.append(json.loads(line))
            except Exception:
                pass
    return {"call_id": safe_id, "events": events}


@app.get("/admin/script")
async def admin_get_script():
    """Return current sales_script.yaml content."""
    if not SCRIPT_FILE.exists():
        raise HTTPException(404, "sales_script.yaml not found")
    return {"content": SCRIPT_FILE.read_text(encoding="utf-8")}


class ScriptUpdateRequest(BaseModel):
    content: str


@app.post("/admin/script")
async def admin_save_script(req: ScriptUpdateRequest):
    """Save edited sales_script.yaml."""
    if not req.content.strip():
        raise HTTPException(400, "content is empty")
    # Basic YAML validation before saving
    try:
        import yaml
        yaml.safe_load(req.content)
    except Exception as exc:
        raise HTTPException(400, f"Invalid YAML: {exc}") from exc
    SCRIPT_FILE.write_text(req.content, encoding="utf-8")
    await _broadcast_admin_event({"type": "script_reload", "message": "Script saved"})
    return {"status": "saved"}


class BotsUpdateRequest(BaseModel):
    max_bots: int


@app.post("/admin/bots")
async def admin_set_bots(req: BotsUpdateRequest):
    """Update MAX_BOTS in .env (takes effect on next worker restart)."""
    if not (1 <= req.max_bots <= 50):
        raise HTTPException(400, "max_bots must be 1-50")
    if ENV_FILE.exists():
        set_key(str(ENV_FILE), "MAX_BOTS", str(req.max_bots))
    return {"status": "updated", "max_bots": req.max_bots,
            "note": "Restart agent workers for change to take effect"}


class SettingsUpdateRequest(BaseModel):
    vad_threshold: Optional[float] = None
    confidence_threshold: Optional[float] = None
    min_endpointing_delay: Optional[float] = None


@app.post("/admin/settings")
async def admin_save_settings(req: SettingsUpdateRequest):
    """Save tuning parameters to .env."""
    if not ENV_FILE.exists():
        raise HTTPException(500, ".env not found")
    updated = {}
    if req.vad_threshold is not None:
        set_key(str(ENV_FILE), "VAD_THRESHOLD", str(req.vad_threshold))
        updated["VAD_THRESHOLD"] = req.vad_threshold
    if req.confidence_threshold is not None:
        set_key(str(ENV_FILE), "CONFIDENCE_THRESHOLD",
                str(req.confidence_threshold))
        updated["CONFIDENCE_THRESHOLD"] = req.confidence_threshold
    if req.min_endpointing_delay is not None:
        set_key(str(ENV_FILE), "MIN_ENDPOINTING_DELAY_S",
                str(req.min_endpointing_delay))
        updated["MIN_ENDPOINTING_DELAY_S"] = req.min_endpointing_delay
    return {
        "status": "saved",
        "updated": updated,
        "note": "Takes effect on next call (next AgentSession creation)",
    }


@app.get("/admin/logs")
async def admin_logs(lines: int = 100):
    """Return last N lines of the main log file."""
    log_file = LOGS_DIR / "voicebot.log"
    if not log_file.exists():
        return {"lines": []}
    all_lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"lines": all_lines[-lines:]}


# ---------------------------------------------------------------------------
# Admin WebSocket — live event stream
# ---------------------------------------------------------------------------

@app.websocket("/admin/ws")
async def admin_ws(ws: WebSocket):
    await ws.accept()
    _admin_clients.append(ws)
    try:
        # Send initial status
        status = await health()
        await ws.send_json({"type": "status", "data": status})
        # Keep alive — client receives real-time events via broadcast
        while True:
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        if ws in _admin_clients:
            _admin_clients.remove(ws)


# ---------------------------------------------------------------------------
# Mock Sarah WebSocket — replaces LiveKit room in MOCK_MODE
#
# The browser JS normally connects to a LiveKit WebRTC room.  In mock mode
# we point `url` at this server and `room` carries the name.  The JS in
# index.html detects `mock: true` and connects to /mock/ws?room=<name>
# instead of a LiveKit room.  We send JSON events that the browser renders
# as transcript bubbles and audio (Web Speech API TTS).
# ---------------------------------------------------------------------------

@app.websocket("/mock/ws")
async def mock_room_ws(ws: WebSocket, room: str = "mock"):
    """Simulate a LiveKit Sarah agent over plain WebSocket for Windows dev."""
    await ws.accept()
    _reply_index = 0

    async def _sarah(text: str) -> None:
        await ws.send_json({
            "type": "transcript",
            "speaker": "sarah",
            "text": text,
        })

    try:
        # Send opening line immediately
        await asyncio.sleep(0.5)
        await _sarah(_MOCK_OPENING)

        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=60)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "ping"})
                continue

            msg = {}
            try:
                msg = json.loads(raw)
            except Exception:
                pass

            if msg.get("type") == "user_speech":
                user_text = msg.get("text", "")
                if user_text:
                    await ws.send_json({
                        "type": "transcript",
                        "speaker": "user",
                        "text": user_text,
                    })
                await asyncio.sleep(1.2)  # simulate 1-2s thinking
                reply = _MOCK_REPLIES[_reply_index % len(_MOCK_REPLIES)]
                _reply_index += 1
                await _sarah(reply)

            elif msg.get("type") == "ping":
                await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass


async def _broadcast_admin_event(payload: dict) -> None:
    dead = []
    for ws in _admin_clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _admin_clients:
            _admin_clients.remove(ws)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _check_http(url: str, name: str) -> bool:
    try:
        timeout = aiohttp.ClientTimeout(total=3)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                return resp.status < 500
    except Exception:
        return False


def _summarise_call(call_id: str, events: list[dict]) -> dict:
    start_ts = events[0].get("ts") if events else None
    end_event = next((e for e in reversed(events) if e.get("event") == "call_ended"), None)
    turns = sum(1 for e in events if e.get("event") == "turn")
    stage = None
    for e in reversed(events):
        stage = e.get("stage") or e.get("final_stage")
        if stage:
            break
    return {
        "call_id": call_id,
        "started_at": start_ts,
        "duration_s": end_event.get("duration_s") if end_event else None,
        "turns": turns,
        "stage": stage,
        "outcome": end_event.get("reason") if end_event else "in_progress",
    }


# ---------------------------------------------------------------------------
# Entry point (for direct run, not needed when started via uvicorn)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")
