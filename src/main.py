"""
Main entry point for the Final Expense Voice Bot.

Starts the FastAPI management server, initialises all subsystems (STT, LLM,
TTS, SIP, VICIdial), and begins listening for incoming calls.

Run with::

    python -m src.main            # development
    uvicorn src.main:app --host 0.0.0.0 --port 9000   # production
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

import audioop
import base64
import json

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Bootstrap logging before any component import
# ---------------------------------------------------------------------------
from config.settings import (
    AppConfig,
    get_config,
    get_llm_config,
    get_rag_config,
    get_sip_config,
    get_stt_config,
    get_tts_config,
    get_vicidial_config,
)

_cfg = get_config()
logging.basicConfig(
    level=getattr(logging, _cfg.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("voicebot")


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------

def _validate_required_config() -> None:
    """Validate that all required environment variables are set before startup."""
    cfg = get_config()
    sip_cfg = get_sip_config()
    vicidial_cfg = get_vicidial_config()
    
    issues = []
    
    # SIP configuration
    if not sip_cfg.server or sip_cfg.server == "your_sip_server_here":
        issues.append("SIP_SERVER not set (required for Twilio/Asterisk integration)")
    if not sip_cfg.username or sip_cfg.username == "your_sip_username_here":
        issues.append("SIP_USERNAME not set")
    if not sip_cfg.password or sip_cfg.password == "your_sip_password_here":
        issues.append("SIP_PASSWORD not set (required for SIP registration)")
    
    # VICIdial configuration (warnings only — not required for Twilio-only testing)
    if not vicidial_cfg.api_user or vicidial_cfg.api_user == "your_vicidial_user_here":
        logger.warning("VICIDIAL_API_USER not set — VICIdial call routing disabled")
    if not vicidial_cfg.api_pass or vicidial_cfg.api_pass == "your_vicidial_pass_here":
        logger.warning("VICIDIAL_API_PASS not set — VICIdial call routing disabled")

    # LLM/TTS/STT configuration
    llm_cfg = get_llm_config()
    if not llm_cfg.vllm_api_url or llm_cfg.vllm_api_url == "http://127.0.0.1:9999":
        issues.append("VLLM_API_URL not set correctly (points to vLLM server)")
    
    tts_cfg = get_tts_config()
    if not tts_cfg.api_url or tts_cfg.api_url == "http://127.0.0.1:9998":
        issues.append("TTS_API_URL not set correctly (points to CosyVoice server)")
    
    if issues:
        logger.error("=" * 60)
        logger.error("CONFIGURATION VALIDATION FAILED:")
        for issue in issues:
            logger.error("  ✗ %s", issue)
        logger.error("=" * 60)
        raise RuntimeError(
            f"Missing or invalid configuration ({len(issues)} issue(s)). "
            f"Check .env file. See .env.example for required fields."
        )
    
    logger.info("Configuration validation passed.")


# ---------------------------------------------------------------------------
# Component imports
# ---------------------------------------------------------------------------
from src.llm.llm_client import LLMClient
from src.llm.rag_engine import RAGEngine
from src.orchestration.call_manager import CallManager
from src.orchestration.transfer_handler import TransferHandler
from src.stt.parakeet_handler import ParakeetSTTHandler
from src.tts.cosyvoice_handler import CosyVoiceTTSHandler
from src.vicidial.agent_api import AgentAPI
from src.vicidial.sip_handler import SIPHandler

# ---------------------------------------------------------------------------
# Shared instances (populated during lifespan)
# ---------------------------------------------------------------------------
stt_handler: ParakeetSTTHandler | None = None
tts_handler: CosyVoiceTTSHandler | None = None
llm_client: LLMClient | None = None
rag_engine: RAGEngine | None = None
call_manager: CallManager | None = None
sip_handler: SIPHandler | None = None
agent_api: AgentAPI | None = None
transfer_handler: TransferHandler | None = None


# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle for all voice bot components."""
    global stt_handler, tts_handler, llm_client, rag_engine
    global call_manager, sip_handler, agent_api, transfer_handler

    logger.info("=" * 60)
    logger.info("  Final Expense Voice Bot – Starting")
    logger.info("=" * 60)

    try:
        # Validate configuration before initializing anything
        _validate_required_config()

        # -- Initialise components --
        logger.info("Initializing STT handler (Parakeet TDT) …")
        stt_handler = ParakeetSTTHandler(get_stt_config())
        
        logger.info("Initializing TTS handler (CosyVoice 2) …")
        tts_handler = CosyVoiceTTSHandler(get_tts_config())
        
        logger.info("Initializing LLM client (vLLM OpenAI API) …")
        llm_client = LLMClient(get_llm_config())
        
        logger.info("Initializing RAG engine …")
        rag_engine = RAGEngine(get_rag_config())

        logger.info("Initializing VICIdial agent API …")
        agent_api = AgentAPI(get_vicidial_config())
        
        logger.info("Initializing transfer handler …")
        transfer_handler = TransferHandler(get_vicidial_config())
        transfer_handler.set_agent_api(agent_api)

        logger.info("Initializing SIP handler …")
        sip_handler = SIPHandler(get_sip_config())

        logger.info("Initializing call manager …")
        call_manager = CallManager(
            stt=stt_handler,
            tts=tts_handler,
            llm_client=llm_client,
            rag_engine=rag_engine,
        )

        # Load knowledge base and sales script
        logger.info("Loading knowledge base and sales script …")
        rag_engine.load()
        
        logger.info("Initializing subsystems …")
        await call_manager.initialize()
        await agent_api.initialize()
        
        # Health checks before accepting calls
        logger.info("Running health checks …")
        
        llm_ok = await llm_client.health_check()
        if not llm_ok:
            raise RuntimeError(
                f"vLLM server not reachable at {get_llm_config().vllm_api_url}. "
                "Check that voicebot-vllm.service is running."
            )
        logger.info("  ✓ LLM server responds")
        
        tts_ok = await tts_handler.health_check()
        if not tts_ok:
            raise RuntimeError(
                f"TTS server not reachable at {get_tts_config().api_url}. "
                "Check that voicebot-tts.service is running."
            )
        logger.info("  ✓ TTS server responds")

        # Wire incoming-call handler BEFORE start() to avoid race condition
        sip_handler.on_incoming_call = _handle_incoming_sip_call

        # Start SIP: sets event loop, registers, starts listener + monitor tasks
        logger.info("Registering with SIP server …")
        sip_ok = await sip_handler.start()
        if not sip_ok:
            logger.warning(
                "SIP registration failed (SIP_SERVER=%s, SIP_USERNAME=%s). "
                "Continuing without SIP — Twilio webhook flow will still work.",
                get_sip_config().server, get_sip_config().username
            )
        else:
            logger.info("  ✓ SIP registered. Listening for calls.")

        logger.info("=" * 60)
        logger.info("Voice bot ready. Management API at http://%s:%d", _cfg.api_host, _cfg.api_port)
        logger.info("=" * 60)

        yield  # ------------- application runs here --------

        # -- Shutdown --
        logger.info("Shutting down …")
        await call_manager.shutdown()
        await agent_api.shutdown()
        await sip_handler.stop()
        logger.info("Shutdown complete.")
    
    except Exception as exc:
        logger.error("Fatal error during startup: %s", exc, exc_info=True)
        raise


app = FastAPI(
    title="Final Expense Voice Bot",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Incoming-call callback (wired to SIP handler)
# ---------------------------------------------------------------------------

async def _handle_incoming_sip_call(sip_call, audio_stream, send_audio) -> None:
    """Called by the SIP handler whenever a new call arrives."""
    logger.info("Incoming call: %s from %s", sip_call.call_id, sip_call.remote_uri)

    # Fetch lead info if available
    lead_data: dict[str, str] = {}
    if agent_api:
        info = await agent_api.get_lead_info(sip_call.call_id)
        if info:
            lead_data = {
                "first_name": info.get("first_name", "there"),
                "state": info.get("state", ""),
            }

    async def on_transfer(call_id: str, state) -> None:
        """Callback when the bot decides to transfer."""
        if transfer_handler:
            result = await transfer_handler.warm_transfer(call_id, state)
            if result.success:
                await sip_handler.transfer(call_id, get_vicidial_config().transfer_extension)

    await call_manager.start_call(
        lead_data=lead_data,
        audio_source=audio_stream,
        audio_sink=send_audio,
        on_transfer=on_transfer,
    )


# ---------------------------------------------------------------------------
# Management API endpoints
# ---------------------------------------------------------------------------

@app.post("/twilio/voice")
async def twilio_voice_webhook(request: Request):
    """TwiML webhook — Twilio calls this when (318) 610-9787 receives a call.
    Returns TwiML that opens a Media Stream WebSocket so the bot handles audio
    directly (STT → LLM → TTS) without needing a SIP registration."""
    host = request.headers.get("host", "")
    stream_url = f"wss://{host}/twilio/stream"
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<Stream url="{stream_url}"/>'
        "</Connect>"
        "</Response>"
    )
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/twilio/stream")
async def twilio_media_stream(ws: WebSocket):
    """Twilio Media Streams WebSocket — real-time bidirectional audio bridge.

    Twilio sends μ-law 8 kHz audio; we need PCM 16 kHz for Parakeet STT.
    CosyVoice TTS produces PCM 22050 Hz; we encode to μ-law 8 kHz for Twilio.
    """
    await ws.accept()

    # Queues for inbound (Twilio→STT) and outbound (TTS→Twilio) audio
    inbound_q: asyncio.Queue = asyncio.Queue()
    outbound_q: asyncio.Queue = asyncio.Queue()

    stream_sid: str | None = None
    call_id: str | None = None
    tts_sr = get_tts_config().sample_rate  # default 22050

    # audioop.ratecv state handles for sample-accurate incremental resampling
    _in_state = None   # 8 kHz → 16 kHz
    _out_state = None  # tts_sr → 8 kHz

    async def _audio_source():
        """Yield PCM 16 kHz frames for the STT handler."""
        while True:
            chunk = await inbound_q.get()
            if chunk is None:
                return
            yield chunk

    def _audio_sink(pcm_bytes: bytes) -> None:
        """Receive PCM from TTS, convert and queue for sending to Twilio."""
        nonlocal _out_state
        resampled, _out_state = audioop.ratecv(
            pcm_bytes, 2, 1, tts_sr, 8000, _out_state
        )
        mulaw = audioop.lin2ulaw(resampled, 2)
        outbound_q.put_nowait(mulaw)

    async def _outbound_sender():
        """Drain the outbound queue and forward encoded audio to Twilio."""
        while True:
            mulaw = await outbound_q.get()
            if mulaw is None:
                break
            payload = base64.b64encode(mulaw).decode("ascii")
            await ws.send_text(
                json.dumps(
                    {"event": "media", "streamSid": stream_sid,
                     "media": {"payload": payload}}
                )
            )

    sender_task = asyncio.create_task(_outbound_sender())

    try:
        async for raw in ws.iter_text():
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "start":
                stream_sid = msg["start"]["streamSid"]
                call_sid = msg["start"].get("callSid", "unknown")
                logger.info("Twilio stream started — callSid=%s streamSid=%s", call_sid, stream_sid)
                lead_data = {"first_name": "there", "phone": call_sid}
                if call_manager:
                    call_id = await call_manager.start_call(
                        lead_data=lead_data,
                        audio_source=_audio_source(),
                        audio_sink=_audio_sink,
                    )

            elif event == "media":
                mulaw = base64.b64decode(msg["media"]["payload"])
                pcm_8k = audioop.ulaw2lin(mulaw, 2)
                pcm_16k, new_state = audioop.ratecv(
                    pcm_8k, 2, 1, 8000, 16000, _in_state
                )
                _in_state = new_state
                inbound_q.put_nowait(pcm_16k)

            elif event == "stop":
                logger.info("Twilio stream stopped — streamSid=%s", stream_sid)
                break

    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected — streamSid=%s", stream_sid)
    except Exception:
        logger.exception("Error in Twilio media stream — streamSid=%s", stream_sid)
    finally:
        inbound_q.put_nowait(None)
        outbound_q.put_nowait(None)
        try:
            await asyncio.wait_for(sender_task, timeout=2.0)
        except (asyncio.TimeoutError, Exception):
            sender_task.cancel()
        if call_id and call_manager:
            await call_manager.end_call(call_id, reason="completed")


class StartCallRequest(BaseModel):
    lead_name: str = "there"
    state: str = ""
    phone: str = ""


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    llm_ok = await llm_client.health_check() if llm_client else False
    tts_ok = await tts_handler.health_check() if tts_handler else False
    return {
        "status": "healthy",
        "components": {
            "llm": "ok" if llm_ok else "degraded",
            "tts": "ok" if tts_ok else "degraded",
            "sip": "registered" if (sip_handler and sip_handler._registered) else "disconnected",
        },
        "active_calls": len(call_manager.get_active_calls()) if call_manager else 0,
    }


@app.get("/calls")
async def list_calls():
    """List active call IDs."""
    return {"calls": call_manager.get_active_calls() if call_manager else []}


@app.get("/calls/{call_id}/metrics")
async def call_metrics(call_id: str):
    """Get real-time metrics for a specific call."""
    if not call_manager:
        raise HTTPException(503, "Call manager not ready.")
    metrics = call_manager.get_metrics(call_id)
    if metrics is None:
        raise HTTPException(404, "Call not found.")
    return {
        "call_id": metrics.call_id,
        "duration_s": metrics.duration_s,
        "total_turns": metrics.total_turns,
        "final_stage": metrics.final_stage,
    }


@app.post("/calls/{call_id}/end")
async def end_call(call_id: str):
    """Force-end a call."""
    if not call_manager:
        raise HTTPException(503, "Call manager not ready.")
    metrics = await call_manager.end_call(call_id, reason="api_request")
    if metrics is None:
        raise HTTPException(404, "Call not found.")
    return {"status": "ended", "call_id": call_id}


@app.post("/reload/script")
async def reload_script():
    """Hot-reload the sales script YAML."""
    if call_manager:
        call_manager._engine.reload_script()
    return {"status": "reloaded"}


@app.post("/reload/knowledge")
async def reload_knowledge():
    """Hot-reload the knowledge base JSON."""
    if rag_engine:
        rag_engine.reload()
    return {"status": "reloaded"}


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the voice bot with uvicorn."""
    uvicorn.run(
        "src.main:app",
        host=_cfg.api_host,
        port=_cfg.api_port,
        log_level=_cfg.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
