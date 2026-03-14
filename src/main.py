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
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

import audioop
import base64
import json
from math import gcd

import numpy as np
from scipy.signal import resample_poly

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
    
    # SIP configuration — warnings only; bot runs in Twilio-only mode without SIP
    if not sip_cfg.server or sip_cfg.server == "your_sip_server_here":
        logger.warning("SIP_SERVER not set — SIP/Asterisk integration disabled (Twilio-only mode)")
    if not sip_cfg.username or sip_cfg.username == "your_sip_username_here":
        logger.warning("SIP_USERNAME not set — SIP integration disabled")
    if not sip_cfg.password or sip_cfg.password == "your_sip_password_here":
        logger.warning("SIP_PASSWORD not set — SIP integration disabled (Twilio-only mode)")
    
    # VICIdial configuration (warnings only — not required for Twilio-only testing)
    if not vicidial_cfg.api_user or vicidial_cfg.api_user == "your_vicidial_user_here":
        logger.warning("VICIDIAL_API_USER not set — VICIdial call routing disabled")
    if not vicidial_cfg.api_pass or vicidial_cfg.api_pass == "your_vicidial_pass_here":
        logger.warning("VICIDIAL_API_PASS not set — VICIdial call routing disabled")

    # LLM/TTS/STT configuration (hard errors — bot cannot run without these)
    llm_cfg = get_llm_config()
    if not llm_cfg.vllm_api_url or llm_cfg.vllm_api_url == "http://127.0.0.1:9999":
        issues.append("VLLM_API_URL not set correctly (points to vLLM server)")
    
    tts_cfg = get_tts_config()
    if not tts_cfg.api_url:
        issues.append("TTS_API_URL is empty — must point to the Kokoro TTS server (e.g. http://127.0.0.1:8001)")
    elif tts_cfg.api_url in ("http://127.0.0.1:9998", "http://127.0.0.1:9000", "http://0.0.0.0:9000"):
        issues.append(
            f"TTS_API_URL={tts_cfg.api_url} is pointing at THE VOICEBOT ITSELF, not Kokoro. "
            "Set TTS_API_URL=http://127.0.0.1:8001 in .env"
        )
    
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
from src.tts.kokoro_handler import KokoroTTSHandler
from src.vicidial.agent_api import AgentAPI
from src.vicidial.sip_handler import SIPHandler

# ---------------------------------------------------------------------------
# Shared instances (populated during lifespan)
# ---------------------------------------------------------------------------
stt_handler: ParakeetSTTHandler | None = None
tts_handler: KokoroTTSHandler | None = None
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
        
        logger.info("Initializing TTS handler (Kokoro) …")
        tts_handler = KokoroTTSHandler(get_tts_config())
        
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

    Twilio sends μ-law 8 kHz audio; we resample to PCM 16 kHz for Parakeet STT.
    Kokoro TTS produces PCM 24 kHz; we resample to μ-law 8 kHz for Twilio.
    Both directions use scipy.signal.resample_poly for anti-aliased resampling.
    """
    await ws.accept()

    # Queues for inbound (Twilio→STT) and outbound (TTS→Twilio) audio
    inbound_q: asyncio.Queue = asyncio.Queue()
    outbound_q: asyncio.Queue = asyncio.Queue()

    stream_sid: str | None = None
    call_id: str | None = None
    tts_sr = get_tts_config().sample_rate   # 24 000 Hz

    # Pre-compute resample ratios for the outbound direction so we
    # don't call gcd() on every audio chunk.
    # Kokoro outputs 24 kHz → Twilio expects μ-law 8 kHz.
    # gcd(24000, 8000) = 8000  →  up=1, down=3.
    _out_g    = gcd(tts_sr, 8000)
    _out_up   = 8000 // _out_g    # = 1  (for 24 000 → 8 000)
    _out_down = tts_sr // _out_g  # = 3

    # 80 ms of μ-law silence at 8 kHz (0x7F = silence in μ-law).
    # Injected after each sentence to prevent polyphase FIR boundary
    # ringing from being audible at sentence transitions.
    _SILENCE_80MS = bytes([0x7F] * 640)

    # Per-stream sentence counter (for mark events).
    _sentence_seq: list[int] = [0]

    async def _audio_source():
        """Yield PCM 16 kHz frames for the STT handler."""
        while True:
            chunk = await inbound_q.get()
            if chunk is None:
                return
            yield chunk

    def _audio_sink(pcm_bytes: bytes) -> None:
        """Receive PCM-16 from TTS (at tts_sr), downsample to 8 kHz μ-law, queue.

        Each call corresponds to one sentence from the TTS engine.

        FIX B — explicit float32 clip: prevents wrap-around distortion if any
                 sample exceeds ±1.0 before int16 conversion.
        FIX C — 80 ms μ-law silence appended after every sentence chunk so
                 Twilio has a gap between sentences instead of directly
                 concatenating them; eliminates boundary pops from the polyphase
                 FIR filter settling transient.
        FIX E — Twilio "mark" event after each sentence so barge-in detection
                 and upstream logic know precisely when each sentence ends.
        """
        # FIX A: 24 kHz → 8 kHz at ratio 1:3 (verified correct for Kokoro 0.9.4)
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # FIX B: hard-clip float32 to [-1, 1] before int16 conversion.
        # resample_poly can introduce slight overshoot; clipping here prevents
        # the wrap-around aliasing that produces the alien-noise symptom.
        samples = np.clip(samples, -1.0, 1.0)

        samples_8k = resample_poly(samples, _out_up, _out_down)

        # Post-resample fade: smooth the polyphase FIR startup/settling
        # transient (≈60 taps / 8 kHz ≈ 7.5 ms) that causes boundary pops
        # when short sentence chunks are resampled independently.
        n = len(samples_8k)
        if n > 320:
            fi = min(80, n // 8)   # ≤10 ms fade-in  @ 8 kHz
            fo = min(80, n // 8)   # ≤10 ms fade-out @ 8 kHz
            samples_8k = samples_8k.copy()
            samples_8k[:fi]  *= np.linspace(0.0, 1.0, fi)
            samples_8k[-fo:] *= np.linspace(1.0, 0.0, fo)

        pcm_8k = (samples_8k * 32768.0).clip(-32768, 32767).astype(np.int16).tobytes()
        mulaw = audioop.lin2ulaw(pcm_8k, 2)

        # Enqueue audio, then FIX C silence, then FIX E mark.
        outbound_q.put_nowait(mulaw)
        outbound_q.put_nowait(_SILENCE_80MS)
        _sentence_seq[0] += 1
        outbound_q.put_nowait({"event": "mark", "seq": _sentence_seq[0]})

    def _filler_sink(mulaw_bytes: bytes) -> None:
        """Direct μ-law bypass for pre-rendered filler clips.

        Filler clips are stored as 8 kHz μ-law bytes.  They must NOT go
        through _audio_sink (which expects 24 kHz int16 PCM) — doing so
        treats every two μ-law bytes as one int16 sample, resamples at the
        wrong ratio and re-encodes, producing the 'alien/tic-tic-tic' noise.
        Putting bytes directly into outbound_q skips the resample/encode path.
        """
        outbound_q.put_nowait(mulaw_bytes)

    async def _outbound_sender():
        """Drain the outbound queue and forward audio + marks to Twilio.

        FIX D: every audio payload is base64-encoded inside the Twilio
               Media Streams JSON envelope — no raw bytes ever sent directly.
        FIX E: mark events are forwarded as Twilio mark messages.
        """
        while True:
            item = await outbound_q.get()
            if item is None:
                break
            if isinstance(item, dict):
                # FIX E — Twilio mark event.
                await ws.send_text(json.dumps({
                    "event": "mark",
                    "streamSid": stream_sid,
                    "mark": {"name": f"sentence_{item['seq']}"},
                }))
            else:
                # FIX D — audio payload correctly base64-encoded in envelope.
                payload = base64.b64encode(item).decode("ascii")
                await ws.send_text(json.dumps({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": payload},
                }))

    sender_task = asyncio.create_task(_outbound_sender())
    call_ended_event: asyncio.Event | None = None
    _chunk_count: list[int] = [0]   # diagnostic: count inbound media chunks

    try:
        while True:
            # Use a short timeout so we can detect when the call ends and
            # close the WebSocket, rather than waiting forever for Twilio.
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                # Periodically check if the call loop has finished.
                if call_ended_event and call_ended_event.is_set():
                    logger.info("Call loop ended — closing stream %s", stream_sid)
                    break
                continue

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
                        filler_sink=_filler_sink,
                    )
                    call_ended_event = call_manager.get_call_ended_event(call_id)

            elif event == "media":
                mulaw = base64.b64decode(msg["media"]["payload"])
                # μ-law decode → int16 PCM @ 8 kHz
                pcm_8k_bytes = audioop.ulaw2lin(mulaw, 2)
                # Upsample 8 → 16 kHz with anti-aliased polyphase FIR
                samples = (
                    np.frombuffer(pcm_8k_bytes, dtype=np.int16)
                    .astype(np.float32)
                    / 32768.0
                )
                samples_16k = resample_poly(samples, 2, 1)
                pcm_16k = (
                    (samples_16k * 32768.0)
                    .clip(-32768, 32767)
                    .astype(np.int16)
                    .tobytes()
                )
                inbound_q.put_nowait(pcm_16k)
                _chunk_count[0] += 1
                if _chunk_count[0] % 50 == 0:
                    logger.debug(
                        "[AUDIO] chunk #%d bytes=%d call=%s",
                        _chunk_count[0], len(mulaw), call_id or "none",
                    )
                # Also check call-ended here for fast response (fires every ~20 ms)
                if call_ended_event and call_ended_event.is_set():
                    logger.info("Call loop ended — closing stream %s", stream_sid)
                    break

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
    """Health check endpoint — returns JSON status of all services."""
    llm_ok = await llm_client.health_check() if llm_client else False
    tts_ok = await tts_handler.health_check() if tts_handler else False
    stt_ok = stt_handler is not None and stt_handler._is_initialized if stt_handler else False
    sip_ok = sip_handler and getattr(sip_handler, "_registered", False)
    active = call_manager.get_active_calls() if call_manager else []

    components = {
        "stt": "ok" if stt_ok else "degraded",
        "llm": "ok" if llm_ok else "degraded",
        "tts": "ok" if tts_ok else "degraded",
        "sip": "registered" if sip_ok else "disconnected",
    }
    all_ok = stt_ok and llm_ok and tts_ok
    return {
        "status": "healthy" if all_ok else "degraded",
        "components": components,
        "active_calls": len(active),
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


@app.get("/test-audio")
async def test_audio():
    """Generate a 3-second TTS sample and save as test_audio.wav at 8 kHz.

    Synthesises a known phrase through the full Kokoro → resample → PCM
    pipeline (without μ-law encoding) and writes a WAV file to the repo
    root for offline quality inspection.

    Returns:
        JSON with sample count, duration, sample rate, and filename.
    """
    import wave
    from math import gcd as _gcd

    if not tts_handler:
        raise HTTPException(503, "TTS handler not ready.")

    phrase = (
        "Hi, this is Sarah with American Beneficiary. "
        "Did I catch you at a good time?"
    )

    tts_sr = get_tts_config().sample_rate  # 24 000 Hz
    _g = _gcd(tts_sr, 8000)
    up_ratio   = 8000 // _g    # 1
    down_ratio = tts_sr // _g  # 3

    pcm_chunks: list[bytes] = []
    async for chunk in tts_handler.synthesize_stream(phrase):
        # Resample from TTS sample rate to 8 kHz (same path as _audio_sink)
        samples = np.frombuffer(chunk.audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        samples = np.clip(samples, -1.0, 1.0)
        samples_8k = resample_poly(samples, up_ratio, down_ratio)
        pcm_8k = (samples_8k * 32768.0).clip(-32768, 32767).astype(np.int16).tobytes()
        pcm_chunks.append(pcm_8k)

    if not pcm_chunks:
        raise HTTPException(500, "TTS returned no audio.")

    all_pcm = b"".join(pcm_chunks)
    total_samples = len(all_pcm) // 2  # int16 = 2 bytes/sample
    duration_s = total_samples / 8000.0

    wav_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_audio.wav")
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)     # 16-bit
        wf.setframerate(8000)  # 8 kHz
        wf.writeframes(all_pcm)

    return {
        "status": "ok",
        "samples": total_samples,
        "duration_s": round(duration_s, 3),
        "sample_rate": 8000,
        "file": "test_audio.wav",
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
