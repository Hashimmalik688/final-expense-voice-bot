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

import uvicorn
from fastapi import FastAPI, HTTPException
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
# Component imports
# ---------------------------------------------------------------------------
from src.llm.mimo_vllm import MimoVLLMClient
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
llm_client: MimoVLLMClient | None = None
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

    # -- Initialise components --
    stt_handler = ParakeetSTTHandler(get_stt_config())
    tts_handler = CosyVoiceTTSHandler(get_tts_config())
    llm_client = MimoVLLMClient(get_llm_config())
    rag_engine = RAGEngine(get_rag_config())

    agent_api = AgentAPI(get_vicidial_config())
    transfer_handler = TransferHandler(get_vicidial_config())
    transfer_handler.set_agent_api(agent_api)

    sip_handler = SIPHandler(get_sip_config())

    call_manager = CallManager(
        stt=stt_handler,
        tts=tts_handler,
        llm_client=llm_client,
        rag_engine=rag_engine,
    )

    # Load knowledge base and sales script
    rag_engine.load()
    await call_manager.initialize()
    await agent_api.initialize()

    # Register SIP
    sip_ok = await sip_handler.register()
    if sip_ok:
        logger.info("SIP registered. Listening for calls.")
    else:
        logger.warning("SIP registration failed – running in API-only mode.")

    # Wire incoming-call handler
    sip_handler.on_incoming_call = _handle_incoming_sip_call

    logger.info("Voice bot ready. Management API at http://%s:%d", _cfg.api_host, _cfg.api_port)

    yield  # ------------- application runs here -------------

    # -- Shutdown --
    logger.info("Shutting down …")
    await call_manager.shutdown()
    await agent_api.shutdown()
    await sip_handler.unregister()
    logger.info("Shutdown complete.")


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
