"""src/agent.py — LiveKit Agent entrypoint for the Final Expense Voice Bot.

REPLACES src/main.py (Twilio+FastAPI pipeline) entirely.

Architecture:
  LiveKit Server (self-hosted ws://localhost:7880)
      ↕  WebRTC
  LiveKit AgentSession  ← manages speaking/listening state machine
      ↕  Plugin interfaces
  ParakeetSTT  |  VllmLLM  |  QwenTTS
      ↕
  ConversationEngine  ← sales logic, stage management (UNCHANGED)
      ↕
  Browser UI (this file) / Telnyx SIP (added later via main_twilio.py.bak)

Why this eliminates every bug from the Twilio pipeline:
  - turns=0 / stuck-SPEAKING: LiveKit AgentSession manages the state machine
  - echo-gate mess: WebRTC echo cancellation runs in the browser; no μ-law path
  - barge-in: LiveKit handles interruptions natively (allow_interruptions=True)
  - scaling: each worker serves one call; run N workers for N concurrent calls

Latency target: 1-2 s on happy path (acceptable — sounds human).
  min_endpointing_delay=1.0 s → Sarah waits 1 s of silence before responding.
  This is intentional: real salespeople pause before answering.

Run:
  python src/agent.py dev     ← development mode (connects to local LiveKit)
  python src/agent.py start   ← production mode
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from livekit import agents, rtc  # noqa: E402
from livekit.agents import AgentSession, Agent, JobContext  # noqa: E402
from livekit.plugins import silero  # noqa: E402

from src.plugins.parakeet_stt import ParakeetSTT  # noqa: E402
from src.plugins.vllm_llm import VllmLLM  # noqa: E402
from src.plugins.qwen_tts import QwenTTS  # noqa: E402
from src.orchestration.conversation_engine import (  # noqa: E402
    ConversationEngine,
    ConversationState,
    CallerIntent,
)
from src.llm.rag_engine import RAGEngine  # noqa: E402
from src.utils.call_logger import CallLogger  # noqa: E402

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("voicebot.agent")

# ---------------------------------------------------------------------------
# Hardcoded opening line — spoken verbatim by Sarah, LLM never touches it.
# This ensures the call always begins consistently regardless of model state.
# ---------------------------------------------------------------------------
OPENING_LINE = (
    "Hi, this is Sarah with American Beneficiary — "
    "I'm calling about the final expense coverage inquiry "
    "we received. Did I catch you at an okay time?"
)


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class SarahAgent(Agent):
    """Sarah — a warm, professional final-expense insurance voice agent.

    Wraps ConversationEngine for stage/intent management while delegating
    the actual turn-taking state machine to LiveKit AgentSession.
    """

    def __init__(
        self,
        engine: ConversationEngine,
        state: ConversationState,
        call_logger: CallLogger,
        rag: RAGEngine,
    ) -> None:
        self._engine = engine
        self._state = state
        self._call_logger = call_logger
        self._rag = rag
        system_prompt = engine._build_system_prompt(state, "")
        super().__init__(
            instructions=system_prompt,
        )

    async def on_enter(self) -> None:
        """Deliver hardcoded opening line when Sarah joins the room.

        Brief 0.5 s pause first — simulates the phone ring-to-answer
        transition and feels natural to the caller.
        """
        await asyncio.sleep(0.5)
        logger.info("[OPENING] Delivering opening line")
        try:
            await self.session.say(OPENING_LINE, allow_interruptions=True)
        except AttributeError:
            # Newer API: session.generate_reply with locked instructions
            await self.session.generate_reply(
                instructions=(
                    f"Say EXACTLY the following verbatim, do not change a word: "
                    f'"{OPENING_LINE}"'
                )
            )


# ---------------------------------------------------------------------------
# Job entrypoint — called per room (per call)
# ---------------------------------------------------------------------------

async def entrypoint(ctx: JobContext) -> None:
    """Entry point for each LiveKit job (one call = one job)."""
    call_id = ctx.room.name
    logger.info("[CALL] Started: %s", call_id)

    # Connect to the LiveKit room
    await ctx.connect()

    # Initialise the sales brain for this call
    engine = ConversationEngine()
    state = engine.create_state(call_id, {})
    call_logger = CallLogger(call_id)

    # RAG engine for domain knowledge retrieval
    try:
        from config.settings import get_rag_config
        rag = RAGEngine(get_rag_config())
    except Exception:
        rag = None

    # Build the LLM plugin — system_prompt is refreshed per-turn
    system_prompt = engine._build_system_prompt(state, "")
    llm_plugin = VllmLLM(system_prompt=system_prompt)

    call_logger.event("call_started", call_id=call_id)

    # Build the AgentSession — LiveKit manages the entire speaking/listening
    # state machine, interruptions, VAD, and TTS pacing.
    session = AgentSession(
        stt=ParakeetSTT(),
        llm=llm_plugin,
        tts=QwenTTS(),
        vad=silero.VAD.load(),
        # 1.0 s silence before Sarah responds — INTENTIONAL.
        # Sounds human. Real salespeople pause before answering objections.
        min_endpointing_delay=1.0,
        max_endpointing_delay=2.0,
    )

    # ------------------------------------------------------------------
    # Per-turn hook: update ConversationEngine state when user speaks
    # ------------------------------------------------------------------

    @session.on("user_speech_committed")
    async def on_user_speech(event) -> None:
        """Advance the conversation stage and refresh the LLM system prompt."""
        # Handle both older (msg.content) and newer (event object) API shapes
        text = (
            getattr(event, "content", None)
            or getattr(event, "transcript", None)
            or getattr(getattr(event, "userMessage", None), "content", None)
            or ""
        )
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if not text:
            return

        # Advance stage via ConversationEngine
        intent: CallerIntent = engine._classify_intent(text)
        engine._advance_stage(state, text, "")
        engine._update_collected_info(state, text)

        # Retrieve RAG context for the new stage
        rag_context = ""
        if rag is not None:
            try:
                chunks = rag.retrieve(text)
                rag_context = RAGEngine.format_context(chunks)
            except Exception as rag_exc:
                logger.debug("RAG retrieve error: %s", rag_exc)

        # Refresh LLM system prompt with updated stage and RAG context
        new_prompt = engine._build_system_prompt(state, rag_context)
        llm_plugin._system_prompt = new_prompt

        # Log the turn
        call_logger.event(
            "turn",
            text=text[:200],
            intent=intent.value if intent else None,
            stage=state.current_stage,
        )
        logger.info(
            "[TURN] call=%s stage=%s intent=%s text=%r",
            call_id,
            state.current_stage,
            intent.value if intent else "?",
            text[:60],
        )

    # ------------------------------------------------------------------
    # Start the session — this is where LiveKit takes over the state machine
    # ------------------------------------------------------------------
    agent = SarahAgent(engine, state, call_logger, rag)
    await session.start(ctx.room, agent=agent)

    logger.info("[CALL] Session running: %s", call_id)


# ---------------------------------------------------------------------------
# Worker startup — try AgentServer (v1.x), fall back to WorkerOptions (v0.x)
# ---------------------------------------------------------------------------

def _build_server():
    """Return an AgentServer (new API) or WorkerOptions (old API) wrapper."""
    try:
        from livekit.agents import AgentServer  # type: ignore
        server = AgentServer()

        @server.rtc_session(agent_name="sarah-voice-bot")
        async def _session(ctx: JobContext):
            await entrypoint(ctx)

        return server
    except (ImportError, AttributeError):
        # Fall back to WorkerOptions for livekit-agents < 1.0
        from livekit.agents import WorkerOptions  # type: ignore
        return WorkerOptions(entrypoint_fnc=entrypoint)


if __name__ == "__main__":
    server_or_options = _build_server()
    agents.cli.run_app(server_or_options)
