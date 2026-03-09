"""
Call manager — orchestrates the full lifecycle of a single voice call.

Wires together STT → LLM → TTS in a streaming pipeline, manages the
audio I/O loop, tracks timing metrics, and emits call-level events
(connected, silence, transfer, hangup).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Optional

from config.settings import AppConfig, get_config
from src.llm.mimo_vllm import MimoVLLMClient
from src.llm.rag_engine import RAGEngine
from src.orchestration.conversation_engine import (
    CallAction,
    ConversationEngine,
    ConversationState,
    TurnResult,
)
from src.stt.parakeet_handler import ParakeetSTTHandler
from src.tts.cosyvoice_handler import CosyVoiceTTSHandler

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Types & internal containers
# ------------------------------------------------------------------

class CallStatus(str, Enum):
    INITIALIZING = "initializing"
    RINGING = "ringing"
    CONNECTED = "connected"
    IN_PROGRESS = "in_progress"
    TRANSFERRING = "transferring"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CallMetrics:
    """Real-time metrics collected during a call."""

    call_id: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0
    total_turns: int = 0
    avg_turn_latency_ms: float = 0.0
    max_turn_latency_ms: float = 0.0
    stt_latency_ms: list[float] = field(default_factory=list)
    llm_latency_ms: list[float] = field(default_factory=list)
    tts_latency_ms: list[float] = field(default_factory=list)
    final_stage: str = ""
    outcome: str = ""

    @property
    def duration_s(self) -> float:
        if self.ended_at and self.started_at:
            return self.ended_at - self.started_at
        return 0.0


# Callback type for sending audio back to the SIP channel
AudioSinkCallback = Callable[[bytes], Any]


@dataclass
class _CallSession:
    call_id: str
    state: ConversationState
    audio_source: AsyncIterator[bytes]
    audio_sink: AudioSinkCallback
    on_transfer: Optional[Callable] = None
    task: Optional[asyncio.Task] = None
    status: CallStatus = CallStatus.INITIALIZING
    metrics: CallMetrics = field(default_factory=CallMetrics)

    def __post_init__(self) -> None:
        self.metrics.call_id = self.call_id


class CallManager:
    """Manages the full audio-in / audio-out loop for one call.

    Typical flow::

        manager = CallManager(stt, tts, llm_client, rag_engine)
        await manager.initialize()

        call_id = await manager.start_call(
            lead_data={"first_name": "John", "state": "TX"},
            audio_source=sip_audio_stream,
            audio_sink=sip_send_audio,
        )

        # ... the call runs asynchronously ...

        metrics = manager.get_metrics(call_id)
    """

    def __init__(
        self,
        stt: ParakeetSTTHandler,
        tts: CosyVoiceTTSHandler,
        llm_client: MimoVLLMClient,
        rag_engine: RAGEngine,
        config: Optional[AppConfig] = None,
    ) -> None:
        self._stt = stt
        self._tts = tts
        self._llm = llm_client
        self._rag = rag_engine
        self._config = config or get_config()
        self._engine = ConversationEngine(llm_client, rag_engine, self._config)
        self._active_calls: dict[str, _CallSession] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load models and scripts."""
        self._engine.load_script()
        self._rag.load()
        await self._stt.initialize()
        await self._tts.initialize()
        await self._llm.initialize()
        logger.info("CallManager initialised.")

    async def shutdown(self) -> None:
        """Graceful shutdown — hang up active calls then release resources."""
        for call_id in list(self._active_calls):
            await self.end_call(call_id, reason="shutdown")
        await self._stt.shutdown()
        await self._tts.shutdown()
        await self._llm.shutdown()
        logger.info("CallManager shut down.")

    # ------------------------------------------------------------------
    # Call lifecycle
    # ------------------------------------------------------------------

    async def start_call(
        self,
        lead_data: dict[str, str],
        audio_source: AsyncIterator[bytes],
        audio_sink: AudioSinkCallback,
        *,
        on_transfer: Optional[Callable[[str, ConversationState], Any]] = None,
    ) -> str:
        """Start processing a new call.

        Parameters
        ----------
        lead_data:
            Prospect information (first_name, state, phone, etc.).
        audio_source:
            Async iterator yielding raw PCM audio from the SIP channel.
        audio_sink:
            Callback that sends raw PCM audio back to the caller.
        on_transfer:
            Optional callback invoked when the bot wants to transfer.

        Returns
        -------
        str
            A unique call ID.
        """
        call_id = str(uuid.uuid4())
        state = self._engine.new_call(call_id, lead_data)

        session = _CallSession(
            call_id=call_id,
            state=state,
            audio_source=audio_source,
            audio_sink=audio_sink,
            on_transfer=on_transfer,
        )
        self._active_calls[call_id] = session

        # Launch the audio loop as a background task
        session.task = asyncio.create_task(self._run_call(session))
        logger.info("Call %s started for lead '%s'", call_id, state.lead_name)
        return call_id

    async def end_call(self, call_id: str, *, reason: str = "normal") -> Optional[CallMetrics]:
        """Terminate a call and return its metrics."""
        session = self._active_calls.pop(call_id, None)
        if session is None:
            return None

        session.status = CallStatus.COMPLETED
        session.metrics.ended_at = time.time()
        session.metrics.outcome = reason
        session.metrics.final_stage = session.state.current_stage

        if session.task and not session.task.done():
            session.task.cancel()
            try:
                await session.task
            except asyncio.CancelledError:
                pass

        logger.info(
            "Call %s ended – reason=%s  duration=%.1fs  turns=%d",
            call_id,
            reason,
            session.metrics.duration_s,
            session.metrics.total_turns,
        )
        return session.metrics

    def get_metrics(self, call_id: str) -> Optional[CallMetrics]:
        session = self._active_calls.get(call_id)
        return session.metrics if session else None

    def get_active_calls(self) -> list[str]:
        return list(self._active_calls.keys())

    # ------------------------------------------------------------------
    # Core audio loop
    # ------------------------------------------------------------------

    async def _run_call(self, session: _CallSession) -> None:
        """Main loop: stream audio → STT → LLM → TTS → audio out."""
        session.status = CallStatus.CONNECTED
        session.metrics.started_at = time.time()

        try:
            # 1. Send opening greeting
            opening = await self._engine.get_opening(session.state)
            await self._speak(session, opening.bot_text)

            # 2. Listen-respond loop
            silence_start: Optional[float] = None

            async for result in self._stt.transcribe_stream(session.audio_source):
                if not result.text.strip():
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > self._config.silence_timeout_s:
                        logger.info("Silence timeout on call %s", session.call_id)
                        await self._speak(session, "Are you still there?")
                        silence_start = None
                    continue

                silence_start = None
                session.metrics.stt_latency_ms.append(result.latency_ms)

                if not result.is_final:
                    continue

                # Process the turn
                turn = await self._engine.process_turn(session.state, result.text)
                session.metrics.total_turns += 1
                session.metrics.llm_latency_ms.append(turn.latency_ms)

                # Speak the response
                await self._speak(session, turn.bot_text)

                # Handle actions
                if turn.action == CallAction.TRANSFER_TO_CLOSER:
                    session.status = CallStatus.TRANSFERRING
                    if session.on_transfer:
                        if asyncio.iscoroutinefunction(session.on_transfer):
                            await session.on_transfer(session.call_id, session.state)
                        else:
                            session.on_transfer(session.call_id, session.state)
                    break

                if turn.action in (CallAction.END_CALL, CallAction.DNC_AND_END_CALL):
                    break

                # Guard max call duration
                elapsed = time.time() - session.metrics.started_at
                if elapsed > self._config.max_call_duration_s:
                    logger.warning("Max call duration reached on %s", session.call_id)
                    break

        except asyncio.CancelledError:
            logger.info("Call %s cancelled.", session.call_id)
        except Exception:
            logger.exception("Error in call loop for %s", session.call_id)
            session.status = CallStatus.FAILED
        finally:
            if session.call_id in self._active_calls:
                await self.end_call(session.call_id, reason=session.status.value)

    async def _speak(self, session: _CallSession, text: str) -> None:
        """Synthesise *text* and send audio to the SIP channel."""
        start = time.perf_counter()
        async for chunk in self._tts.synthesize_stream(text):
            session.audio_sink(chunk.audio_bytes)
        elapsed = (time.perf_counter() - start) * 1000
        session.metrics.tts_latency_ms.append(elapsed)
