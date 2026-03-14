"""
Call manager — orchestrates the full lifecycle of a single voice call.

Wires together STT → LLM → TTS in a streaming pipeline, manages the
audio I/O loop, tracks timing metrics, and emits call-level events
(connected, silence, transfer, hangup).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Optional

from config.settings import AppConfig, get_config
from src.llm.llm_client import LLMClient
from src.llm.rag_engine import RAGEngine
from src.orchestration.conversation_engine import (
    CallAction,
    ConversationEngine,
    ConversationState,
    TurnResult,
)
from src.stt.parakeet_handler import ParakeetSTTHandler
from src.tts.kokoro_handler import KokoroTTSHandler
from src.tts.filler_player import FillerPlayer
from src.utils.call_logger import CallLogger

# Transcript injected when STT confidence falls below threshold
_CLARIFY_TEXT = "[UNCLEAR]"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Env-configurable audio-pipeline constants
# ---------------------------------------------------------------------------
# Duration (s) to discard STT results after bot finishes speaking.
# Echo from the PSTN arrives 50–150 ms after bot stops; 300 ms is enough
# to reject it while NOT blocking a real caller who starts speaking ~350ms later.
_ECHO_GATE_S: float = float(os.environ.get("ECHO_GATE_MS", "2000")) / 1000.0
# Between-barge-in refractory period (s) — prevents rapid re-triggers from echo.
_BARGE_IN_REFRACTORY_S: float = float(os.environ.get("BARGE_IN_COOLDOWN_S", "0.8"))
# Minimum RMS energy (0–1) on an inbound chunk to count as a direct barge-in
# while the bot is speaking (energy-based path, bypasses Parakeet STT).
_BARGE_IN_ENERGY_THRESHOLD: float = float(os.environ.get("BARGE_IN_ENERGY", "0.015"))
# Pause (s) between receiving an STT result and starting the LLM response.
# Gives the caller a moment to finish their thought before the bot replies.
_EOT_PAUSE_S: float = float(os.environ.get("EOT_PAUSE_MS", "1000")) / 1000.0


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


class BotCallState(str, Enum):
    """Fine-grained state of the bot within a single turn cycle.

    Transitions::

        LISTENING → RESPONDING → SPEAKING → RESPONDING → … → LISTENING
                        ↑                      ↓
                        └── INTERRUPTED ←──────┘  (barge-in)
    """
    LISTENING   = "listening"    # waiting for / processing user utterance
    RESPONDING  = "responding"   # LLM generating tokens
    SPEAKING    = "speaking"     # TTS audio is being sent to caller
    INTERRUPTED = "interrupted"  # caller spoke while bot was playing audio


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
    # Barge-in support: set by the parallel STT task when the user speaks
    # while the bot is playing audio.  _speak() checks this event each chunk.
    barge_in_event: asyncio.Event = field(default_factory=asyncio.Event)
    bot_speaking: bool = False   # True only while audio is being sent to sink
    bot_call_state: BotCallState = field(default_factory=lambda: BotCallState.LISTENING)
    # Fired when _run_call() finishes — tells the WebSocket handler to close.
    call_ended_event: asyncio.Event = field(default_factory=asyncio.Event)
    # Filler audio task — plays a short clip while the LLM generates; cancelled
    # the moment the first real TTS chunk is sent.
    filler_task: Optional[asyncio.Task] = None
    # Structured per-call JSONL logger
    call_logger: Optional["CallLogger"] = None
    # Prevents "sorry I didn't catch that" retry loop: True after first
    # UNCLEAR in a row so the next consecutive UNCLEAR just listens again.
    unclear_retry_sent: bool = False
    # Separate sink for pre-rendered filler clips (μ-law 8 kHz bytes).
    # When set, the filler player sends to this sink; when None, fillers are
    # disabled.  Required because the primary audio_sink expects 24 kHz PCM
    # (Kokoro output format) and silently corrupts μ-law bytes.
    filler_sink: Optional[Callable[[bytes], None]] = None
    # Epoch timestamp until which STT results are discarded after bot speech.
    # Phone-network echo of the bot's own voice arrives in the inbound stream
    # for ~300-600 ms after the bot stops speaking; without this gate the STT
    # picks up "Hi, this is Sarah …" and the bot responds to itself.
    post_speak_gate_until: float = 0.0

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

    # Minimum pause (seconds) after a barge-in before re-entering the
    # LLM→TTS pipeline.  Gives the caller a natural gap after finishing.
    _BARGE_IN_COOLDOWN_S: float = 0.30

    def __init__(
        self,
        stt: ParakeetSTTHandler,
        tts: KokoroTTSHandler,
        llm_client: LLMClient,
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
        self._filler_player = FillerPlayer()

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
        self._filler_player.load()
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
        filler_sink: Optional[Callable[[bytes], None]] = None,
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
            filler_sink=filler_sink,
            on_transfer=on_transfer,
            call_logger=CallLogger(call_id),
        )
        self._active_calls[call_id] = session
        session.call_logger.event("call_started", lead_name=state.lead_name)

        # Launch the audio loop as a background task
        session.task = asyncio.create_task(self._run_call(session))
        logger.info("Call %s started for lead '%s'", call_id, state.lead_name)
        return call_id

    def get_call_ended_event(self, call_id: str) -> Optional[asyncio.Event]:
        """Return the event that fires when the call loop finishes."""
        session = self._active_calls.get(call_id)
        return session.call_ended_event if session else None

    async def end_call(self, call_id: str, *, reason: str = "normal") -> Optional[CallMetrics]:
        """Terminate a call and return its metrics."""
        session = self._active_calls.pop(call_id, None)
        if session is None:
            return None

        session.status = CallStatus.COMPLETED
        session.metrics.ended_at = time.time()
        session.metrics.outcome = reason
        session.metrics.final_stage = session.state.current_stage

        # Only cancel the task when called from outside the task itself.
        # Calling end_call() from within _run_call()'s finally block would
        # otherwise try to await the currently-running task → deadlock.
        current = asyncio.current_task()
        if session.task and not session.task.done() and session.task is not current:
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
        if session.call_logger:
            session.call_logger.event(
                "call_ended",
                reason=reason,
                duration_s=round(session.metrics.duration_s, 1),
                turns=session.metrics.total_turns,
                stage=session.metrics.final_stage,
            )
            session.call_logger.close()
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
        """Main loop: STT runs in parallel so barge-in interrupts bot speech."""
        session.status = CallStatus.CONNECTED
        session.metrics.started_at = time.time()

        # Queue that receives finalised transcriptions from the background STT
        # task.  The main loop pulls from here without blocking the STT path.
        transcript_queue: asyncio.Queue = asyncio.Queue()

        # Single-word filler sounds that should never trigger barge-in on their
        # own.  The caller saying "Mm." while the bot is talking is a social
        # acknowledgment, not an answer — barge-in on it causes rapid TTS
        # start/stop (audible as clicking/alien-voice artefacts).
        _BARGE_IN_NOISE = {"mm", "mmm", "mhm", "hmm", "uh", "um", "ah", "oh",
                           "yeah", "yep", "yup", "ok", "okay", "right", "sure"}
        _last_barge_in_t: list[float] = [0.0]   # mutable for closure
        _gated_chunk_n: list[int] = [0]          # diagnostic chunk counter

        async def _gated_audio_source() -> AsyncIterator[bytes]:
            """Feed Parakeet only when the bot is NOT speaking.

            While the bot sends audio to Twilio the PSTN echoes it back through
            the inbound stream.  Feeding that echo to Parakeet causes it to
            transcribe the bot's own voice as caller speech.  This generator:

              • Discards inbound audio while ``session.bot_speaking`` is True
                so Parakeet only ever sees real caller audio.
              • Runs a raw energy check on each discarded chunk for immediate
                barge-in detection (no round-trip through Parakeet needed).
            """
            async for chunk in session.audio_source:
                _gated_chunk_n[0] += 1
                echo_gated = time.time() < session.post_speak_gate_until
                if session.bot_speaking or echo_gated:
                    samples = (
                        np.frombuffer(chunk, dtype=np.int16)
                        .astype(np.float32) / 32768.0
                    )
                    rms = float(np.sqrt(np.mean(samples ** 2)))
                    if _gated_chunk_n[0] % 100 == 0:
                        gate_reason = "speaking" if session.bot_speaking else "echo-gate"
                        logger.debug(
                            "[GATE] blocked=%s state=%s rms=%.4f passing=SILENCE chunk=#%d call=%s",
                            gate_reason, session.bot_call_state.name, rms, _gated_chunk_n[0], session.call_id,
                        )
                    if session.bot_speaking and rms > _BARGE_IN_ENERGY_THRESHOLD:
                        now_e = time.time()
                        if (now_e - _last_barge_in_t[0]) > _BARGE_IN_REFRACTORY_S:
                            logger.info(
                                "Energy barge-in on call %s (rms=%.4f)",
                                session.call_id, rms,
                            )
                            session.barge_in_event.set()
                            self._set_state(session, BotCallState.INTERRUPTED, reason="energy-barge-in")
                            _last_barge_in_t[0] = now_e
                    continue  # never feed echo to Parakeet
                if _gated_chunk_n[0] % 100 == 0:
                    logger.debug(
                        "[GATE] blocked=no state=%s passing=AUDIO chunk=#%d call=%s",
                        session.bot_call_state.name, _gated_chunk_n[0], session.call_id,
                    )
                yield chunk

        async def _stt_background() -> None:
            """Run STT continuously.  Signal barge-in when user speaks mid-bot."""
            async for result in self._stt.transcribe_stream(_gated_audio_source()):
                if not result.is_final or not result.text.strip():
                    continue

                txt = result.text.strip()
                logger.debug(
                    "[STT] text='%s' conf=%.2f latency_ms=%.0f "
                    "speaking=%s gate_ms=%.0f",
                    txt[:80], result.confidence, result.latency_ms,
                    session.bot_speaking,
                    max(0.0, session.post_speak_gate_until - time.time()) * 1000,
                )
                words = [w.strip(".,!?").lower() for w in txt.split()]
                is_filler_only = (
                    len(words) == 1
                    and words[0] in _BARGE_IN_NOISE
                )

                if session.bot_speaking:
                    now = time.time()
                    barge_in_cooldown_ok = (now - _last_barge_in_t[0]) > _BARGE_IN_REFRACTORY_S
                    # Suppress barge-in for single-word filler sounds and for
                    # repeated barge-ins within 1.5 s (prevents rapid TTS
                    # start/stop clicking when room noise is transcribed as
                    # acknowledgment phrases).
                    if is_filler_only or not barge_in_cooldown_ok:
                        logger.debug(
                            "Suppressing barge-in noise '%s' on call %s",
                            txt, session.call_id,
                        )
                        continue   # don't enqueue, don't interrupt
                    logger.info(
                        "Barge-in detected on call %s: '%s'", session.call_id, txt[:40]
                    )
                    session.barge_in_event.set()
                    self._set_state(session, BotCallState.INTERRUPTED, reason="stt-barge-in")
                    _last_barge_in_t[0] = now
                else:
                    # Post-speak echo gate: discard STT results that land inside
                    # the window immediately after the bot stops talking.
                    # The phone network echoes the bot's audio back through the
                    # inbound Twilio stream; without this gate, STT transcribes
                    # the echo ("Hi, this is Sarah …") as a caller utterance and
                    # the bot responds to itself in a tight loop.
                    now = time.time()
                    if now < session.post_speak_gate_until:
                        logger.debug(
                            "Echo gate active (%.0f ms remain) — discarding '%s' on call %s",
                            (session.post_speak_gate_until - now) * 1000,
                            txt[:40], session.call_id,
                        )
                        continue
                    # Extended echo suppression: single-word filler sounds
                    # arriving within 3s of bot speech are almost always echo
                    # remnants (Parakeet buffers audio and fires late).
                    if len(words) <= 3 and (now - session.post_speak_gate_until) < 4.0:
                        logger.debug(
                            "Post-echo suppressed (short phrase): '%s' (%.1fs after gate) call=%s",
                            txt, now - session.post_speak_gate_until, session.call_id,
                        )
                        continue

                # --- Safety pre-checks before confidence scoring ---
                # Empty / None / only punctuation / under 2 chars → UNCLEAR
                if not txt or len(txt) < 2 or not re.search(r'[a-zA-Z0-9]', txt):
                    from dataclasses import replace as _dc_replace
                    result = _dc_replace(result, text=_CLARIFY_TEXT)
                    await transcript_queue.put(result)
                    continue

                # Gate on confidence: very low scores mean noise or unintelligible
                # speech — replace with a clarification marker so the engine
                # responds with "Sorry, could you say that again?" rather than
                # hallucinating an answer to garbled input.
                if result.confidence < self._stt.CONFIDENCE_THRESHOLD:
                    logger.info(
                        "Low STT confidence (%.2f) on call %s — gating to clarification",
                        result.confidence, session.call_id,
                    )
                    from dataclasses import replace as _dc_replace
                    result = _dc_replace(result, text=_CLARIFY_TEXT)
                # Enqueue so the main loop can process the utterance.
                await transcript_queue.put(result)

        stt_task = asyncio.create_task(_stt_background())

        try:
            # 1. Opening greeting
            opening = await self._engine.get_opening(session.state)
            await self._speak(session, opening.bot_text)

            # Drain any STT results that accumulated while the bot was speaking
            # the opening (phone echo, line noise, STT artefacts).  The echo
            # gate in _stt_background suppresses *new* results for 650 ms, but
            # results that were already queued before the gate was set are
            # discarded here so the first real caller utterance is handled fresh.
            while not transcript_queue.empty():
                try:
                    transcript_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            # 2. Listen-respond loop
            silence_start: Optional[float] = None

            while True:
                # Wait up to 1 s for the next user utterance
                try:
                    result = await asyncio.wait_for(transcript_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > self._config.silence_timeout_s:
                        logger.info("Silence timeout on call %s", session.call_id)
                        await self._speak(session, "Are you still there?")
                        silence_start = None
                    # Safety: if bot_call_state is stuck in SPEAKING but bot_speaking
                    # is False, the state machine is wedged — force it back.
                    if (session.bot_call_state == BotCallState.SPEAKING
                            and not session.bot_speaking):
                        logger.warning(
                            "[STATE] Forced SPEAKING→LISTENING after safety timeout call=%s",
                            session.call_id,
                        )
                        self._set_state(session, BotCallState.LISTENING, reason="safety-timeout")
                    continue

                silence_start = None
                session.metrics.stt_latency_ms.append(result.latency_ms)
                if session.call_logger:
                    session.call_logger.event(
                        "stt_result",
                        text=result.text[:200],
                        confidence=round(result.confidence, 3),
                        latency_ms=round(result.latency_ms, 1),
                    )

                # End-of-turn patience pause — wait briefly before responding
                # so the caller can finish their thought.  During the window we
                # collect any additional Parakeet results (caller kept talking)
                # and use the LATEST one so we respond to what they just said.
                _eot_deadline = time.time() + _EOT_PAUSE_S
                while True:
                    _remaining = _eot_deadline - time.time()
                    if _remaining <= 0:
                        break
                    try:
                        _extra = await asyncio.wait_for(
                            transcript_queue.get(), timeout=_remaining
                        )
                        result = _extra   # prefer the latest utterance
                    except asyncio.TimeoutError:
                        break

                # ----------------------------------------------------------
                # Post-barge-in cooldown
                # SPEAKING → INTERRUPTED → (wait) → LISTENING → RESPONDING
                # ----------------------------------------------------------
                if session.bot_call_state == BotCallState.INTERRUPTED:
                    # Wait until in-flight TTS audio has finished draining.
                    while session.bot_speaking:
                        await asyncio.sleep(0.02)
                    # Hold 300 ms so the caller does not feel cut off.
                    await asyncio.sleep(self._BARGE_IN_COOLDOWN_S)
                    self._set_state(session, BotCallState.LISTENING, reason="barge-in-cooldown")
                    logger.debug("Barge-in cooldown complete on call %s", session.call_id)
                    # Drain any stale results that arrived during the bot turn
                    # (echo of current bot speech, queued noise words, etc.)
                    while not transcript_queue.empty():
                        try:
                            transcript_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break

                # Process the turn using streaming LLM → sentence-level TTS
                self._set_state(session, BotCallState.RESPONDING, reason="turn-start")
                # Start a short filler clip to fill dead-air while LLM generates.
                # _speak() will cancel it the moment the first real chunk is sent.
                # IMPORTANT: filler clips are pre-rendered μ-law 8 kHz bytes and
                # MUST go through session.filler_sink (a direct μ-law bypass),
                # NOT session.audio_sink which expects 24 kHz PCM from Kokoro.
                # Use a 400 ms delay so the filler only plays when the LLM
                # actually takes that long — fast responses produce no filler
                # blip/click.
                if session.filler_sink is not None:
                    session.filler_task = self._filler_player.start_delayed(
                        session.filler_sink, delay_s=0.40
                    )
                else:
                    session.filler_task = None
                turn = await self._process_turn_streamed(session, result.text)
                self._set_state(session, BotCallState.LISTENING, reason="turn-done")
                # Drain STT results that piled up while the bot was generating
                # + speaking (echo, early caller speech that is now stale).
                # Without this drain, old utterances replay immediately as if
                # they were fresh input on the very next loop iteration.
                while not transcript_queue.empty():
                    try:
                        transcript_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                session.metrics.total_turns += 1
                session.metrics.llm_latency_ms.append(turn.latency_ms)
                if session.call_logger:
                    session.call_logger.event(
                        "turn_complete",
                        bot_text=turn.bot_text[:200],
                        action=str(turn.action),
                        stage=turn.current_stage,
                        latency_ms=round(turn.latency_ms, 1),
                    )

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
            stt_task.cancel()
            try:
                await stt_task
            except (asyncio.CancelledError, Exception):
                pass
            # Signal the WebSocket handler that the call loop is done so it
            # can close the stream.  Do NOT call end_call() here — calling it
            # from within the task itself would try to await the task → deadlock.
            session.call_ended_event.set()

    async def _process_turn_streamed(self, session: _CallSession, prospect_text: str) -> "TurnResult":
        """Stream LLM tokens, TTS each sentence as it completes (lowest latency).

        Stops generating if a barge-in is detected mid-response so the user's
        next utterance is processed immediately.
        """
        from src.llm.rag_engine import RAGEngine
        start = time.perf_counter()
        state = session.state
        state.turn_count += 1

        # ------------------------------------------------------------------
        # Low-confidence / unintelligible audio — skip LLM, ask to repeat.
        # Prevents retry loop: first UNCLEAR = speak retry phrase; consecutive
        # UNCLEAR in the same turn streak = stay silent and keep listening.
        # ------------------------------------------------------------------
        if prospect_text == _CLARIFY_TEXT:
            from src.orchestration.conversation_engine import TurnResult, CallAction
            if not session.unclear_retry_sent:
                clarify_text = "Sorry, I didn't quite catch that — could you say that again?"
                await self._speak(session, clarify_text)
                session.unclear_retry_sent = True
            else:
                clarify_text = ""  # stay silent, keep listening
                logger.debug("Consecutive UNCLEAR on call %s — staying silent", session.call_id)
            return TurnResult(
                bot_text=clarify_text,
                action=CallAction.CONTINUE,
                current_stage=state.current_stage,
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        # Valid transcript received — reset the UNCLEAR streak flag
        session.unclear_retry_sent = False
        state.history.append({"role": "user", "content": prospect_text})

        rag_chunks = self._rag.retrieve(prospect_text)
        rag_context = RAGEngine.format_context(rag_chunks)
        system_prompt = self._engine._build_system_prompt(state, rag_context)

        # Clear any stale barge-in flag before we start speaking
        session.barge_in_event.clear()

        buf = ""
        full_response = ""
        _sent = re.compile(r'(?<=[.!?])(?:\s|$)')
        # Enforce 3-sentence max and 1 question-mark max per bot turn.
        _turn_sentences = 0
        _turn_questions = 0
        _speech_parts: list[str] = []   # accumulate sentences; speak all at once

        try:
            async for token in self._llm.generate_stream(
                system_prompt=system_prompt, messages=state.history
            ):
                buf += token
                full_response += token
                while True:
                    m = _sent.search(buf)
                    if not m:
                        break
                    sentence = buf[:m.end()].strip()
                    buf = buf[m.end():]
                    if not sentence:
                        continue
                    # --- Sentence-count cap: max 3 sentences per turn ---
                    if _turn_sentences >= 3:
                        buf = ""   # discard the rest of the LLM stream
                        break
                    # --- Question-mark cap: at most 1 question per turn ---
                    if "?" in sentence:
                        if _turn_questions >= 1:
                            # Strip everything from the second "?" onward
                            sentence = sentence[: sentence.index("?") + 1]
                            buf = ""  # discard anything that would follow
                        _turn_questions += 1
                    _speech_parts.append(sentence)
                    _turn_sentences += 1
                if _turn_sentences >= 3:
                    break
        except Exception:
            logger.exception("Streaming LLM failed on call %s – falling back", session.call_id)
            from src.orchestration.conversation_engine import TurnResult, CallAction
            turn = await self._engine.process_turn(state, prospect_text)
            state.history.pop()
            state.history.pop()
            return turn

        # Flush any remaining text (under the sentence cap)
        if buf.strip() and _turn_sentences < 3:
            _speech_parts.append(buf.strip())

        # Speak the entire response as one TTS call so Kokoro has full
        # context for natural prosody — no per-sentence gaps or clicks.
        if _speech_parts and not session.barge_in_event.is_set():
            await self._speak(session, " ".join(_speech_parts))

        action = self._engine._resolve_action(state, prospect_text, full_response)
        self._engine._advance_stage(state, prospect_text, full_response)
        # Persist any caller facts extracted this turn (prevents duplicate questions)
        self._engine._update_collected_info(state, prospect_text)
        state.history.append({"role": "assistant", "content": full_response})

        from src.orchestration.conversation_engine import TurnResult
        return TurnResult(
            bot_text=full_response,
            action=action,
            current_stage=state.current_stage,
            latency_ms=(time.perf_counter() - start) * 1000,
            rag_chunks_used=len(rag_chunks),
        )

    def _set_state(self, session: "_CallSession", new_state: BotCallState, reason: str = "") -> None:
        """Transition bot_call_state, keep bot_speaking in sync, and log every change."""
        old = session.bot_call_state
        session.bot_call_state = new_state
        session.bot_speaking = (new_state == BotCallState.SPEAKING)
        logger.debug(
            "[STATE] %s → %s reason=%s call=%s",
            old.name, new_state.name, reason, session.call_id,
        )

    async def _speak(self, session: "_CallSession", text: str) -> None:
        """Synthesise *text* and send audio, stopping immediately on barge-in."""
        prev_state = session.bot_call_state
        self._set_state(session, BotCallState.SPEAKING, reason="speak-start")
        start = time.perf_counter()
        first_chunk = True
        try:
            async for chunk in self._tts.synthesize_stream(text):
                if session.barge_in_event.is_set():
                    # Caller interrupted — flush audio buffer and stop.
                    break
                # Cancel filler the moment the first real TTS audio arrives.
                if first_chunk:
                    first_chunk = False
                    if session.filler_task and not session.filler_task.done():
                        session.filler_task.cancel()
                        session.filler_task = None
                session.audio_sink(chunk.audio_bytes)
        finally:
            # Always restore previous state — even on exception or barge-in.
            if session.bot_call_state == BotCallState.SPEAKING:
                self._set_state(session, prev_state, reason="speak-end")
            else:
                # State was already changed by barge-in handler; ensure flag is clear.
                session.bot_speaking = False
                logger.debug(
                    "[STATE] _speak() finally state=%s (barge-in) call=%s",
                    session.bot_call_state.name, session.call_id,
                )
            # -------------------------------------------------------------------
            # Post-speak echo gate: discard STT results for ECHO_GATE_S after
            # bot finishes to prevent phone-echo self-response loop.
            # -------------------------------------------------------------------
            session.post_speak_gate_until = time.time() + _ECHO_GATE_S
        elapsed = (time.perf_counter() - start) * 1000
        session.metrics.tts_latency_ms.append(elapsed)
