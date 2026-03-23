"""ParakeetSTT — LiveKit STT plugin wrapping NVIDIA Parakeet TDT 0.6B v3.

Interface pattern follows CoreWorxLab/local-livekit-plugins FasterWhisperSTT:
  - Extends livekit.agents.stt.STT
  - Non-streaming (LiveKit VAD + Silero detect speech boundaries, then we
    receive a complete AudioBuffer for batch transcription)
  - Audio arrives at 16 kHz int16 PCM — exactly what Parakeet expects, so
    no resampling is needed here (unlike the Twilio μ-law path)

The handler is loaded lazily on first call to avoid blocking process startup.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import numpy as np
from livekit.agents import stt, utils
from livekit.agents import APIConnectOptions
from livekit.agents.types import NOT_GIVEN, NotGivenOr

__all__ = ["ParakeetSTT"]

logger = logging.getLogger(__name__)


class ParakeetSTT(stt.STT):
    """LiveKit STT plugin wrapping NVIDIA Parakeet TDT 0.6B v3.

    Non-streaming: LiveKit + Silero VAD accumulate audio until the speaker
    pauses, then call _recognize_impl with the buffered AudioBuffer.
    """

    def __init__(self) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self._handler: Optional[object] = None  # ParakeetSTTHandler, loaded lazily
        self._load_lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        """Load Parakeet model on first use (lazy init to avoid blocking startup)."""
        if self._handler is not None:
            return
        async with self._load_lock:
            if self._handler is not None:
                return
            logger.info("Loading Parakeet TDT model (first call) …")
            from src.stt.parakeet_handler import ParakeetSTTHandler
            handler = ParakeetSTTHandler()
            await handler.initialize()
            self._handler = handler
            logger.info("Parakeet TDT model ready.")

    # ------------------------------------------------------------------
    # Core plugin interface (called by LiveKit after VAD speech boundary)
    # ------------------------------------------------------------------

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """Transcribe a complete utterance (called after speaker stops)."""
        await self._ensure_loaded()

        # Convert AudioBuffer (single frame or list of frames) → raw bytes
        # LiveKit delivers 16 kHz int16 PCM — exactly what Parakeet needs.
        if isinstance(buffer, list):
            raw_bytes = b"".join(bytes(frame.data) for frame in buffer)
        else:
            raw_bytes = bytes(buffer.data)

        if not raw_bytes:
            return _empty_event()

        start = time.perf_counter()
        try:
            result = await self._handler.transcribe_buffer(raw_bytes)
        except Exception as exc:
            logger.error("Parakeet transcription failed: %s", exc, exc_info=True)
            return _empty_event()

        elapsed_ms = (time.perf_counter() - start) * 1000
        text = (result.text or "").strip() if result else ""
        confidence = float(result.confidence) if result else 0.0

        logger.debug(
            "ParakeetSTT: %.0f ms | conf=%.2f | text=%r",
            elapsed_ms, confidence, text[:80],
        )

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    text=text,
                    start_time=0.0,
                    end_time=0.0,
                    language="en",
                    confidence=confidence,
                )
            ],
        )


def _empty_event() -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[stt.SpeechData(text="", start_time=0.0, end_time=0.0, language="en")],
    )
