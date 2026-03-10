"""
Parakeet TDT Speech-to-Text handler.

Wraps NVIDIA's Parakeet TDT model for real-time streaming speech recognition.
Audio chunks arrive from the SIP channel and are transcribed with minimal
latency to feed the conversation engine.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import numpy as np

from config.settings import STTConfig, get_stt_config

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Immutable container for a single transcription segment."""

    text: str
    is_final: bool
    confidence: float
    start_time: float
    end_time: float
    latency_ms: float


class ParakeetSTTHandler:
    """Stream audio to the Parakeet TDT model and yield transcriptions.

    The handler operates in *streaming* mode:  audio chunks are buffered and
    processed in overlapping windows so partial hypotheses can be returned as
    quickly as possible.

    Usage::

        handler = ParakeetSTTHandler()
        await handler.initialize()

        async for result in handler.transcribe_stream(audio_chunks):
            print(result.text, result.is_final)
    """

    def __init__(self, config: Optional[STTConfig] = None) -> None:
        self._config = config or get_stt_config()
        self._model = None
        self._is_initialized = False
        self._audio_buffer: list[np.ndarray] = []
        self._buffer_duration_s: float = 0.0
        # Minimum audio to accumulate before running inference (seconds)
        self._min_chunk_s: float = 0.3
        # Maximum buffer before forcing a flush
        self._max_buffer_s: float = 10.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load the Parakeet TDT model onto the configured device."""
        if self._is_initialized:
            return

        logger.info(
            "Loading Parakeet TDT model '%s' on device '%s' …",
            self._config.model_name,
            self._config.device,
        )

        try:
            import nemo.collections.asr as nemo_asr  # type: ignore[import-untyped]

            self._model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self._config.model_name,
            )
            if self._config.device == "cuda":
                self._model = self._model.cuda()
            self._model.eval()
            self._is_initialized = True
            logger.info("Parakeet TDT model loaded successfully.")
        except Exception:
            logger.exception("Failed to load Parakeet TDT model.")
            raise

    async def shutdown(self) -> None:
        """Release model resources."""
        self._model = None
        self._is_initialized = False
        self._audio_buffer.clear()
        self._buffer_duration_s = 0.0
        logger.info("Parakeet STT handler shut down.")

    # ------------------------------------------------------------------
    # Streaming transcription
    # ------------------------------------------------------------------

    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
    ) -> AsyncIterator[TranscriptionResult]:
        """Consume raw PCM audio chunks and yield transcription results.

        Parameters
        ----------
        audio_chunks:
            An async iterator of raw PCM-16 LE audio bytes at
            ``self._config.sample_rate`` Hz.

        Yields
        ------
        TranscriptionResult
            Partial and final transcription segments.
        """
        if not self._is_initialized:
            raise RuntimeError("Handler not initialised – call initialize() first.")

        async for chunk in audio_chunks:
            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            self._audio_buffer.append(samples)
            self._buffer_duration_s += len(samples) / self._config.sample_rate

            if self._buffer_duration_s >= self._min_chunk_s:
                result = await self._run_inference(is_final=False)
                if result and result.text.strip():
                    yield result

            if self._buffer_duration_s >= self._max_buffer_s:
                result = await self._run_inference(is_final=True)
                if result and result.text.strip():
                    yield result
                self._reset_buffer()

    async def transcribe_buffer(self, audio: bytes) -> TranscriptionResult:
        """One-shot transcription of a complete audio buffer.

        Useful for voicemail detection or post-call analysis.
        """
        if not self._is_initialized:
            raise RuntimeError("Handler not initialised – call initialize() first.")

        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        self._audio_buffer = [samples]
        self._buffer_duration_s = len(samples) / self._config.sample_rate
        result = await self._run_inference(is_final=True)
        self._reset_buffer()
        return result

    # ------------------------------------------------------------------
    # Silence / end-of-speech detection
    # ------------------------------------------------------------------

    def detect_silence(self, audio: np.ndarray, threshold: float = 0.01) -> bool:
        """Return ``True`` if the RMS energy of *audio* is below *threshold*."""
        rms = float(np.sqrt(np.mean(audio ** 2)))
        return rms < threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_inference(self, *, is_final: bool) -> TranscriptionResult:
        """Run the model on the current audio buffer."""
        start = time.perf_counter()
        waveform = np.concatenate(self._audio_buffer)

        try:
            # Offload blocking inference to a thread so the event loop stays free.
            # Add timeout to prevent event loop stalling on stuck inference.
            transcription = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    self._infer_sync,
                    waveform,
                ),
                timeout=10.0  # 10-second max for STT inference
            )
        except asyncio.TimeoutError:
            logger.error("STT inference timeout (>10s) – returning empty transcription")
            transcription = ""
        except Exception as exc:
            logger.error("STT inference failed: %s", exc, exc_info=True)
            transcription = ""

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(
            "STT inference: %.1f ms | final=%s | text='%s'",
            elapsed_ms,
            is_final,
            transcription,
        )

        return TranscriptionResult(
            text=transcription,
            is_final=is_final,
            confidence=1.0,  # Parakeet doesn't expose per-segment scores
            start_time=0.0,
            end_time=self._buffer_duration_s,
            latency_ms=elapsed_ms,
        )

    def _infer_sync(self, waveform: np.ndarray) -> str:
        """Synchronous inference wrapper (runs inside an executor thread)."""
        import tempfile
        import soundfile as sf  # type: ignore[import-untyped]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, waveform, self._config.sample_rate)
            transcriptions = self._model.transcribe([tmp.name])
            # Parakeet returns a list of Hypothesis objects or plain strings.
            if transcriptions and isinstance(transcriptions[0], str):
                return transcriptions[0]
            if transcriptions and hasattr(transcriptions[0], "text"):
                return transcriptions[0].text
            return str(transcriptions[0]) if transcriptions else ""

    def _reset_buffer(self) -> None:
        self._audio_buffer.clear()
        self._buffer_duration_s = 0.0
