"""
Parakeet TDT Speech-to-Text handler.

Wraps NVIDIA's Parakeet TDT model for real-time streaming speech recognition.
Audio chunks arrive from the SIP channel and are transcribed with minimal
latency to feed the conversation engine.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional

import numpy as np
import os

from config.settings import STTConfig, get_stt_config

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Post-processing: correct common insurance-term misrecognitions.
# NeMo TDT doesn't expose a native hot-word boost API without LM
# integration, so we apply a deterministic substitution pass to the
# raw transcript text instead.
# ------------------------------------------------------------------
_INSURANCE_CORRECTIONS: list[tuple[re.Pattern, str]] = [
    # Proper nouns / product names
    (re.compile(r"\bfinal expense?s?\b",            re.I), "final expense"),
    (re.compile(r"\bbenefic[ie]a?ry\b",             re.I), "beneficiary"),
    (re.compile(r"\bpremi+um\b",                    re.I), "premium"),
    (re.compile(r"\bpolicyholder\b",                re.I), "policyholder"),
    (re.compile(r"\bunderwriting\b",                re.I), "underwriting"),
    (re.compile(r"\bwhole[- ]?life\b",              re.I), "whole life"),
    (re.compile(r"\bterm[- ]?life\b",               re.I), "term life"),
    (re.compile(r"\bburial insurance\b",            re.I), "burial insurance"),
    (re.compile(r"\bdeath benefit\b",               re.I), "death benefit"),
    (re.compile(r"\bcoverage amount\b",             re.I), "coverage amount"),
    # Numbers that TDT sometimes mis-transcribes
    (re.compile(r"\bten thousand\b",                re.I), "ten thousand"),
    (re.compile(r"\bthirty (dollars?|a month)\b",   re.I), r"thirty \1"),
]


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

    # Minimum confidence score to treat an STT result as valid.
    # Results below this threshold trigger a clarification request.
    # Configurable via CONFIDENCE_THRESHOLD env var.
    CONFIDENCE_THRESHOLD: float = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.55"))

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
        # How many consecutive silent chunks before we flush.
        # At 20 ms/chunk: 50 = 1 000 ms  ← end-of-speech silence threshold.
        # Raising this from the old 35 (~700 ms) gives callers enough time
        # to finish a sentence before the bot responds.
        self._silence_threshold: float = 0.002  # RMS amplitude gate
        self._silence_chunks_needed: int = 50   # 1 000 ms at 20 ms/chunk
        # Extra chunks to keep buffering after EOS is confirmed (speech pad).
        # 10 chunks × 20 ms = 200 ms — captures trailing syllables/breaths.
        self._speech_pad_chunks: int = 10
        self._silence_counter: int = 0

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
            import torch
            import nemo.collections.asr as nemo_asr  # type: ignore[import-untyped]

            device = torch.device(self._config.device)
            model_name = self._config.model_name
            # Always load weights to CPU first, then convert to FP16 before
            # moving to CUDA — avoids a 3 GB FP32 spike that exhausts VRAM.
            cpu_device = torch.device("cpu")
            if model_name.endswith(".nemo") or model_name.startswith("/"):
                # Local .nemo file — use restore_from() instead of from_pretrained()
                import os
                abs_path = os.path.abspath(model_name)
                logger.info("Loading from local .nemo file: %s → cpu → %s", abs_path, device)
                self._model = nemo_asr.models.ASRModel.restore_from(
                    restore_path=abs_path, map_location=cpu_device
                )
            else:
                self._model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=model_name, map_location=cpu_device,
                )
            # Keep FP32 on CUDA — Parakeet TDT v3's NeMo TDT decoder returns
            # different internal tuple structures under FP16, causing a
            # "not enough values to unpack" crash during transcribe_generator.
            # On a 24 GB GPU the extra ~1.5 GB is negligible.
            self._model = self._model.to(device)
            self._model.eval()
            # Disable CUDA graphs — NeMo's TDT label-looping decoder uses
            # cudaStreamGetCaptureInfo which is incompatible with CUDA 12.8 /
            # PyTorch 2.10 (returns a different tuple length). Falling back
            # to the plain PyTorch path is functionally identical.
            self._disable_cuda_graphs()
            self._is_initialized = True
            logger.info("Parakeet TDT model loaded successfully (device=%s, dtype=%s).",
                        device, next(self._model.parameters()).dtype)
        except Exception:
            logger.exception("Failed to load Parakeet TDT model.")
            raise

        logger.info("VAD disabled — using energy-based silence detection only.")

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
        *,
        on_partial: Optional[Callable[[str], None]] = None,
    ) -> AsyncIterator[TranscriptionResult]:
        """Consume raw PCM audio chunks and yield transcription results.

        Parameters
        ----------
        audio_chunks:
            An async iterator of raw PCM-16 LE audio bytes at
            ``self._config.sample_rate`` Hz.
        on_partial:
            Optional callback invoked with a partial transcript string
            while the user is still speaking (non-final, low confidence).
            Useful to start a filler player or a typing indicator.

        Yields
        ------
        TranscriptionResult
            Final transcription segments (emitted after end-of-speech silence).
        """
        if not self._is_initialized:
            raise RuntimeError("Handler not initialised – call initialize() first.")

        # No VAD — buffer all audio and flush after silence or max size.
        self._silence_counter = 0
        _partial_timer: float = 0.0   # track when we last emitted a partial

        async for chunk in audio_chunks:
            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(samples ** 2)))

            self._audio_buffer.append(samples)
            self._buffer_duration_s += len(samples) / self._config.sample_rate

            if rms < self._silence_threshold:
                self._silence_counter += 1
            else:
                self._silence_counter = 0

            # Optional partial-transcript callback: fire every ~1.5 s of
            # active speech to allow external logic to start a filler sooner.
            if (
                on_partial is not None
                and self._buffer_duration_s >= 1.5
                and self._silence_counter <= 5   # still talking
                and (self._buffer_duration_s - _partial_timer) >= 1.5
            ):
                _partial_timer = self._buffer_duration_s
                try:
                    partial_result = await self._run_inference(is_final=False)
                    if partial_result and partial_result.text.strip():
                        on_partial(partial_result.text)
                except Exception:
                    pass  # partial callback failure is non-fatal

            # Flush when we have enough audio AND a sustained silence run.
            if (
                self._buffer_duration_s >= self._min_chunk_s
                and self._silence_counter >= self._silence_chunks_needed + self._speech_pad_chunks
            ):
                result = await self._run_inference(is_final=True)
                if result and result.text.strip():
                    yield result
                self._reset_buffer()
                self._silence_counter = 0
                _partial_timer = 0.0
                continue

            # Force-flush if the buffer is growing too large
            if self._buffer_duration_s >= self._max_buffer_s:
                result = await self._run_inference(is_final=True)
                if result and result.text.strip():
                    yield result
                self._reset_buffer()
                self._silence_counter = 0
                _partial_timer = 0.0

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

    def _disable_cuda_graphs(self) -> None:
        """Walk the model's decoding chain and disable CUDA graphs."""
        for obj in (self._model, getattr(self._model, 'decoding', None)):
            inner = getattr(obj, 'decoding', None) if obj is not None else None
            if inner is not None:
                computer = getattr(inner, 'decoding_computer', None)
                if computer is not None and hasattr(computer, 'disable_cuda_graphs'):
                    computer.disable_cuda_graphs()
                    logger.info("Disabled CUDA graphs on decoding_computer (%s)", type(computer).__name__)
                    return
        logger.warning("Could not find decoding_computer to disable CUDA graphs")

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
            text, confidence = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    self._infer_sync,
                    waveform,
                ),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.error("STT inference timeout (>10s) – returning empty transcription")
            text, confidence = "", 0.0
        except Exception as exc:
            logger.error("STT inference failed: %s", exc, exc_info=True)
            text, confidence = "", 0.0

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            "STT inference: %.1f ms | final=%s | conf=%.2f | text='%s'",
            elapsed_ms, is_final, confidence, text,
        )

        return TranscriptionResult(
            text=text,
            is_final=is_final,
            confidence=confidence,
            start_time=0.0,
            end_time=self._buffer_duration_s,
            latency_ms=elapsed_ms,
        )

    def _infer_sync(self, waveform: np.ndarray) -> tuple[str, float]:
        """Synchronous inference wrapper (runs inside an executor thread).

        Returns (text, confidence) where confidence is derived from NeMo
        log-prob scores when available, falling back to a heuristic.
        """
        import tempfile
        import soundfile as sf  # type: ignore[import-untyped]

        # Guard against very short buffers that confuse the NeMo dataloader.
        min_samples = int(self._config.sample_rate * 0.2)  # 200 ms minimum
        if len(waveform) < min_samples:
            return ("", 0.0)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            sf.write(tmp.name, waveform, self._config.sample_rate)
            tmp.close()
            hypotheses = self._model.transcribe(
                [tmp.name], return_hypotheses=True, verbose=False
            )
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

        if not hypotheses:
            return ("", 0.0)

        result = hypotheses[0]
        # NeMo returns a list-of-lists when return_hypotheses=True
        if isinstance(result, list):
            result = result[0] if result else None
        if result is None:
            return ("", 0.0)

        # Extract text
        if hasattr(result, "text"):
            text = result.text or ""
        else:
            text = str(result) if result else ""

        # Compute confidence via word-density heuristic.
        # NeMo's Hypothesis.score (mean log-prob) is unreliable for real-time
        # phone audio — values cluster around -0.6…-1.0 making exp() always
        # land below any useful threshold.  The speech-rate heuristic is more
        # robust: if the caller produced a reasonable number of words for the
        # audio duration, we trust the transcript.
        raw_score = None
        if hasattr(result, "score") and result.score is not None:
            try:
                raw_score = float(result.score)
            except (ValueError, TypeError):
                raw_score = None

        words = len(text.split())
        duration_s = len(waveform) / self._config.sample_rate

        if duration_s < 0.3 or words == 0:
            confidence = 0.0
        elif words / duration_s < 0.5:       # too sparse — likely noise
            confidence = 0.45
        elif words / duration_s > 8.0:       # too fast — likely decoder artefact
            confidence = 0.45
        else:
            confidence = 0.85

        logger.info(
            "[CONFIDENCE DEBUG] raw_score=%s, conf=%.3f, text=\"%s\"",
            raw_score, confidence, text,
        )

        text = self._apply_corrections(text)
        return (text, confidence)

    @staticmethod
    def _apply_corrections(text: str) -> str:
        """Post-process ASR output to fix common insurance-term misrecognitions.

        NeMo TDT doesn't support native hot-word biasing without LM integration,
        so we use a deterministic substitution pass tuned to the insurance domain.
        """
        for pattern, replacement in _INSURANCE_CORRECTIONS:
            text = pattern.sub(replacement, text)
        return text

    def _reset_buffer(self) -> None:
        self._audio_buffer.clear()
        self._buffer_duration_s = 0.0
