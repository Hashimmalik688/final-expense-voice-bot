"""
Silero Voice Activity Detection (VAD) handler.

Filters non-speech segments before passing audio to Parakeet STT.
Without VAD the STT model runs inference on silence, wasting ~40 % of
inference calls during a typical phone call.

Adopted from ShayneP/local-voice-ai's Silero VAD approach
(https://github.com/ShayneP/local-voice-ai) which uses silero.VAD.load()
via the LiveKit plugin.  Here we drive the silero-vad PyPI package directly
so we're not tied to LiveKit.

Install::

    pip install silero-vad>=5.1.2
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class SileroVADHandler:
    """Lightweight speech/silence classifier using the Silero VAD model.

    The handler fails **open** — if the ``silero-vad`` package is unavailable
    (not installed, or failed to load), every ``is_speech()`` call returns
    ``True`` so the STT pipeline continues normally without VAD filtering.

    Usage::

        vad = SileroVADHandler()
        vad.load()

        # Per-chunk check (fast, runs on every incoming 20–30 ms PCM packet)
        if vad.is_speech(raw_pcm_bytes):
            await stt.buffer(raw_pcm_bytes)

        # Strip silence from a full buffered utterance before STT inference
        clean = vad.filter_speech(full_float32_array)
        if len(clean):
            transcript = await stt.infer(clean)
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        threshold: float = float(os.environ.get("VAD_THRESHOLD", "0.35")),
        min_speech_ms: int = int(os.environ.get("VAD_MIN_SPEECH_MS", "200")),
        min_silence_ms: int = int(os.environ.get("VAD_MIN_SILENCE_MS", "300")),
        energy_threshold: float = float(os.environ.get("VAD_ENERGY_THRESHOLD", "0.002")),
        min_speech_duration_ms: int = int(os.environ.get("VAD_MIN_SPEECH_DURATION_MS", "150")),
    ) -> None:
        """
        Parameters
        ----------
        sample_rate:
            Audio sample rate in Hz.  Parakeet expects 16 kHz.
        threshold:
            Probability threshold above which a chunk is classified as speech.
            0.5 is the recommended default (Silero documentation).
        min_speech_ms:
            Minimum duration (ms) to constitute a speech segment when
            calling ``filter_speech()``.  Raised to 400 ms to ignore
            short non-speech sounds (coughs, background TV, etc.).
        min_silence_ms:
            Minimum silence gap (ms) between speech segments.
        energy_threshold:
            RMS energy gate.  Frames quieter than this (< 0.003 ≈ -50 dBFS)
            are classified as silence regardless of Silero's VAD output.
            This rejects distant background voices and low-level noise.
        min_speech_duration_ms:
            Minimum sustained speech duration (ms) before ``is_speech()``
            returns True.  Short bursts (< 300 ms) such as mouth clicks,
            coughs, and background pops are rejected.
        """
        self._sample_rate = sample_rate
        self._threshold = threshold
        self._min_speech_ms = min_speech_ms
        self._min_silence_ms = min_silence_ms
        self._energy_threshold = energy_threshold
        self._min_speech_duration_ms = min_speech_duration_ms
        self._model = None
        self._available: bool = False
        # Track sustained speech duration for the min-duration gate
        self._speech_onset_ms: float = 0.0
        self._is_in_speech: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the Silero VAD model weights (downloads ~2 MB on first run)."""
        if self._model is not None:
            return
        try:
            from silero_vad import load_silero_vad  # type: ignore[import-untyped]

            self._model = load_silero_vad()
            self._available = True
            logger.info("Silero VAD loaded (threshold=%.2f).", self._threshold)
        except ImportError:
            logger.warning(
                "silero-vad not installed — VAD disabled (all audio passes to STT). "
                "Install with: pip install silero-vad>=5.1.2"
            )
        except Exception:
            logger.exception("Silero VAD failed to load — VAD disabled.")

    # ------------------------------------------------------------------
    # Per-chunk speech detection
    # ------------------------------------------------------------------

    def is_speech(self, audio_chunk: "bytes | np.ndarray") -> bool:
        """Return ``True`` if the chunk likely contains speech.

        This is designed to run on every small PCM packet (20–100 ms) that
        arrives from the SIP channel.  It is fast enough to run synchronously
        in the receive loop.

        A minimum-duration gate (default 300 ms) ensures that short bursts
        (mouth clicks, coughs) are not classified as speech.

        Parameters
        ----------
        audio_chunk:
            Either raw PCM-16 LE *bytes* or a float32 numpy array in [-1, 1].
        """
        if not self._available:
            return True  # fail open

        import torch  # type: ignore[import-untyped]
        import time

        if isinstance(audio_chunk, (bytes, bytearray)):
            samples = (
                np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
                / 32768.0
            )
        else:
            samples = np.asarray(audio_chunk, dtype=np.float32)

        if samples.size == 0:
            return False

        # Energy gate: reject frames that are too quiet to be foreground speech.
        rms = float(np.sqrt(np.mean(samples ** 2)))
        if rms < self._energy_threshold:
            self._is_in_speech = False
            self._speech_onset_ms = 0.0
            return False

        tensor = torch.from_numpy(samples).unsqueeze(0)
        try:
            with torch.no_grad():
                confidence: float = self._model(tensor, self._sample_rate).item()
        except Exception:
            logger.debug("VAD inference error — passing audio through.", exc_info=True)
            return True  # fail open

        if confidence >= self._threshold:
            now_ms = time.monotonic() * 1000
            if not self._is_in_speech:
                self._is_in_speech = True
                self._speech_onset_ms = now_ms
            # Enforce minimum speech duration gate
            elapsed_ms = now_ms - self._speech_onset_ms
            if elapsed_ms < self._min_speech_duration_ms:
                logger.debug(
                    "[VAD] prob=%.3f energy=%.4f duration_ms=%.0f result=False(too-short)",
                    confidence, rms, elapsed_ms,
                )
                return False  # too short — wait for more sustained speech
            logger.debug(
                "[VAD] prob=%.3f energy=%.4f duration_ms=%.0f result=True",
                confidence, rms, elapsed_ms,
            )
            return True
        else:
            self._is_in_speech = False
            self._speech_onset_ms = 0.0
            logger.debug(
                "[VAD] prob=%.3f energy=%.4f result=False(low-prob)",
                confidence, rms,
            )
            return False

    # ------------------------------------------------------------------
    # Full-utterance silence stripping
    # ------------------------------------------------------------------

    def filter_speech(self, audio: np.ndarray) -> np.ndarray:
        """Strip silence from a full buffered utterance before STT inference.

        Returns concatenated speech segments.  If VAD is unavailable or no
        speech is detected, returns the original array unchanged so the
        caller can decide what to do.

        Parameters
        ----------
        audio:
            Float32 numpy array at ``self._sample_rate`` Hz.
        """
        if not self._available or self._model is None:
            return audio

        try:
            import torch  # type: ignore[import-untyped]
            from silero_vad import get_speech_timestamps  # type: ignore[import-untyped]
        except ImportError:
            return audio

        tensor = torch.from_numpy(audio.astype(np.float32))
        timestamps = get_speech_timestamps(
            tensor,
            self._model,
            sampling_rate=self._sample_rate,
            threshold=self._threshold,
            min_speech_duration_ms=self._min_speech_ms,
            min_silence_duration_ms=self._min_silence_ms,
        )
        if not timestamps:
            return np.array([], dtype=np.float32)
        return np.concatenate([audio[ts["start"] : ts["end"]] for ts in timestamps])
