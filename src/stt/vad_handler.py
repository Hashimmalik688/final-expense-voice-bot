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
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 100,
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
            calling ``filter_speech()``.
        min_silence_ms:
            Minimum silence gap (ms) between speech segments.
        """
        self._sample_rate = sample_rate
        self._threshold = threshold
        self._min_speech_ms = min_speech_ms
        self._min_silence_ms = min_silence_ms
        self._model = None
        self._available: bool = False

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

        Parameters
        ----------
        audio_chunk:
            Either raw PCM-16 LE *bytes* or a float32 numpy array in [-1, 1].
        """
        if not self._available:
            return True  # fail open

        import torch  # type: ignore[import-untyped]

        if isinstance(audio_chunk, (bytes, bytearray)):
            samples = (
                np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
                / 32768.0
            )
        else:
            samples = np.asarray(audio_chunk, dtype=np.float32)

        if samples.size == 0:
            return False

        tensor = torch.from_numpy(samples).unsqueeze(0)
        try:
            with torch.no_grad():
                confidence: float = self._model(tensor, self._sample_rate).item()
            return confidence >= self._threshold
        except Exception:
            logger.debug("VAD inference error — passing audio through.", exc_info=True)
            return True  # fail open

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
