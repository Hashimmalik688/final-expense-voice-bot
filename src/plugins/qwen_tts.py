"""QwenTTS — LiveKit TTS plugin calling a local TTS server.

Calls src/tts/qwen_server.py (which itself tries Qwen3-TTS → XTTS v2 → Kokoro).

Interface pattern follows CoreWorxLab/local-livekit-plugins PiperTTS:
  - Extends livekit.agents.tts.TTS
  - Inner ChunkedStream._run(emitter) synthesises audio and pushes to LiveKit
  - Audio is returned as 24 kHz int16 PCM; LiveKit handles downstream resampling
    (no μ-law conversion needed — that was only required for Twilio)

Server URL is read from TTS_SERVER_URL env var (default: http://127.0.0.1:8002).
The plugin will fall back to in-process XTTS v2 synthesis if the server is
unreachable, so the agent still works during development without running the
separate TTS server.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
import uuid
import wave
from typing import TYPE_CHECKING

import aiohttp
from livekit.agents import tts, APIConnectOptions

if TYPE_CHECKING:
    from livekit.agents.tts.tts import AudioEmitter

__all__ = ["QwenTTS"]

logger = logging.getLogger(__name__)

_TTS_SERVER_URL = os.environ.get("TTS_SERVER_URL", "http://127.0.0.1:8002")
_TTS_SPEAKER = os.environ.get("TTS_SPEAKER", "Ana Florence")
_SAMPLE_RATE = 24000
_NUM_CHANNELS = 1


class QwenTTS(tts.TTS):
    """LiveKit TTS plugin — calls local Qwen3-TTS / XTTS v2 server."""

    def __init__(self, voice_name: str = _TTS_SPEAKER) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=_SAMPLE_RATE,
            num_channels=_NUM_CHANNELS,
        )
        self._voice = voice_name
        self._server_url = _TTS_SERVER_URL
        # Fallback in-process TTS model (loaded lazily only if server is down)
        self._fallback_model = None
        self._fallback_lock = asyncio.Lock()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = APIConnectOptions(),
    ) -> "_QwenChunkedStream":
        return _QwenChunkedStream(self, text, conn_options=conn_options)


class _QwenChunkedStream(tts.ChunkedStream):
    """Fetch audio from TTS server and push PCM frames to LiveKit."""

    def __init__(
        self,
        tts_plugin: QwenTTS,
        text: str,
        *,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts_plugin, input_text=text, conn_options=conn_options)
        self._tts = tts_plugin
        self._text = text

    async def _run(self, emitter: "AudioEmitter") -> None:
        emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=_SAMPLE_RATE,
            num_channels=_NUM_CHANNELS,
            mime_type="audio/pcm",
        )

        start = time.perf_counter()
        try:
            audio_bytes = await self._fetch_from_server(self._text)
        except Exception as exc:
            logger.warning(
                "TTS server unreachable (%s) — using in-process fallback", exc
            )
            audio_bytes = await self._synthesize_fallback(self._text)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(
            "TTS: %.0f ms for %d chars → %d bytes",
            elapsed_ms, len(self._text), len(audio_bytes),
        )

        if audio_bytes:
            emitter.push(audio_bytes)

    # ------------------------------------------------------------------
    # HTTP client to TTS server
    # ------------------------------------------------------------------

    async def _fetch_from_server(self, text: str) -> bytes:
        """POST text to the TTS server, receive raw PCM bytes."""
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self._tts._server_url}/synthesize",
                json={"text": text, "speaker": self._tts._voice},
            ) as resp:
                resp.raise_for_status()
                return await resp.read()

    # ------------------------------------------------------------------
    # In-process fallback (used only when TTS server is down)
    # ------------------------------------------------------------------

    async def _synthesize_fallback(self, text: str) -> bytes:
        """Load XTTS v2 in-process and synthesise (expensive, single-use guard)."""
        model = await self._ensure_fallback_loaded()
        if model is None:
            logger.error("No TTS fallback available — returning silence")
            return b""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._infer_blocking, model, text)

    async def _ensure_fallback_loaded(self) -> object:
        if self._tts._fallback_model is not None:
            return self._tts._fallback_model
        async with self._tts._fallback_lock:
            if self._tts._fallback_model is not None:
                return self._tts._fallback_model
            logger.info("Loading XTTS v2 fallback model …")
            try:
                from TTS.api import TTS as CoquiTTS  # type: ignore
                model = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: CoquiTTS(
                        "tts_models/multilingual/multi-dataset/xtts_v2",
                        gpu=True,
                    ),
                )
                self._tts._fallback_model = model
                logger.info("XTTS v2 fallback ready.")
                return model
            except Exception as exc:
                logger.error("XTTS v2 fallback load failed: %s", exc)
                return None

    @staticmethod
    def _infer_blocking(model, text: str) -> bytes:
        """Run XTTS v2 synchronously (called in executor)."""
        import numpy as np
        try:
            audio_list = model.tts(
                text=text,
                speaker=_TTS_SPEAKER,
                language="en",
            )
            audio = np.array(audio_list, dtype=np.float32)
            # Convert float32 [-1,1] → int16 PCM
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            return pcm
        except Exception as exc:
            logger.error("XTTS v2 inference error: %s", exc)
            return b""
