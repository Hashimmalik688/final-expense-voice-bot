"""
CosyVoice 2 Text-to-Speech handler.

Sends text to the CosyVoice 2 API and receives streaming audio back.
Supports sentence-level chunking so the first audio bytes reach the caller
before the full response has been synthesised, keeping end-to-end latency
under the 500 ms target.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import aiohttp

from config.settings import TTSConfig, get_tts_config

logger = logging.getLogger(__name__)


@dataclass
class TTSAudioChunk:
    """A piece of synthesised audio returned by the TTS engine."""

    audio_bytes: bytes
    sample_rate: int
    is_last: bool
    latency_ms: float


class CosyVoiceTTSHandler:
    """Async client for the CosyVoice 2 TTS service.

    The CosyVoice 2 server is expected to expose a REST endpoint that
    accepts JSON ``{"text": "…", "voice_id": "…", "stream": true}``
    and returns chunked PCM audio (or WAV) over the HTTP response body.

    Usage::

        handler = CosyVoiceTTSHandler()
        await handler.initialize()

        async for chunk in handler.synthesize_stream("Hello, how are you?"):
            play(chunk.audio_bytes)
    """

    # Regex that splits text at sentence boundaries while keeping the
    # delimiter attached to the preceding segment.
    _SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+")

    def __init__(self, config: Optional[TTSConfig] = None) -> None:
        self._config = config or get_tts_config()
        self._session: Optional[aiohttp.ClientSession] = None
        self._base_url = self._config.api_url.rstrip("/")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the HTTP session used for all TTS requests."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
            logger.info(
                "CosyVoice TTS initialised – endpoint=%s  voice=%s",
                self._base_url,
                self._config.voice_id,
            )

    async def shutdown(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("CosyVoice TTS handler shut down.")

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Return ``True`` if the TTS server is reachable."""
        if not self._session or self._session.closed:
            return False
        try:
            async with self._session.get(f"{self._base_url}/health") as resp:
                return resp.status == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Streaming synthesis
    # ------------------------------------------------------------------

    async def synthesize_stream(self, text: str) -> AsyncIterator[TTSAudioChunk]:
        """Synthesise *text* and yield audio chunks as they become available.

        If the TTS server supports native streaming, we consume its chunked
        response directly.  Otherwise we split the text into sentences and
        fire one request per sentence, yielding each result as it arrives.
        """
        if self._config.streaming:
            async for chunk in self._stream_native(text):
                yield chunk
        else:
            async for chunk in self._stream_by_sentence(text):
                yield chunk

    async def synthesize_full(self, text: str) -> bytes:
        """Return the complete synthesised audio as a single bytes object."""
        parts: list[bytes] = []
        async for chunk in self.synthesize_stream(text):
            parts.append(chunk.audio_bytes)
        return b"".join(parts)

    # ------------------------------------------------------------------
    # Internal: native server-side streaming
    # ------------------------------------------------------------------

    async def _stream_native(self, text: str) -> AsyncIterator[TTSAudioChunk]:
        url = f"{self._base_url}/v1/tts/stream"
        payload = {
            "text": text,
            "voice_id": self._config.voice_id,
            "sample_rate": self._config.sample_rate,
            "speed": self._config.speed,
            "stream": True,
        }
        start = time.perf_counter()
        first_chunk = True

        try:
            async with self._session.post(url, json=payload) as resp:
                resp.raise_for_status()
                buffer = bytearray()
                async for data in resp.content.iter_any():
                    buffer.extend(data)
                    # Yield in ~4 KB chunks to match typical audio frame sizes
                    while len(buffer) >= 4096:
                        chunk_bytes = bytes(buffer[:4096])
                        del buffer[:4096]
                        elapsed = (time.perf_counter() - start) * 1000
                        if first_chunk:
                            logger.debug("TTS first-byte latency: %.1f ms", elapsed)
                            first_chunk = False
                        yield TTSAudioChunk(
                            audio_bytes=chunk_bytes,
                            sample_rate=self._config.sample_rate,
                            is_last=False,
                            latency_ms=elapsed,
                        )

                # Flush remaining buffer
                if buffer:
                    elapsed = (time.perf_counter() - start) * 1000
                    yield TTSAudioChunk(
                        audio_bytes=bytes(buffer),
                        sample_rate=self._config.sample_rate,
                        is_last=True,
                        latency_ms=elapsed,
                    )
        except Exception:
            logger.exception("TTS streaming request failed.")
            raise

    # ------------------------------------------------------------------
    # Internal: sentence-level chunking fallback
    # ------------------------------------------------------------------

    async def _stream_by_sentence(self, text: str) -> AsyncIterator[TTSAudioChunk]:
        sentences = self._split_sentences(text)
        for idx, sentence in enumerate(sentences):
            is_last = idx == len(sentences) - 1
            audio = await self._synthesize_one(sentence)
            yield TTSAudioChunk(
                audio_bytes=audio,
                sample_rate=self._config.sample_rate,
                is_last=is_last,
                latency_ms=0.0,
            )

    async def _synthesize_one(self, text: str) -> bytes:
        """Send a single non-streaming TTS request and return audio bytes."""
        url = f"{self._base_url}/v1/tts/generate"
        payload = {
            "text": text,
            "voice_id": self._config.voice_id,
            "sample_rate": self._config.sample_rate,
            "speed": self._config.speed,
        }
        start = time.perf_counter()
        try:
            async with self._session.post(url, json=payload) as resp:
                resp.raise_for_status()
                audio = await resp.read()
                elapsed = (time.perf_counter() - start) * 1000
                logger.debug("TTS single synthesis: %.1f ms | %d bytes", elapsed, len(audio))
                return audio
        except Exception:
            logger.exception("TTS single synthesis failed.")
            raise

    # ------------------------------------------------------------------
    # Text utilities
    # ------------------------------------------------------------------

    @classmethod
    def _split_sentences(cls, text: str) -> list[str]:
        """Split text at sentence boundaries, keeping short segments merged."""
        parts = cls._SENTENCE_SPLIT.split(text.strip())
        merged: list[str] = []
        buf = ""
        for part in parts:
            if len(buf) + len(part) < 120:
                buf = f"{buf} {part}".strip() if buf else part
            else:
                if buf:
                    merged.append(buf)
                buf = part
        if buf:
            merged.append(buf)
        return merged or [text]
