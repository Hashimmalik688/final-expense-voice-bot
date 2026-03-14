"""
Kokoro v0.9 Text-to-Speech handler.

Sends text to the Kokoro TTS HTTP server and receives streaming audio back.
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
import numpy as np

from config.settings import TTSConfig, get_tts_config

logger = logging.getLogger(__name__)


@dataclass
class TTSAudioChunk:
    """A piece of synthesised audio returned by the TTS engine."""

    audio_bytes: bytes
    sample_rate: int
    is_last: bool
    latency_ms: float


class KokoroTTSHandler:
    """Async client for the Kokoro TTS service.

    The Kokoro server exposes a REST endpoint that accepts JSON
    ``{"text": "…", "voice_id": "…", "speed": 1.0}`` and returns raw
    int16 PCM audio at 24 kHz over the HTTP response body.

    Usage::

        handler = KokoroTTSHandler()
        await handler.initialize()

        async for chunk in handler.synthesize_stream("Hello, how are you?"):
            play(chunk.audio_bytes)
    """

    # Regex that splits text at sentence boundaries while keeping the
    # delimiter attached to the preceding segment.
    _SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+")

    # Matches greeting words at the start of a clause that are NOT already
    # followed by a comma so we can inject a natural breath-pause.
    _GREETING_RE = re.compile(
        r"\b(Hello|Hi\s+there|Hi|Hey\s+there|Hey|"
        r"Good\s+(?:morning|afternoon|evening))\s+(?=[A-Za-z])",
        re.IGNORECASE,
    )
    # Matches a long clause (>=40 chars) ending just before a coordinating
    # conjunction — used to inject a soft pause inside run-on sentences.
    _CONJUNCTION_RE = re.compile(
        r"([^.!?,\n]{40,})\s+\b(and|but|so)\b\s+([a-z])",
        re.IGNORECASE,
    )

    # ── Contraction expansion (formal → spoken) ──────────────────────────
    _CONTRACTIONS: list[tuple[re.Pattern, str]] = [
        (re.compile(r"\bwill not\b",          re.I), "won't"),
        (re.compile(r"\bcannot\b",            re.I), "can't"),
        (re.compile(r"\bdo not\b",            re.I), "don't"),
        (re.compile(r"\bdoes not\b",          re.I), "doesn't"),
        (re.compile(r"\bdid not\b",           re.I), "didn't"),
        (re.compile(r"\bshould not\b",        re.I), "shouldn't"),
        (re.compile(r"\bwould not\b",         re.I), "wouldn't"),
        (re.compile(r"\bcould not\b",         re.I), "couldn't"),
        (re.compile(r"\bhas not\b",           re.I), "hasn't"),
        (re.compile(r"\bhave not\b",          re.I), "haven't"),
        (re.compile(r"\bhad not\b",           re.I), "hadn't"),
        (re.compile(r"\bI am\b",              re.I), "I'm"),
        (re.compile(r"\bI will\b",            re.I), "I'll"),
        (re.compile(r"\bI have\b",            re.I), "I've"),
        (re.compile(r"\bI would\b",           re.I), "I'd"),
        (re.compile(r"\bwe are\b",            re.I), "we're"),
        (re.compile(r"\bwe will\b",           re.I), "we'll"),
        (re.compile(r"\bwe have\b",           re.I), "we've"),
        (re.compile(r"\byou are\b",           re.I), "you're"),
        (re.compile(r"\byou will\b",          re.I), "you'll"),
        (re.compile(r"\byou have\b",          re.I), "you've"),
        (re.compile(r"\bthey are\b",          re.I), "they're"),
        (re.compile(r"\bthat is\b",           re.I), "that's"),
        (re.compile(r"\bit is\b",             re.I), "it's"),
        (re.compile(r"\bthere is\b",          re.I), "there's"),
        (re.compile(r"\bhe is\b",             re.I), "he's"),
        (re.compile(r"\bshe is\b",            re.I), "she's"),
        (re.compile(r"\bwhat is\b",           re.I), "what's"),
        (re.compile(r"\bhow is\b",            re.I), "how's"),
        (re.compile(r"\blet us\b",            re.I), "let's"),
    ]

    # ── Abbreviation → spoken form ───────────────────────────────────────
    _DOLLAR_RE   = re.compile(r"\$(\d[\d,]*)")
    _PCT_RE      = re.compile(r"(\d+(?:\.\d+)?)\s*%")
    _ABBREV: list[tuple[re.Pattern, str]] = [
        (re.compile(r"\bFE\b"),              "final expense"),
        (re.compile(r"\bfe\b"),              "final expense"),
        (re.compile(r"\byrs\b",    re.I),   "years"),
        (re.compile(r"\byr\b",     re.I),   "year"),
        (re.compile(r"\bmos?\b",   re.I),   "months"),
        (re.compile(r"\bapprox\.?",re.I),   "approximately"),
        (re.compile(r"\bvs\.?",    re.I),   "versus"),
        (re.compile(r"&"),                   "and"),
    ]

    # ── AI-slop openers to strip before TTS ────────────────────────────
    # LLMs habitually start with acknowledgment phrases that sound robotic
    # when spoken aloud.  Strip them so the actual content plays first.
    _AI_SLOP_RE = re.compile(
        r'^(?:Certainly|Absolutely|Of\s+course|Sure(?:\s+thing)?'
        r'|Great(?:\s+question)?|Fantastic|Awesome|No\s+problem'
        r'|Understood|Got\s+it|I\s+understand|I\s+hear\s+you'
        r'|That\s+makes\s+sense|I\s+appreciate\s+(?:that|your\s+\w+)'
        r'|Thank\s+you\s+for\s+(?:sharing|that|asking)'
        r'|I\'d\s+be\s+(?:happy|glad|delighted)\s+to(?:\s+help)?'
        r'|Allow\s+me\s+to|Let\s+me\s+(?:help|explain|clarify)'
        r'|I\s+totally\s+(?:understand|get\s+that)'
        r')(?:[,!.]\s*|\s+)',
        re.IGNORECASE,
    )

    # ── Markdown / formatting strip patterns ─────────────────────────────
    _MD_STRIP = re.compile(
        r"(\*{1,3}|_{1,3}|`+|~~)"
        r"|^\s*[-*•]\s+"
        r"|^\s*\d+[.):]\s+",
        re.MULTILINE
    )

    def __init__(self, config: Optional[TTSConfig] = None) -> None:
        self._config = config or get_tts_config()
        self._session: Optional[aiohttp.ClientSession] = None
        self._base_url = self._config.api_url.rstrip("/")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the HTTP session used for all TTS requests.

        Performs a mandatory health-check against the configured TTS URL
        so that a port misconfiguration causes an immediate, loud failure
        at startup rather than silently forwarding error-JSON as audio.
        """
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)

        # ── Startup health guard ────────────────────────────────────────────
        # Fail loudly if the TTS URL is wrong (e.g. pointing to the voicebot
        # itself instead of the Kokoro server).
        try:
            health_url = f"{self._base_url}/health"
            async with self._session.get(
                health_url, timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(
                        f"TTS health check at {health_url} returned HTTP {resp.status}: {body[:200]}"
                    )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"FATAL: Cannot reach TTS server at {self._base_url}/health — {exc}\n"
                f"Check TTS_API_URL in .env and confirm the Kokoro server is running on port 8001."
            ) from exc

        logger.info(
            "Kokoro TTS initialised – endpoint=%s  voice=%s",
            self._base_url,
            self._config.voice_id,
        )

    async def shutdown(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("Kokoro TTS handler shut down.")

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

        Text is pre-processed for natural prosody (pause injection,
        conjunction breaks) and then split at sentence boundaries so the
        first audio bytes reach the caller as soon as the first sentence is
        synthesised.  Each sentence is sent as its own TTS request.
        """
        preprocessed = self._preprocess_text(text)
        async for chunk in self._stream_by_sentence(preprocessed):
            yield chunk

    async def synthesize_full(self, text: str) -> bytes:
        """Return the complete synthesised audio as a single bytes object."""
        parts: list[bytes] = []
        async for chunk in self.synthesize_stream(text):
            parts.append(chunk.audio_bytes)
        return b"".join(parts)

    # ------------------------------------------------------------------
    # Internal: sentence-level chunking
    # ------------------------------------------------------------------

    async def _stream_by_sentence(self, text: str) -> AsyncIterator[TTSAudioChunk]:
        sentences = self._split_sentences(text)
        for idx, sentence in enumerate(sentences):
            is_last = idx == len(sentences) - 1
            audio = await self._synthesize_one(sentence)
            # Only apply a fade-out on the final chunk; mid-sentence chunks
            # get no fade-out so there is no audible silence gap between them.
            audio = self._apply_fades(
                audio, self._config.sample_rate,
                fade_out_ms=5.0 if is_last else 0.0,
            )
            yield TTSAudioChunk(
                audio_bytes=audio,
                sample_rate=self._config.sample_rate,
                is_last=is_last,
                latency_ms=0.0,
            )

    async def _synthesize_one(self, text: str) -> bytes:
        """Send a single TTS request and return raw PCM int16 bytes."""
        url = f"{self._base_url}/v1/tts/stream"
        payload = {
            "text": text,
            "voice_id": self._config.voice_id,
            "speed": self._config.speed,
        }
        start = time.perf_counter()
        try:
            async with self._session.post(url, json=payload) as resp:
                resp.raise_for_status()
                audio = await resp.read()
                elapsed = (time.perf_counter() - start) * 1000
                logger.debug("TTS synthesis: %.1f ms | %d bytes", elapsed, len(audio))
                return audio
        except Exception:
            logger.exception("TTS synthesis failed.")
            raise

    # ------------------------------------------------------------------
    # Text utilities
    # ------------------------------------------------------------------

    @classmethod
    def _preprocess_text(cls, text: str) -> str:
        """Normalise LLM output to natural spoken text before TTS synthesis.

        Passes in order:
        1. Strip markdown formatting (asterisks, bullets, numbered lists).
        2. Expand formal phrases to contractions for spoken naturalness.
        3. Replace domain abbreviations with spoken equivalents.
        4. Inject breath-pause commas after greetings and before long-clause
           conjunctions.
        5. Final cleanup (double commas/spaces).
        """
        # Pass 1 — strip AI slop openers (LLM filler that sounds robotic aloud)
        text = cls._AI_SLOP_RE.sub("", text).lstrip(", !.")
        # Pass 2 — strip markdown
        text = cls._MD_STRIP.sub("", text)

        # Pass 3 — expand contractions (formal → spoken)
        for pattern, replacement in cls._CONTRACTIONS:
            text = pattern.sub(replacement, text)

        # Pass 4 — abbreviations + currency/percent
        text = cls._DOLLAR_RE.sub(lambda m: m.group(1) + " dollars", text)
        text = cls._PCT_RE.sub(lambda m: m.group(1) + " percent", text)
        for pattern, replacement in cls._ABBREV:
            text = pattern.sub(replacement, text)

        # Pass 5a — greeting pause
        text = cls._GREETING_RE.sub(lambda m: m.group(0).rstrip() + ", ", text)

        # Pass 5b — conjunction pause in long clauses
        def _maybe_break(m: re.Match) -> str:
            before, conj, first_char = m.group(1), m.group(2), m.group(3)
            if len(before.strip()) >= 40:
                return f"{before}, {conj} {first_char}"
            return m.group(0)

        text = cls._CONJUNCTION_RE.sub(_maybe_break, text)

        # Pass 6 — cleanup
        text = re.sub(r",\s*,", ",", text)
        text = re.sub(r"  +", " ", text)
        return text.strip()

    @classmethod
    def _split_sentences(cls, text: str) -> list[str]:
        """Split text at sentence boundaries, hard-limiting chunks to 180 chars."""
        parts = cls._SENTENCE_SPLIT.split(text.strip())
        merged: list[str] = []
        buf = ""
        for part in parts:
            # Merge sentences up to 300 chars so that short responses are
            # synthesised in one request — fewer HTTP round-trips means
            # no audible gaps between sentences.
            if len(buf) + len(part) < 300:
                buf = f"{buf} {part}".strip() if buf else part
            else:
                if buf:
                    merged.append(buf)
                buf = part
        if buf:
            merged.append(buf)

        result: list[str] = []
        for seg in merged or [text]:
            while len(seg) > 180:
                cut = 180
                for delim_re in (
                    re.compile(r",(?=[^,]{0,60}$)"),
                    re.compile(r"\b(?:and|but|so)\b(?=[^.!?]{0,60}$)", re.I),
                ):
                    m = None
                    for m in delim_re.finditer(seg[:180]):
                        pass
                    if m:
                        cut = m.end()
                        break
                result.append(seg[:cut].strip())
                seg = seg[cut:].strip()
            if seg:
                result.append(seg)
        return result

    # ------------------------------------------------------------------
    # Audio post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_fades(
        pcm_bytes: bytes,
        sample_rate: int,
        fade_in_ms: float = 3.0,
        fade_out_ms: float = 5.0,
    ) -> bytes:
        """Apply a linear fade-in and fade-out to raw int16 PCM.

        Eliminates click/pop artefacts at sentence boundaries.
        """
        if not pcm_bytes:
            return pcm_bytes

        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        n = len(samples)

        fade_in_n  = min(int(sample_rate * fade_in_ms  / 1000), n // 4)
        fade_out_n = min(int(sample_rate * fade_out_ms / 1000), n // 4)

        if fade_in_n > 0:
            samples[:fade_in_n] *= np.linspace(0.0, 1.0, fade_in_n)
        if fade_out_n > 0:
            samples[-fade_out_n:] *= np.linspace(1.0, 0.0, fade_out_n)

        return samples.clip(-32768, 32767).astype(np.int16).tobytes()
