"""
Filler (thinking) audio player.

When the LLM is generating tokens and no TTS audio has been sent yet, the
call_manager plays a short pre-recorded "thinking" clip to fill dead air.
The filler cancels the moment the first real TTS sentence is ready.

Clips are stored as μ-law 8 kHz mono .wav files under src/tts/fillers/.
They are generated once by running  scripts/generate_fillers.py  against the
live TTS server, then committed and reused on every restart.

If the fillers directory is empty or missing, playback is silently skipped —
the bot still works, it just has the small dead-air gap.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import wave
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ── Locate the fillers directory relative to this file ──────────────────────
_FILLERS_DIR = Path(__file__).resolve().parent / "fillers"

# μ-law 8 kHz chunk size (20 ms = 160 samples)
_CHUNK_SIZE = 160


class FillerPlayer:
    """Manages a rotating set of pre-recorded filler audio clips.

    The player loads all *.wav files from the fillers/ directory at startup.
    During a call, ``play_filler()`` starts playing a randomly selected clip
    through the provided *audio_sink*.  ``cancel()`` stops playback
    immediately regardless of position in the clip.

    All clips MUST be:
    - μ-law encoded
    - 8 000 Hz sample rate
    - mono
    (i.e. the same format Twilio expects on the outbound path)
    """

    def __init__(self) -> None:
        self._clips: list[bytes] = []
        self._current_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all *.wav clips from the fillers/ directory."""
        if not _FILLERS_DIR.exists():
            logger.warning("Fillers directory %s not found — filler playback disabled.", _FILLERS_DIR)
            return

        loaded = 0
        for wav_path in sorted(_FILLERS_DIR.glob("*.wav")):
            try:
                with wave.open(str(wav_path), "rb") as wf:
                    raw = wf.readframes(wf.getnframes())
                    self._clips.append(raw)
                    loaded += 1
            except Exception:
                logger.warning("Could not load filler clip %s", wav_path, exc_info=True)

        if loaded:
            logger.info("Filler player: loaded %d clip(s) from %s", loaded, _FILLERS_DIR)
        else:
            logger.warning("Filler player: no usable clips in %s", _FILLERS_DIR)

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    async def play_filler(
        self,
        audio_sink: Callable[[bytes], None],
        *,
        chunk_ms: float = 20.0,
    ) -> None:
        """Play a randomly chosen filler clip through *audio_sink*.

        Streams the clip in *chunk_ms* increments (default 20 ms = 160 bytes
        at 8 kHz μ-law) so it can be cancelled between chunks when real TTS
        audio arrives.

        This coroutine is normally run as an ``asyncio.Task`` so it can be
        cancelled cleanly from ``cancel()``.
        """
        if not self._clips:
            return

        clip = random.choice(self._clips)
        chunk_size = int(8000 * chunk_ms / 1000)  # samples = bytes for μ-law

        try:
            pos = 0
            while pos < len(clip):
                chunk = clip[pos : pos + chunk_size]
                pos += chunk_size
                if not chunk:
                    break
                audio_sink(chunk)
                # Yield control so cancel() can interrupt between chunks
                await asyncio.sleep(chunk_ms / 1000)
        except asyncio.CancelledError:
            # Clean cancel — normal operation when real TTS arrives
            pass

    def start(self, audio_sink: Callable[[bytes], None]) -> Optional[asyncio.Task]:
        """Launch filler playback as a background task.

        Returns the task so the caller can await it or cancel it.
        Returns ``None`` if no clips are loaded.
        """
        if not self._clips:
            return None
        self._current_task = asyncio.create_task(self.play_filler(audio_sink))
        return self._current_task

    def start_delayed(
        self,
        audio_sink: Callable[[bytes], None],
        delay_s: float = 0.40,
    ) -> Optional[asyncio.Task]:
        """Launch filler playback only after *delay_s* seconds of dead air.

        When the LLM responds in less than *delay_s* (common for short yes/no
        answers), ``_speak()`` cancels the task before it ever starts playing.
        This prevents the blip/click that occurred when filler started then was
        immediately cancelled after a single 20 ms chunk.

        Returns the task so the caller can await or cancel it.
        Returns ``None`` if no clips are loaded.
        """
        if not self._clips:
            return None

        async def _delayed() -> None:
            try:
                await asyncio.sleep(delay_s)
                await self.play_filler(audio_sink)
            except asyncio.CancelledError:
                pass  # cancelled before delay expired — totally normal

        self._current_task = asyncio.create_task(_delayed())
        return self._current_task

    def cancel(self) -> None:
        """Stop filler playback immediately (no-op if not playing)."""
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
        self._current_task = None
