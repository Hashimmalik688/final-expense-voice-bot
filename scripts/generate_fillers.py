#!/usr/bin/env python3
"""
Generate filler audio clips for the FillerPlayer.

Run this script ONCE after the TTS server is live to pre-synthesise
short "thinking" phrases and save them as μ-law 8 kHz mono WAV files
in src/tts/fillers/.

Usage (from repo root, with TTS server running):
    python scripts/generate_fillers.py

The script uses the same KokoroTTSHandler that the main bot uses,
so the filler voice will match the bot's persona exactly.
"""

from __future__ import annotations

import asyncio
import audioop
import os
import sys
import wave
from math import gcd
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tts.kokoro_handler import KokoroTTSHandler

# Output directory — relative to this script's parent/parent (repo root)
OUT_DIR = Path(__file__).resolve().parent.parent / "src" / "tts" / "fillers"

TTS_SAMPLE_RATE = 24_000   # Kokoro default; must match config/settings.py
OUTPUT_SAMPLE_RATE = 8_000  # Twilio μ-law rate

# Short "thinking" phrases — these fill the dead-air while the LLM generates.
# Keep them under 1 second; the average chosen by _speak will cancel this.
FILLERS = {
    "mm_hmm": "Mm-hmm.",
    "sure": "Sure.",
    "let_me_check": "Let me check that.",
    "one_moment": "One moment.",
    "i_see": "I see.",
}


def pcm_to_mulaw_wav(pcm_bytes: bytes, src_rate: int, out_path: Path) -> None:
    """Downsample PCM-16 from *src_rate* to 8 kHz and write μ-law WAV."""
    from scipy.signal import resample_poly
    import numpy as np

    g = gcd(src_rate, OUTPUT_SAMPLE_RATE)
    up = OUTPUT_SAMPLE_RATE // g
    down = src_rate // g

    samples = (
        np.frombuffer(pcm_bytes, dtype=np.int16)
        .astype(np.float32)
        / 32768.0
    )
    samples_8k = resample_poly(samples, up, down)
    pcm_8k = (
        (samples_8k * 32768.0)
        .clip(-32768, 32767)
        .astype(np.int16)
        .tobytes()
    )

    mulaw = audioop.lin2ulaw(pcm_8k, 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)          # μ-law = 1 byte per sample
        wf.setframerate(OUTPUT_SAMPLE_RATE)
        wf.writeframes(mulaw)

    dur_ms = int(len(mulaw) / OUTPUT_SAMPLE_RATE * 1000)
    print(f"  Wrote {out_path.name}  ({dur_ms} ms)")


async def main() -> None:
    print("Connecting to TTS server …")
    handler = KokoroTTSHandler()
    await handler.initialize()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for key, text in FILLERS.items():
        out_path = OUT_DIR / f"{key}.wav"
        print(f"Synthesising '{text}' → {out_path.name} …")
        pcm_chunks: list[bytes] = []
        async for chunk in handler.synthesize_stream(text):
            pcm_chunks.append(chunk.audio_bytes)
        if pcm_chunks:
            pcm_bytes = b"".join(pcm_chunks)
            pcm_to_mulaw_wav(pcm_bytes, TTS_SAMPLE_RATE, out_path)
        else:
            print(f"  WARNING: no audio returned for '{text}'")

    print("\nDone.  Clips are in:", OUT_DIR)


if __name__ == "__main__":
    asyncio.run(main())
