"""
Kokoro TTS HTTP server  (replaces CosyVoice).

Exposes:
  GET  /health            — liveness probe
  POST /v1/tts/stream     — synthesise speech, return raw int16 PCM at 24 kHz

Environment variables:
  KOKORO_VOICE   — Kokoro voice ID (default: af_heart  — American female)
  TTS_PORT       — port to listen on (default: 8001)
  TTS_DEVICE     — cuda or cpu (default: cuda)
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
import uvicorn
from scipy.signal import resample_poly
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("kokoro_server")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
import os
VOICE       = os.environ.get("KOKORO_VOICE", "sarah_warm")  # warm+clear American female blend
SAMPLE_RATE = 24000   # Kokoro native output rate
DEVICE      = os.environ.get("TTS_DEVICE", "cuda")         # use GPU when available

# ---------------------------------------------------------------------------
# Load pipeline at startup (once, thread-safe)
# ---------------------------------------------------------------------------
_lock     = threading.Lock()
_pipeline = None   # kokoro.KPipeline instance


def _load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _lock:
        if _pipeline is not None:
            return _pipeline
        logger.info("Loading Kokoro pipeline (voice=%s, device=%s) …", VOICE, DEVICE)
        from kokoro import KPipeline  # type: ignore
        _pipeline = KPipeline(lang_code="a", device=DEVICE)  # 'a' = American English
        logger.info("Kokoro pipeline ready.")
    return _pipeline


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Kokoro TTS Server")


@app.on_event("startup")
def on_startup():
    t = threading.Thread(target=_load_pipeline, daemon=True)
    t.start()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _pipeline is not None,
            "voice": VOICE, "sample_rate": SAMPLE_RATE}


@app.get("/v1/voices")
def list_voices():
    """Return available voice IDs and their underlying Kokoro voice strings."""
    return {
        "voices": [
            {"id": "sarah_warm",    "kokoro": "af_heart,af_sarah", "description": "Warm, clear American female — Sarah persona (default)"},
            {"id": "heart",         "kokoro": "af_heart",                   "description": "af_heart — warmest American female"},
            {"id": "sarah",         "kokoro": "af_sarah",                   "description": "af_sarah — clear, natural American female"},
            {"id": "sky",           "kokoro": "af_sky",                    "description": "af_sky — lighter American female"},
            {"id": "bella",         "kokoro": "af_bella",                  "description": "af_bella"},
            {"id": "nova",          "kokoro": "af_nova",                   "description": "af_nova"},
            {"id": "friendly_female","kokoro": "af_heart,af_sarah", "description": "Alias for sarah_warm"},
        ]
    }


class TTSRequest(BaseModel):
    text:        str
    voice_id:    Optional[str]   = None    # None → use server default (VOICE env var)
    speed:       Optional[float] = 1.0
    sample_rate: Optional[int]   = None   # resample output if != 24000


@app.post("/v1/tts/stream")
def synthesise(req: TTSRequest) -> Response:
    """Synthesise *text* and return raw int16 PCM at 24 kHz mono."""
    pipeline = _load_pipeline()

    # voice_id from request, else env default
    voice = req.voice_id or VOICE
    # Map friendly names → Kokoro voice IDs (including voice blends)
    _voice_map = {
        "sarah_warm":     "af_heart,af_sarah",   # warm + clear Sarah blend (default)
        "friendly_female":"af_heart,af_sarah",   # alias
        "heart":          "af_heart",
        "sarah":          "af_sarah",
        "sky":            "af_sky",
        "bella":          "af_bella",
        "nova":           "af_nova",
        "default":        "af_heart,af_sarah",
    }
    voice = _voice_map.get(voice.lower(), voice)

    try:
        chunks: list[np.ndarray] = []
        for _, _, audio in pipeline(req.text, voice=voice,
                                    speed=req.speed or 1.0):
            if hasattr(audio, "numpy"):
                audio = audio.numpy()
            chunks.append(np.squeeze(audio))

        if not chunks:
            raise HTTPException(status_code=500, detail="Empty TTS output")

        pcm = np.concatenate(chunks).astype(np.float32)

        # ── 1. Trim leading silence / breath artifact ──────────────────────
        peak = float(np.max(np.abs(pcm)))
        if peak > 0:
            # find first sample above 2 % of peak, back off 5 ms
            threshold = peak * 0.02
            onset_indices = np.where(np.abs(pcm) > threshold)[0]
            if len(onset_indices):
                onset = max(0, onset_indices[0] - int(SAMPLE_RATE * 0.005))
                pcm = pcm[onset:]

        # ── 2. Normalise to -1 dBFS for consistent, clear volume ───────────
        peak = float(np.max(np.abs(pcm)))
        if peak > 0:
            pcm = pcm * (0.891 / peak)   # 0.891 ≈ -1 dBFS

        # ── 3. Soft-knee limiter: eliminate any stray clips ─────────────────
        pcm = np.tanh(pcm * 1.5) / np.tanh(np.array(1.5, dtype=np.float32))

        # ── 4. 8 ms linear fade-in to avoid hard transients at start ────────
        fade_samples = int(SAMPLE_RATE * 0.008)
        if len(pcm) > fade_samples * 2:
            pcm[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)

        # ── 5. Convert to int16 ─────────────────────────────────────────────
        pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)

        # ── 6. Resample if the caller requested a different rate ─────────────
        out_sr = req.sample_rate or SAMPLE_RATE
        if out_sr != SAMPLE_RATE:
            from math import gcd
            g = gcd(SAMPLE_RATE, out_sr)
            up, down = out_sr // g, SAMPLE_RATE // g
            resampled = resample_poly(
                pcm_int16.astype(np.float32) / 32768.0, up, down
            )
            resampled_int16 = (resampled * 32768.0).clip(-32768, 32767).astype(np.int16)
            return Response(content=resampled_int16.tobytes(),
                            media_type="application/octet-stream")

        return Response(content=pcm_int16.tobytes(),
                        media_type="application/octet-stream")

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("TTS synthesis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("TTS_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
