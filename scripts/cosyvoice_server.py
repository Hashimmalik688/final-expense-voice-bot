"""
Kokoro TTS HTTP server  (replaces CosyVoice).

Exposes:
  GET  /health            — liveness probe
  POST /v1/tts/stream     — synthesise speech, return raw int16 PCM at 24 kHz

Environment variables:
  KOKORO_VOICE   — Kokoro voice ID (default: af_sarah  — American female)
  TTS_PORT       — port to listen on (default: 8001)
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
import uvicorn
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
VOICE       = os.environ.get("KOKORO_VOICE", "af_heart")  # warmest/most natural American female
SAMPLE_RATE = 24000   # Kokoro always outputs 24 kHz
DEVICE      = "cpu"   # vLLM occupies the GPU; keep TTS on CPU

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
app = FastAPI(title="CosyVoice2 TTS Server")


@app.on_event("startup")
def on_startup():
    t = threading.Thread(target=_load_pipeline, daemon=True)
    t.start()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _pipeline is not None,
            "voice": VOICE, "sample_rate": SAMPLE_RATE}


class TTSRequest(BaseModel):
    text:     str
    voice_id: Optional[str] = "af_sarah"
    speed:    Optional[float] = 1.0


@app.post("/v1/tts/stream")
def synthesise(req: TTSRequest) -> Response:
    """Synthesise *text* and return raw int16 PCM at 24 kHz mono."""
    pipeline = _load_pipeline()

    # voice_id from request, else env default
    voice = req.voice_id or VOICE
    # Map friendly names → Kokoro voice IDs
    _voice_map = {
        "friendly_female": "af_heart",
        "heart":           "af_heart",
        "sarah":           "af_sarah",
        "sky":             "af_sky",
        "bella":           "af_bella",
        "nova":            "af_nova",
        "default":         "af_heart",
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

        pcm = np.concatenate(chunks)

        # Strip the leading breath/aspirate that Kokoro sometimes prepends.
        # Find the first sample where amplitude exceeds 2 % of peak, then
        # back-off 5 ms so we don't clip the onset of the first word.
        peak = np.max(np.abs(pcm))
        if peak > 0:
            threshold = peak * 0.02           # 2 % of peak
            onset_indices = np.where(np.abs(pcm) > threshold)[0]
            if len(onset_indices):
                onset = max(0, onset_indices[0] - int(SAMPLE_RATE * 0.005))  # 5 ms back-off
                pcm = pcm[onset:]

        # Apply a short 8 ms linear fade-in to eliminate any remaining hard click
        fade_samples = int(SAMPLE_RATE * 0.008)
        if len(pcm) > fade_samples * 2:
            fade = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            pcm[:fade_samples] *= fade

        pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
        return Response(content=pcm_int16.tobytes(),
                        media_type="application/octet-stream")

    except Exception as exc:
        logger.exception("TTS synthesis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("TTS_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
