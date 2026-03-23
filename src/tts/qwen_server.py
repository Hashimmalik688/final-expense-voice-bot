"""TTS model server — Qwen3-TTS > XTTS v2 > Kokoro fallback chain.

Runs as a standalone FastAPI service on port 8002.
Multiple LiveKit agent workers share this one model instance.

Start: python src/tts/qwen_server.py
Health: curl http://127.0.0.1:8002/health

POST /synthesize  { "text": "...", "speaker": "Ana Florence" }
  → returns raw int16 PCM audio at 24 kHz, Content-Type: audio/pcm

The model chain is tried at startup (first import wins):
  1. Qwen/Qwen3-TTS  (transformers ≥4.42, ~1.5 GB)
  2. tts_models/multilingual/multi-dataset/xtts_v2  (TTS package, ~2 GB)
  3. Kokoro v0.9 (kokoro-onnx, local fallback)
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import sys
import wave

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("tts_server")

PORT = int(os.environ.get("TTS_SERVER_PORT", "8002"))
DEFAULT_SPEAKER = os.environ.get("TTS_SPEAKER", "Ana Florence")
SAMPLE_RATE = 24000

app = FastAPI(title="Local TTS Server", version="2.0.0")

# Global model and backend name
_tts_model = None
_backend = "none"


# ---------------------------------------------------------------------------
# Load model at startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def load_model() -> None:
    global _tts_model, _backend
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _load_best_model)
    _tts_model, _backend = result
    logger.info("TTS server ready — backend=%s", _backend)


def _load_best_model():
    """Try each TTS backend in order. Returns (model, backend_name)."""

    # --- 1. Qwen3-TTS ---
    try:
        logger.info("Trying Qwen3-TTS …")
        from transformers import AutoTokenizer, AutoModel  # type: ignore
        import torch  # type: ignore
        model_id = os.environ.get("QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS")
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        logger.info("Qwen3-TTS loaded on %s", device)
        return ({"type": "qwen3", "model": model, "tokenizer": tok, "device": device},
                "qwen3-tts")
    except Exception as exc:
        logger.warning("Qwen3-TTS not available: %s", exc)

    # --- 2. XTTS v2 (Coqui TTS) ---
    try:
        logger.info("Trying XTTS v2 …")
        from TTS.api import TTS  # type: ignore
        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
        logger.info("XTTS v2 loaded.")
        return ({"type": "xtts2", "model": model}, "xtts_v2")
    except Exception as exc:
        logger.warning("XTTS v2 not available: %s", exc)

    # --- 3. Kokoro (onnx, existing install) ---
    try:
        logger.info("Trying Kokoro ONNX …")
        from kokoro_onnx import Kokoro  # type: ignore
        voice = os.environ.get("KOKORO_VOICE", "af_heart")
        model_path = os.environ.get(
            "KOKORO_MODEL", "/workspace/models/kokoro/kokoro-v0_19.onnx"
        )
        voices_path = os.environ.get(
            "KOKORO_VOICES", "/workspace/models/kokoro/voices.bin"
        )
        model = Kokoro(model_path, voices_path)
        logger.info("Kokoro loaded.")
        return ({"type": "kokoro", "model": model, "voice": voice}, "kokoro")
    except Exception as exc:
        logger.warning("Kokoro not available: %s", exc)

    logger.error("All TTS backends failed — server will return 503 on synthesis.")
    return (None, "none")


# ---------------------------------------------------------------------------
# Synthesis helpers
# ---------------------------------------------------------------------------

def _synthesize_qwen3(state: dict, text: str, speaker: str) -> bytes:
    import torch  # type: ignore
    model = state["model"]
    tok = state["tokenizer"]
    device = state["device"]
    with torch.no_grad():
        inputs = tok(text, return_tensors="pt").to(device)
        audio = model.generate(**inputs)
        audio_np = audio.squeeze().cpu().float().numpy()
    return _float32_to_pcm(audio_np)


def _synthesize_xtts2(state: dict, text: str, speaker: str) -> bytes:
    model = state["model"]
    audio_list = model.tts(text=text, speaker=speaker, language="en")
    audio_np = np.array(audio_list, dtype=np.float32)
    return _float32_to_pcm(audio_np)


def _synthesize_kokoro(state: dict, text: str, speaker: str) -> bytes:
    model = state["model"]
    voice = state.get("voice", "af_heart")
    samples, sr = model.create(text, voice=voice, speed=1.0, lang="en-us")
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly  # type: ignore
        from math import gcd
        g = gcd(sr, SAMPLE_RATE)
        samples = resample_poly(samples, SAMPLE_RATE // g, sr // g)
    return _float32_to_pcm(samples.astype(np.float32))


def _float32_to_pcm(audio: np.ndarray) -> bytes:
    """Convert float32 [-1,1] numpy array to int16 PCM bytes at SAMPLE_RATE."""
    clamped = np.clip(audio, -1.0, 1.0)
    return (clamped * 32767).astype(np.int16).tobytes()


_SYNTHESIZERS = {
    "qwen3": _synthesize_qwen3,
    "xtts2": _synthesize_xtts2,
    "kokoro": _synthesize_kokoro,
}


def _run_synthesis(text: str, speaker: str) -> bytes:
    """Blocking synthesis — runs in executor."""
    if _tts_model is None:
        raise RuntimeError("No TTS backend loaded")
    synth_fn = _SYNTHESIZERS.get(_tts_model.get("type", ""))
    if synth_fn is None:
        raise RuntimeError(f"Unknown backend type: {_tts_model.get('type')}")
    return synth_fn(_tts_model, text, speaker)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

class SynthRequest(BaseModel):
    text: str
    speaker: str = DEFAULT_SPEAKER


@app.post("/synthesize")
async def synthesize(req: SynthRequest) -> Response:
    """Synthesise text → raw int16 PCM at 24 kHz."""
    if not req.text.strip():
        raise HTTPException(400, "text is empty")
    if _tts_model is None:
        raise HTTPException(503, f"TTS backend not ready (backend={_backend})")

    try:
        loop = asyncio.get_running_loop()
        pcm_bytes = await loop.run_in_executor(
            None, _run_synthesis, req.text, req.speaker
        )
    except Exception as exc:
        logger.error("TTS synthesis error: %s", exc, exc_info=True)
        raise HTTPException(500, f"Synthesis failed: {exc}") from exc

    return Response(content=pcm_bytes, media_type="audio/pcm")


@app.get("/health")
async def health():
    ok = _tts_model is not None
    return {
        "status": "ok" if ok else "degraded",
        "backend": _backend,
        "sample_rate": SAMPLE_RATE,
    }


@app.get("/speakers")
async def list_speakers():
    """List available speakers (XTTS v2 only)."""
    if _tts_model and _tts_model.get("type") == "xtts2":
        model = _tts_model["model"]
        speakers = getattr(model, "speakers", []) or []
        return {"speakers": speakers}
    return {"speakers": [DEFAULT_SPEAKER]}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting TTS server on port %d …", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
