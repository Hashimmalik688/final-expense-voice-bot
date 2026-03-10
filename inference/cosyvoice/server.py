"""
CosyVoice 2 FastAPI inference server.

Exposes:
  GET  /health         → {"status": "ok", "model": "<model_id>"}
  POST /synthesize     → WAV audio bytes (streaming or one-shot)

Environment variables:
  COSYVOICE_MODEL   HuggingFace model id  (default: FunAudioLLM/CosyVoice2-0.5B)
  HF_HOME           HuggingFace cache dir (default: ~/.cache/huggingface)
  PORT              Listen port           (default: 8001)
"""

from __future__ import annotations

import io
import logging
import os
import struct
import wave
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("cosyvoice-server")

MODEL_ID = os.getenv("COSYVOICE_MODEL", "FunAudioLLM/CosyVoice2-0.5B")
PORT = int(os.getenv("PORT", "8001"))

cosyvoice_model = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global cosyvoice_model
    logger.info("Loading CosyVoice 2 model: %s …", MODEL_ID)
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore[import-untyped]

        cosyvoice_model = CosyVoice2(MODEL_ID)
        logger.info("CosyVoice 2 model loaded.")
    except Exception:
        logger.exception("CosyVoice 2 failed to load — /synthesize will return 503.")
    yield
    cosyvoice_model = None


app = FastAPI(title="CosyVoice 2 TTS Server", lifespan=lifespan)


class SynthesizeRequest(BaseModel):
    text: str
    voice_id: str = "Sarah"
    stream: bool = False
    speed: float = 1.0


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 22050, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


@app.get("/health")
async def health():
    if cosyvoice_model is None:
        return Response(status_code=503, content='{"status":"loading"}', media_type="application/json")
    return {"status": "ok", "model": MODEL_ID}


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    if cosyvoice_model is None:
        return Response(status_code=503, content='{"error":"model not loaded"}', media_type="application/json")

    try:
        import numpy as np  # type: ignore[import-untyped]

        audio_chunks = []
        for result in cosyvoice_model.inference_sft(req.text, req.voice_id, stream=False, speed=req.speed):
            audio_chunks.append(result["tts_speech"].numpy())

        if not audio_chunks:
            return Response(status_code=500, content='{"error":"no audio generated"}', media_type="application/json")

        audio = np.concatenate(audio_chunks)
        # Convert float32 [-1,1] → int16
        pcm = (audio * 32767).clip(-32767, 32767).astype("int16").tobytes()
        wav_bytes = _pcm_to_wav(pcm, sample_rate=cosyvoice_model.sample_rate)
        return Response(content=wav_bytes, media_type="audio/wav")

    except Exception as exc:
        logger.exception("Synthesis failed: %s", exc)
        return Response(status_code=500, content=f'{{"error":"{exc}"}}', media_type="application/json")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
