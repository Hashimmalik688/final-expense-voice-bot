"""Local LiveKit plugins for the Final Expense Voice Bot.

Wraps on-device models as LiveKit Agents plugin interfaces:
  - ParakeetSTT  → NVIDIA Parakeet TDT 0.6B v3
  - VllmLLM      → Llama 3.1 8B via vLLM (OpenAI-compat API)
  - QwenTTS      → Qwen3-TTS / XTTS v2 via local TTS server

Pattern based on CoreWorxLab/local-livekit-plugins (FasterWhisperSTT + PiperTTS).
"""

from src.plugins.parakeet_stt import ParakeetSTT
from src.plugins.vllm_llm import VllmLLM
from src.plugins.qwen_tts import QwenTTS

__all__ = ["ParakeetSTT", "VllmLLM", "QwenTTS"]
