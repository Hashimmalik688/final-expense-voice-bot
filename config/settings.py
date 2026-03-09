"""
Configuration management for the Final Expense Voice Bot.

Loads settings from environment variables and .env files with
sensible defaults for local development and production deployment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
SALES_SCRIPT_PATH = CONFIG_DIR / "sales_script.yaml"
KNOWLEDGE_BASE_PATH = CONFIG_DIR / "knowledge_base.json"

load_dotenv(PROJECT_ROOT / ".env")


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int = 0) -> int:
    return int(os.getenv(key, str(default)))


def _env_float(key: str, default: float = 0.0) -> float:
    return float(os.getenv(key, str(default)))


def _env_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Dataclass configuration groups
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SIPConfig:
    """SIP / Asterisk connection settings."""

    server: str = field(default_factory=lambda: _env("SIP_SERVER", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("SIP_PORT", 5060))
    local_ip: str = field(default_factory=lambda: _env("SIP_LOCAL_IP", "127.0.0.1"))
    username: str = field(default_factory=lambda: _env("SIP_USERNAME", "voicebot"))
    password: str = field(default_factory=lambda: _env("SIP_PASSWORD", ""))
    transport: str = field(default_factory=lambda: _env("SIP_TRANSPORT", "udp"))
    codec: str = field(default_factory=lambda: _env("SIP_CODEC", "opus"))


@dataclass(frozen=True)
class VICIdialConfig:
    """VICIdial API settings."""

    api_url: str = field(default_factory=lambda: _env("VICIDIAL_API_URL", "http://127.0.0.1/vicidial"))
    api_user: str = field(default_factory=lambda: _env("VICIDIAL_API_USER", ""))
    api_pass: str = field(default_factory=lambda: _env("VICIDIAL_API_PASS", ""))
    campaign_id: str = field(default_factory=lambda: _env("VICIDIAL_CAMPAIGN_ID", ""))
    agent_user: str = field(default_factory=lambda: _env("VICIDIAL_AGENT_USER", "voicebot"))
    transfer_extension: str = field(default_factory=lambda: _env("VICIDIAL_TRANSFER_EXT", "8300"))
    closer_ingroup: str = field(default_factory=lambda: _env("VICIDIAL_CLOSER_INGROUP", "CLOSERS"))


@dataclass(frozen=True)
class STTConfig:
    """Parakeet TDT speech-to-text settings."""

    model_name: str = field(default_factory=lambda: _env("STT_MODEL", "nvidia/parakeet-tdt-0.6b-v2"))
    device: str = field(default_factory=lambda: _env("STT_DEVICE", "cuda"))
    sample_rate: int = field(default_factory=lambda: _env_int("STT_SAMPLE_RATE", 16000))
    chunk_duration_ms: int = field(default_factory=lambda: _env_int("STT_CHUNK_MS", 160))
    beam_size: int = field(default_factory=lambda: _env_int("STT_BEAM_SIZE", 4))
    language: str = field(default_factory=lambda: _env("STT_LANGUAGE", "en"))


@dataclass(frozen=True)
class LLMConfig:
    """Mimo v2 Flash LLM / vLLM settings."""

    model_name: str = field(default_factory=lambda: _env("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct"))
    vllm_api_url: str = field(default_factory=lambda: _env("VLLM_API_URL", "http://127.0.0.1:8000"))
    max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 256))
    temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.6))
    top_p: float = field(default_factory=lambda: _env_float("LLM_TOP_P", 0.9))
    max_concurrent: int = field(default_factory=lambda: _env_int("LLM_MAX_CONCURRENT", 25))
    timeout_s: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_S", 5.0))


@dataclass(frozen=True)
class TTSConfig:
    """CosyVoice 2 text-to-speech settings."""

    model_name: str = field(default_factory=lambda: _env("TTS_MODEL", "CosyVoice2-0.5B"))
    api_url: str = field(default_factory=lambda: _env("TTS_API_URL", "http://127.0.0.1:8001"))
    voice_id: str = field(default_factory=lambda: _env("TTS_VOICE_ID", "friendly_female"))
    sample_rate: int = field(default_factory=lambda: _env_int("TTS_SAMPLE_RATE", 22050))
    speed: float = field(default_factory=lambda: _env_float("TTS_SPEED", 1.0))
    streaming: bool = field(default_factory=lambda: _env_bool("TTS_STREAMING", True))


@dataclass(frozen=True)
class RAGConfig:
    """Retrieval-Augmented Generation settings."""

    knowledge_base_path: Path = field(default=KNOWLEDGE_BASE_PATH)
    top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 3))
    similarity_threshold: float = field(default_factory=lambda: _env_float("RAG_SIMILARITY_THRESHOLD", 0.4))


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    latency_target_ms: int = field(default_factory=lambda: _env_int("LATENCY_TARGET_MS", 500))
    api_host: str = field(default_factory=lambda: _env("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: _env_int("API_PORT", 9000))
    max_call_duration_s: int = field(default_factory=lambda: _env_int("MAX_CALL_DURATION_S", 600))
    silence_timeout_s: int = field(default_factory=lambda: _env_int("SILENCE_TIMEOUT_S", 10))
    sales_script_path: Path = field(default=SALES_SCRIPT_PATH)
    knowledge_base_path: Path = field(default=KNOWLEDGE_BASE_PATH)


# ---------------------------------------------------------------------------
# Singleton-style helpers
# ---------------------------------------------------------------------------

def get_config() -> AppConfig:
    """Return the application configuration."""
    return AppConfig()


def get_sip_config() -> SIPConfig:
    return SIPConfig()


def get_vicidial_config() -> VICIdialConfig:
    return VICIdialConfig()


def get_stt_config() -> STTConfig:
    return STTConfig()


def get_llm_config() -> LLMConfig:
    return LLMConfig()


def get_tts_config() -> TTSConfig:
    return TTSConfig()


def get_rag_config() -> RAGConfig:
    return RAGConfig()
