"""
Mimo v2 Flash LLM client served via vLLM.

Provides an async interface to the vLLM OpenAI-compatible API running the
Mimo v2 Flash model.  Supports streaming token generation so the TTS engine
can begin synthesising speech before the full response is ready.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import aiohttp

from config.settings import LLMConfig, get_llm_config

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Container for an LLM completion result."""

    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class MimoVLLMClient:
    """Async client for the Mimo model running on a vLLM server.

    The vLLM server exposes an **OpenAI-compatible** ``/v1/chat/completions``
    endpoint.  This client calls that endpoint with the system prompt,
    conversation history, and any RAG context injected by the orchestrator.

    Usage::

        client = MimoVLLMClient()
        await client.initialize()

        response = await client.generate(
            system_prompt="You are a helpful insurance agent …",
            messages=[{"role": "user", "content": "Tell me about coverage."}],
        )
        print(response.text)
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self._config = config or get_llm_config()
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent)
        self._base_url = self._config.vllm_api_url.rstrip("/")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._config.timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout)
            logger.info(
                "MimoVLLM client initialised – endpoint=%s  model=%s",
                self._base_url,
                self._config.model_name,
            )

    async def shutdown(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("MimoVLLM client shut down.")

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Return ``True`` if the vLLM server is reachable."""
        try:
            async with self._session.get(f"{self._base_url}/health") as resp:
                return resp.status == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Non-streaming generation
    # ------------------------------------------------------------------

    async def generate(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate a full (non-streamed) response.

        Parameters
        ----------
        system_prompt:
            The system-level instruction (persona + rules).
        messages:
            The conversation history in OpenAI chat format.
        temperature / max_tokens:
            Override the defaults from config if provided.
        """
        async with self._semaphore:
            return await self._call_chat(
                system_prompt=system_prompt,
                messages=messages,
                stream=False,
                temperature=temperature or self._config.temperature,
                max_tokens=max_tokens or self._config.max_tokens,
            )

    # ------------------------------------------------------------------
    # Streaming generation
    # ------------------------------------------------------------------

    async def generate_stream(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Yield tokens as they are generated (SSE stream).

        This allows the TTS engine to start synthesising speech before the
        entire response has been produced, reducing end-to-end latency.
        """
        async with self._semaphore:
            payload = self._build_payload(
                system_prompt=system_prompt,
                messages=messages,
                stream=True,
                temperature=temperature or self._config.temperature,
                max_tokens=max_tokens or self._config.max_tokens,
            )
            url = f"{self._base_url}/v1/chat/completions"

            try:
                async with self._session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.content:
                        decoded = line.decode("utf-8").strip()
                        if not decoded or not decoded.startswith("data: "):
                            continue
                        data_str = decoded[len("data: "):]
                        if data_str == "[DONE]":
                            break
                        import json
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
            except Exception:
                logger.exception("Streaming generation failed.")
                raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        stream: bool,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        return {
            "model": self._config.model_name,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": self._config.top_p,
            "stream": stream,
        }

    async def _call_chat(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        stream: bool,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        payload = self._build_payload(
            system_prompt=system_prompt,
            messages=messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        url = f"{self._base_url}/v1/chat/completions"
        start = time.perf_counter()

        try:
            async with self._session.post(url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except aiohttp.ClientError:
            logger.exception("vLLM request failed.")
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        choice = data["choices"][0]
        usage = data.get("usage", {})

        result = LLMResponse(
            text=choice["message"]["content"],
            finish_reason=choice.get("finish_reason", "unknown"),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=elapsed_ms,
        )
        logger.debug(
            "LLM response: %.1f ms | tokens=%d | reason=%s",
            result.latency_ms,
            result.completion_tokens,
            result.finish_reason,
        )
        return result
