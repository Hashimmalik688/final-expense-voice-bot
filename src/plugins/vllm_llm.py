"""VllmLLM — LiveKit LLM plugin wrapping Llama 3.1 8B via vLLM.

Wraps the existing LLMClient (which calls vLLM's OpenAI-compatible API)
as a livekit.agents.llm.LLM plugin.

Key design decisions:
  - system_prompt is stored as an instance attribute so the agent can
    refresh it per-turn as ConversationEngine advances the stage
  - Sentences are yielded as soon as each sentence boundary is detected
    (first-sentence-dispatch) to minimise perceived TTS latency
  - Hard cap of 3 sentences / 1 question per turn is enforced here,
    matching the original call_manager.py behaviour

Latency philosophy: we do NOT optimise for minimum TTFB. A 1-2 s pause
before Sarah responds sounds human and natural. The vLLM server is
already running warm so generation is fast; any extra latency comes
from the deliberate min_endpointing_delay=1.0 in AgentSession.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from livekit.agents import llm

__all__ = ["VllmLLM"]

logger = logging.getLogger(__name__)

# Sentence boundary pattern (period/question/exclamation followed by space or EOL)
_SENT_RE = re.compile(r'(?<=[.!?])(?:\s|$)')


class VllmLLM(llm.LLM):
    """LiveKit LLM plugin wrapping Llama 3.1 8B via vLLM."""

    def __init__(self, system_prompt: str = "") -> None:
        super().__init__()
        # Mutable: refreshed by the agent on each turn as stage advances
        self._system_prompt = system_prompt

        from src.llm.llm_client import LLMClient
        from config.settings import get_llm_config
        self._client = LLMClient(get_llm_config())

    def chat(
        self,
        chat_ctx: llm.ChatContext,
        fnc_ctx: Optional[llm.FunctionContext] = None,
        **kwargs,
    ) -> "VllmLLMStream":
        return VllmLLMStream(self, chat_ctx)


class VllmLLMStream(llm.LLMStream):
    """Streaming token output from vLLM, yielded sentence-by-sentence."""

    def __init__(self, llm_instance: VllmLLM, chat_ctx: llm.ChatContext) -> None:
        super().__init__(llm_instance, chat_ctx=chat_ctx, fnc_ctx=None)
        self._llm = llm_instance

    async def _run(self) -> None:
        messages = self._build_messages()
        buf = ""
        turn_sentences = 0
        turn_questions = 0

        try:
            async for token in self._llm._client.generate_stream(
                system_prompt=self._llm._system_prompt,
                messages=messages,
            ):
                buf += token

                # Dispatch complete sentences immediately (first-sentence dispatch)
                while True:
                    m = _SENT_RE.search(buf)
                    if not m:
                        break

                    sentence = buf[: m.start() + 1].strip()
                    buf = buf[m.end():]

                    if not sentence:
                        continue

                    # --- 3-sentence cap per turn ---
                    if turn_sentences >= 3:
                        buf = ""  # discard rest of stream
                        break

                    # --- 1-question cap per turn ---
                    if "?" in sentence:
                        if turn_questions >= 1:
                            sentence = sentence[: sentence.index("?") + 1]
                            buf = ""
                        turn_questions += 1

                    turn_sentences += 1
                    self._event_ch.send_nowait(
                        llm.ChatChunk(
                            choices=[
                                llm.Choice(
                                    delta=llm.ChoiceDelta(
                                        role="assistant",
                                        content=sentence + " ",
                                    )
                                )
                            ]
                        )
                    )

        except Exception as exc:
            logger.error("VllmLLMStream error: %s", exc, exc_info=True)

        # Flush any remaining buffer text (respects caps)
        if buf.strip() and turn_sentences < 3:
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    choices=[
                        llm.Choice(
                            delta=llm.ChoiceDelta(
                                role="assistant",
                                content=buf.strip(),
                            )
                        )
                    ]
                )
            )

    def _build_messages(self) -> list[dict]:
        """Convert LiveKit ChatContext to vLLM messages list.

        The system prompt is NOT included here — LLMClient prepends it
        internally via _build_payload(system_prompt=..., messages=...).
        """
        msgs: list[dict] = []
        for msg in self._chat_ctx.messages:
            role_str = str(getattr(msg, "role", "")).lower()
            content = str(getattr(msg, "content", "") or "")
            if "user" in role_str:
                msgs.append({"role": "user", "content": content})
            elif "assistant" in role_str:
                msgs.append({"role": "assistant", "content": content})
            # system messages are handled by LLMClient via _system_prompt
        return msgs
