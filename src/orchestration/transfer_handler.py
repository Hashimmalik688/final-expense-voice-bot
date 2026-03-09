"""
Transfer handler — manages warm-transfer of calls to human closers.

Coordinates with VICIdial to route the call to the correct in-group,
plays hold music / bridge audio, and passes lead context so the human
agent sees the conversation summary on screen-pop.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from config.settings import VICIdialConfig, get_vicidial_config
from src.orchestration.conversation_engine import ConversationState

logger = logging.getLogger(__name__)


@dataclass
class TransferResult:
    """Outcome of a transfer attempt."""

    success: bool
    closer_agent: str
    transfer_type: str  # "warm" | "blind" | "ivr"
    error: str = ""


class TransferHandler:
    """Encapsulates the logic for transferring a live call.

    Supports:
    - **Warm transfer**: bot stays on the line briefly to introduce the
      closer, then drops.
    - **Blind transfer**: call is sent directly to the closer queue.
    - **IVR transfer**: call is sent to an IVR extension (e.g. for
      after-hours fallback).

    The handler interacts with the VICIdial Agent API (via ``agent_api``)
    and the SIP handler to execute the actual media transfer.
    """

    def __init__(
        self,
        vicidial_config: Optional[VICIdialConfig] = None,
    ) -> None:
        self._config = vicidial_config or get_vicidial_config()
        self._agent_api = None  # Injected after initialization

    def set_agent_api(self, agent_api: "AgentAPI") -> None:  # noqa: F821
        """Inject the VICIdial agent API client."""
        self._agent_api = agent_api

    # ------------------------------------------------------------------
    # Transfer methods
    # ------------------------------------------------------------------

    async def warm_transfer(
        self,
        call_id: str,
        state: ConversationState,
    ) -> TransferResult:
        """Perform a warm (attended) transfer to a human closer.

        Steps:
        1. Build a screen-pop summary from conversation state.
        2. Request an available closer from VICIdial.
        3. Bridge the prospect into the closer's call.
        4. Drop the bot leg after a brief overlap.
        """
        logger.info("Initiating warm transfer for call %s", call_id)

        summary = self._build_lead_summary(state)

        try:
            # Request transfer via VICIdial API
            if self._agent_api:
                transfer_ok = await self._agent_api.transfer_call(
                    call_id=call_id,
                    ingroup=self._config.closer_ingroup,
                    extension=self._config.transfer_extension,
                    lead_data=summary,
                )
            else:
                logger.warning("Agent API not available – simulating transfer.")
                transfer_ok = True

            if transfer_ok:
                logger.info("Warm transfer succeeded for call %s", call_id)
                return TransferResult(
                    success=True,
                    closer_agent=self._config.closer_ingroup,
                    transfer_type="warm",
                )
            else:
                logger.error("Warm transfer failed for call %s", call_id)
                return TransferResult(
                    success=False,
                    closer_agent="",
                    transfer_type="warm",
                    error="VICIdial transfer request failed.",
                )

        except Exception as exc:
            logger.exception("Transfer error for call %s", call_id)
            return TransferResult(
                success=False,
                closer_agent="",
                transfer_type="warm",
                error=str(exc),
            )

    async def blind_transfer(
        self,
        call_id: str,
        extension: str,
    ) -> TransferResult:
        """Perform a blind (unattended) transfer to *extension*."""
        logger.info("Blind transfer: call %s → ext %s", call_id, extension)
        try:
            if self._agent_api:
                ok = await self._agent_api.blind_transfer(call_id, extension)
            else:
                ok = True
            return TransferResult(success=ok, closer_agent=extension, transfer_type="blind")
        except Exception as exc:
            logger.exception("Blind transfer error for call %s", call_id)
            return TransferResult(
                success=False, closer_agent="", transfer_type="blind", error=str(exc)
            )

    async def ivr_transfer(self, call_id: str, ivr_extension: str) -> TransferResult:
        """Transfer to an IVR/voicemail extension for after-hours."""
        logger.info("IVR transfer: call %s → %s", call_id, ivr_extension)
        try:
            if self._agent_api:
                ok = await self._agent_api.blind_transfer(call_id, ivr_extension)
            else:
                ok = True
            return TransferResult(success=ok, closer_agent=ivr_extension, transfer_type="ivr")
        except Exception as exc:
            logger.exception("IVR transfer error for call %s", call_id)
            return TransferResult(
                success=False, closer_agent="", transfer_type="ivr", error=str(exc)
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_lead_summary(state: ConversationState) -> dict:
        """Build a concise summary dict to pass to the closer's screen-pop."""
        recent_history = state.history[-6:]  # Last 3 exchanges
        conversation_snippet = "\n".join(
            f"{'Prospect' if m['role'] == 'user' else 'Bot'}: {m['content']}"
            for m in recent_history
        )
        return {
            "call_id": state.call_id,
            "lead_name": state.lead_name,
            "state": state.state,
            "stage_reached": state.current_stage,
            "turn_count": state.turn_count,
            "beneficiary": state.beneficiary,
            "collected_fields": state.collected_fields,
            "conversation_snippet": conversation_snippet,
        }
