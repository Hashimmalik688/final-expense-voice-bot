"""
Conversation engine — the "brain" of the voice bot.

Loads the editable ``sales_script.yaml``, manages conversation state, builds
LLM prompts with RAG context, and decides when to transition between script
stages (greeting → qualifying → presentation → transfer, etc.).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml

from config.settings import AppConfig, get_config
from src.llm.llm_client import LLMClient, LLMResponse
from src.llm.rag_engine import RAGEngine

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Script stage model
# ------------------------------------------------------------------

class CallAction(str, Enum):
    """Actions the conversation engine can emit to the call manager."""

    CONTINUE = "CONTINUE"
    TRANSFER_TO_CLOSER = "TRANSFER_TO_CLOSER"
    END_CALL = "END_CALL"
    DNC_AND_END_CALL = "DNC_AND_END_CALL"
    SCHEDULE_CALLBACK = "SCHEDULE_CALLBACK"


@dataclass
class TurnResult:
    """What the engine produces after processing one prospect turn."""

    bot_text: str
    action: CallAction
    current_stage: str
    latency_ms: float
    rag_chunks_used: int = 0


@dataclass
class ConversationState:
    """Mutable state carried across the lifetime of a single call."""

    call_id: str
    lead_name: str = "there"
    state: str = ""
    beneficiary: str = "your loved ones"
    coverage_amount: str = "$10,000"
    monthly_premium: str = "$30"
    current_stage: str = "greeting"
    turn_count: int = 0
    history: list[dict[str, str]] = field(default_factory=list)
    collected_fields: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)


class ConversationEngine:
    """Drives the conversation through the sales script stages.

    The engine is stateless across calls — each call receives its own
    ``ConversationState``.  The engine:

    1. Looks up the current stage in the loaded script.
    2. Queries the RAG engine for relevant knowledge.
    3. Builds a rich LLM prompt with persona, stage instructions, history,
       and RAG context.
    4. Calls the LLM and returns the bot's spoken text plus any action
       (transfer, end call, etc.).
    """

    def __init__(
        self,
        llm_client: LLMClient,
        rag_engine: RAGEngine,
        config: Optional[AppConfig] = None,
    ) -> None:
        self._llm = llm_client
        self._rag = rag_engine
        self._config = config or get_config()
        self._script: dict[str, Any] = {}
        self._persona: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_script(self, path: Optional[Path] = None) -> None:
        """Load the sales script YAML from disk."""
        script_path = path or self._config.sales_script_path
        try:
            raw = yaml.safe_load(script_path.read_text(encoding="utf-8"))
            self._script = raw.get("stages", {})
            self._persona = raw.get("bot_persona", {})
            logger.info("Sales script loaded: %d stages", len(self._script))
        except Exception:
            logger.exception("Failed to load sales script from %s", script_path)
            raise

    def reload_script(self) -> None:
        """Hot-reload the sales script (safe between calls)."""
        self.load_script()

    # ------------------------------------------------------------------
    # Per-call helpers
    # ------------------------------------------------------------------

    def new_call(self, call_id: str, lead_data: dict[str, str] | None = None) -> ConversationState:
        """Create a fresh conversation state for a new call."""
        lead = lead_data or {}
        return ConversationState(
            call_id=call_id,
            lead_name=lead.get("first_name", "there"),
            state=lead.get("state", ""),
            beneficiary=lead.get("beneficiary", "your loved ones"),
            coverage_amount=lead.get("coverage_amount", "$10,000"),
            monthly_premium=lead.get("monthly_premium", "$30"),
        )

    # ------------------------------------------------------------------
    # Main turn processing
    # ------------------------------------------------------------------

    async def process_turn(
        self,
        state: ConversationState,
        prospect_text: str,
    ) -> TurnResult:
        """Process one prospect utterance and return the bot's response.

        Parameters
        ----------
        state:
            The mutable call state (updated in-place).
        prospect_text:
            What the prospect said (from STT).

        Returns
        -------
        TurnResult
            The bot's spoken response, any call action, and timing.
        """
        start = time.perf_counter()
        state.turn_count += 1
        state.history.append({"role": "user", "content": prospect_text})

        # 1. RAG retrieval
        rag_chunks = self._rag.retrieve(prospect_text)
        rag_context = RAGEngine.format_context(rag_chunks)

        # 2. Build system prompt
        system_prompt = self._build_system_prompt(state, rag_context)

        # 3. Call LLM
        llm_response: LLMResponse = await self._llm.generate(
            system_prompt=system_prompt,
            messages=state.history,
        )

        bot_text = llm_response.text.strip()

        # 4. Determine action from script stage
        action = self._resolve_action(state, prospect_text, bot_text)

        # 5. Advance stage based on LLM + heuristic analysis
        self._advance_stage(state, prospect_text, bot_text)

        state.history.append({"role": "assistant", "content": bot_text})
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Turn %d | stage=%s | action=%s | latency=%.0f ms | call=%s",
            state.turn_count,
            state.current_stage,
            action.value,
            elapsed_ms,
            state.call_id,
        )

        return TurnResult(
            bot_text=bot_text,
            action=action,
            current_stage=state.current_stage,
            latency_ms=elapsed_ms,
            rag_chunks_used=len(rag_chunks),
        )

    async def get_opening(self, state: ConversationState) -> TurnResult:
        """Generate the bot's opening line without any prospect input."""
        start = time.perf_counter()
        stage_data = self._script.get(state.current_stage, {})
        opening_lines = stage_data.get("lines", [])

        if opening_lines:
            bot_text = self._fill_variables(opening_lines[0], state)
        else:
            system_prompt = self._build_system_prompt(state, "")
            llm_resp = await self._llm.generate(
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": "[Call connected – greet the prospect.]"}],
            )
            bot_text = llm_resp.text.strip()

        state.history.append({"role": "assistant", "content": bot_text})
        elapsed_ms = (time.perf_counter() - start) * 1000

        return TurnResult(
            bot_text=bot_text,
            action=CallAction.CONTINUE,
            current_stage=state.current_stage,
            latency_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_system_prompt(self, state: ConversationState, rag_context: str) -> str:
        persona = self._persona
        stage_data = self._script.get(state.current_stage, {})
        stage_goal = stage_data.get("goal", "Continue the conversation naturally.")

        rules_block = "\n".join(f"- {r}" for r in persona.get("rules", []))

        # Stage-specific lines or questions for few-shot guidance
        guidance_lines = stage_data.get("lines", [])
        questions = stage_data.get("questions", [])
        objections = stage_data.get("objections", {})

        guidance = ""
        if guidance_lines:
            filled = [self._fill_variables(l, state) for l in guidance_lines]
            guidance = "Example lines you can adapt:\n" + "\n".join(f"  - {l}" for l in filled)
        if questions:
            q_texts = [self._fill_variables(q["text"], state) for q in questions]
            guidance += "\nQuestions to ask:\n" + "\n".join(f"  - {q}" for q in q_texts)
        if objections:
            obj_lines = []
            for key, obj in objections.items():
                # New format: empathy / education / reframe / question
                if "empathy" in obj:
                    empathy = self._fill_variables(obj.get("empathy", ""), state)
                    education = self._fill_variables(obj.get("education", ""), state)
                    reframe = self._fill_variables(obj.get("reframe", ""), state)
                    question = self._fill_variables(obj.get("question", ""), state)
                    obj_lines.append(f"  [{key}]:\n    Empathy: {empathy}\n    Education: {education}\n    Reframe: {reframe}\n    Question: {question}")
                elif "response" in obj:
                    obj_lines.append(f"  [{key}]: {self._fill_variables(obj['response'], state)}")
            guidance += "\nObjection responses:\n" + "\n".join(obj_lines)

        prompt = f"""You are {persona.get('name', 'Sarah')}, a {persona.get('role', 'benefits coordinator')} at {persona.get('company', 'Senior Life Services')}.
Tone: {persona.get('tone', 'warm, empathetic, professional')}

RULES:
{rules_block}

CURRENT STAGE: {state.current_stage}
STAGE GOAL: {stage_goal}

{guidance}

{rag_context}

INSTRUCTIONS:
- This is a real phone call. Speak exactly as a warm, natural human would — NOT like a script being read.
- Keep your reply SHORT: 1-2 sentences max. Short answers feel more human on a phone call.
- NEVER start your reply with "Certainly", "Absolutely", "Of course", "Great question", or similar filler.
- Do NOT use markdown, lists, or any formatting — pure spoken words only.
- Mirror the prospect's pacing. If they gave a short answer, give a short reply.
- Use the prospect's name ({state.lead_name}) at most once per reply, and only when it feels natural.
- If the prospect wants to be removed from the call list, comply warmly and immediately.
- Do not repeat information you already said in the conversation.
"""
        return prompt.strip()

    # ------------------------------------------------------------------
    # Stage advancement heuristics
    # ------------------------------------------------------------------

    def _advance_stage(
        self,
        state: ConversationState,
        prospect_text: str,
        bot_text: str,
    ) -> None:
        """Move to the next stage based on simple keyword heuristics."""
        text_lower = prospect_text.lower()
        stage = state.current_stage

        # DNC detection (highest priority)
        dnc_phrases = ["do not call", "stop calling", "remove my number", "don't call"]
        if any(p in text_lower for p in dnc_phrases):
            state.current_stage = "dnc_removal"
            return

        transitions = self._script.get(stage, {}).get("transitions", {})

        if stage == "greeting":
            if self._is_negative(text_lower):
                state.current_stage = "not_interested"
            else:
                state.current_stage = transitions.get("confirmed_identity", "qualification")

        elif stage == "qualification":
            if state.turn_count > 4:
                state.current_stage = transitions.get("all_qualified", "value_proposition")

        elif stage == "value_proposition":
            if self._is_positive(text_lower):
                state.current_stage = transitions.get("positive_response", "info_collection")
            elif self._has_objection(text_lower) or self._is_negative(text_lower):
                state.current_stage = transitions.get("has_questions", "objection_handling")

        elif stage == "info_collection":
            if state.turn_count > 8:
                state.current_stage = transitions.get("all_collected", "transfer_criteria")
            elif self._is_negative(text_lower):
                state.current_stage = transitions.get("wants_to_stop", "not_interested")

        elif stage == "objection_handling":
            if self._is_positive(text_lower):
                state.current_stage = transitions.get("objection_overcome", "info_collection")
            elif self._is_negative(text_lower):
                state.current_stage = transitions.get("still_objecting", "not_interested")

        elif stage == "transfer_criteria":
            state.current_stage = transitions.get("ready_for_transfer", "transfer_script")

        elif stage == "transfer_script":
            state.current_stage = "transfer_script"

    def _resolve_action(
        self,
        state: ConversationState,
        prospect_text: str,
        bot_text: str,
    ) -> CallAction:
        """Determine the call action based on the current script stage."""
        stage_data = self._script.get(state.current_stage, {})
        action_str = stage_data.get("action")
        if action_str:
            try:
                return CallAction(action_str)
            except ValueError:
                pass
        return CallAction.CONTINUE

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_positive(text: str) -> bool:
        positives = ["yes", "yeah", "sure", "okay", "ok", "yep", "absolutely", "of course", "sounds good", "go ahead"]
        return any(p in text for p in positives)

    @staticmethod
    def _is_negative(text: str) -> bool:
        negatives = ["no", "nah", "not interested", "no thanks", "no thank you", "don't want"]
        return any(n in text for n in negatives)

    @staticmethod
    def _has_objection(text: str) -> bool:
        triggers = [
            "can't afford", "too expensive", "already have", "think about it",
            "not sure", "call me back", "talk to my", "don't trust",
        ]
        return any(t in text for t in triggers)

    def _fill_variables(self, text: str, state: ConversationState) -> str:
        """Replace {placeholders} with state values."""
        replacements = {
            "{customer_name}": state.lead_name,
            "{lead_name}": state.lead_name,
            "{agent_name}": self._persona.get("name", "Sarah"),
            "{company}": self._persona.get("company", "Senior Life Services"),
            "{beneficiary}": state.beneficiary,
            "{coverage_amount}": state.coverage_amount,
            "{monthly_premium}": state.monthly_premium,
            "{state}": state.state,
        }
        result = text
        for key, val in replacements.items():
            result = result.replace(key, val)
        return result.strip()
