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


class CallerIntent(str, Enum):
    """Classified intent of the most recent prospect utterance."""

    INTERESTED         = "interested"
    NOT_INTERESTED     = "not_interested"
    OBJECTION_PRICE    = "objection_price"
    OBJECTION_HAVE_IT  = "objection_have_it"   # already has coverage
    OBJECTION_SPOUSE   = "objection_spouse"    # need to talk to spouse
    QUESTION_PRODUCT   = "question_product"    # what does it cover?
    QUESTION_COMPANY   = "question_company"    # who are you?
    READY_TO_TRANSFER  = "ready_to_transfer"
    CONFUSED           = "confused"
    HOSTILE            = "hostile"
    UNKNOWN            = "unknown"


# Anchor phrases used for keyword + embedding similarity classification.
_INTENT_ANCHORS: dict[CallerIntent, list[str]] = {
    CallerIntent.INTERESTED: [
        "yes", "yeah", "sounds good", "tell me more", "I'm interested",
        "okay", "go ahead", "sure", "that works",
    ],
    CallerIntent.NOT_INTERESTED: [
        "not interested", "no thank you", "don't call me", "remove me",
        "I don't want", "please don't", "stop calling",
    ],
    CallerIntent.OBJECTION_PRICE: [
        "can't afford", "too expensive", "too much", "don't have the money",
        "that's a lot", "monthly payment", "cost too much",
    ],
    CallerIntent.OBJECTION_HAVE_IT: [
        "already have", "I have coverage", "I have insurance", "already covered",
        "got a policy", "I have a plan",
    ],
    CallerIntent.OBJECTION_SPOUSE: [
        "talk to my wife", "talk to my husband", "ask my spouse",
        "need to discuss", "check with my partner",
    ],
    CallerIntent.QUESTION_PRODUCT: [
        "what does it cover", "how does it work", "what's included",
        "is it whole life", "any medical exam", "health questions",
    ],
    CallerIntent.QUESTION_COMPANY: [
        "who are you", "what company", "is this legit", "never heard of",
        "how long have you been",
    ],
    CallerIntent.READY_TO_TRANSFER: [
        "I'm ready", "sign me up", "let's do it", "I want to apply",
        "how do I get started",
    ],
    CallerIntent.CONFUSED: [
        "what", "I don't understand", "can you repeat", "say that again",
        "huh", "pardon",
    ],
    CallerIntent.HOSTILE: [
        "shut up", "leave me alone", "scam", "fraud", "I'll report you",
        "go to hell",
    ],
}


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
    # Qualifying information extracted from the conversation.
    # Keys: name, dob, state, health, smoking_status.
    # Injected into the system prompt so the LLM never re-asks for data
    # it has already collected.
    collected_info: dict[str, str] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    # Set by _classify_intent() each turn so the system prompt can include it.
    last_intent: str = ""


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
        self._objection_responses: dict[str, str] = {}

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
            self._objection_responses = raw.get("objection_responses", {})
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

        # Capture prospect's name from the greeting stage response
        if state.current_stage == "greeting" and state.lead_name == "there":
            name = self._extract_name(prospect_text)
            if name:
                state.lead_name = name
                logger.info("Captured prospect name: %s", name)

        # Incrementally populate collected_info so the LLM never re-asks for
        # information the caller has already provided.
        self._update_collected_info(state, prospect_text)

        # 1. RAG retrieval
        rag_chunks = self._rag.retrieve(prospect_text)
        rag_context = RAGEngine.format_context(rag_chunks)

        # 1a. Classify caller intent for prompt enrichment + stage routing
        intent = self._classify_intent(prospect_text, state)
        state.last_intent = intent.value

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

        # 5. Advance stage using intent-based routing
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

    # Hardcoded opening utterance spoken verbatim on every call connect.
    # This BYPASSES the LLM so the line can never be reworded by the model.
    # Written as a single breath so TTS doesn't split into two chunks with an
    # awkward pause between them.
    _HARDCODED_OPENING = (
        "Hi, this is Sarah from American Beneficiary — "
        "we received a request for final expense coverage in your area, "
        "and I just wanted to reach out real quick. "
        "Is now an okay time?"
    )

    async def get_opening(self, state: ConversationState) -> TurnResult:
        """Return the bot's hardcoded opening line — bypasses LLM entirely.

        The opening is never generated by the model so it can never be
        reworded, hallucinated, or replaced with call-centre clichés.
        """
        start = time.perf_counter()
        bot_text = self._HARDCODED_OPENING

        state.history.append({"role": "assistant", "content": bot_text})
        elapsed_ms = (time.perf_counter() - start) * 1000

        return TurnResult(
            bot_text=bot_text,
            action=CallAction.CONTINUE,
            current_stage=state.current_stage,
            latency_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    def _classify_intent(self, text: str, state: ConversationState) -> CallerIntent:
        """Classify the prospect's intent from *text*.

        Uses two-pass classification:
        1. Fast keyword scan (no GPU, < 1 ms).
        2. If ambiguous, cosine similarity against anchor embeddings via the
           RAG engine's sentence encoder.

        This runs synchronously (called inside an async turn) but is cheap
        enough not to need an executor.
        """
        text_lower = text.lower().strip()

        # --- Pass 1: keyword matching ---
        matches: dict[CallerIntent, int] = {}
        for intent, anchors in _INTENT_ANCHORS.items():
            score = sum(1 for phrase in anchors if phrase in text_lower)
            if score:
                matches[intent] = score

        # DNC / hostile always wins immediately
        for intent in (CallerIntent.HOSTILE, CallerIntent.NOT_INTERESTED):
            if matches.get(intent, 0) > 0:
                return intent

        if matches:
            # Return the highest-scoring keyword match
            best = max(matches, key=lambda k: matches[k])
            # If the score is unambiguous (>1) or no embedder available, use it.
            if matches[best] > 1 or not hasattr(self._rag, "_model"):
                return best

        # --- Pass 2: embedding similarity (only when keyword score == 1) ---
        try:
            text_vec = self._rag._model.encode([text], convert_to_numpy=True)[0]
            best_intent = CallerIntent.UNKNOWN
            best_sim = 0.35  # minimum threshold

            for intent, anchors in _INTENT_ANCHORS.items():
                anchor_vecs = self._rag._model.encode(anchors, convert_to_numpy=True)
                # Mean cosine similarity across anchors
                sims = anchor_vecs @ text_vec / (
                    (anchor_vecs ** 2).sum(axis=1) ** 0.5 * (text_vec ** 2).sum() ** 0.5
                    + 1e-9
                )
                mean_sim = float(sims.mean())
                if mean_sim > best_sim:
                    best_sim = mean_sim
                    best_intent = intent

            return best_intent
        except Exception:
            # Classifier failure is non-fatal — return keyword result or UNKNOWN.
            return next(iter(matches), CallerIntent.UNKNOWN)

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_system_prompt(self, state: ConversationState, rag_context: str) -> str:
        persona    = self._persona
        stage_data = self._script.get(state.current_stage, {})
        stage_goal = stage_data.get("goal", "Continue the conversation naturally.")

        name    = persona.get("name",    "Sarah")
        company = persona.get("company", "American Beneficiary")
        role    = persona.get("role",    "benefits specialist")
        tone    = persona.get("tone",    "warm, conversational, confident")

        # Stage-specific guidance lines
        guidance_lines = stage_data.get("lines", [])
        questions      = stage_data.get("questions", [])
        guidance_parts: list[str] = []
        if guidance_lines:
            filled = [self._fill_variables(l, state) for l in guidance_lines]
            guidance_parts.append(
                "Example lines you can adapt (do NOT quote verbatim):\n"
                + "\n".join(f"  - {l}" for l in filled)
            )
        if questions:
            q_texts = [self._fill_variables(q["text"], state) for q in questions]
            guidance_parts.append(
                "Questions to ask:\n" + "\n".join(f"  - {q}" for q in q_texts)
            )

        # Inject a pre-written objection response when the intent matches
        intent_block = ""
        if state.last_intent:
            obj_key = {
                CallerIntent.OBJECTION_PRICE.value:   "price",
                CallerIntent.OBJECTION_HAVE_IT.value: "already_have",
                CallerIntent.OBJECTION_SPOUSE.value:  "spouse",
            }.get(state.last_intent)
            if obj_key and obj_key in self._objection_responses:
                intent_block = (
                    f"DETECTED OBJECTION ({state.last_intent}).\n"
                    f"Suggested response (adapt naturally):\n"
                    f"  {self._objection_responses[obj_key]}"
                )
            elif state.last_intent == CallerIntent.CONFUSED.value:
                intent_block = "The caller seems confused — clarify simply and slowly."
            elif state.last_intent == CallerIntent.HOSTILE.value:
                intent_block = (
                    "The caller is hostile. Apologise briefly and offer to remove "
                    "them from the list if they wish."
                )

        guidance = "\n\n".join(guidance_parts)
        rag_block = rag_context if rag_context.strip() else ""

        # collected_info block: tells the LLM what has already been gathered
        # so it cannot ask the same question twice.
        collected_block = ""
        if state.collected_info:
            items = ", ".join(f"{k}={v}" for k, v in state.collected_info.items())
            collected_block = (
                f"INFORMATION ALREADY COLLECTED: {items}\n"
                "Do NOT ask for any piece of information listed above — it is already known."
            )

        prompt = f"""You are {name}, a {role} at {company}. You're on a live phone call.
Tone: {tone}

SPEAKING RULES — every rule is non-negotiable:
1. MAX 2 sentences per reply (3rd only for direct factual questions). Never 4+.
2. Contractions always: don't / I'm / we'll / it's. NEVER the full form.
3. BANNED openers (starting your reply with any of these = failure):
   Certainly, Absolutely, Of course, Great, Fantastic, Awesome, Sure thing,
   I understand, I hear you, That makes sense, I appreciate, I'd be happy,
   Thank you for sharing, Allow me, Let me explain, Noted, Understood, Got it.
4. Jump straight to the response. No acknowledgment, no filler, no cushioning.
5. Never echo what the caller said ("So you're saying…" / "It sounds like…").
6. Not interested? Acknowledge briefly and let them go. Never re-pitch.
7. If asked if you're AI: “I'm just here to help make sure families are taken care of.”
8. End every reply with ONE question OR a clear next step. Never two questions.
9. Mirror the caller's pace — if brief, you be brief. Match their energy.
10. NEVER ask for info already collected (see below).

ONE QUESTION AT A TIME — CRITICAL:
WRONG: “Can I get your name? And your date of birth?”
RIGHT: “What's your first name?”  ← stop, wait, one thing at a time.

CURRENT STAGE : {state.current_stage}
STAGE GOAL    : {stage_goal}
CALLER NAME   : {state.lead_name}

{collected_block}
{guidance}
{intent_block}
{rag_block}"""
        return prompt.strip()

    # ------------------------------------------------------------------
    # Stage advancement
    # ------------------------------------------------------------------

    def _advance_stage(
        self,
        state: ConversationState,
        prospect_text: str,
        bot_text: str,
    ) -> None:
        """Move to the next stage using CallerIntent + turn-count heuristics."""
        intent = CallerIntent(state.last_intent) if state.last_intent else CallerIntent.UNKNOWN
        stage  = state.current_stage
        transitions = self._script.get(stage, {}).get("transitions", {})

        # DNC detection is always highest priority
        if intent == CallerIntent.NOT_INTERESTED and any(
            p in prospect_text.lower()
            for p in ("do not call", "stop calling", "remove my number", "take me off")
        ):
            state.current_stage = "dnc_removal"
            return

        # Hostile caller — end gracefully
        if intent == CallerIntent.HOSTILE:
            state.current_stage = transitions.get("not_interested", "not_interested")
            return

        if stage == "greeting":
            if intent == CallerIntent.NOT_INTERESTED:
                state.current_stage = transitions.get("not_interested", "not_interested")
            else:
                state.current_stage = transitions.get("confirmed_identity", "qualification")

        elif stage == "qualification":
            if state.turn_count > 4:
                state.current_stage = transitions.get("all_qualified", "health_screening")

        elif stage == "health_screening":
            if intent == CallerIntent.NOT_INTERESTED:
                state.current_stage = transitions.get("not_interested", "not_interested")
            elif state.turn_count > 7:
                state.current_stage = transitions.get("screening_complete", "coverage_options")

        elif stage == "coverage_options":
            if intent in (CallerIntent.INTERESTED, CallerIntent.READY_TO_TRANSFER):
                state.current_stage = transitions.get("amount_selected", "info_collection")
            elif intent in (
                CallerIntent.OBJECTION_PRICE,
                CallerIntent.OBJECTION_HAVE_IT,
                CallerIntent.OBJECTION_SPOUSE,
            ):
                state.current_stage = transitions.get("has_questions", "objection_handling")
            elif intent == CallerIntent.NOT_INTERESTED:
                state.current_stage = transitions.get("not_interested", "not_interested")

        elif stage == "value_proposition":
            if intent in (CallerIntent.INTERESTED, CallerIntent.READY_TO_TRANSFER):
                state.current_stage = transitions.get("positive_response", "info_collection")
            elif intent in (
                CallerIntent.OBJECTION_PRICE,
                CallerIntent.OBJECTION_HAVE_IT,
                CallerIntent.OBJECTION_SPOUSE,
            ):
                state.current_stage = transitions.get("has_questions", "objection_handling")
            elif intent == CallerIntent.NOT_INTERESTED:
                state.current_stage = transitions.get("not_interested", "not_interested")

        elif stage == "info_collection":
            if intent == CallerIntent.READY_TO_TRANSFER or state.turn_count > 12:
                state.current_stage = transitions.get("all_collected", "transfer_criteria")
            elif intent == CallerIntent.NOT_INTERESTED:
                state.current_stage = transitions.get("wants_to_stop", "not_interested")

        elif stage == "objection_handling":
            if intent == CallerIntent.INTERESTED:
                state.current_stage = transitions.get("objection_overcome", "coverage_options")
            elif intent == CallerIntent.NOT_INTERESTED:
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

    def _update_collected_info(self, state: ConversationState, text: str) -> None:
        """Extract call‑relevant facts from caller utterance and persist in collected_info."""
        import re

        # Name
        if "name" not in state.collected_info and state.lead_name != "there":
            state.collected_info["name"] = state.lead_name

        # Date of birth — matches MM/DD/YYYY, MM-DD-YYYY, or "January 5 1960" style
        if "dob" not in state.collected_info:
            dob_m = re.search(
                r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
                r'|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2},?\s*\d{4})\b',
                text, re.IGNORECASE,
            )
            if dob_m:
                state.collected_info["dob"] = dob_m.group(1)

        # US state — two-letter abbreviation or full name
        if "state" not in state.collected_info:
            _abbrs = (
                "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD "
                "MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC "
                "SD TN TX UT VT VA WA WV WI WY"
            ).split()
            for abbr in _abbrs:
                if re.search(r'\b' + abbr + r'\b', text.upper()):
                    state.collected_info["state"] = abbr
                    break

        # Beneficiary name (after "beneficiary is …" or "my beneficiary …")
        if "beneficiary" not in state.collected_info:
            ben_m = re.search(
                r'(?:beneficiary\s+(?:is|would be|will be)|name(?:d)?\s+as\s+beneficiary)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
                text, re.IGNORECASE,
            )
            if ben_m:
                state.collected_info["beneficiary"] = ben_m.group(1).title()

    @staticmethod
    def _extract_name(text: str) -> str:
        """Try to extract a first name from a greeting response like 'this is John' or 'my name is Mary'."""
        import re
        text_clean = text.strip().rstrip(".!?,")
        # Patterns: "this is X", "my name is X", "I'm X", "it's X", or just a standalone name
        for pattern in [
            r"(?:this is|my name is|i'm|i am|it's|name's|they call me)\s+(\w+)",
            r"^(\w+)$",  # single word = likely just their name
        ]:
            m = re.search(pattern, text_clean, re.IGNORECASE)
            if m:
                name = m.group(1).capitalize()
                # Filter out common non-name words
                if name.lower() not in {"yes", "yeah", "no", "nah", "hi", "hello", "hey", "good", "fine", "ok", "okay"}:
                    return name
        return ""

    def _fill_variables(self, text: str, state: ConversationState) -> str:
        """Replace {placeholders} with state values."""
        replacements = {
            "{customer_name}": state.lead_name,
            "{lead_name}": state.lead_name,
            "{agent_name}": self._persona.get("name", "Sarah"),
            "{company}": self._persona.get("company", "American Beneficiary"),
            "{beneficiary}": state.beneficiary,
            "{coverage_amount}": state.coverage_amount,
            "{monthly_premium}": state.monthly_premium,
            "{state}": state.state,
        }
        result = text
        for key, val in replacements.items():
            result = result.replace(key, val)
        return result.strip()
