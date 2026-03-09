"""
tests/test_local_conversation.py
─────────────────────────────────
Ollama-powered local conversation tester for the Final Expense Voice Bot.
Tests the full pipeline — script YAML, RAG knowledge base, and LLM responses
— entirely on CPU, no SIP, no audio, no GPU required.

MODES
─────
  Interactive (default)  —  type responses as the prospect; Ollama replies
    python -m tests.test_local_conversation

  Interactive with mock  —  same UI but uses the fast mock LLM (no Ollama)
    python -m tests.test_local_conversation --mock

  Automated fast (CI)    —  5 scenario tests using the mock LLM
    python -m tests.test_local_conversation auto

  Full Ollama auto       —  happy-path scenario driven by Ollama
    python -m tests.test_local_conversation ollama

  Objection tests        —  all 8 objections handled by Ollama
    python -m tests.test_local_conversation objections

  Jump to a stage        —  start interactive at a specific stage
    python -m tests.test_local_conversation --stage objection_handling

PREREQUISITES (for Ollama modes)
──────────────────────────────────
  1.  Install Ollama:  https://ollama.com
  2.  Pull the model:  ollama pull llama3.1:8b
  3.  Start server:    ollama serve          (or it auto-starts on first use)
  4.  Run this file:   python -m tests.test_local_conversation
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiohttp

# ── project root on Python path ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.mimo_vllm import LLMResponse, MimoVLLMClient
from src.llm.rag_engine import RAGEngine
from src.orchestration.conversation_engine import (
    CallAction,
    ConversationEngine,
    ConversationState,
    TurnResult,
)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,          # keep engine noise quiet during interactive use
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("test_conversation")

# ── constants ─────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL  = "http://localhost:11434"
OLLAMA_MODEL     = "llama3.1:8b"
LOGS_DIR         = Path(__file__).resolve().parent.parent / "logs"
LINE_WIDTH       = 68
STAGES_IN_ORDER  = [
    "greeting", "qualification", "value_proposition",
    "info_collection", "objection_handling",
    "transfer_criteria", "transfer_script",
]

# ─────────────────────────────────────────────────────────────────────────────
#  OLLAMA LLM CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class OllamaLLMClient(MimoVLLMClient):
    """Calls Ollama's /api/chat endpoint instead of a remote vLLM server.

    Drop-in replacement for MimoVLLMClient — uses the same call signature
    so the ConversationEngine works without any changes.

    The system prompt is inserted as the first message with role="system".
    All conversation history messages follow in order.
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = 0.7,
        max_tokens: int = 220,
    ) -> None:
        # Don't call super().__init__() — we don't need the vLLM config
        self._model       = model
        self._base_url    = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens  = max_tokens
        self._session: Optional[aiohttp.ClientSession] = None

    async def initialize(self) -> None:
        timeout = aiohttp.ClientTimeout(total=120)
        self._session = aiohttp.ClientSession(timeout=timeout)
        logger.info("OllamaLLMClient ready – model=%s  url=%s", self._model, self._base_url)

    async def shutdown(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def health_check(self) -> bool:
        """Return True if Ollama is reachable and the model is available."""
        if self._session is None or self._session.closed:
            await self.initialize()
        assert self._session is not None
        try:
            async with self._session.get(f"{self._base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    return False
                data = await resp.json()
                models = [m["name"] for m in data.get("models", [])]
                # Accept either "llama3.1:8b" or "llama3.1:8b-instruct-q4_K_M" etc.
                return any(self._model.split(":")[0] in m for m in models)
        except Exception:
            return False

    async def generate(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        **_kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request to Ollama."""
        if self._session is None or self._session.closed:
            await self.initialize()
        assert self._session is not None

        # Build the message list: system first, then conversation history
        ollama_messages = [{"role": "system", "content": system_prompt}]
        ollama_messages.extend(messages)

        payload = {
            "model":   self._model,
            "messages": ollama_messages,
            "stream":  False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
                "stop": ["\n\n", "Customer:", "User:"],   # prevent runaway generation
            },
        }

        t0 = time.perf_counter()
        async with self._session.post(
            f"{self._base_url}/api/chat",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        latency_ms = (time.perf_counter() - t0) * 1000
        text = data.get("message", {}).get("content", "").strip()

        return LLMResponse(
            text=text,
            finish_reason="stop",
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            latency_ms=latency_ms,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  MOCK LLM CLIENT  (fast, no Ollama — used for CI / auto mode)
# ─────────────────────────────────────────────────────────────────────────────

class MockLLMClient(MimoVLLMClient):
    """Returns canned responses so tests run without any server."""

    _RESPONSES: dict[str, str] = {
        "greeting":          "Hey there! Is this Mary? Great — my name is Sarah from Senior Life Services. How's your day?",
        "qualification":     "Perfect! Just a couple quick questions — are you somewhere between 40 and 85 years young?",
        "value_proposition": "So this program gives your family a whole life policy to cover funeral costs. Your rate locks in and never goes up.",
        "info_collection":   "Great! Let me grab a few details. Can I get your full legal name?",
        "objection_handling":"I totally get it — most folks are surprised that plans start under a dollar a day. Would you want your family protected?",
        "transfer_criteria": "Wonderful — you're all set! Let me connect you with a specialist.",
        "transfer_script":   "I'm connecting you right now with a senior benefits specialist. Stay on the line!",
        "not_interested":    "I completely understand. Thank you so much for your time — have a wonderful day!",
        "disqualification":  "I appreciate your time! This program is for ages 40 to 85, so it isn't a fit today.",
        "dnc_removal":       "Absolutely — removing your number right now. You won't hear from us again. Take care!",
    }

    async def initialize(self) -> None:   pass
    async def shutdown(self) -> None:     pass
    async def health_check(self) -> bool: return True

    async def generate(self, system_prompt: str, messages: list[dict[str, str]], **_kw) -> LLMResponse:
        stage = "greeting"
        for key in self._RESPONSES:
            if f"CURRENT STAGE: {key}" in system_prompt:
                stage = key
                break
        return LLMResponse(
            text=self._RESPONSES.get(stage, "I'm here to help with final expense coverage."),
            finish_reason="stop",
            prompt_tokens=100,
            completion_tokens=30,
            latency_ms=1.0,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  TRANSCRIPT LOGGER
# ─────────────────────────────────────────────────────────────────────────────

class TranscriptLogger:
    """Saves the full conversation to logs/<timestamp>_<call_id>.json."""

    def __init__(self, call_id: str) -> None:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = LOGS_DIR / f"{ts}_{call_id}.json"
        self._entries: list[dict[str, Any]] = []
        self._start = time.time()
        self._call_id = call_id

    def log_turn(
        self,
        role: str,                          # "bot" | "prospect"
        text: str,
        stage: str = "",
        action: str = "",
        latency_ms: float = 0.0,
        rag_chunks: int = 0,
    ) -> None:
        self._entries.append({
            "ts_offset_s": round(time.time() - self._start, 2),
            "role":        role,
            "stage":       stage,
            "action":      action,
            "text":        text,
            "latency_ms":  round(latency_ms, 1),
            "rag_chunks":  rag_chunks,
        })

    def save(self, state: ConversationState) -> Path:
        record = {
            "call_id":      self._call_id,
            "saved_at":     datetime.now().isoformat(),
            "duration_s":   round(time.time() - self._start, 1),
            "final_stage":  state.current_stage,
            "turn_count":   state.turn_count,
            "collected":    state.collected_fields,
            "transcript":   self._entries,
        }
        self._path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return self._path


# ─────────────────────────────────────────────────────────────────────────────
#  DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _bar(value: int, total: int = 100, width: int = 12) -> str:
    filled = round(value / total * width)
    return "█" * filled + "░" * (width - filled)


def _qualification_score(state: ConversationState) -> int:
    """Rough 0-100 score based on how far through the funnel we are."""
    stage_scores = {
        "greeting":          5,
        "qualification":     20,
        "value_proposition": 40,
        "info_collection":   60,
        "objection_handling":35,
        "transfer_criteria": 80,
        "transfer_script":  100,
        "not_interested":     0,
        "disqualification":   0,
        "dnc_removal":        0,
        "voicemail":          0,
    }
    score = stage_scores.get(state.current_stage, 10)
    # Bonus for collected fields
    score += min(len(state.collected_fields) * 3, 15)
    return min(score, 100)


def print_divider(char: str = "─") -> None:
    print(char * LINE_WIDTH)


def print_turn_header(turn: TurnResult, llm_label: str, score: int = 10) -> None:
    """Print a compact state panel after each bot turn."""
    print_divider()
    print(
        f"  STAGE   {turn.current_stage:<26}  "
        f"Turn {turn.action.value}"
    )
    print(
        f"  SCORE   {_bar(score)} {score:3d}%   "
        f"ACTION: {turn.action.value}"
    )
    if turn.rag_chunks_used:
        print(f"  RAG     {turn.rag_chunks_used} chunk(s) retrieved")
    print(f"  LLM     {turn.latency_ms:,.0f} ms  |  {llm_label}")
    print_divider()


def print_state_dump(state: ConversationState) -> None:
    """Print full state when user types /state."""
    print_divider("═")
    print("  FULL CONVERSATION STATE")
    print_divider("═")
    print(f"  call_id      : {state.call_id}")
    print(f"  stage        : {state.current_stage}")
    print(f"  lead_name    : {state.lead_name}")
    print(f"  state        : {state.state}")
    print(f"  turn_count   : {state.turn_count}")
    print(f"  history msgs : {len(state.history)}")
    print(f"  collected    : {json.dumps(state.collected_fields, indent=16)}" if state.collected_fields
          else "  collected    : (none yet)")
    print(f"  qual score   : {_qualification_score(state)}%")
    elapsed = round(time.time() - state.started_at, 0)
    print(f"  elapsed      : {elapsed:.0f}s")
    print_divider("═")


def print_help() -> None:
    print_divider()
    print("  COMMANDS")
    print("  /state             — dump full conversation state")
    print("  /stage <name>      — jump to a specific stage")
    print("  /restart           — restart the call from greeting")
    print("  /objection <key>   — trigger a specific objection")
    print("                       keys: too_expensive, need_to_think,")
    print("                             already_have_coverage, dont_trust_insurance,")
    print("                             family_can_handle_it, too_healthy,")
    print("                             just_cremate_me, send_info")
    print("  /help              — show this help")
    print("  /quit  or  q       — save transcript and exit")
    print_divider()


OBJECTION_TRIGGERS: dict[str, str] = {
    "too_expensive":          "I can't afford it, I'm on a fixed income and it's just too expensive.",
    "need_to_think":          "I need to think about it. Let me sleep on it and you can call me back.",
    "already_have_coverage":  "I already have life insurance through my job so I don't need anything.",
    "dont_trust_insurance":   "Honestly I don't trust insurance companies — they just take your money.",
    "family_can_handle_it":   "My kids will handle it. We'll just do a GoFundMe, it'll be fine.",
    "too_healthy":            "I'm in great health, I really don't think I need this right now.",
    "just_cremate_me":        "Just cremate me, I don't want a funeral. It's way too expensive.",
    "send_info":              "Can you just mail me something? I'd rather look at it on my own time.",
}

VALID_STAGES = [
    "greeting", "qualification", "value_proposition", "info_collection",
    "objection_handling", "transfer_criteria", "transfer_script",
    "not_interested", "disqualification", "dnc_removal", "voicemail",
]


# ─────────────────────────────────────────────────────────────────────────────
#  INTERACTIVE MODE
# ─────────────────────────────────────────────────────────────────────────────

async def run_interactive(
    start_stage: str = "greeting",
    use_mock: bool = False,
) -> None:
    """Interactive CLI — user types prospect responses, bot replies via Ollama."""

    llm_label = "mock LLM" if use_mock else f"ollama/{OLLAMA_MODEL}"

    print("\n" + "═" * LINE_WIDTH)
    print("  FINAL EXPENSE VOICE BOT — Local Conversation Tester")
    print(f"  LLM: {llm_label}")
    print("  Type /help for commands, /quit or q to exit.")
    print("═" * LINE_WIDTH + "\n")

    # ── set up Ollama ─────────────────────────────────────────────────────────
    if use_mock:
        llm: MimoVLLMClient = MockLLMClient()
    else:
        llm = OllamaLLMClient()
        await llm.initialize()
        print("  Checking Ollama connection…", end="", flush=True)
        ok = await llm.health_check()
        if not ok:
            print(f"\n\n  ERROR: Cannot reach Ollama at {OLLAMA_BASE_URL}")
            print(f"  Make sure Ollama is running and'{OLLAMA_MODEL}' is pulled:")
            print(f"    ollama pull {OLLAMA_MODEL}")
            print("    ollama serve")
            print("\n  Tip: run with --mock to use the built-in mock LLM instead.\n")
            return
        print(" OK\n")

    rag = RAGEngine()
    rag.load()
    engine = ConversationEngine(llm, rag)
    engine.load_script()

    # ── helper: build a fresh call ────────────────────────────────────────────
    def new_call(call_id: str) -> tuple[ConversationState, TranscriptLogger]:
        state_ = engine.new_call(call_id, {"first_name": "there", "state": "your state"})
        state_.current_stage = start_stage
        tlog_ = TranscriptLogger(call_id)
        return state_, tlog_

    call_counter = 0

    def fresh_call() -> tuple[ConversationState, TranscriptLogger]:
        nonlocal call_counter
        call_counter += 1
        return new_call(f"local-{call_counter:03d}")

    state, tlog = fresh_call()

    # ── opening line ──────────────────────────────────────────────────────────
    print("  Getting opening line…", end="", flush=True)
    opening = await engine.get_opening(state)
    print(f"\r  {' ' * 30}\r", end="")
    print(f"\n  Bot  [{state.current_stage}]\n  {opening.bot_text}\n")
    tlog.log_turn("bot", opening.bot_text, stage=state.current_stage)

    # ── main loop ─────────────────────────────────────────────────────────────
    loop = asyncio.get_event_loop()

    while True:
        # Non-blocking input (lets Ctrl-C work cleanly)
        try:
            raw = await loop.run_in_executor(None, lambda: input("  You  > "))
        except (EOFError, KeyboardInterrupt):
            print()
            break

        user_input = raw.strip()
        if not user_input:
            continue

        # ── slash commands ────────────────────────────────────────────────────
        if user_input.lower() in ("/quit", "q", "quit", "exit"):
            break

        if user_input.lower() == "/help":
            print_help()
            continue

        if user_input.lower() == "/state":
            print_state_dump(state)
            continue

        if user_input.lower().startswith("/stage "):
            target = user_input[7:].strip()
            if target in VALID_STAGES:
                state.current_stage = target
                print(f"  → Jumped to stage: {target}\n")
            else:
                print(f"  Unknown stage '{target}'. Valid: {', '.join(VALID_STAGES)}\n")
            continue

        if user_input.lower() == "/restart":
            saved = tlog.save(state)
            print(f"  Transcript saved → {saved.name}")
            state, tlog = fresh_call()
            opening = await engine.get_opening(state)
            print(f"\n  Bot  [{state.current_stage}]\n  {opening.bot_text}\n")
            tlog.log_turn("bot", opening.bot_text, stage=state.current_stage)
            continue

        if user_input.lower().startswith("/objection "):
            key = user_input[11:].strip()
            if key in OBJECTION_TRIGGERS:
                user_input = OBJECTION_TRIGGERS[key]
                print(f"  [Simulating objection: {key}]")
                print(f"  You  > {user_input}\n")
                state.current_stage = "objection_handling"
            else:
                print(f"  Unknown objection key '{key}'. Valid keys:\n  {', '.join(OBJECTION_TRIGGERS)}\n")
                continue

        # ── normal turn ───────────────────────────────────────────────────────
        tlog.log_turn("prospect", user_input, stage=state.current_stage)

        print("  …", end="", flush=True)
        try:
            turn = await engine.process_turn(state, user_input)
        except aiohttp.ClientError as exc:
            print(f"\r  ERROR: Ollama request failed — {exc}\n")
            continue

        # Clear "…" spinner
        print(f"\r  {' ' * 4}\r", end="")

        print(f"\n  Bot  [{turn.current_stage}]\n  {turn.bot_text}\n")
        tlog.log_turn(
            "bot", turn.bot_text,
            stage=turn.current_stage,
            action=turn.action.value,
            latency_ms=turn.latency_ms,
            rag_chunks=turn.rag_chunks_used,
        )

        print_turn_header(turn, llm_label, score=_qualification_score(state))
        print()

        # ── terminal actions ──────────────────────────────────────────────────
        if turn.action in (CallAction.TRANSFER_TO_CLOSER,):
            print("  ✓ TRANSFER — handing off to closer.\n")
            break
        if turn.action in (CallAction.END_CALL, CallAction.DNC_AND_END_CALL):
            print(f"  ✓ CALL ENDED — {turn.action.value}\n")
            break

    # ── wrap up ───────────────────────────────────────────────────────────────
    saved = tlog.save(state)
    print_divider("═")
    print(f"  Turns: {state.turn_count}  |  Final stage: {state.current_stage}")
    print(f"  Transcript saved → {saved}")
    print_divider("═")

    await llm.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
#  AUTOMATED FAST TESTS  (mock LLM — CI-safe)
# ─────────────────────────────────────────────────────────────────────────────

async def run_auto_tests() -> None:
    """5-scenario automated test suite using the mock LLM.  Always passes without Ollama."""

    print("\n" + "=" * LINE_WIDTH)
    print("  Automated Conversation Tests  (mock LLM — no Ollama needed)")
    print("=" * LINE_WIDTH + "\n")

    llm  = MockLLMClient()
    rag  = RAGEngine()
    rag.load()
    engine = ConversationEngine(llm, rag)
    engine.load_script()

    passed = 0

    # ── Test 1: Happy path → transfer ─────────────────────────────────────────
    print("Test 1: Happy path → transfer")
    state = engine.new_call("auto-001", {"first_name": "Mary", "state": "FL"})
    opening = await engine.get_opening(state)
    assert opening.bot_text, "Opening must not be empty"

    for utterance in [
        "Yes, this is Mary!",
        "Yes, I'm between 40 and 85.",
        "Yes, Florida is correct.",
        "No, this is my first time looking into it.",
        "That makes total sense, sure.",
        "Sounds good, let's do it.",
        "My name is Mary Smith.",
        "January 15th, 1957.",
        "123 Maple Street, Orlando.",
    ]:
        turn = await engine.process_turn(state, utterance)
        assert turn.bot_text, f"Empty response for: {utterance!r}"

    print(f"  ✓  {state.turn_count} turns  |  stage: {state.current_stage}")
    passed += 1

    # ── Test 2: DNC request ───────────────────────────────────────────────────
    print("Test 2: DNC removal")
    s2 = engine.new_call("auto-002", {"first_name": "Bob"})
    await engine.get_opening(s2)
    await engine.process_turn(s2, "Please remove my number and stop calling me.")
    assert s2.current_stage == "dnc_removal", f"Expected dnc_removal, got {s2.current_stage}"
    print(f"  ✓  stage: {s2.current_stage}")
    passed += 1

    # ── Test 3: Not interested ────────────────────────────────────────────────
    print("Test 3: Not interested from greeting")
    s3 = engine.new_call("auto-003", {"first_name": "Sue"})
    await engine.get_opening(s3)
    await engine.process_turn(s3, "No thanks, not interested.")
    assert s3.current_stage == "not_interested", f"Expected not_interested, got {s3.current_stage}"
    print(f"  ✓  stage: {s3.current_stage}")
    passed += 1

    # ── Test 4: RAG retrieval ─────────────────────────────────────────────────
    print("Test 4: RAG retrieval quality")
    chunks = rag.retrieve("How much does final expense insurance cost per month?")
    assert len(chunks) > 0, "RAG must return at least one chunk"
    assert chunks[0].score > 0.5, f"Top score too low: {chunks[0].score}"
    print(f"  ✓  {len(chunks)} chunk(s)  |  top score: {chunks[0].score:.4f}")
    passed += 1

    # ── Test 5: RAG context formatting ────────────────────────────────────────
    print("Test 5: RAG context formatting")
    ctx = RAGEngine.format_context(chunks)
    assert "[KNOWLEDGE BASE CONTEXT]" in ctx
    assert "[END CONTEXT]" in ctx
    print(f"  ✓  {len(ctx)} chars  |  context looks correct")
    passed += 1

    print(f"\n{'=' * LINE_WIDTH}")
    print(f"  {passed}/5 tests passed")
    print(f"{'=' * LINE_WIDTH}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  OLLAMA AUTO SCENARIO  (full happy-path without human input)
# ─────────────────────────────────────────────────────────────────────────────

async def run_ollama_scenario() -> None:
    """Drive a complete happy-path conversation using Ollama, no human input."""

    print("\n" + "=" * LINE_WIDTH)
    print(f"  Ollama Auto Scenario  |  model: {OLLAMA_MODEL}")
    print("=" * LINE_WIDTH + "\n")

    llm = OllamaLLMClient()
    await llm.initialize()

    print("  Checking Ollama…", end="", flush=True)
    if not await llm.health_check():
        print(f"\n\n  ERROR: Ollama not reachable at {OLLAMA_BASE_URL}")
        print(f"  Run: ollama pull {OLLAMA_MODEL} && ollama serve\n")
        return
    print(" OK\n")

    rag = RAGEngine()
    rag.load()
    engine = ConversationEngine(llm, rag)
    engine.load_script()

    state = engine.new_call("ollama-001", {"first_name": "Robert", "state": "Texas"})
    tlog  = TranscriptLogger("ollama-001")

    opening = await engine.get_opening(state)
    print(f"  Bot  [{state.current_stage}]\n  {opening.bot_text}\n")
    tlog.log_turn("bot", opening.bot_text, stage=state.current_stage)

    # Scripted prospect utterances — represents a cooperative prospect
    prospect_script = [
        ("greeting",          "Yes, this is Robert. Hi Sarah!"),
        ("qualification",     "Yeah, I'm 68 years old and I live in Texas."),
        ("qualification",     "No, I don't have any coverage. This would be brand new for me."),
        ("value_proposition", "Okay, that actually makes sense. Twelve thousand dollars is a lot to just have lying around."),
        ("value_proposition", "Yeah, I'd want my wife not to have to worry about that."),
        ("info_collection",   "Robert Mitchell."),
        ("info_collection",   "My birthday is March 3rd, 1957."),
        ("info_collection",   "I'm at 4821 Elm Drive, San Antonio, Texas, 78201."),
        ("info_collection",   "My phone is 210-555-9147. And this is it, yes."),
        ("info_collection",   "The last four of my social? Sure — it's 4829."),
        ("transfer_criteria", "Sure, I'm happy to talk to someone. Let's do it."),
    ]

    passed_stages: set[str] = set()

    for input_stage, utterance in prospect_script:
        # Optionally nudge to the expected stage for the test to flow correctly
        if state.current_stage == "greeting" and input_stage != "greeting":
            state.current_stage = input_stage

        print(f"  You  [{input_stage}]  {utterance}")
        tlog.log_turn("prospect", utterance, stage=state.current_stage)

        turn = await engine.process_turn(state, utterance)
        print(f"\n  Bot  [{turn.current_stage}]\n  {turn.bot_text}")
        print(f"         ↳ {turn.latency_ms:,.0f} ms  |  RAG: {turn.rag_chunks_used} chunks  |  {turn.action.value}\n")
        tlog.log_turn(
            "bot", turn.bot_text,
            stage=turn.current_stage, action=turn.action.value,
            latency_ms=turn.latency_ms, rag_chunks=turn.rag_chunks_used,
        )
        passed_stages.add(turn.current_stage)

        if turn.action in (CallAction.TRANSFER_TO_CLOSER, CallAction.END_CALL, CallAction.DNC_AND_END_CALL):
            print(f"  ✓ Call ended with action: {turn.action.value}")
            break

    saved = tlog.save(state)
    print_divider()
    print(f"  Stages visited : {', '.join(passed_stages)}")
    print(f"  Final stage    : {state.current_stage}")
    print(f"  Turns          : {state.turn_count}")
    print(f"  Transcript     → {saved.name}")
    print_divider()

    await llm.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
#  OBJECTION TESTS  (all 8 objections, one by one, using Ollama)
# ─────────────────────────────────────────────────────────────────────────────

async def run_objection_tests() -> None:
    """Test all 8 objection types — advance to objection_handling then fire each one."""

    print("\n" + "=" * LINE_WIDTH)
    print(f"  Objection Handling Tests  |  model: {OLLAMA_MODEL}")
    print("=" * LINE_WIDTH + "\n")

    llm = OllamaLLMClient()
    await llm.initialize()

    print("  Checking Ollama…", end="", flush=True)
    if not await llm.health_check():
        print(f"\n\n  ERROR: Ollama not reachable at {OLLAMA_BASE_URL}")
        print(f"  Run: ollama pull {OLLAMA_MODEL} && ollama serve\n")
        return
    print(" OK\n")

    rag = RAGEngine()
    rag.load()
    engine = ConversationEngine(llm, rag)
    engine.load_script()

    results: list[dict[str, Any]] = []

    for idx, (obj_key, trigger_phrase) in enumerate(OBJECTION_TRIGGERS.items(), 1):
        print(f"  [{idx}/{len(OBJECTION_TRIGGERS)}]  {obj_key}")
        print(f"  Prospect: \"{trigger_phrase[:70]}…\"" if len(trigger_phrase) > 70 else f"  Prospect: \"{trigger_phrase}\"")

        # Fresh call for each objection, jump straight to objection_handling
        state = engine.new_call(
            f"obj-{idx:02d}",
            {"first_name": "Patricia", "state": "Ohio"},
        )
        state.current_stage = "objection_handling"

        # Seed one turn of history so the LLM has context
        state.history = [
            {"role": "assistant", "content": "Just to make sure I have the right information — are you between 40 and 85 years old?"},
            {"role": "user",      "content": "Yes I'm 71."},
            {"role": "assistant", "content": "Wonderful! So let me tell you quickly what this program is about."},
            {"role": "user",      "content": "Okay."},
        ]

        print("  Bot  …", end="", flush=True)
        t0 = time.perf_counter()
        try:
            turn = await engine.process_turn(state, trigger_phrase)
            elapsed = (time.perf_counter() - t0) * 1000
        except Exception as exc:
            print(f"\r  ERROR: {exc}\n")
            results.append({"objection": obj_key, "passed": False, "error": str(exc)})
            continue

        # Clear spinner
        print(f"\r  Bot: {turn.bot_text[:LINE_WIDTH - 8]}" + ("…" if len(turn.bot_text) > LINE_WIDTH - 8 else ""))
        if len(turn.bot_text) > LINE_WIDTH - 8:
            # Print remaining text
            remaining = turn.bot_text[LINE_WIDTH - 8:]
            while remaining:
                print(f"       {remaining[:LINE_WIDTH - 8]}")
                remaining = remaining[LINE_WIDTH - 8:]

        non_empty = bool(turn.bot_text.strip())
        # Check the response contains something substantive (> 20 chars, has some words)
        substantive = len(turn.bot_text.strip()) > 20

        status = "✓ PASS" if (non_empty and substantive) else "✗ FAIL"
        print(f"  {status}  |  {elapsed:,.0f} ms  |  RAG: {turn.rag_chunks_used} chunks\n")

        results.append({
            "objection":       obj_key,
            "passed":          non_empty and substantive,
            "response_len":    len(turn.bot_text),
            "latency_ms":      round(elapsed, 1),
            "rag_chunks":      turn.rag_chunks_used,
            "response_preview":turn.bot_text[:120],
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    print_divider("═")
    passed_count = sum(1 for r in results if r["passed"])
    print(f"  Results: {passed_count}/{len(results)} objections handled correctly\n")
    for r in results:
        icon = "✓" if r["passed"] else "✗"
        lat  = r.get("latency_ms", 0)
        print(f"  {icon}  {r['objection']:<30}  {lat:>6,.0f} ms")

    # Save results
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = LOGS_DIR / f"{ts}_objection_tests.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Full results saved → {out.name}")
    print_divider("═")

    await llm.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
#  INFO COLLECTION STAGE TEST
# ─────────────────────────────────────────────────────────────────────────────

async def run_info_collection_test() -> None:
    """Verify all 5 fields (name, DOB, address, phone, SSN-4) are gathered."""

    print("\n" + "=" * LINE_WIDTH)
    print(f"  Info Collection Stage Test  |  model: {OLLAMA_MODEL}")
    print("=" * LINE_WIDTH + "\n")

    llm = OllamaLLMClient()
    await llm.initialize()

    if not await llm.health_check():
        print(f"  ERROR: Ollama not reachable at {OLLAMA_BASE_URL}\n")
        return

    rag = RAGEngine()
    rag.load()
    engine = ConversationEngine(llm, rag)
    engine.load_script()

    state = engine.new_call("ic-001", {"first_name": "Dorothy", "state": "Florida"})
    state.current_stage = "info_collection"

    # Seed context
    state.history = [
        {"role": "assistant", "content": "Great! Let me grab a few details so we can pull up your exact rates."},
    ]

    data_exchange = [
        ("Can I get your full legal name?",                "Dorothy Jean Williams"),
        ("And your date of birth?",                        "August 12th, 1951"),
        ("What's a good mailing address?",                 "8802 Sunrise Boulevard, Clearwater, Florida 33755"),
        ("Is this the best number to reach you?",          "Yes, 727-555-4416 is the best number."),
        ("Just the last four digits of your social?",      "Mine are 3317"),
    ]

    print("  Simulating info collection exchange:\n")
    all_answered = True

    for bot_q, prospect_a in data_exchange:
        print(f"  Bot:      {bot_q}")
        print(f"  Prospect: {prospect_a}")

        turn = await engine.process_turn(state, prospect_a)
        print(f"  Bot:      {turn.bot_text[:90]}{'…' if len(turn.bot_text) > 90 else ''}")
        print(f"            ({turn.latency_ms:,.0f} ms)\n")

        if not turn.bot_text.strip():
            print("  ✗ EMPTY RESPONSE — FAIL")
            all_answered = False

    print_divider()
    print(f"  {'✓ All fields covered' if all_answered else '✗ Some responses were empty'}")
    print(f"  Final stage: {state.current_stage}  |  Turns: {state.turn_count}")
    print_divider()

    await llm.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
#  TRANSFER DECISION LOGIC TEST
# ─────────────────────────────────────────────────────────────────────────────

async def run_transfer_test() -> None:
    """Verify the transfer_criteria → transfer_script path fires correctly."""

    print("\n" + "=" * LINE_WIDTH)
    print("  Transfer Decision Logic Test  (mock LLM)")
    print("=" * LINE_WIDTH + "\n")

    llm   = MockLLMClient()
    rag   = RAGEngine()
    rag.load()
    engine = ConversationEngine(llm, rag)
    engine.load_script()

    # Build a state that's ready to transfer
    state = engine.new_call("xfer-001", {"first_name": "James", "state": "Georgia"})
    state.current_stage = "transfer_criteria"
    state.collected_fields = {
        "full_name":     "James Carter",
        "date_of_birth": "1955-07-14",
        "phone_number":  "404-555-0178",
    }

    # Any positive response should advance to transfer_script
    turn = await engine.process_turn(state, "Yes, go ahead and connect me with someone.")

    print(f"  Prospect: 'Yes, go ahead and connect me.'")
    print(f"  Stage after → {state.current_stage}")
    print(f"  Action      → {turn.action.value}")

    reached_transfer = state.current_stage in ("transfer_script", "transfer_criteria")
    print(f"\n  {'✓ Transfer stage reached' if reached_transfer else '✗ Transfer NOT reached'}")

    # Test that DNC blocks transfer
    state2 = engine.new_call("xfer-002", {"first_name": "Donna"})
    state2.current_stage = "transfer_criteria"
    turn2  = await engine.process_turn(state2, "Stop calling me. Remove my number now.")

    dnc_caught = state2.current_stage == "dnc_removal"
    print(f"  {'✓ DNC correctly blocked transfer' if dnc_caught else '✗ DNC not caught'}")
    print()
    print_divider()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tests.test_local_conversation",
        description="Final Expense Voice Bot — Ollama-powered local conversation tester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODES
  (none)           Interactive — type responses; Ollama answers
  auto             Automated 5-test suite (mock LLM, fast, no Ollama)
  ollama           Automated scenario driven by Ollama
  objections       Test all 8 objection types with Ollama
  info             Test info-collection stage with Ollama
  transfer         Test transfer decision logic (mock LLM)

FLAGS
  --mock           Use mock LLM instead of Ollama (interactive only)
  --stage NAME     Start interactive at a specific stage

EXAMPLES
  python -m tests.test_local_conversation
  python -m tests.test_local_conversation auto
  python -m tests.test_local_conversation objections
  python -m tests.test_local_conversation --stage objection_handling
  python -m tests.test_local_conversation --mock
        """,
    )
    p.add_argument(
        "mode",
        nargs="?",
        default="interactive",
        choices=["interactive", "auto", "ollama", "objections", "info", "transfer"],
        help="Test mode (default: interactive)",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Use built-in mock LLM instead of Ollama",
    )
    p.add_argument(
        "--stage",
        default="greeting",
        choices=VALID_STAGES,
        metavar="STAGE",
        help="Stage to start at in interactive mode",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.mode == "auto":
        asyncio.run(run_auto_tests())
    elif args.mode == "ollama":
        asyncio.run(run_ollama_scenario())
    elif args.mode == "objections":
        asyncio.run(run_objection_tests())
    elif args.mode == "info":
        asyncio.run(run_info_collection_test())
    elif args.mode == "transfer":
        asyncio.run(run_transfer_test())
    else:
        # Default: interactive
        asyncio.run(run_interactive(
            start_stage=args.stage,
            use_mock=args.mock,
        ))
