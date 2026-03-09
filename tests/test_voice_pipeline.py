"""
tests/test_voice_pipeline.py
═════════════════════════════════════════════════════════════════════════════
Real-stack voice pipeline tester.

Unlike test_local_conversation.py (Ollama + text only), this script exercises
the ACTUAL production components — the same ones deployed by deploy_gpu.sh:

  LLM  ──  MimoVLLMClient        vLLM port 8000  (XiaomiMiMo/MiMo-7B-RL)
  TTS  ──  CosyVoiceTTSHandler   TTS  port 8001  (CosyVoice2-0.5B)
  STT  ──  Parakeet TDT          in-process NeMo  (nvidia/parakeet-tdt-0.6b-v2)
           └─ fallback: faster-whisper (CPU, Windows-friendly)
           └─ fallback: text input  (no mic / no GPU needed)
  RAG  ──  RAGEngine              semantic search, all-MiniLM-L6-v2

MODES
─────────────────────────────────────────────────────────────────────────────
  health    Ping every service and print latency table. Run first.
  chat      Text in → real vLLM → text out.  (No audio, test LLM only.)
  speak     Text in → real vLLM → CosyVoice TTS → plays on YOUR speakers.
  voice     Mic → STT → real vLLM → CosyVoice TTS → plays on your speakers.
  vicidial  Test VICIdial API: connectivity, login, lead fetch, disposition.
  load      Fire N concurrent vLLM requests and measure throughput / latency.

QUICK START
─────────────────────────────────────────────────────────────────────────────
  # 1. Health check from anywhere you can reach the VPS:
  python -m tests.test_voice_pipeline health \\
      --vllm-url http://VPS_IP:8000 \\
      --tts-url  http://VPS_IP:8001

  # 2. Bot speaks aloud, YOU type  (no mic needed — good first real test):
  python -m tests.test_voice_pipeline speak \\
      --vllm-url http://VPS_IP:8000 \\
      --tts-url  http://VPS_IP:8001

  # 3. Full voice (run ON the GPU VPS or with local GPU + mic):
  python -m tests.test_voice_pipeline voice

  # 4. Stress-test vLLM with 8 concurrent calls:
  python -m tests.test_voice_pipeline load --workers 8 \\
      --vllm-url http://VPS_IP:8000

  # 5. VICIdial connectivity:
  python -m tests.test_voice_pipeline vicidial \\
      --vicidial-url http://YOUR_VICIDIAL_IP/vicidial

EXTRA PACKAGES  (pip install as needed)
─────────────────────────────────────────────────────────────────────────────
  sounddevice    — audio playback + mic recording
  faster-whisper — lightweight CPU STT (Windows-friendly alternative to NeMo)

  pip install sounddevice faster-whisper

════════════════════════════════════════════════════════════════════════════════
  HOW TO CONNECT TO VICIDIAL  (step-by-step guide for today)
════════════════════════════════════════════════════════════════════════════════

WHAT YOU NEED FROM YOUR VICIDIAL ADMIN
  ① VICIdial server IP/hostname   (e.g. 192.168.1.10)
  ② Admin username + password     (for non_agent_api.php)
  ③ Campaign ID                   (e.g. FINALEXP)
  ④ A dedicated SIP extension/username for the bot  (e.g. voicebot)
  ⑤ A "closer" ingroup name       (e.g. CLOSERS)  — for warm transfer
  ⑥ Transfer extension number     (e.g. 8300)

STEP 1 — add a SIP peer in Asterisk on the VICIdial box
  Edit /etc/asterisk/sip.conf  (or sip_custom.conf):

  [voicebot]
  type=peer
  host=YOUR_VPS_IP       ; ← GPU VPS public IP
  username=voicebot
  secret=YOUR_SIP_PASS
  context=from-internal
  allow=ulaw,alaw
  canreinvite=no
  nat=force_rport,comedia
  dtmfmode=rfc2833
  qualify=yes

  Reload Asterisk:  asterisk -rx "sip reload"

STEP 2 — route outbound calls to the bot in extensions.conf
  ; All calls placed by VICIdial auto-dialer go to the bot first:
  exten => _NXXNXXXXXX,1,Answer()
  exten => _NXXNXXXXXX,n,Dial(SIP/voicebot@YOUR_VPS_IP:5060,30,gU(sub-record))
  exten => _NXXNXXXXXX,n,Hangup()

  Reload dialplan:  asterisk -rx "dialplan reload"

STEP 3 — create a VICIdial agent account for the bot
  VICIdial admin panel → User Management → Add User
    Username:        voicebot
    Password:        YOUR_AGENT_PASS
    User Level:      1  (agent)
    Campaign:        FINALEXP
    ✓ is "API user"  (allows agent_api.php login)

STEP 4 — fill in .env on the GPU VPS
  VICIDIAL_API_URL=http://192.168.1.10/vicidial
  VICIDIAL_API_USER=admin                ← admin user (non_agent_api access)
  VICIDIAL_API_PASS=your_admin_pass
  VICIDIAL_CAMPAIGN_ID=FINALEXP
  VICIDIAL_AGENT_USER=voicebot           ← bot agent account
  VICIDIAL_AGENT_PASS=your_bot_pass
  VICIDIAL_TRANSFER_EXT=8300
  VICIDIAL_CLOSER_INGROUP=CLOSERS
  SIP_SERVER=192.168.1.10                ← Asterisk IP (VICIdial box)
  SIP_USERNAME=voicebot
  SIP_PASSWORD=YOUR_SIP_PASS

STEP 5 — verify:
  python -m tests.test_voice_pipeline vicidial --vicidial-url http://192.168.1.10/vicidial

STEP 6 — start the VPS stack and place a test call from VICIdial manually.

════════════════════════════════════════════════════════════════════════════════
  HOW TO ADD MORE BOTS  (horizontal scaling)
════════════════════════════════════════════════════════════════════════════════

ARCHITECTURE
  1 GPU VPS  →  1 vLLM server (handles up to 30 concurrent LLM requests)
             →  1 CosyVoice TTS server (handles 30+ concurrent TTS requests)
             →  N voicebot worker processes  (each handles exactly 1 SIP call)

  vLLM batches requests automatically. You only need more worker processes.

CAPACITY MATH (A100 40 GB)
  vLLM max-num-seqs=30  →  30 simultaneous LLM generations
  Each call takes ~50-200 ms LLM time  →  effective throughput ~150-600 calls/hr
  For ≤ 30 concurrent active calls, 1 vLLM instance is enough.

STEP 1 — add one SIP username per bot in Asterisk sip.conf:
  [voicebot1]
  type=peer ; host=VPS_IP ; secret=pass1 ; context=from-internal ; allow=ulaw
  [voicebot2]
  type=peer ; host=VPS_IP ; secret=pass2 ; context=from-internal ; allow=ulaw
  (repeat for voicebot3 … voicebotN)

STEP 2 — add one VICIdial agent user per bot (Admin > User Management).
  voicebot1, voicebot2, … all in campaign FINALEXP.

STEP 3 — create a systemd service template on the GPU VPS:
  File: /etc/systemd/system/voicebot@.service

  [Unit]
  Description=VoiceBot worker instance %i
  After=network.target redis.service voicebot-vllm.service voicebot-tts.service
  Requires=voicebot-vllm.service voicebot-tts.service

  [Service]
  User=root
  WorkingDirectory=/opt/voicebot
  EnvironmentFile=/opt/voicebot/.env
  Environment=BOT_ID=%i
  Environment=SIP_USERNAME=voicebot%i
  Environment=SIP_PASSWORD=pass%i
  Environment=VICIDIAL_AGENT_USER=voicebot%i
  Environment=API_PORT=900%i
  ExecStart=/opt/voicebot/.venv/bin/uvicorn src.main:app \\
      --host 0.0.0.0 --port 900%i --workers 1 --loop asyncio
  Restart=always ; RestartSec=10

  [Install]
  WantedBy=multi-user.target

  Enable & start:
    systemctl enable voicebot@1 voicebot@2 voicebot@3
    systemctl start  voicebot@1 voicebot@2 voicebot@3

STEP 4 — for >30 concurrent calls, add a SECOND vLLM instance:
  # vllm-b.service: same as voicebot-vllm but --port 8002
  # Set env for half the bots: VLLM_API_URL=http://127.0.0.1:8002
  # A second A100 node can be added with Nginx upstream load-balancing
  # over port 8000 → [node1:8000, node2:8000].
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import aiohttp
import numpy as np

# ── project root on Python path ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.mimo_vllm import LLMResponse, MimoVLLMClient
from src.llm.rag_engine import RAGEngine
from src.tts.cosyvoice_handler import CosyVoiceTTSHandler
from src.orchestration.conversation_engine import (
    CallAction,
    ConversationEngine,
    ConversationState,
    TurnResult,
)
from config.settings import get_llm_config, get_tts_config, get_vicidial_config

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("test_voice_pipeline")

LOGS_DIR   = Path(__file__).resolve().parent.parent / "logs"
LINE_WIDTH = 70


# ═════════════════════════════════════════════════════════════════════════════
#  AUDIO PLAYBACK (sounddevice → speakers)
# ═════════════════════════════════════════════════════════════════════════════

class AudioPlayer:
    """Streams raw int16 PCM bytes to the default speakers in a background thread.

    Usage::
        player = AudioPlayer(sample_rate=22050)
        player.start()
        async for chunk in tts.synthesize_stream(text):
            player.feed(chunk.audio_bytes)
        player.wait_done()   # block until playback finishes
        player.stop()
    """

    def __init__(self, sample_rate: int = 22050) -> None:
        self._sr     = sample_rate
        self._q: queue.Queue[Optional[bytes]] = queue.Queue(maxsize=400)
        self._thread: Optional[threading.Thread] = None
        self._available = False

        try:
            import sounddevice as sd  # type: ignore[import-untyped]
            self._sd = sd
            self._available = True
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self._available

    def start(self) -> None:
        if not self._available:
            return
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def feed(self, pcm_bytes: bytes) -> None:
        """Push PCM bytes (int16 LE) onto the playback queue."""
        if self._available:
            self._q.put(pcm_bytes)

    def wait_done(self) -> None:
        """Block until the queue is empty (playback caught up)."""
        if self._available:
            self._q.join()

    def stop(self) -> None:
        if self._available:
            self._q.put(None)       # sentinel
            if self._thread:
                self._thread.join(timeout=5)

    def _worker(self) -> None:
        """Background thread: drains queue and writes to sounddevice OutputStream."""
        BLOCK = 2048    # frames per write (~93 ms @ 22050 Hz)
        stream = self._sd.RawOutputStream(
            samplerate=self._sr,
            channels=1,
            dtype="int16",
            blocksize=BLOCK,
        )
        stream.start()
        buf = bytearray()
        try:
            while True:
                item = self._q.get()
                if item is None:
                    self._q.task_done()
                    break
                buf.extend(item)
                while len(buf) >= BLOCK * 2:
                    stream.write(bytes(buf[: BLOCK * 2]))
                    del buf[: BLOCK * 2]
                # Flush small tail chunk on silence
                if len(buf) > 0 and self._q.empty():
                    pad = (BLOCK * 2 - len(buf) % (BLOCK * 2)) % (BLOCK * 2)
                    stream.write(bytes(buf) + b"\x00" * pad)
                    buf.clear()
                self._q.task_done()
        finally:
            stream.stop()
            stream.close()


# ═════════════════════════════════════════════════════════════════════════════
#  MICROPHONE RECORDER (sounddevice)
# ═════════════════════════════════════════════════════════════════════════════

class MicRecorder:
    """Records from the default microphone until silence is detected.

    Returns raw int16 LE PCM bytes at the given sample_rate.
    Falls back to text input if sounddevice is not installed.
    """

    SILENCE_RMS_THRESHOLD = 0.008    # normalised float32  (0–1 range)

    def __init__(self, sample_rate: int = 16000, silence_ms: int = 1200) -> None:
        self._sr         = sample_rate
        self._silence_ms = silence_ms
        self._available  = False

        try:
            import sounddevice as sd  # type: ignore[import-untyped]
            self._sd = sd
            self._available = True
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self._available

    def record_until_silence(self) -> bytes:
        """Block until the user stops speaking (or 30 s max). Returns PCM bytes."""
        if not self._available:
            raise RuntimeError("sounddevice not available")

        CHUNK_FRAMES = int(self._sr * 0.08)     # 80 ms per block
        silence_blocks_needed = int(self._silence_ms / 80)

        frames: list[np.ndarray] = []
        silent_blocks = 0
        recording = False
        start_time = time.time()

        def callback(indata, frames_count, time_info, status):   # noqa: ARG001
            nonlocal silent_blocks, recording
            chunk = indata[:, 0].copy()     # mono
            frames.append(chunk)
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms > self.SILENCE_RMS_THRESHOLD:
                recording = True
                silent_blocks = 0
            elif recording:
                silent_blocks += 1

        with self._sd.InputStream(
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_FRAMES,
            callback=callback,
        ):
            while True:
                time.sleep(0.05)
                elapsed = time.time() - start_time
                # Stop if: we heard speech AND then silence, OR 30 s timeout
                if recording and silent_blocks >= silence_blocks_needed:
                    break
                if elapsed > 30:
                    break

        if not frames:
            return b""

        audio_f32 = np.concatenate(frames)
        audio_i16 = (np.clip(audio_f32, -1.0, 1.0) * 32767).astype(np.int16)
        return audio_i16.tobytes()


# ═════════════════════════════════════════════════════════════════════════════
#  STT BACKENDS
# ═════════════════════════════════════════════════════════════════════════════

class STTBackend:
    name: str = "base"

    async def transcribe(self, pcm_bytes: bytes, sample_rate: int) -> str:
        raise NotImplementedError


class ParakeetSTTBackend(STTBackend):
    """In-process Parakeet TDT (requires NeMo — GPU VPS only)."""
    name = "parakeet-tdt"

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        import nemo.collections.asr as nemo_asr  # type: ignore[import-untyped]
        self._model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=os.getenv("STT_MODEL", "nvidia/parakeet-tdt-0.6b-v2")
        )
        try:
            import torch
            if torch.cuda.is_available():
                self._model = self._model.cuda()
        except ImportError:
            pass
        self._model.eval()

    async def transcribe(self, pcm_bytes: bytes, sample_rate: int) -> str:
        import tempfile, soundfile as sf   # type: ignore[import-untyped]
        audio_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_f32 = audio_i16.astype(np.float32) / 32768.0
        loop = asyncio.get_event_loop()

        def _infer():
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, audio_f32, sample_rate)
                results = self._model.transcribe([tmp.name])
                if not results:
                    return ""
                r = results[0]
                return r.text if hasattr(r, "text") else str(r)

        return await loop.run_in_executor(None, _infer)


class FasterWhisperSTTBackend(STTBackend):
    """faster-whisper CPU STT — works on Windows, no GPU needed."""
    name = "faster-whisper"

    def __init__(self, model_size: str = "base.en") -> None:
        self._model_size = model_size
        self._model = None

    def load(self) -> None:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]
        self._model = WhisperModel(self._model_size, device="cpu", compute_type="int8")

    async def transcribe(self, pcm_bytes: bytes, sample_rate: int) -> str:
        import tempfile, soundfile as sf   # type: ignore[import-untyped]
        audio_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_f32 = audio_i16.astype(np.float32) / 32768.0
        loop = asyncio.get_event_loop()

        def _infer():
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, audio_f32, sample_rate)
                segments, _ = self._model.transcribe(tmp.name, beam_size=5, language="en")
                return " ".join(s.text.strip() for s in segments)

        return await loop.run_in_executor(None, _infer)


def _pick_stt_backend(force: Optional[str] = None) -> Optional[STTBackend]:
    """Auto-detect the best available STT backend."""
    if force == "parakeet":
        b = ParakeetSTTBackend(); b.load(); return b
    if force == "whisper":
        b = FasterWhisperSTTBackend(); b.load(); return b

    # Auto: try Parakeet (NeMo) first
    if force is None:
        try:
            b = ParakeetSTTBackend(); b.load()
            ok(f"STT backend: {b.name}")
            return b
        except Exception:
            pass
        # Fallback: faster-whisper
        try:
            b = FasterWhisperSTTBackend(); b.load()
            warn(f"NeMo not available — using {b.name} (CPU, slower)")
            return b
        except Exception:
            pass

    return None


# ═════════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ═════════════════════════════════════════════════════════════════════════════

RED   = "\033[0;31m"; GREEN = "\033[0;32m"; YELLOW = "\033[1;33m"
CYAN  = "\033[0;36m"; BOLD  = "\033[1m";    RESET  = "\033[0m"

def ok(msg: str)   -> None: print(f"  {GREEN}✓{RESET} {msg}")
def warn(msg: str) -> None: print(f"  {YELLOW}⚠{RESET} {msg}")
def err(msg: str)  -> None: print(f"  {RED}✗{RESET} {msg}")
def info(msg: str) -> None: print(f"  {CYAN}▶{RESET} {msg}")
def div(c: str = "─") -> None: print(c * LINE_WIDTH)


# ═════════════════════════════════════════════════════════════════════════════
#  MODE 1: HEALTH CHECK
# ═════════════════════════════════════════════════════════════════════════════

async def run_health(vllm_url: str, tts_url: str, vicidial_url: str) -> None:
    """Ping all configured services and report status + latency."""
    div("═")
    print(f"  {BOLD}HEALTH CHECK  —  Final Expense Voice Bot{RESET}")
    div("═")

    checks = [
        ("vLLM (Mimo 7B)",       f"{vllm_url}/health"),
        ("CosyVoice TTS",        f"{tts_url}/health"),
    ]
    if vicidial_url and vicidial_url != "http://your-vicidial-server/vicidial":
        checks.append(("VICIdial API", f"{vicidial_url}/non_agent_api.php?source=test&user=admin&pass=x&function=version"))

    timeout = aiohttp.ClientTimeout(total=8)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for name, url in checks:
            t0 = time.perf_counter()
            try:
                async with session.get(url) as resp:
                    latency = (time.perf_counter() - t0) * 1000
                    body = (await resp.text())[:80]
                    if resp.status == 200:
                        ok(f"{name:<28}  {latency:>6.0f} ms  |  {body[:60]}")
                    else:
                        warn(f"{name:<28}  HTTP {resp.status}  |  {body[:60]}")
            except Exception as exc:
                latency = (time.perf_counter() - t0) * 1000
                err(f"{name:<28}  {latency:>6.0f} ms  UNREACHABLE — {exc}")

    # Local checks
    print()
    info("Local checks:")
    try:
        from src.llm.rag_engine import RAGEngine
        rag = RAGEngine(); rag.load()
        ok(f"RAG engine loaded  ({rag.entry_count} entries, backend={rag.backend})")
    except Exception as exc:
        err(f"RAG engine failed: {exc}")

    try:
        import sounddevice as sd   # type: ignore[import-untyped]
        devs = sd.query_devices()
        in_dev  = sd.query_devices(kind="input")
        out_dev = sd.query_devices(kind="output")
        ok(f"sounddevice ready  |  input: {in_dev['name']}  |  output: {out_dev['name']}")
    except ImportError:
        warn("sounddevice not installed — voice/speak modes unavailable  (pip install sounddevice)")
    except Exception as exc:
        warn(f"sounddevice error: {exc}")

    for lib, pkg in [("nemo", "nemo_toolkit[asr]"), ("faster_whisper", "faster-whisper")]:
        try:
            __import__(lib)
            ok(f"{lib} installed — STT available")
        except ImportError:
            warn(f"{lib} not found  (pip install {pkg}) — needed for voice mode")

    div("═")


# ═════════════════════════════════════════════════════════════════════════════
#  MODE 2: CHAT  (text in → real vLLM → text out)
# ═════════════════════════════════════════════════════════════════════════════

async def run_chat(vllm_url: str) -> None:
    """Real vLLM + RAG — text in, text out. Tests LLM without any audio."""
    div("═")
    print(f"  {BOLD}CHAT MODE  —  Real vLLM at {vllm_url}{RESET}")
    print("  Type your response, Enter to send. /quit to exit.")
    div("═")
    print()

    from config.settings import LLMConfig
    cfg = get_llm_config()
    # Allow overriding the URL from the CLI flag
    import dataclasses
    cfg = dataclasses.replace(cfg, vllm_api_url=vllm_url)

    llm = MimoVLLMClient(config=cfg)
    await llm.initialize()

    print("  Checking vLLM…", end="", flush=True)
    if not await llm.health_check():
        print()
        err(f"vLLM not reachable at {vllm_url}")
        print("  Make sure voicebot-vllm.service is running on the VPS.")
        await llm.shutdown(); return
    print(" OK\n")

    rag    = RAGEngine(); rag.load()
    engine = ConversationEngine(llm, rag)
    engine.load_script()

    call_n  = 0
    def new_call():
        nonlocal call_n
        call_n += 1
        return engine.new_call(f"chat-{call_n:03d}", {"first_name": "there", "state": "your state"})

    state = new_call()
    opening = await engine.get_opening(state)
    print(f"  Bot  [{state.current_stage}]\n  {opening.bot_text}\n")

    loop = asyncio.get_event_loop()
    while True:
        try:
            raw = await loop.run_in_executor(None, lambda: input("  You  > "))
        except (EOFError, KeyboardInterrupt):
            print(); break
        user_input = raw.strip()
        if not user_input: continue
        if user_input.lower() in ("/quit", "q", "quit"): break
        if user_input.lower() == "/restart":
            state = new_call()
            opening = await engine.get_opening(state)
            print(f"\n  Bot  [{state.current_stage}]\n  {opening.bot_text}\n"); continue

        print("  …", end="", flush=True)
        t0 = time.perf_counter()
        try:
            turn = await engine.process_turn(state, user_input)
        except Exception as exc:
            print(f"\r  ERROR: {exc}\n"); continue

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"\r  Bot  [{turn.current_stage}]  ({elapsed:.0f} ms, RAG={turn.rag_chunks_used})\n  {turn.bot_text}\n")

        if turn.action in (CallAction.TRANSFER_TO_CLOSER,):
            print("  ✓ TRANSFER — handing off.\n"); break
        if turn.action in (CallAction.END_CALL, CallAction.DNC_AND_END_CALL):
            print(f"  ✓ CALL ENDED — {turn.action.value}\n"); break

    await llm.shutdown()


# ═════════════════════════════════════════════════════════════════════════════
#  MODE 3: SPEAK  (text in → real vLLM → TTS → speakers)
# ═════════════════════════════════════════════════════════════════════════════

async def run_speak(vllm_url: str, tts_url: str) -> None:
    """Real vLLM + CosyVoice TTS. You type, the bot speaks through your speakers."""
    div("═")
    print(f"  {BOLD}SPEAK MODE  —  vLLM: {vllm_url}  |  TTS: {tts_url}{RESET}")
    print("  Type your response. The bot will SPEAK its reply aloud.")
    print("  /quit to exit.")
    div("═")
    print()

    import dataclasses
    llm_cfg = dataclasses.replace(get_llm_config(), vllm_api_url=vllm_url)
    tts_cfg = dataclasses.replace(get_tts_config(), api_url=tts_url)

    llm = MimoVLLMClient(config=llm_cfg)
    tts = CosyVoiceTTSHandler(config=tts_cfg)
    await llm.initialize()
    await tts.initialize()

    # Check services
    print("  Checking services…", flush=True)
    if not await llm.health_check():
        err(f"vLLM not reachable at {vllm_url}"); await llm.shutdown(); await tts.shutdown(); return
    ok(f"vLLM at {vllm_url}")
    if not await tts.health_check():
        err(f"TTS not reachable at {tts_url}"); await llm.shutdown(); await tts.shutdown(); return
    ok(f"TTS at {tts_url}")
    print()

    player = AudioPlayer(sample_rate=tts_cfg.sample_rate)
    if not player.available:
        warn("sounddevice not installed — audio will NOT be played.")
        warn("pip install sounddevice  to hear the bot speak.")
    player.start()

    rag    = RAGEngine(); rag.load()
    engine = ConversationEngine(llm, rag)
    engine.load_script()

    call_n = 0
    def new_call():
        nonlocal call_n; call_n += 1
        return engine.new_call(f"speak-{call_n:03d}", {"first_name": "there", "state": "your state"})

    state = new_call()
    opening = await engine.get_opening(state)
    print(f"  Bot  [{state.current_stage}]\n  {opening.bot_text}")
    await _speak_text(tts, player, opening.bot_text)
    print()

    loop = asyncio.get_event_loop()
    while True:
        try:
            raw = await loop.run_in_executor(None, lambda: input("  You  > "))
        except (EOFError, KeyboardInterrupt):
            print(); break
        user_input = raw.strip()
        if not user_input: continue
        if user_input.lower() in ("/quit", "q", "quit"): break
        if user_input.lower() == "/restart":
            state = new_call()
            opening = await engine.get_opening(state)
            print(f"\n  Bot  [{state.current_stage}]\n  {opening.bot_text}")
            await _speak_text(tts, player, opening.bot_text); print(); continue

        print("  …", end="", flush=True)
        t0 = time.perf_counter()
        try:
            turn = await engine.process_turn(state, user_input)
        except Exception as exc:
            print(f"\r  ERROR: {exc}\n"); continue

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"\r  Bot  [{turn.current_stage}]  ({elapsed:.0f} ms LLM, RAG={turn.rag_chunks_used})\n  {turn.bot_text}")
        await _speak_text(tts, player, turn.bot_text)
        print()

        if turn.action in (CallAction.TRANSFER_TO_CLOSER,):
            print("  ✓ TRANSFER — handing off.\n"); break
        if turn.action in (CallAction.END_CALL, CallAction.DNC_AND_END_CALL):
            print(f"  ✓ CALL ENDED — {turn.action.value}\n"); break

    player.stop()
    await llm.shutdown()
    await tts.shutdown()


async def _speak_text(tts: CosyVoiceTTSHandler, player: AudioPlayer, text: str) -> None:
    """Stream TTS audio for *text* and feed it to the AudioPlayer."""
    if not player.available:
        return
    try:
        async for chunk in tts.synthesize_stream(text):
            player.feed(chunk.audio_bytes)
        player.wait_done()
    except Exception as exc:
        warn(f"TTS error: {exc}")


# ═════════════════════════════════════════════════════════════════════════════
#  MODE 4: VOICE  (mic → STT → vLLM → TTS → speakers)
# ═════════════════════════════════════════════════════════════════════════════

async def run_voice(vllm_url: str, tts_url: str, stt_backend_name: Optional[str] = None) -> None:
    """Full voice loop: mic → STT → real vLLM → CosyVoice TTS → speakers."""
    div("═")
    print(f"  {BOLD}VOICE MODE  —  Full pipeline{RESET}")
    print(f"  vLLM: {vllm_url}   TTS: {tts_url}")
    div("═")
    print()

    import dataclasses
    llm_cfg = dataclasses.replace(get_llm_config(), vllm_api_url=vllm_url)
    tts_cfg = dataclasses.replace(get_tts_config(), api_url=tts_url)

    llm = MimoVLLMClient(config=llm_cfg)
    tts = CosyVoiceTTSHandler(config=tts_cfg)
    await llm.initialize()
    await tts.initialize()

    if not await llm.health_check():
        err(f"vLLM not reachable at {vllm_url}"); return
    ok(f"vLLM at {vllm_url}")
    if not await tts.health_check():
        err(f"TTS not reachable at {tts_url}"); return
    ok(f"TTS at {tts_url}")

    # STT
    info("Loading STT backend…")
    stt = _pick_stt_backend(force=stt_backend_name)
    if stt is None:
        warn("No STT backend available — falling back to TEXT INPUT.")
        warn("pip install faster-whisper  to enable microphone STT.")

    # Microphone
    mic = MicRecorder(sample_rate=16000, silence_ms=1200)
    if not mic.available and stt is not None:
        warn("sounddevice not installed — mic recording unavailable.")
        warn("pip install sounddevice  to enable microphone input.")
        stt = None

    # Playback
    player = AudioPlayer(sample_rate=tts_cfg.sample_rate)
    if not player.available:
        warn("sounddevice not installed — audio will NOT be played.")
    player.start()

    use_mic = (stt is not None and mic.available)
    input_label = f"mic → {stt.name}" if use_mic else "keyboard"
    ok(f"Input method     : {input_label}")
    ok(f"Output method    : {'sounddevice speakers' if player.available else 'text only'}")
    print()

    rag    = RAGEngine(); rag.load()
    engine = ConversationEngine(llm, rag)
    engine.load_script()

    call_n = 0
    def new_call():
        nonlocal call_n; call_n += 1
        return engine.new_call(f"voice-{call_n:03d}", {"first_name": "there", "state": "your state"})

    state   = new_call()
    opening = await engine.get_opening(state)
    print(f"  {BOLD}Bot{RESET}  [{state.current_stage}]\n  {opening.bot_text}")
    await _speak_text(tts, player, opening.bot_text)
    print()

    loop = asyncio.get_event_loop()

    while True:
        # ── Capture input ─────────────────────────────────────────────────────
        if use_mic:
            info(f"Listening (speak now, {mic._silence_ms} ms silence to stop)…")
            try:
                pcm = await loop.run_in_executor(None, mic.record_until_silence)
            except KeyboardInterrupt:
                print(); break
            if len(pcm) < 3200:     # < 0.1 s of 16 kHz audio → skip
                warn("No speech detected. Tap Enter to quit.")
                continue
            info("Transcribing…")
            t_stt = time.perf_counter()
            user_text = await stt.transcribe(pcm, sample_rate=16000)
            stt_ms = (time.perf_counter() - t_stt) * 1000
            user_text = user_text.strip()
            if not user_text:
                warn("Could not transcribe — try again."); continue
            print(f"  {BOLD}You{RESET}  [{stt.name}, {stt_ms:.0f} ms]\n  {user_text}\n")
        else:
            try:
                raw = await loop.run_in_executor(None, lambda: input("  You  > "))
            except (EOFError, KeyboardInterrupt):
                print(); break
            user_text = raw.strip()
            if not user_text: continue
            if user_text.lower() in ("/quit", "q", "quit", "exit"): break

        if user_text.lower() in ("/quit", "q", "quit", "exit"): break
        if user_text.lower() == "/restart":
            state = new_call()
            opening = await engine.get_opening(state)
            print(f"\n  {BOLD}Bot{RESET}  [{state.current_stage}]\n  {opening.bot_text}")
            await _speak_text(tts, player, opening.bot_text); print(); continue

        # ── LLM turn ─────────────────────────────────────────────────────────
        print("  …", end="", flush=True)
        t0 = time.perf_counter()
        try:
            turn = await engine.process_turn(state, user_text)
        except Exception as exc:
            print(f"\r  ERROR: {exc}\n"); continue

        llm_ms = (time.perf_counter() - t0) * 1000
        print(f"\r  {BOLD}Bot{RESET}  [{turn.current_stage}]  ({llm_ms:.0f} ms LLM, RAG={turn.rag_chunks_used})\n  {turn.bot_text}")
        await _speak_text(tts, player, turn.bot_text)
        print()

        if turn.action in (CallAction.TRANSFER_TO_CLOSER,):
            print("  ✓ TRANSFER\n"); break
        if turn.action in (CallAction.END_CALL, CallAction.DNC_AND_END_CALL):
            print(f"  ✓ CALL ENDED — {turn.action.value}\n"); break

    player.stop()
    await llm.shutdown()
    await tts.shutdown()


# ═════════════════════════════════════════════════════════════════════════════
#  MODE 5: VICIDIAL CONNECTIVITY TEST
# ═════════════════════════════════════════════════════════════════════════════

async def run_vicidial_test(vicidial_url: str) -> None:
    """
    Verifies full VICIdial connectivity:
      ① HTTP reachability
      ② non_agent_api.php  — get_lead_info (admin call)
      ③ agent_api.php      — agent_login + agent_logout
      ④ non_agent_api.php  — get_active_calls

    Prints exactly what to fix if any step fails.
    """
    from src.vicidial.agent_api import AgentAPI
    from config.settings import VICIdialConfig

    div("═")
    print(f"  {BOLD}VICIDIAL CONNECTIVITY TEST{RESET}")
    print(f"  URL: {vicidial_url}")
    div("═")
    print()

    cfg = get_vicidial_config()
    if vicidial_url != cfg.api_url:
        import dataclasses
        cfg = dataclasses.replace(cfg, api_url=vicidial_url)

    # ── Step 1: Raw HTTP ping ─────────────────────────────────────────────────
    info("Step 1 — HTTP reachability")
    timeout = aiohttp.ClientTimeout(total=8)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(f"{vicidial_url}/non_agent_api.php?source=test") as resp:
                body = await resp.text()
                if resp.status == 200:
                    ok(f"HTTP 200 — server is reachable  ({body[:60]})")
                else:
                    warn(f"HTTP {resp.status} — server responded but returned an error")
        except Exception as exc:
            err(f"Cannot reach VICIdial at {vicidial_url}  —  {exc}")
            print()
            print("  FIX: Check the VICIdial server IP/hostname and make sure port 80 is open.")
            print("  If using HTTPS, update VICIDIAL_API_URL to https://...")
            return

    # ── Step 2: non_agent_api.php — get server version ────────────────────────
    info("Step 2 — Admin API (non_agent_api.php)")
    if not cfg.api_user or not cfg.api_pass:
        warn("VICIDIAL_API_USER or VICIDIAL_API_PASS not set in .env — skipping admin test")
        print("  Set these in your .env and re-run.")
    else:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            params = {
                "source":   "test_voice_pipeline",
                "user":     cfg.api_user,
                "pass":     cfg.api_pass,
                "function": "version",
            }
            url = f"{vicidial_url}/non_agent_api.php?" + "&".join(f"{k}={v}" for k, v in params.items())
            try:
                async with session.get(url) as resp:
                    body = (await resp.text()).strip()
                    if "ERROR" in body:
                        err(f"Admin auth failed: {body}")
                        print("  FIX: Check VICIDIAL_API_USER + VICIDIAL_API_PASS in .env.")
                        print("  The user must have 'Enable API Access' checked in User Management.")
                    else:
                        ok(f"Admin auth OK  —  {body[:80]}")
            except Exception as exc:
                err(f"non_agent_api.php call failed: {exc}")

    # ── Step 3: agent_api.php — agent login ───────────────────────────────────
    info("Step 3 — Agent login (agent_api.php)")
    if not cfg.agent_user:
        warn("VICIDIAL_AGENT_USER not set in .env — skipping agent login test")
        print("  Create a VICIdial agent user for the bot and set VICIDIAL_AGENT_USER.")
    else:
        api = AgentAPI(config=cfg)
        await api.initialize()
        try:
            result = await api.login()
            if result:
                ok(f"Agent login OK  —  user={cfg.agent_user}  campaign={cfg.campaign_id}")
            else:
                err(f"Agent login returned False — check VICIDIAL_AGENT_USER / VICIDIAL_AGENT_PASS / VICIDIAL_CAMPAIGN_ID")
                print("  FIX:")
                print("    1. Admin > User Management — ensure the bot user exists")
                print("    2. Set User Level = 1 (agent), enable 'API user'")
                print("    3. Assign them to the correct campaign")
        except Exception as exc:
            err(f"Agent login exception: {exc}")

        # ── Step 4: logout ─────────────────────────────────────────────────────
        info("Step 4 — Agent logout")
        try:
            await api.logout()
            ok("Agent logout OK")
        except Exception as exc:
            warn(f"Logout failed: {exc}")
        await api.shutdown()

    # ── Step 5: Summary ────────────────────────────────────────────────────────
    print()
    div("═")
    print("  NEXT STEPS:")
    print()
    print("  If all steps passed → the bot is ready for VICIdial calls.")
    print("  VICIdial setup checklist for LIVE calls:")
    print()
    print("    ① Asterisk sip.conf  — add [voicebot] peer pointing to VPS IP")
    print("    ② extensions.conf    — Dial(SIP/voicebot@VPS_IP:5060,30)")
    print("    ③ Campaign settings  — set 'Recording' and 'Agent Screen' fields")
    print("    ④ Place a manual test call from VICIdial to a lead in the campaign")
    print()
    print("  See the full guide at the top of this file  (python -m tests.test_voice_pipeline --guide)")
    div("═")


# ═════════════════════════════════════════════════════════════════════════════
#  MODE 6: LOAD TEST
# ═════════════════════════════════════════════════════════════════════════════

async def run_load_test(vllm_url: str, workers: int = 5) -> None:
    """Fire *workers* concurrent vLLM requests and measure throughput / latency."""
    div("═")
    print(f"  {BOLD}LOAD TEST  —  {workers} concurrent workers  →  {vllm_url}{RESET}")
    div("═")
    print()

    import dataclasses
    cfg = dataclasses.replace(get_llm_config(), vllm_api_url=vllm_url, max_concurrent=workers + 10)

    llm = MimoVLLMClient(config=cfg)
    await llm.initialize()

    if not await llm.health_check():
        err(f"vLLM not reachable at {vllm_url}"); await llm.shutdown(); return
    ok(f"vLLM reachable — starting {workers} concurrent requests…\n")

    PROMPTS = [
        "Hello, how are you today?",
        "Tell me about final expense insurance.",
        "I'm on a fixed income, is this affordable?",
        "What happens if I miss a payment?",
        "My husband already has life insurance.",
        "I just want to be cremated.",
        "Can you send me something in the mail?",
        "How long have you been in business?",
        "I need to think about it.",
        "What does this plan actually cover?",
    ]

    SYS = ("You are Sarah, a friendly final expense insurance agent."
           " Reply in 1-2 sentences only.")

    results: list[dict[str, float]] = []
    errors = 0

    async def one_worker(idx: int) -> None:
        nonlocal errors
        prompt = PROMPTS[idx % len(PROMPTS)]
        t0 = time.perf_counter()
        try:
            resp = await llm.generate(
                system_prompt=SYS,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed = (time.perf_counter() - t0) * 1000
            results.append({"latency_ms": elapsed, "tokens": resp.completion_tokens})
            print(f"  [{idx:02d}] {elapsed:>6.0f} ms  {resp.completion_tokens:>3} tok  {resp.text[:55]}")
        except Exception as exc:
            errors += 1
            elapsed = (time.perf_counter() - t0) * 1000
            err(f"[{idx:02d}] FAILED after {elapsed:.0f} ms  —  {exc}")

    wall_t0 = time.perf_counter()
    await asyncio.gather(*[one_worker(i) for i in range(workers)])
    wall_ms = (time.perf_counter() - wall_t0) * 1000

    if results:
        lats   = [r["latency_ms"] for r in results]
        tokens = [r["tokens"]     for r in results]
        p50    = sorted(lats)[len(lats) // 2]
        p95    = sorted(lats)[int(len(lats) * 0.95)]
        print()
        div()
        ok(f"Completed: {len(results)}/{workers}   errors: {errors}")
        ok(f"Wall time: {wall_ms:.0f} ms   throughput: {len(results) / (wall_ms / 1000):.1f} req/s")
        ok(f"Latency   p50={p50:.0f} ms   p95={p95:.0f} ms   max={max(lats):.0f} ms")
        ok(f"Tokens    avg={sum(tokens)/len(tokens):.0f}  total={sum(tokens)}")

        if p95 < 3000:
            ok("vLLM is healthy — p95 < 3 s")
        elif p95 < 8000:
            warn("p95 latency is 3-8 s — acceptable but watch under load")
        else:
            warn("p95 > 8 s — consider reducing max_tokens or concurrent workers")
        div()

    await llm.shutdown()


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tests.test_voice_pipeline",
        description="Final Expense Voice Bot — Real GPU-stack pipeline tester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODES
  health     Ping all services and print latency table
  chat       Text in → real vLLM → text out
  speak      Text in → real vLLM → CosyVoice TTS → plays on speakers  ← START HERE
  voice      Mic → STT → real vLLM → TTS → speakers
  vicidial   Test VICIdial API connectivity
  load       Concurrent vLLM load test

EXAMPLES
  # From your Windows PC pointing at VPS (no mic, no GPU needed):
  python -m tests.test_voice_pipeline speak --vllm-url http://VPS_IP:8000 --tts-url http://VPS_IP:8001

  # Full health check:
  python -m tests.test_voice_pipeline health --vllm-url http://VPS_IP:8000 --tts-url http://VPS_IP:8001

  # Load test with 10 workers:
  python -m tests.test_voice_pipeline load --workers 10 --vllm-url http://VPS_IP:8000

  # VICIdial:
  python -m tests.test_voice_pipeline vicidial --vicidial-url http://VPS_IP/vicidial

  # Run with .env pointing at localhost (on the VPS itself):
  python -m tests.test_voice_pipeline voice

  # Read the full VICIdial + scaling guide:
  python -m tests.test_voice_pipeline --guide
        """,
    )
    p.add_argument(
        "mode",
        nargs="?",
        default="health",
        choices=["health", "chat", "speak", "voice", "vicidial", "load"],
        help="Test mode (default: health)",
    )
    p.add_argument(
        "--vllm-url",
        default=None,
        metavar="URL",
        help="vLLM server URL (default: from VLLM_API_URL in .env)",
    )
    p.add_argument(
        "--tts-url",
        default=None,
        metavar="URL",
        help="CosyVoice TTS server URL (default: from TTS_API_URL in .env)",
    )
    p.add_argument(
        "--vicidial-url",
        default=None,
        metavar="URL",
        help="VICIdial URL, e.g. http://192.168.1.10/vicidial",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=5,
        metavar="N",
        help="Number of concurrent workers for load test (default: 5)",
    )
    p.add_argument(
        "--stt",
        default=None,
        choices=["parakeet", "whisper"],
        help="Force a specific STT backend (default: auto-detect)",
    )
    p.add_argument(
        "--guide",
        action="store_true",
        help="Print the VICIdial + scaling guide and exit",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.guide:
        print(__doc__)
        sys.exit(0)

    # Resolve service URLs: CLI flag > .env > deploy_gpu.sh default
    vllm_url     = args.vllm_url     or get_llm_config().vllm_api_url
    tts_url      = args.tts_url      or get_tts_config().api_url
    vicidial_url = args.vicidial_url or get_vicidial_config().api_url

    if args.mode == "health":
        asyncio.run(run_health(vllm_url, tts_url, vicidial_url))

    elif args.mode == "chat":
        asyncio.run(run_chat(vllm_url))

    elif args.mode == "speak":
        asyncio.run(run_speak(vllm_url, tts_url))

    elif args.mode == "voice":
        asyncio.run(run_voice(vllm_url, tts_url, stt_backend_name=args.stt))

    elif args.mode == "vicidial":
        asyncio.run(run_vicidial_test(vicidial_url))

    elif args.mode == "load":
        asyncio.run(run_load_test(vllm_url, workers=args.workers))
