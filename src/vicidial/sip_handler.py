"""
src/vicidial/sip_handler.py
════════════════════════════
SIP endpoint for the Final Expense Voice Bot.

Receives inbound calls transferred by VICIdial/Asterisk, bridges bidirectional
RTP audio to the STT/TTS pipeline, and performs SIP REFER transfers when the
lead is qualified for a human closer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW VICIDIAL + BOT INTEGRATION WORKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌──────────────────┐    ┌─────────────────────────┐    ┌────────────────┐
  │  Prospect Phone  │    │   VICIdial / Asterisk   │    │   Voice Bot    │
  │  (any phone)     │    │   (192.168.1.10)        │    │ (192.168.1.20) │
  └────────┬─────────┘    └──────────┬──────────────┘    └───────┬────────┘
           │                         │                            │
           │  ① VICIdial auto-dials  │                            │
           │◄────────────────────────│                            │
           │                         │                            │
           │  ② Prospect answers     │                            │
           │────────────────────────►│                            │
           │                         │                            │
           │    AMD = HUMAN VOICE    │  ③ Asterisk INVITE         │
           │                         │───────────────────────────►│
           │◄════════ RTP ═══════════╪══════════ RTP ════════════►│
           │   (prospect hears bot)  │                            │
           │                         │                 ④ Qualified│
           │                         │◄─── SIP REFER ─────────────│
           │                         │                            │
  ┌────────┴────────┐    ┌───────────┴─────────────┐
  │    Prospect     │◄═══╪══════ Closer Extension  │   (bot hangs up)
  └─────────────────┘    └─────────────────────────┘

STEP 1  VICIdial campaign dials the prospect using predictive pacing.

STEP 2  Prospect answers. Asterisk's AMD (Answering Machine Detector) confirms
        a human voice.

STEP 3  Asterisk executes the dialplan (see ASTERISK_SETUP below) and sends a
        SIP INVITE to the bot:
            INVITE sip:voicebot@BOT_IP:5060 SIP/2.0
        The bot answers with 200 OK containing an SDP offer for RTP.
        Bidirectional audio flows:
          • Prospect audio → RTP → Bot → STT decoder → text
          • TTS synthesis → PCM → Bot → RTP → Prospect

STEP 4  When the conversation engine decides to transfer, the bot sends a
        SIP REFER back to Asterisk:
            REFER sip:voicebot@BOT_IP SIP/2.0
            Refer-To: sip:8300@ASTERISK_IP
        Asterisk bridges the prospect to extension 8300 (closer queue).
        The bot receives a NOTIFY 200 confirming the transfer and hangs up.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASTERISK CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ── sip.conf ──────────────────────────────────────────────────────────────
  [voicebot]
  type=friend                    ; both peer and user
  host=BOT_IP_ADDRESS            ; static IP of the bot server
  secret=your_sip_password       ; must match SIP_PASSWORD in .env
  context=voicebot-campaign      ; dialplan context the bot sees
  disallow=all
  allow=ulaw                     ; G.711 µ-law (8 kHz, 8-bit)
  allow=alaw                     ; G.711 A-law  (optional)
  dtmfmode=rfc2833
  nat=force_rport,comedia        ; required if bot is behind NAT
  qualify=yes                    ; Asterisk sends periodic OPTIONS pings

  ── extensions.conf ───────────────────────────────────────────────────────
  ; Called when VICIdial decides to bridge to the bot:
  [voicebot-campaign]
  exten => voicebot,1,NoOp(=== Bridging to Voice Bot ===)
  exten => voicebot,n,Answer()
  exten => voicebot,n,Dial(SIP/voicebot,60,tT)   ; t=called-party transfer
  exten => voicebot,n,HangUp()

  ; Closer queue (bot transfers here when qualified):
  exten => 8300,1,NoOp(=== Closer Queue ===)
  exten => 8300,n,Queue(CLOSERS,tT,,,120)
  exten => 8300,n,HangUp()

  ── VICIdial Campaign Settings ────────────────────────────────────────────
  • After AMD detects human: transfer to SIP/voicebot@BOT_IP
  • Closer In-Group: CLOSERS (all human closers log into this group)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIRED PYTHON PACKAGES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pip install pyVoIP>=1.6.8     # pure-Python SIP/RTP library
                                #   https://github.com/tayler6000/pyVoIP

  Alternative (heavier, faster codec support):
  pip install pjsua2            # Python bindings for PJSIP
  (See PJSUA2_NOTES at end of file for the pjsua2 equivalent implementation)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUDIO FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • RTP codec : G.711 µ-law (PCMU) — 8 kHz, 8-bit, mono
  • Bot needs : PCM-16 LE @ 16 kHz (Parakeet STT) and 22 050 Hz (CosyVoice)
  • Resampling: scipy.signal.resample or audioop are used inline
  • Frame size : 160 samples per RTP packet (20 ms at 8 kHz)
"""

from __future__ import annotations

import asyncio
import audioop
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncIterator, Callable, Optional

from config.settings import SIPConfig, get_sip_config

logger = logging.getLogger(__name__)

# ── audio constants ────────────────────────────────────────────────────────────
RTP_SAMPLE_RATE   = 8_000        # G.711 over RTP
STT_SAMPLE_RATE   = 16_000       # Parakeet TDT input
TTS_SAMPLE_RATE   = 22_050       # CosyVoice 2 output
FRAME_SAMPLES     = 160          # 20 ms per RTP packet @ 8 kHz
FRAME_BYTES_ULAW  = FRAME_SAMPLES          # 1 byte per µ-law sample
FRAME_BYTES_PCM16 = FRAME_SAMPLES * 2      # 2 bytes per 16-bit sample
AUDIO_QUEUE_MAXSIZE = 1_000      # ~20 seconds of audio before dropping
DROPOUT_THRESHOLD_S = 3.0        # declare audio dropout after this silence


# ─────────────────────────────────────────────────────────────────────────────
# Call state machine
# ─────────────────────────────────────────────────────────────────────────────

class SIPCallState(Enum):
    RINGING      = auto()   # INVITE received, not yet answered
    CONNECTING   = auto()   # 200 OK sent, waiting for ACK
    CONNECTED    = auto()   # Media flowing
    ON_HOLD      = auto()   # Local hold (re-INVITE with sendonly)
    TRANSFERRING = auto()   # REFER sent, awaiting NOTIFY
    ENDING       = auto()   # BYE sent/received
    TERMINATED   = auto()   # Call fully over


@dataclass
class SIPCall:
    """All state for one active SIP call leg."""

    call_id:       str
    remote_uri:    str                     # who called us (VICIdial/Asterisk)
    local_uri:     str                     # our SIP URI
    lead_phone:    str        = ""         # prospect's original phone number
    lead_id:       str        = ""         # VICIdial lead_id (from SIP header)
    state:         SIPCallState = SIPCallState.RINGING
    # backward-compat string alias for state (used by test_sip_connection.py)
    status:        str        = ""
    rtp_local_port:int        = 0
    rtp_remote_port:int       = 0
    started_at:    float      = field(default_factory=time.time)
    connected_at:  float      = 0.0
    transfer_target: str      = ""         # extension when transferring
    last_audio_ts: float      = field(default_factory=time.time)

    @property
    def duration_s(self) -> float:
        if self.connected_at:
            return time.time() - self.connected_at
        return 0.0

    @property
    def has_audio_dropout(self) -> bool:
        return (time.time() - self.last_audio_ts) > DROPOUT_THRESHOLD_S


# ─────────────────────────────────────────────────────────────────────────────
# Thread-safe audio bridge  (pyVoIP callback threads ↔ asyncio event loop)
# ─────────────────────────────────────────────────────────────────────────────

class AudioBridge:
    """Thread-safe ring-buffer connecting pyVoIP's read/write threads
    to asyncio-native code in the conversation pipeline.

    Inbound path  (prospect → bot):
        pyVoIP RTP thread  ──put_inbound()──►  asyncio.Queue ──►  STT

    Outbound path (bot → prospect):
        TTS synthesis      ──put_outbound()──►  queue.Queue  ──►  pyVoIP RTP
    """

    def __init__(self) -> None:
        # asyncio queue for inbound audio (filled by pyVoIP thread, drained
        # by async code) — created lazily inside the correct event loop
        self._loop:    Optional[asyncio.AbstractEventLoop] = None
        self._inbound: Optional[asyncio.Queue[bytes]]      = None
        # Regular thread-safe queue for outbound audio
        self._outbound: queue.Queue[bytes]  = queue.Queue(maxsize=AUDIO_QUEUE_MAXSIZE)
        self._active  = threading.Event()   # set while call is connected
        self._active.set()

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind this bridge to *loop* so we can safely call put_nowait."""
        self._loop    = loop
        self._inbound = asyncio.Queue(maxsize=AUDIO_QUEUE_MAXSIZE)

    def put_inbound_threadsafe(self, ulaw_frame: bytes) -> None:
        """Called from pyVoIP's RTP receive thread.

        Converts µ-law → PCM-16 LE @ 16 kHz then pushes onto the asyncio
        queue using call_soon_threadsafe so it's safe to call from any
        thread.
        """
        if not self._active.is_set() or self._loop is None or self._inbound is None:
            return
        pcm8 = audioop.ulaw2lin(ulaw_frame, 2)              # µ-law → PCM-16 @8 kHz
        pcm16 = audioop.ratecv(pcm8, 2, 1, RTP_SAMPLE_RATE, STT_SAMPLE_RATE, None)[0]  # type: ignore
        try:
            self._loop.call_soon_threadsafe(
                self._inbound.put_nowait, pcm16
            )
        except asyncio.QueueFull:
            pass    # drop the oldest frame — backpressure from slow STT

    async def get_inbound_stream(self) -> AsyncIterator[bytes]:
        """Async generator that yields PCM-16 LE @ STT_SAMPLE_RATE frames."""
        if self._inbound is None:
            raise RuntimeError("AudioBridge not attached to an event loop.")
        inbound: asyncio.Queue[bytes] = self._inbound
        while self._active.is_set():
            try:
                chunk = await asyncio.wait_for(inbound.get(), timeout=0.5)
                yield chunk
            except asyncio.TimeoutError:
                continue

    def put_outbound(self, pcm_tts: bytes) -> None:
        """Called from TTS/async context.

        Converts PCM-16 LE @ TTS_SAMPLE_RATE → µ-law @ 8 kHz and enqueues
        for the pyVoIP write thread to pull.
        """
        if not self._active.is_set():
            return
        pcm8 = audioop.ratecv(pcm_tts, 2, 1, TTS_SAMPLE_RATE, RTP_SAMPLE_RATE, None)[0]  # type: ignore
        ulaw = audioop.lin2ulaw(pcm8, 2)
        # Split into RTP-sized frames
        for offset in range(0, len(ulaw), FRAME_BYTES_ULAW):
            frame = ulaw[offset:offset + FRAME_BYTES_ULAW]
            if len(frame) < FRAME_BYTES_ULAW:
                # Pad the final frame with silence (µ-law 0xFF = silence)
                frame = frame.ljust(FRAME_BYTES_ULAW, b"\xff")
            try:
                self._outbound.put_nowait(frame)
            except queue.Full:
                pass

    def read_outbound(self) -> Optional[bytes]:
        """Called from pyVoIP's RTP send thread.

        Returns one µ-law RTP frame or ``None`` if the queue is empty
        (pyVoIP will send a silence frame automatically).
        """
        try:
            return self._outbound.get_nowait()
        except queue.Empty:
            return None

    def drain(self) -> None:
        """Discard all buffered audio and signal end-of-call."""
        self._active.clear()
        while not self._outbound.empty():
            try:
                self._outbound.get_nowait()
            except queue.Empty:
                break


# ─────────────────────────────────────────────────────────────────────────────
# Main SIP Handler
# ─────────────────────────────────────────────────────────────────────────────

class SIPHandler:
    """Manages SIP signalling and RTP media for the voice bot.

    Registers with Asterisk as a SIP peer, listens for incoming INVITEs
    from VICIdial, and bridges audio between the RTP streams and the
    bot's STT/TTS pipeline.

    In production, set ``simulation_mode=False`` and ensure pyVoIP is
    installed (``pip install pyVoIP``).  For local testing without an
    Asterisk server, leave ``simulation_mode=True`` (the default when no
    real server is reachable).

    Usage::

        handler = SIPHandler()
        handler.on_incoming_call = my_async_call_handler
        await handler.start()          # register + start listener
        ...
        await handler.stop()
    """

    # Reconnect back-off: wait 5 s, 10 s, 20 s … up to 60 s
    BACKOFF_SEQUENCE = [5, 10, 20, 40, 60]

    def __init__(
        self,
        config: Optional[SIPConfig] = None,
    ) -> None:
        self._config   = config or get_sip_config()

        self._registered = False
        self._active_calls: dict[str, SIPCall]      = {}
        self._bridges:      dict[str, AudioBridge]  = {}

        # Injected by call_manager after construction
        self._on_incoming_call: Optional[Callable] = None

        # asyncio infrastructure
        self._loop:           Optional[asyncio.AbstractEventLoop] = None
        self._listener_task:  Optional[asyncio.Task]              = None
        self._monitor_task:   Optional[asyncio.Task]              = None

        # ── pyVoIP phone object (created in start()) ──────────────────────
        # In production this is a pyVoIP.VoIP.VoIPPhone instance.
        # We keep it as Any to avoid a hard import at module level; pyVoIP
        # is only imported when simulation_mode=False.
        self._phone = None

    # ─────────────────────────────────────────────────────────────────────
    # Property / injection
    # ─────────────────────────────────────────────────────────────────────

    @property
    def on_incoming_call(self) -> Optional[Callable]:
        return self._on_incoming_call

    @on_incoming_call.setter
    def on_incoming_call(self, handler: Callable) -> None:
        """Register a coroutine that is awaited when a new call arrives.

        Signature::

            async def handler(
                call:         SIPCall,
                audio_in:     AsyncIterator[bytes],   # PCM-16 LE @ 16 kHz
                send_audio:   Callable[[bytes], None],# PCM-16 LE @ 22050 Hz
            ) -> None: ...
        """
        self._on_incoming_call = handler

    # ─────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────

    async def start(self) -> bool:
        """Register with Asterisk and start the listener and monitor tasks.

        Returns ``True`` if registration succeeded.
        """
        self._loop = asyncio.get_event_loop()
        ok = await self.register()
        if ok:
            self._listener_task = asyncio.create_task(self._listener_loop())
            self._monitor_task  = asyncio.create_task(self._audio_monitor_loop())
        return ok

    async def stop(self) -> None:
        """Hang up all active calls, unregister, and clean up."""
        for call_id in list(self._active_calls):
            await self.hangup(call_id, reason="Server shutting down")

        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()

        await self.unregister()

    # ─────────────────────────────────────────────────────────────────────
    # Registration
    # ─────────────────────────────────────────────────────────────────────

    async def register(self) -> bool:
        """Register the bot as a SIP peer with Asterisk.

        In simulation mode this always succeeds.  In production mode it
        creates a pyVoIP VoIPPhone and calls start().
        """
        logger.info(
            "SIP registering: %s@%s:%d",
            self._config.username, self._config.server,
            self._config.port
        )
        try:
            # ── PRODUCTION: pyVoIP ─────────────────────────────────────
            # pyVoIP handles SIP registration automatically when start()
            # is called.  The callCallback is invoked in a pyVoIP thread
            # (NOT the asyncio thread) whenever an INVITE arrives.
            
            from pyVoIP.VoIP import VoIPPhone, PhoneStatus  # type: ignore
            self._phone = VoIPPhone(
                server   = self._config.server,
                port     = self._config.port,
                username = self._config.username,
                password = self._config.password,
                myIP     = self._config.local_ip,  # add to .env: SIP_LOCAL_IP
                myPort   = self._config.port,
                callCallback = self._pyvoip_call_callback,
            )
            self._phone.start()
            self._registered = True
            logger.info("SIP registration successful.")
            return True

        except Exception:
            logger.exception("SIP registration failed.")
            self._registered = False
            return False

    async def unregister(self) -> None:
        """De-register from Asterisk and stop pyVoIP."""
        if self._phone is not None:
            self._phone.stop()
            self._phone = None
        self._registered = False
        logger.info("SIP endpoint unregistered.")

    # ─────────────────────────────────────────────────────────────────────
    # Listener loop (handles reconnection)
    # ─────────────────────────────────────────────────────────────────────

    async def _listener_loop(self) -> None:
        """Watch registration health and reconnect on failure.

        pyVoIP runs its own listener threads — we just need to monitor
        the registration state and re-register if Asterisk drops us
        (e.g. after a server restart).
        """
        backoff_iter = iter(self._BACKOFF_SEQUENCE())
        consecutive_failures = 0

        while True:
            try:
                # ── PRODUCTION ──────────────────────────────────────────
                from pyVoIP.VoIP import PhoneStatus  # type: ignore
                if self._phone and self._phone.status != PhoneStatus.REGISTERED:
                    logger.warning("SIP registration lost — reconnecting …")
                    await self.unregister()
                    ok = await self.register()
                    if not ok:
                        raise RuntimeError("Re-registration failed")
                await asyncio.sleep(5)
                consecutive_failures = 0

            except asyncio.CancelledError:
                break
            except Exception:
                consecutive_failures += 1
                wait = self._backoff_wait(consecutive_failures)
                logger.exception("SIP listener error; retry in %ds", wait)
                await asyncio.sleep(wait)

    def _BACKOFF_SEQUENCE(self):  # noqa: N802
        for s in self.BACKOFF_SEQUENCE:
            yield s
        while True:
            yield 60

    @staticmethod
    def _backoff_wait(attempt: int) -> int:
        sequence = SIPHandler.BACKOFF_SEQUENCE
        return sequence[min(attempt - 1, len(sequence) - 1)]

    # ─────────────────────────────────────────────────────────────────────
    # Audio dropout monitor
    # ─────────────────────────────────────────────────────────────────────

    async def _audio_monitor_loop(self) -> None:
        """Periodically check all active calls for audio dropouts.

        If no audio has been received for DROPOUT_THRESHOLD_S seconds on
        a connected call, an event is logged.  The call_manager can hook
        into this to escalate (renegotiate or hang up).
        """
        while True:
            try:
                await asyncio.sleep(1.0)
                for call_id, call in list(self._active_calls.items()):
                    if call.state == SIPCallState.CONNECTED and call.has_audio_dropout:
                        logger.warning(
                            "Audio dropout detected on call %s "
                            "(last audio %.1fs ago)",
                            call_id,
                            time.time() - call.last_audio_ts,
                        )
                        # In production, optionally send a re-INVITE to
                        # renegotiate media or trigger a hangup:
                        # await self.hangup(call_id, reason="Audio dropout")
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Audio monitor error")

    # ─────────────────────────────────────────────────────────────────────
    # Incoming call handling
    # ─────────────────────────────────────────────────────────────────────

    def _pyvoip_call_callback(self, call) -> None:
        """Invoked by pyVoIP's internal thread when an INVITE arrives.

        IMPORTANT: This runs in a NON-asyncio thread.  Do NOT await
        anything here — use asyncio.run_coroutine_threadsafe() instead.

        The call object is a pyVoIP.VoIP.VoIPCall instance.
        """
        # ── PRODUCTION CODE ────────────────────────────────────────────────
        from pyVoIP.VoIP import CallState  # type: ignore
        
        if call.state == CallState.RINGING:
            call_id = call.call_id
        
            # Parse the From header to extract lead info VICIdial may
            # embed in SIP User-Agent or X-headers:
            remote_uri = call.call_from  # e.g. "sip:2145550001@asterisk"
        
            # Answer the call — this sends 200 OK + SDP
            call.answer()
        
            sip_call = SIPCall(
                call_id=call_id,
                remote_uri=remote_uri,
                local_uri=f"sip:{self._config.username}@{self._config.server}",
                state=SIPCallState.CONNECTED,
                connected_at=time.time(),
            )
            self._active_calls[call_id] = sip_call
        
            bridge = AudioBridge()
            bridge.attach_loop(self._loop)
            self._bridges[call_id] = bridge
        
            # pyVoIP read thread: pull audio from RTP and push to bridge
            def _rtp_reader():
                while call.state == CallState.ANSWERED:
                    frame = call.read_audio(FRAME_BYTES_ULAW, blocking=True)
                    if frame:
                        sip_call.last_audio_ts = time.time()
                        bridge.put_inbound_threadsafe(frame)
            threading.Thread(target=_rtp_reader, daemon=True).start()
        
            # pyVoIP write thread: pull audio from bridge and write to RTP
            def _rtp_writer():
                while call.state == CallState.ANSWERED:
                    frame = bridge.read_outbound()
                    if frame:
                        call.write_audio(frame)
                    else:
                        # Send silence to keep RTP alive
                        call.write_audio(b"\xff" * FRAME_BYTES_ULAW)
                    time.sleep(0.02)   # 20 ms pacing
            threading.Thread(target=_rtp_writer, daemon=True).start()
        
            # Dispatch to the asyncio handler (non-blocking)
            if self._on_incoming_call and self._loop:
                asyncio.run_coroutine_threadsafe(
                    self._dispatch_call(sip_call, bridge), self._loop
                )

    async def _dispatch_call(self, call: SIPCall, bridge: AudioBridge) -> None:
        """Invoke the user-supplied on_incoming_call handler."""
        if self._on_incoming_call is None:
            logger.warning("Incoming call %s but no handler set — hanging up.", call.call_id)
            await self.hangup(call.call_id, reason="No handler")
            return

        audio_in = bridge.get_inbound_stream()

        def send_audio(pcm_tts: bytes) -> None:
            """Called by the TTS pipeline to push audio to the prospect."""
            bridge.put_outbound(pcm_tts)

        try:
            await self._on_incoming_call(call, audio_in, send_audio)
        except Exception:
            logger.exception("Error in call handler for %s", call.call_id)
        finally:
            await self.hangup(call.call_id, reason="Handler completed")

    # ─────────────────────────────────────────────────────────────────────
    # Answering a call
    # ─────────────────────────────────────────────────────────────────────

    async def answer_call(self, call_id: str) -> tuple[AsyncIterator[bytes], Callable[[bytes], None]]:
        """Answer a ringing call and return the audio I/O handles.

        Returns
        -------
        audio_in : AsyncIterator[bytes]
            Async generator of PCM-16 LE @ 16 kHz frames from the prospect.
        send_audio : Callable[[bytes], None]
            Call with PCM-16 LE @ 22 050 Hz to speak to the prospect.
        """
        call = self._active_calls.get(call_id)
        if call is None:
            raise ValueError(f"Unknown call ID: {call_id!r}")

        call.state = SIPCallState.CONNECTED
        call.connected_at = time.time()

        bridge = AudioBridge()
        bridge.attach_loop(asyncio.get_event_loop())
        self._bridges[call_id] = bridge

        logger.info("Answered call %s  remote=%s", call_id, call.remote_uri)
        return bridge.get_inbound_stream(), bridge.put_outbound

    # ─────────────────────────────────────────────────────────────────────
    # Hangup
    # ─────────────────────────────────────────────────────────────────────

    async def hangup(self, call_id: str, *, reason: str = "Normal clearing") -> None:
        """Terminate a call and clean up all associated resources.

        Sends a SIP BYE (or 486 Busy if still ringing) and drains the
        audio bridge.
        """
        call = self._active_calls.pop(call_id, None)
        bridge = self._bridges.pop(call_id, None)

        if bridge:
            bridge.drain()

        if call:
            prev_state = call.state
            call.state = SIPCallState.TERMINATED
            logger.info(
                "Hangup call %s  prev_state=%s  duration=%.1fs  reason=%s",
                call_id, prev_state.name, call.duration_s, reason,
            )
            # ── PRODUCTION ───────────────────────────────────────────
            pyvoip_call = self._get_live_call(call_id)
            if pyvoip_call:
                pyvoip_call.hangup()

    # ─────────────────────────────────────────────────────────────────────
    # Blind transfer  (SIP REFER)
    # ─────────────────────────────────────────────────────────────────────

    async def blind_transfer(
        self,
        call_id: str,
        target_extension: str,
        target_domain: Optional[str] = None,
    ) -> bool:
        """Perform a blind SIP REFER transfer to *target_extension*.

        How it works:
        ─────────────
        1.  Bot sends:  REFER sip:voicebot@BOT_IP SIP/2.0
                        Refer-To: sip:<target_extension>@<asterisk_ip>
        2.  Asterisk receives the REFER and immediately bridges the
            prospect to the target extension / queue.
        3.  Asterisk sends back:  NOTIFY  (with body "SIP/2.0 200 OK")
        4.  Bot sends:  BYE (we're no longer needed in the call path).

        Parameters
        ----------
        call_id:
            The call to transfer.
        target_extension:
            The closer's extension or in-group number (e.g. "8300").
        target_domain:
            The Asterisk server IP.  Defaults to ``SIP_SERVER`` from config.
        """
        call = self._active_calls.get(call_id)
        if call is None:
            logger.error("blind_transfer: unknown call %s", call_id)
            return False

        domain = target_domain or self._config.server
        refer_to = f"sip:{target_extension}@{domain}"

        call.state = SIPCallState.TRANSFERRING
        call.transfer_target = target_extension

        logger.info("SIP REFER  call=%s  target=%s", call_id, refer_to)

        # ── PRODUCTION ──────────────────────────────────────────────────────
        # pyVoIP exposes call.transfer(target_uri) which sends a SIP REFER.
        
        pyvoip_call = self._get_live_call(call_id)
        if pyvoip_call is None:
            return False
        try:
            pyvoip_call.transfer(refer_to)
            # Wait for NOTIFY confirming the REFER was accepted
            for _ in range(30):                  # 3-second timeout
                if call.state == SIPCallState.TERMINATED:
                    break
                await asyncio.sleep(0.1)
            await self.hangup(call_id, reason=f"Transfer completed → {refer_to}")
            return True
        except Exception:
            logger.exception("SIP REFER failed for call %s → %s", call_id, refer_to)
            call.state = SIPCallState.CONNECTED   # roll back state
            return False

        return False  # unreachable in simulation, but satisfies type checker

    # ─────────────────────────────────────────────────────────────────────
    # Warm transfer  (hold + bridge + REFER)
    # ─────────────────────────────────────────────────────────────────────

    async def warm_transfer(
        self,
        call_id: str,
        target_extension: str,
        whisper_audio: Optional[bytes] = None,
        *,
        target_domain: Optional[str] = None,
    ) -> bool:
        """Place the prospect on hold, whisper to the closer, then connect.

        Sequence
        ────────
        1.  Bot sends re-INVITE with ``a=sendonly`` → prospect hears hold music.
        2.  Bot calls the closer extension and plays *whisper_audio* to brief
            the closer (lead name, stage, collected info, objections).
        3.  Bot sends REFER bridging the prospect to the closer.
        4.  Both legs are connected; bot hangs up.

        Parameters
        ----------
        whisper_audio:
            PCM-16 LE @ 22 050 Hz audio clip to play to the closer
            before bridging.  If ``None``, a silent hold is used.
        """
        call = self._active_calls.get(call_id)
        if call is None:
            logger.error("warm_transfer: unknown call %s", call_id)
            return False

        domain = target_domain or self._config.server
        logger.info(
            "Warm transfer: call=%s → %s  (whisper=%s)",
            call_id, f"sip:{target_extension}@{domain}",
            "yes" if whisper_audio else "none",
        )

        # ── PRODUCTION ──────────────────────────────────────────────────────
        # Step 1: Put prospect on hold
        # await self._send_hold(call_id)
        #
        # Step 2: Dial closer, play whisper
        # closer_call = await self._dial_out(f"sip:{target_extension}@{domain}")
        # if whisper_audio:
        #     bridge = self._bridges.get(call_id)
        #     if bridge:
        #         bridge.put_outbound(whisper_audio)
        # await asyncio.sleep(2)   # give closer a moment to pick up
        #
        # Step 3: REFER prospect to closer
        # success = await self.blind_transfer(call_id, target_extension, target_domain)
        # return success

        return False

    # ─────────────────────────────────────────────────────────────────────
    # Hold / Resume
    # ─────────────────────────────────────────────────────────────────────

    async def put_on_hold(self, call_id: str) -> bool:
        """Send a re-INVITE with ``a=sendonly`` to put the prospect on hold."""
        call = self._active_calls.get(call_id)
        if call is None:
            return False
        # ── PRODUCTION ────────────────────────────────────────────────────
        # Send re-INVITE with modified SDP:
        #   a=sendonly  (bot → prospect)
        # pyVoIP does not expose hold natively; you'd build a raw SIP request
        # using pyVoIP.SIP.SIPMessage or switch to pjsua2 which has hold().
        return False

    async def resume_from_hold(self, call_id: str) -> bool:
        """Send a re-INVITE with ``a=sendrecv`` to resume media."""
        call = self._active_calls.get(call_id)
        if call is None:
            return False
        # ── PRODUCTION ────────────────────────────────────────────────────
        return False

    # ─────────────────────────────────────────────────────────────────────
    # Simulation helpers  (for local testing without Asterisk)
    # ─────────────────────────────────────────────────────────────────────

    def _get_live_call(self, call_id: str):
        """Find the corresponding pyVoIP call object by ID."""
        if not self._phone:
            return None
        for call_obj, _ in self._phone.calls.items():
            if call_obj.call_id == call_id:
                return call_obj
        return None

    async def simulate_incoming_call(
        self,
        call_id: str,
        remote_uri: str = "sip:prospect@vicidial",
        lead_phone: str = "2145550001",
        lead_id: str = "12345",
    ) -> SIPCall:
        """Create a fake incoming call for local testing.

        Populates ``_active_calls``, invokes ``on_incoming_call`` callback,
        and returns the ``SIPCall`` object.

        Example::

            handler = SIPHandler(simulation_mode=True)
            call = await handler.simulate_incoming_call("test-001")
        """
        sip_call = SIPCall(
            call_id    = call_id,
            remote_uri = remote_uri,
            local_uri  = f"sip:{self._config.username}@{self._config.server}",
            lead_phone = lead_phone,
            lead_id    = lead_id,
            state      = SIPCallState.RINGING,
        )
        self._active_calls[call_id] = sip_call

        audio_in, send_audio = await self.answer_call(call_id)

        if self._on_incoming_call:
            asyncio.create_task(
                self._on_incoming_call(sip_call, audio_in, send_audio)
            )

        return sip_call

    def feed_audio(self, call_id: str, audio_pcm16_8khz: bytes) -> None:
        """Push raw PCM audio into a simulated call's inbound queue.

        Use this in tests to simulate the prospect speaking.

        Parameters
        ----------
        audio_pcm16_8khz:
            Raw little-endian 16-bit signed PCM at 8 kHz.
            The bridge will upsample to 16 kHz for the STT engine.
        """
        bridge = self._bridges.get(call_id)
        if bridge:
            # Convert to µ-law first (what the bridge expects from RTP)
            ulaw = audioop.lin2ulaw(audio_pcm16_8khz, 2)
            bridge.put_inbound_threadsafe(ulaw)
        else:
            logger.warning("feed_audio: no bridge for call %s", call_id)

    # ─────────────────────────────────────────────────────────────────────    # Backward-compatible aliases
    # ─────────────────────────────────────────────────────────────────

    async def transfer(self, call_id: str, extension: str) -> bool:
        """Alias for ``blind_transfer`` — kept for test / legacy compatibility."""
        return await self.blind_transfer(call_id, extension)

    # ─────────────────────────────────────────────────────────────────    # Introspection
    # ─────────────────────────────────────────────────────────────────────

    @property
    def active_call_count(self) -> int:
        return len(self._active_calls)

    def get_call(self, call_id: str) -> Optional[SIPCall]:
        return self._active_calls.get(call_id)

    def is_registered(self) -> bool:
        return self._registered


# =============================================================================
# PJSUA2_NOTES — alternative implementation using pjsua2
# =============================================================================
#
# If you need more codec options (Opus, G.729) or better NAT traversal,
# replace pyVoIP with pjsua2.  Install:
#   pip install pjsua2
# (pjsua2 requires the PJSIP shared library to be installed first:
#   apt-get install libpjsip-dev  # Ubuntu/Debian
#   or build from source: https://www.pjsip.org/download.htm )
#
# PJSUA2 REGISTRATION SNIPPET:
# ──────────────────────────────
#   import pjsua2 as pj
#
#   class BotAccount(pj.Account):
#       def __init__(self, handler: SIPHandler):
#           super().__init__()
#           self._handler = handler
#
#       def onRegState(self, prm):
#           ai = self.getInfo()
#           if ai.regIsActive:
#               self._handler._registered = True
#               logger.info("pjsua2: registered OK")
#           else:
#               self._handler._registered = False
#
#       def onIncomingCall(self, prm):
#           call = BotCall(self, self._handler, call_id=prm.callId)
#           call_prm = pj.CallOpParam()
#           call_prm.statusCode = 200
#           call.answer(call_prm)
#
#   class BotCall(pj.Call):
#       def __init__(self, acc, handler, call_id):
#           super().__init__(acc, call_id)
#           self._handler = handler
#
#       def onCallMediaState(self, prm):
#           ci = self.getInfo()
#           for mi in ci.media:
#               if mi.type == pj.PJMEDIA_TYPE_AUDIO:
#                   if mi.status == pj.PJSUA_CALL_MEDIA_ACTIVE:
#                       m = self.getMedia(mi.index)
#                       am = pj.AudioMedia.typecastFromMedia(m)
#                       # Connect to custom media port for bot audio I/O
#
#   # Setup endpoint
#   ep_cfg = pj.EpConfig()
#   ep_cfg.uaConfig.maxCalls = 100
#   ep = pj.Endpoint()
#   ep.libCreate()
#   ep.libInit(ep_cfg)
#   tp_cfg = pj.TransportConfig()
#   tp_cfg.port = 5060
#   ep.transportCreate(pj.PJSIP_TRANSPORT_UDP, tp_cfg)
#   ep.libStart()
#
#   acc_cfg = pj.AccountConfig()
#   acc_cfg.idUri = f"sip:{username}@{server}"
#   acc_cfg.regConfig.registrarUri = f"sip:{server}"
#   cred = pj.AuthCredInfo("digest", "*", username, 0, password)
#   acc_cfg.sipConfig.authCreds.append(cred)
#   acc = BotAccount(self)
#   acc.create(acc_cfg)
