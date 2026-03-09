"""
SIP connection test — verifies SIP registration and simulated calls.

Run::

    python -m tests.test_sip_connection
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_sip_config, get_vicidial_config
from src.vicidial.agent_api import AgentAPI
from src.vicidial.sip_handler import SIPCall, SIPHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("test_sip")


async def test_sip_registration() -> bool:
    """Test SIP registration with the configured Asterisk server."""
    print("\n--- Test: SIP Registration ---")
    handler = SIPHandler()
    result = await handler.register()
    print(f"  Registration: {'OK' if result else 'FAILED'}")

    if result:
        await handler.unregister()
        print("  Unregistration: OK")
    return result


async def test_simulated_call() -> bool:
    """Test a simulated incoming call with dummy audio."""
    print("\n--- Test: Simulated Call ---")
    handler = SIPHandler()
    await handler.register()

    received_audio: list[bytes] = []

    async def on_call(sip_call: SIPCall, audio_stream, send_audio):
        print(f"  Incoming call: {sip_call.call_id}")
        # Read a few chunks
        count = 0
        async for chunk in audio_stream:
            received_audio.append(chunk)
            count += 1
            if count >= 3:
                break
        send_audio(b"\x00" * 320)  # Send silence back
        print(f"  Received {count} audio chunks, sent 1 back.")

    handler.on_incoming_call = on_call

    # Simulate a call
    call_id = "test-sip-001"
    sip_call = SIPCall(
        call_id=call_id,
        remote_uri="sip:prospect@test",
        local_uri="sip:voicebot@test",
        status="ringing",
    )
    handler._active_calls[call_id] = sip_call
    audio_stream, send_audio = await handler.answer_call(call_id)

    # Feed dummy audio
    for _ in range(5):
        handler.feed_audio(call_id, b"\x00" * 320)

    # Consume from audio stream
    count = 0
    async for chunk in audio_stream:
        count += 1
        if count >= 3:
            break

    print(f"  Consumed {count} chunks from audio stream.")

    await handler.hangup(call_id)
    print("  Hangup: OK")

    await handler.unregister()
    return count > 0


async def test_sip_transfer() -> bool:
    """Test SIP REFER (transfer) simulation."""
    print("\n--- Test: SIP Transfer ---")
    handler = SIPHandler()
    await handler.register()

    call_id = "test-transfer-001"
    sip_call = SIPCall(
        call_id=call_id,
        remote_uri="sip:prospect@test",
        local_uri="sip:voicebot@test",
        status="connected",
    )
    handler._active_calls[call_id] = sip_call

    result = await handler.transfer(call_id, "8300")
    print(f"  Transfer to 8300: {'OK' if result else 'FAILED'}")

    await handler.unregister()
    return result


async def test_vicidial_api_format() -> bool:
    """Test VICIdial API parameter formatting (no actual server needed)."""
    print("\n--- Test: VICIdial API Parameter Formatting ---")
    api = AgentAPI()
    await api.initialize()

    # Test response parsing
    sample_response = "first_name=John|last_name=Doe|state=TX|phone=5551234567"
    parsed = api._parse_vicidial_response(sample_response)
    assert parsed["first_name"] == "John", f"Expected John, got {parsed.get('first_name')}"
    assert parsed["state"] == "TX", f"Expected TX, got {parsed.get('state')}"
    print(f"  Parsed response: {parsed}")
    print("  Formatting: OK")

    await api.shutdown()
    return True


async def run_all_tests() -> None:
    """Run all SIP/VICIdial connection tests."""
    print("=" * 60)
    print("  SIP / VICIdial Connection Tests")
    print("=" * 60)

    results = {
        "SIP Registration": await test_sip_registration(),
        "Simulated Call": await test_simulated_call(),
        "SIP Transfer": await test_sip_transfer(),
        "VICIdial API Format": await test_vicidial_api_format(),
    }

    print("\n" + "=" * 60)
    print("  Results:")
    all_pass = True
    for name, passed in results.items():
        symbol = "✓" if passed else "✗"
        print(f"    {symbol} {name}")
        if not passed:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("  All tests passed!")
    else:
        print("  Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
