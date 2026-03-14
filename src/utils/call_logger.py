"""
Structured JSONL call logger.

Writes one JSON line per event to ``logs/calls/{call_id}.jsonl``.
Each line captures a state transition, metric, or notable event during
a call so that post-call analysis can reconstruct the full timeline.

Usage::

    from src.utils.call_logger import CallLogger

    log = CallLogger(call_id)
    log.event("call_started", lead_name="John")
    log.event("stt_result", text="yes", confidence=0.92, latency_ms=120)
    log.event("llm_response", text="Great!", tokens=12, latency_ms=340)
    log.event("tts_chunk", bytes=4096, latency_ms=85)
    log.event("call_ended", reason="completed", duration_s=45.2)
    log.close()
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs" / "calls"


class CallLogger:
    """Per-call structured JSONL logger."""

    def __init__(self, call_id: str) -> None:
        self._call_id = call_id
        self._start = time.time()
        _LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self._path = _LOGS_DIR / f"{call_id}.jsonl"
        self._file = open(self._path, "a", encoding="utf-8")  # noqa: SIM115
        logger.debug("CallLogger opened: %s", self._path)

    def event(self, event_type: str, **data: Any) -> None:
        """Write a single event line."""
        record = {
            "ts": time.time(),
            "elapsed_s": round(time.time() - self._start, 3),
            "call_id": self._call_id,
            "event": event_type,
            **data,
        }
        line = json.dumps(record, default=str)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        """Flush and close the log file."""
        if self._file and not self._file.closed:
            self._file.close()
            logger.debug("CallLogger closed: %s", self._path)
