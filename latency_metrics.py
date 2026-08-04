"""Server-side latency markers for voice-turn experiments.

The metrics in this module stop at the WebSocket send boundary. They do not
measure browser buffering, audio decoding, or the moment sound reaches the
listener.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
import uuid
from collections.abc import Callable
from typing import Any


Clock = Callable[[], float]


@dataclass
class TurnLatencyTrace:
    """Collect monotonic timestamps for one accepted voice turn."""

    clock: Clock = time.perf_counter
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    _events: dict[str, float] = field(default_factory=dict, init=False)
    _emitted: bool = field(default=False, init=False)

    def mark(self, event: str) -> None:
        """Record an event once so retries cannot silently move a milestone."""
        self._events.setdefault(event, self.clock())

    def record(self, *, path: str, outcome: str) -> dict[str, Any]:
        origin = self._events.get("turn_end")
        events_ms = {
            name: _milliseconds(timestamp - origin)
            for name, timestamp in self._events.items()
            if origin is not None
        }

        durations_ms: dict[str, float] = {}
        _add_duration(durations_ms, "asr", self._events, "turn_end", "asr_done")
        _add_duration(
            durations_ms,
            "asr_to_llm_first_text",
            self._events,
            "asr_done",
            "llm_first_text",
        )
        _add_duration(
            durations_ms,
            "llm_first_text_to_first_audio_sent",
            self._events,
            "llm_first_text",
            "audio_first_segment_sent",
        )
        _add_duration(
            durations_ms,
            "turn_end_to_first_audio_sent",
            self._events,
            "turn_end",
            "audio_first_segment_sent",
        )
        _add_duration(
            durations_ms,
            "server_turn_total",
            self._events,
            "turn_end",
            "response_done",
        )

        return {
            "schema_version": 1,
            "turn_id": self.turn_id,
            "mode": "voice",
            "path": path,
            "outcome": outcome,
            "events_ms_from_turn_end": events_ms,
            "durations_ms": durations_ms,
            "boundary": "server_websocket_send",
        }

    def emit(self, *, path: str, outcome: str) -> dict[str, Any] | None:
        """Print one machine-readable record and return it for callers/tests."""
        if self._emitted:
            return None
        self._emitted = True
        payload = self.record(path=path, outcome=outcome)
        print(f"[Latency] {json.dumps(payload, ensure_ascii=False, sort_keys=True)}")
        return payload


def _add_duration(
    target: dict[str, float],
    name: str,
    events: dict[str, float],
    start: str,
    end: str,
) -> None:
    if start in events and end in events:
        target[name] = _milliseconds(events[end] - events[start])


def _milliseconds(seconds: float) -> float:
    return round(seconds * 1000, 1)
