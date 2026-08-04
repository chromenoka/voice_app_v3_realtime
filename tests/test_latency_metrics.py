from latency_metrics import TurnLatencyTrace


class FakeClock:
    def __init__(self, *values: float):
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def test_voice_latency_record_uses_turn_end_as_origin():
    trace = TurnLatencyTrace(
        clock=FakeClock(10.0, 10.25, 10.30, 10.80, 11.10, 11.12, 12.0),
        turn_id="turn-test",
    )
    for event in (
        "turn_end",
        "asr_done",
        "llm_request_started",
        "llm_first_text",
        "tts_first_segment_ready",
        "audio_first_segment_sent",
        "response_done",
    ):
        trace.mark(event)

    record = trace.record(path="direct", outcome="completed")

    assert record["events_ms_from_turn_end"]["llm_first_text"] == 800.0
    assert record["durations_ms"] == {
        "asr": 250.0,
        "asr_to_llm_first_text": 550.0,
        "llm_first_text_to_first_audio_sent": 320.0,
        "turn_end_to_first_audio_sent": 1120.0,
        "server_turn_total": 2000.0,
    }
    assert record["boundary"] == "server_websocket_send"


def test_missing_events_are_omitted_instead_of_invented():
    trace = TurnLatencyTrace(clock=FakeClock(5.0, 5.4), turn_id="turn-empty")
    trace.mark("turn_end")
    trace.mark("asr_done")

    record = trace.record(path="asr_only", outcome="filtered")

    assert record["durations_ms"] == {"asr": 400.0}
    assert "turn_end_to_first_audio_sent" not in record["durations_ms"]


def test_mark_keeps_first_timestamp_for_retried_event():
    trace = TurnLatencyTrace(clock=FakeClock(1.0, 2.0), turn_id="turn-once")
    trace.mark("turn_end")
    trace.mark("turn_end")

    assert trace.record(path="direct", outcome="completed")[
        "events_ms_from_turn_end"
    ]["turn_end"] == 0.0
