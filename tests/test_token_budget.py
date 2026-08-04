from token_budget import (
    build_request_messages,
    compact_tool_result,
    output_token_limit,
    remove_tool_exchange,
    select_tool_names,
)


def test_plain_chat_has_no_tool_schema():
    assert select_tool_names("你好，介绍一下你自己") == []


def test_multilingual_tool_routing():
    assert "get_weather" in select_tool_names("東京の天気を教えて")
    assert "calculate" in select_tool_names("calculate 12 * 8")
    assert set(select_tool_names("搜索这个网页")) == {"search_web", "get_webpage_summary"}


def test_context_is_bounded_and_has_mode_prompt():
    history = [{"role": "system", "content": "old"}]
    for index in range(10):
        history.extend([
            {"role": "user", "content": f"u{index}"},
            {"role": "assistant", "content": f"a{index}"},
        ])

    messages = build_request_messages("base", history, is_typed=False, recent_turns=2)
    assert len(messages) == 6
    assert messages[0] == {"role": "system", "content": "base"}
    assert "模式=语音" in messages[1]["content"]
    assert [message["content"] for message in messages[-4:]] == ["u8", "a8", "u9", "a9"]


def test_tool_results_are_compact():
    assert compact_tool_result("a   b\n\n\n\nc", max_chars=20) == "a b\n\nc"
    assert compact_tool_result("123456", max_chars=4) == "1234…"


def test_completed_tool_protocol_is_removed():
    history = [
        {"role": "user", "content": "天气"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "sunny"},
    ]
    remove_tool_exchange(history, {"call-1"})
    assert history == [{"role": "user", "content": "天气"}]


def test_voice_output_budget_is_smaller():
    assert output_token_limit(False) < output_token_limit(True)
