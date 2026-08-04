"""Token budget helpers for the realtime voice agent.

This module intentionally has no model or audio dependencies so its routing and
context-window behavior can be tested without starting the application.
"""

from __future__ import annotations

import os
import re
from typing import Iterable


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


RECENT_CONTEXT_TURNS = _positive_int_env("RECENT_CONTEXT_TURNS", 6)
VOICE_MAX_OUTPUT_TOKENS = _positive_int_env("VOICE_MAX_OUTPUT_TOKENS", 120)
TEXT_MAX_OUTPUT_TOKENS = _positive_int_env("TEXT_MAX_OUTPUT_TOKENS", 600)
TOOL_RESULT_MAX_CHARS = _positive_int_env("TOOL_RESULT_MAX_CHARS", 350)

VOICE_MODE_PROMPT = "模式=语音；用用户语言回答；最多2句、50个汉字或等量文本；口语化；不列点。"
TEXT_MODE_PROMPT = "模式=文字；用用户语言直接回答；默认不超过300字；仅在用户明确要求时展开。"


TOOL_KEYWORDS = {
    "get_current_time": (
        "几点", "时间", "星期", "日期", "何時", "時間", "曜日", "date", "time",
    ),
    "get_weather": (
        "天气", "气温", "温度", "下雨", "天气预报", "天気", "気温", "雨", "weather", "temperature",
    ),
    "open_application": (
        "打开应用", "打开程序", "启动程序", "记事本", "计算器", "文件管理器", "命令行",
        "アプリを開", "起動", "open app", "launch app", "notepad", "calculator", "explorer",
    ),
    "get_clipboard_content": (
        "读取剪贴板", "剪贴板内容", "クリップボードを読", "read clipboard",
    ),
    "set_clipboard_content": (
        "复制到剪贴板", "写入剪贴板", "クリップボードに", "copy to clipboard",
    ),
    "read_file": (
        "读取文件", "读文件", "文件内容", "ファイルを読", "read file",
    ),
    "list_directory": (
        "列出目录", "目录内容", "文件列表", "フォルダ一覧", "ディレクトリ", "list directory", "list files",
    ),
    "search_web": (
        "搜索", "搜一下", "上网查", "检索", "検索", "調べて", "search", "look up",
    ),
    "get_webpage_summary": (
        "网页摘要", "总结网页", "这个网址", "このページ", "要約", "summarize page", "summarise page", "url",
    ),
    "calculate": (
        "计算", "算一下", "等于多少", "計算", "calculate", "compute",
    ),
    "translate_text": (
        "翻译", "译成", "翻訳", "訳して", "translate",
    ),
}


def select_tool_names(text: str) -> list[str]:
    """Return only the tools that are plausibly needed for this request."""
    normalized = text.casefold()
    selected = [
        name
        for name, keywords in TOOL_KEYWORDS.items()
        if any(keyword.casefold() in normalized for keyword in keywords)
    ]

    # A page summary often needs search as a fallback, while a search result may
    # subsequently need the page reader. Keep this pair together.
    if "search_web" in selected and "get_webpage_summary" not in selected:
        selected.append("get_webpage_summary")
    if "get_webpage_summary" in selected and "search_web" not in selected:
        selected.append("search_web")
    return selected


def select_tool_schemas(text: str, schemas: Iterable[dict]) -> list[dict]:
    selected_names = set(select_tool_names(text))
    return [
        schema
        for schema in schemas
        if schema.get("function", {}).get("name") in selected_names
    ]


def build_request_messages(
    system_prompt: str,
    history: list[dict],
    *,
    is_typed: bool,
    recent_turns: int = RECENT_CONTEXT_TURNS,
) -> list[dict]:
    """Build a bounded context while keeping complete user/assistant turns."""
    clean_history = [
        message
        for message in history
        if message.get("role") in {"user", "assistant"}
        and message.get("content")
        and "tool_calls" not in message
    ]
    recent = clean_history[-(recent_turns * 2):]
    mode_prompt = TEXT_MODE_PROMPT if is_typed else VOICE_MODE_PROMPT
    return [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": mode_prompt},
        *recent,
    ]


def output_token_limit(is_typed: bool) -> int:
    return TEXT_MAX_OUTPUT_TOKENS if is_typed else VOICE_MAX_OUTPUT_TOKENS


def compact_tool_result(result: str, max_chars: int = TOOL_RESULT_MAX_CHARS) -> str:
    """Remove formatting noise and cap tool context sent back to the model."""
    compact = re.sub(r"[ \t]+", " ", str(result))
    compact = re.sub(r"\n{3,}", "\n\n", compact).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "…"


def remove_tool_exchange(history: list[dict], tool_call_ids: set[str]) -> None:
    """Remove a completed tool protocol exchange from future model context."""
    if not tool_call_ids:
        return

    def keep(message: dict) -> bool:
        if message.get("role") == "tool" and message.get("tool_call_id") in tool_call_ids:
            return False
        calls = message.get("tool_calls") or []
        return not any(call.get("id") in tool_call_ids for call in calls)

    history[:] = [message for message in history if keep(message)]


def estimate_tokens(messages: list[dict], tool_schemas: list[dict]) -> int:
    """Cheap provider-independent estimate used only for operational logs."""
    text = str(messages) + str(tool_schemas)
    cjk_chars = len(re.findall(r"[\u3400-\u9fff\u3040-\u30ff]", text))
    other_chars = len(text) - cjk_chars
    return cjk_chars + max(1, other_chars // 4)
