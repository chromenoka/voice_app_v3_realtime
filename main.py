import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import webrtcvad
import uvicorn
import asyncio
import numpy as np
from faster_whisper import WhisperModel
import torch
from openai import AsyncOpenAI
import edge_tts
import json
import os
import re
import ast
import subprocess
import datetime
import webbrowser
import httpx
import audioop  # 提前导入，避免热循环中重复 import
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

# 修复 Windows 终端 GBK 编码问题（emoji 会崩）
if os.name == "nt":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

# === 可选依赖：剪贴板 ===
try:
    import pyperclip
    HAS_CLIPBOARD = True
except ImportError:
    HAS_CLIPBOARD = False
    print("[警告] pyperclip 未安装，剪贴板功能不可用。pip install pyperclip")

# === 可选依赖：DuckDuckGo 搜索 ===
try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False
    print("[警告] duckduckgo-search 未安装，网页搜索功能不可用。pip install duckduckgo-search")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

vad = webrtcvad.Vad(3)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "small" if device == "cuda" else "tiny"
import onnxruntime as ort
dml_available = "DmlExecutionProvider" in ort.get_available_providers()
if dml_available:
    print("[显卡加速] 🚀 已成功开启 DirectML (DirectX 12) 显卡硬件加速！正在驱动 Intel Arc B570 独立显卡 GPU 核心进行并行听写计算！")
else:
    print(f"[加载中] 检测到可用设备: {device.upper()}，正在加载 Whisper 极速模型...")
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

HISTORY_FILE = "chat_memory.json"

# system 提示词抽成常量，保证「新会话」与「加载老记忆」用的是同一份。
# 关键：要求用用户提问的同种语言回复（日/中/英…），否则 TTS 选对声音也没用。
SYSTEM_PROMPT = "你是一个高级人工智能管家。请用非常口语化、像真人聊天一样的话回复。重要：用户用什么语言提问，你就用同一种语言回复（用户说日语就用日语回，说中文就用中文回，说英语就用英语回）。如果处于语音通话模式，每次回答一两句话即可，越简短越好。如果是文字聊天模式，可以详细作答。你拥有工具调用能力。当用户的问题需要实时信息（天气、时间、搜索）或系统操作时，主动使用工具。"

if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        conversation_history = json.load(f)
    # 强制刷新 system：老记忆里可能存的是旧版中文 prompt，这里用最新版覆盖，
    # 保证「用同种语言回复」这条新规则对老会话也立刻生效。
    if conversation_history and conversation_history[0].get("role") == "system":
        conversation_history[0]["content"] = SYSTEM_PROMPT
    else:
        conversation_history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    print(f"[记忆加载] 成功回忆起了 {len(conversation_history)-1} 条历史对话！")
else:
    conversation_history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]


def save_memory():
    # 内存中截断
    if len(conversation_history) > 100:
        conversation_history[:] = [conversation_history[0]] + conversation_history[-40:]
    # 持久化时过滤掉 tool_calls / tool 消息（避免重启后 API 报错）
    saveable = [msg for msg in conversation_history
                if msg.get("role") != "tool" and "tool_calls" not in msg]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(saveable, f, ensure_ascii=False, indent=2)


def clean_text_for_tts(text: str) -> str:
    """过滤掉 URL、代码块等不适合 TTS 朗读的内容"""
    # 去除 URL
    text = re.sub(r'https?://\S+', '链接', text)
    # 去除 Markdown 代码块 ```
    text = re.sub(r'```[\s\S]*?```', '', text)
    # 去除行内代码 ` `
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # 去除 Markdown 粗体/斜体标记
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    return text.strip()


# ============================================================
# 工具名称中文映射（用于前端 TOOL_CALL 通知）
# ============================================================
TOOL_NAMES_CN = {
    "get_current_time": "获取当前时间",
    "get_weather": "查询天气",
    "open_application": "打开应用程序",
    "get_clipboard_content": "读取剪贴板",
    "set_clipboard_content": "设置剪贴板",
    "read_file": "读取文件",
    "list_directory": "列出目录",
    "search_web": "搜索网页",
    "get_webpage_summary": "抓取网页摘要",
    "calculate": "计算表达式",
    "translate_text": "翻译文本",
}

# ============================================================
# OpenAI Function Calling 工具定义 (11个)
# ============================================================
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期、时间和星期几",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气信息，包括温度、天气描述、风力、湿度",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海、Tokyo、London、New York"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_application",
            "description": "打开本地Windows应用程序。支持：记事本(notepad)、计算器(calc)、浏览器(浏览器/browser)、文件管理器(explorer/文件管理器)、命令行(cmd/命令行)",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {
                        "type": "string",
                        "description": "要打开的应用程序名称，支持中英文：记事本/notepad、计算器/calc/calculator、浏览器/browser、文件管理器/explorer、命令行/cmd"
                    }
                },
                "required": ["app_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_clipboard_content",
            "description": "读取当前Windows剪贴板中的文本内容",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_clipboard_content",
            "description": "将指定文本设置到Windows剪贴板中",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "要复制到剪贴板的文本内容"
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取指定文本文件的内容，最多返回前500字。文件不存在时会返回友好提示",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "文件的完整路径，例如：C:/Users/xxx/Desktop/笔记.txt"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "列出指定目录下的文件和文件夹名称",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "目录路径，默认为当前目录。例如：C:/Users/xxx/Documents"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "使用DuckDuckGo搜索引擎搜索网页，返回前3条结果的标题、摘要和链接",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_webpage_summary",
            "description": "抓取指定网页URL并提取正文内容，返回前500字摘要",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "网页URL地址，例如：https://example.com/article"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "安全计算数学表达式，支持加减乘除、括号、幂运算。例如：2+3*4、(100-20)/5、2**10",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，只允许数字和 +-*/()**. 运算符"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "将文本翻译成指定语言。翻译独立进行，不影响主对话",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "需要翻译的文本"
                    },
                    "target_language": {
                        "type": "string",
                        "description": "目标语言，例如：中文、English、日本語、한국어、Français"
                    }
                },
                "required": ["text", "target_language"]
            }
        }
    },
]


# ============================================================
# 11 个工具函数实现
# ============================================================

def get_current_time() -> str:
    """获取当前日期 + 时间 + 星期几"""
    try:
        now = datetime.datetime.now()
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        weekday = weekdays[now.weekday()]
        return f"当前时间：{now.year}年{now.month}月{now.day}日 {now.hour:02d}:{now.minute:02d}:{now.second:02d} {weekday}"
    except Exception as e:
        return f"获取时间失败: {str(e)}"


async def get_weather(city: str) -> str:
    """通过 wttr.in 免费 API 查询城市天气"""
    if not city:
        return "请提供城市名称"
    try:
        async with httpx.AsyncClient(timeout=10.0) as http_client:
            url = f"https://wttr.in/{city}?format=j1"
            resp = await http_client.get(url)
            if resp.status_code != 200:
                return f"查询天气失败，HTTP {resp.status_code}"
            data = resp.json()
            current = data.get("current_condition", [{}])[0]
            temp = current.get("temp_C", "未知")
            desc = current.get("weatherDesc", [{}])[0].get("value", "未知")
            wind = current.get("windspeedKmph", "未知")
            humidity = current.get("humidity", "未知")
            feels_like = current.get("FeelsLikeC", temp)
            area = data.get("nearest_area", [{}])[0].get("areaName", [{}])[0].get("value", city)
            return f"{area}当前天气：温度{temp}°C（体感{feels_like}°C），{desc}，风力{wind}km/h，湿度{humidity}%"
    except httpx.TimeoutException:
        return "查询天气超时，请稍后重试"
    except Exception as e:
        return f"查询天气失败: {str(e)}"


def open_application(app_name: str) -> str:
    """打开本地 Windows 应用程序"""
    if not app_name:
        return "请提供应用程序名称"
    try:
        name_lower = app_name.strip().lower()
        # 应用名映射
        app_map = {
            "记事本": "notepad.exe",
            "notepad": "notepad.exe",
            "计算器": "calc.exe",
            "calc": "calc.exe",
            "calculator": "calc.exe",
            "浏览器": "browser",
            "browser": "browser",
            "文件管理器": "explorer.exe",
            "explorer": "explorer.exe",
            "命令行": "cmd.exe",
            "cmd": "cmd.exe",
        }
        target = app_map.get(name_lower)
        if not target:
            return f"不支持的应用: {app_name}。支持：记事本、计算器、浏览器、文件管理器、命令行"
        if target == "browser":
            webbrowser.open("")
            return "已打开默认浏览器"
        else:
            # 不用 shell=True：target 已是固定白名单里的可执行文件名，
            # 直接以列表形式启动，避免 shell 注入风险。
            subprocess.Popen([target])
            return f"已打开 {app_name}"
    except Exception as e:
        return f"打开应用失败: {str(e)}"


def get_clipboard_content() -> str:
    """读取 Windows 剪贴板文本"""
    if not HAS_CLIPBOARD:
        return "剪贴板功能不可用，请安装 pyperclip: pip install pyperclip"
    try:
        content = pyperclip.paste()
        if not content or not content.strip():
            return "剪贴板当前为空"
        if len(content) > 500:
            content = content[:500] + "\n...(内容已截断，共 " + str(len(pyperclip.paste())) + " 字符)"
        return f"剪贴板内容：\n{content}"
    except Exception as e:
        return f"读取剪贴板失败: {str(e)}"


def set_clipboard_content(text: str) -> str:
    """设置 Windows 剪贴板文本"""
    if not HAS_CLIPBOARD:
        return "剪贴板功能不可用，请安装 pyperclip: pip install pyperclip"
    if not text:
        return "请提供要复制的文本内容"
    try:
        pyperclip.copy(text)
        return f"已将文本复制到剪贴板（{len(text)} 字符）"
    except Exception as e:
        return f"设置剪贴板失败: {str(e)}"


def read_file(file_path: str) -> str:
    """读取文本文件，最多返回前 500 字"""
    if not file_path:
        return "请提供文件路径"
    try:
        if not os.path.exists(file_path):
            return f"文件不存在: {file_path}"
        if not os.path.isfile(file_path):
            return f"路径不是文件: {file_path}"
        # 检查文件大小，超过 1MB 拒绝
        size = os.path.getsize(file_path)
        if size > 1024 * 1024:
            return f"文件过大（{size / 1024:.0f}KB），仅支持读取 1MB 以内的文本文件"
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(500)
            if len(content) >= 500:
                content += "\n...(内容已截断，文件共 " + str(size) + " 字节)"
            return content
    except Exception as e:
        return f"读取文件失败: {str(e)}"


def list_directory(path: str = ".") -> str:
    """列出目录下的文件和文件夹"""
    if not path:
        path = "."
    try:
        if not os.path.exists(path):
            return f"目录不存在: {path}"
        if not os.path.isdir(path):
            return f"路径不是目录: {path}"
        items = os.listdir(path)
        if not items:
            return f"目录 '{path}' 为空"
        # 分类显示
        dirs = []
        files = []
        for item in sorted(items):
            full = os.path.join(path, item)
            if os.path.isdir(full):
                dirs.append(f"📁 {item}/")
            else:
                size = os.path.getsize(full)
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f}KB"
                else:
                    size_str = f"{size / 1024 / 1024:.1f}MB"
                files.append(f"📄 {item} ({size_str})")
        result = f"目录 '{path}' 内容（共 {len(items)} 项）：\n"
        result += "\n".join(dirs + files)
        if len(result) > 1500:
            result = result[:1500] + f"\n...(共 {len(items)} 项，已截断)"
        return result
    except PermissionError:
        return f"没有权限访问目录: {path}"
    except Exception as e:
        return f"列出目录失败: {str(e)}"


async def search_web(query: str) -> str:
    """DuckDuckGo 搜索，返回前 3 条结果"""
    if not query:
        return "请提供搜索关键词"
    if not HAS_DDGS:
        return "网页搜索功能不可用，请安装 duckduckgo-search: pip install duckduckgo-search"
    try:
        def _search():
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=3):
                    title = r.get("title", "无标题")
                    body = r.get("body", "无摘要")
                    href = r.get("href", "无链接")
                    results.append(f"📌 {title}\n   {body}\n   🔗 {href}")
            return results

        results = await asyncio.to_thread(_search)
        if not results:
            return f"未找到与 '{query}' 相关的结果"
        return f"搜索 '{query}' 的结果：\n\n" + "\n\n".join(results)
    except Exception as e:
        return f"搜索失败: {str(e)}"


async def get_webpage_summary(url: str) -> str:
    """抓取网页并提取正文前 500 字"""
    if not url:
        return "请提供网页URL"
    if not url.startswith("http"):
        url = "https://" + url
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as http_client:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            resp = await http_client.get(url, headers=headers)
            if resp.status_code != 200:
                return f"抓取网页失败，HTTP {resp.status_code}"
            soup = BeautifulSoup(resp.text, "html.parser")
            # 移除不需要的标签
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            # 清理多余空白
            text = re.sub(r'\s+', ' ', text)
            title = soup.title.string if soup.title else "无标题"
            if len(text) > 500:
                text = text[:500] + "...(内容已截断)"
            return f"网页标题：{title}\n\n正文摘要：\n{text}"
    except httpx.TimeoutException:
        return "抓取网页超时，请稍后重试"
    except Exception as e:
        return f"抓取网页失败: {str(e)}"


def _safe_eval(node: ast.AST):
    """
    递归地在 AST 上做白名单求值，彻底替代 eval。
    只允许数字、+ - * / ** % 和括号组合。
    遇到任何非算术节点（名字、调用、属性等）直接拒绝，
    从根上杜绝 9**9**9 这类指数爆炸和代码注入。
    """
    # 数字字面量
    if isinstance(node, ast.Constant):  # Python 3.8+ 的数字节点
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("不支持的常量类型")
    if isinstance(node, ast.Num):  # 兼容旧版
        return node.n
    # 二元运算：+ - * / ** % //
    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError()
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise ZeroDivisionError()
            return left // right
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise ZeroDivisionError()
            return left % right
        if isinstance(node.op, ast.Pow):
            # 限制指数避免 9**9**9**9 级别的内存爆炸
            if abs(right) > 1000 or (abs(left) > 1e6 and abs(right) > 6):
                raise ValueError("指数过大，拒绝计算")
            return left ** right
        raise ValueError("不支持的运算符")
    # 一元负号 / 正号
    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        raise ValueError("不支持的一元运算")
    # 顶层允许是表达式
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    # 其余节点（名字 Name、调用 Call、属性 Attribute 等）一律拒绝
    raise ValueError("表达式包含不允许的内容")


def calculate(expression: str) -> str:
    """安全计算数学表达式（基于 AST 白名单，不使用 eval）"""
    if not expression:
        return "请提供数学表达式"
    expr = expression.strip()
    try:
        # 先做字符白名单校验，挡掉明显非法输入
        safe_chars = set("0123456789+-*/().% ^")
        check_expr = expr.replace("**", "^")
        if not all(c in safe_chars for c in check_expr):
            return "表达式包含不允许的字符。只支持数字和 +-*/()**. 运算符"
        # 解析成 AST，再在白名单节点上递归求值
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval(tree)
        return f"{expr} = {result}"
    except ZeroDivisionError:
        return "计算出错: 除数不能为零"
    except SyntaxError:
        return f"表达式语法错误: {expr}"
    except ValueError as e:
        return f"计算出错: {str(e)}"
    except Exception as e:
        return f"计算出错: {str(e)}"


async def translate_text(text: str, target_language: str) -> str:
    """通过 DeepSeek 独立翻译（不影响主对话历史）"""
    if not text:
        return "请提供需要翻译的文本"
    if not target_language:
        target_language = "中文"
    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": f"你是一个翻译助手。请将用户输入的文本翻译成{target_language}。只返回翻译结果，不要加任何解释、注释或引号。"},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
        )
        translated = response.choices[0].message.content.strip()
        return f"翻译结果（→{target_language}）：\n{translated}"
    except Exception as e:
        return f"翻译失败: {str(e)}"


# ============================================================
# 工具调度器
# ============================================================
async def execute_tool(name: str, args: dict) -> str:
    """根据工具名称调用对应的 Python 函数"""
    try:
        if name == "get_current_time":
            return get_current_time()
        elif name == "get_weather":
            return await get_weather(args.get("city", ""))
        elif name == "open_application":
            return open_application(args.get("app_name", ""))
        elif name == "get_clipboard_content":
            return get_clipboard_content()
        elif name == "set_clipboard_content":
            return set_clipboard_content(args.get("text", ""))
        elif name == "read_file":
            return read_file(args.get("file_path", ""))
        elif name == "list_directory":
            return list_directory(args.get("path", "."))
        elif name == "search_web":
            return await search_web(args.get("query", ""))
        elif name == "get_webpage_summary":
            return await get_webpage_summary(args.get("url", ""))
        elif name == "calculate":
            return calculate(args.get("expression", ""))
        elif name == "translate_text":
            return await translate_text(
                args.get("text", ""),
                args.get("target_language", "中文")
            )
        else:
            return f"未知工具: {name}"
    except Exception as e:
        return f"工具执行失败: {str(e)}"


@app.get("/")
async def get():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/history")
async def get_history():
    """向前端返回剔除了系统设定的历史对话列表"""
    return {"history": [msg for msg in conversation_history if msg["role"] != "system"]}


TTS_VOICE_ZH = "zh-CN-XiaoxiaoNeural"
TTS_VOICE_JA = "ja-JP-NanamiNeural"
TTS_VOICE_EN = "en-US-JennyNeural"
TTS_RATE = "+25%"


def detect_tts_voice(text: str) -> str:
    # 1. 假名（平假名/片假名）出现 → 铁证日语
    if re.search(r'[぀-ゟ゠-ヿ]', text):
        return TTS_VOICE_JA
    # 2. 如果对话上下文为日语，即使这句全是汉字/标点，也优先用日文发音人
    if detect_conversation_lang() == "ja":
        return TTS_VOICE_JA
    # 3. 纯英文 → 英语发音人
    if not re.search(r'[一-鿿぀-ヿ]', text) and re.search(r'[A-Za-z]', text):
        return TTS_VOICE_EN
    # 4. 默认中文发音人
    return TTS_VOICE_ZH
    # 日文假名（平假名 + 片假名）是日语的铁证，出现即按日语念
    if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):  # ひらがな + カタカナ
        return TTS_VOICE_JA
    # 纯英文（只有拉丁字母+数字+标点，没有中日字符）→ 英语
    if not re.search(r'[\u4e00-\u9fff\u3040-\u30ff]', text) and re.search(r'[A-Za-z]', text):
        return TTS_VOICE_EN
    # 其余（含中文汉字等）默认中文
    return TTS_VOICE_ZH


def detect_conversation_lang() -> str | None:
    """
    从最近几轮对话中推断用户当前的主导语言。
    返回 whisper 支持的 language 代码（'ja'/'en'/'zh'/...），或 None（自动检测）。

    场景：用户开始说日语后，后续句子 Whisper tiny 模型容易误识别成中文。
    如果检测到最近对话含日文假名，就提示 Whisper 倾向日语，大幅提升识别率。
    """
    # 取最近 6 条用户消息
    recent = [msg["content"] for msg in conversation_history[-12:]
              if msg.get("role") == "user" and msg.get("content")]
    if not recent:
        return None
    # 拼接最近几条，统计字符类别
    combined = "".join(recent[-3:])  # 只看最近 3 条，避免早期对话干扰
    ja_count = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', combined))  # 假名
    en_count = len(re.findall(r'[A-Za-z]', combined))
    zh_count = len(re.findall(r'[\u4e00-\u9fff]', combined))
    # 日语假名占比 > 15% → 倾向日语
    if ja_count > 0 and ja_count > len(combined) * 0.15:
        return "ja"
    # 纯英文（无中日字符）→ 倾向英语
    if en_count > 0 and zh_count == 0 and ja_count == 0:
        return "en"
    # 其余让 Whisper 自动检测
    return None


def split_sentences(text: str) -> list:
    """
    把整段文字按中英文句号/问号/感叹号/换行切成短句。
    用于 TTS 流式：第一句合成完立即推送，大幅降低首字延迟。
    切完后保留原标点（朗读需要）。
    """
    # 用正则在句末标点后插入分隔符，保留标点本身
    # 匹配：。！？!?… 或连续换行
    parts = re.split(r'(?<=[。！？!?…])|(?<=\n)', text)
    sentences = []
    for p in parts:
        p = p.strip()
        if p:
            sentences.append(p)
    return sentences or [text]


async def text_to_speech_stream(text: str, websocket: WebSocket, cancel_event: asyncio.Event):
    if cancel_event.is_set():
        return
    # 过滤掉 URL 和代码块，只念自然语言部分
    clean_text = clean_text_for_tts(text)
    if not clean_text:
        return
    # 【已回退】不再按句切片。原因：拆成 N 句 = N 次 edge-tts 网络请求，
    # 每次都有 HTTP 往返延迟，句子之间播放队列容易「空窗」→ 一卡一卡。
    # 整段一次合成只有 1 次网络请求，合成完一次性推送，前端播整段顺滑。
    # 本机 CPU/内存充足，整段合成延迟可接受，体验比碎片化更稳。
    voice = detect_tts_voice(clean_text)
    communicate = edge_tts.Communicate(clean_text, voice, rate=TTS_RATE)
    audio_data = bytearray()
    try:
        async for chunk in communicate.stream():
            if cancel_event.is_set():
                return
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
        if audio_data and not cancel_event.is_set():
            await websocket.send_bytes(bytes(audio_data))
    except Exception as e:
        print(f"[TTS 错误]: {e}")


async def process_llm_and_tts(text: str, websocket: WebSocket, cancel_event: asyncio.Event, is_typed: bool = False):
    global conversation_history
    conversation_history.append({"role": "user", "content": text})

    if not is_typed:
        await websocket.send_text(f"TRANSCRIPT_USER:{text}")

    full_content = ""
    tool_calls_dict = {}
    has_tool_calls = False

    # 句切分正则：碰到句号、问号、感叹号、换行符切分
    sentence_delimiters = re.compile(r'([。！？!?，,、\n]+)')

    tts_queue = asyncio.Queue()
    
    async def tts_worker():
        while True:
            chunk_text = await tts_queue.get()
            if chunk_text is None:
                tts_queue.task_done()
                break
            if cancel_event.is_set():
                tts_queue.task_done()
                break
            if chunk_text.strip():
                await text_to_speech_stream(chunk_text, websocket, cancel_event)
            tts_queue.task_done()

    tts_worker_task = None

    try:
        stream = await client.chat.completions.create(
            model="deepseek-chat",
            messages=conversation_history,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            stream=True,
            temperature=0.7
        )

        current_sentence_buf = ""

        async for chunk in stream:
            if cancel_event.is_set():
                break
            delta = chunk.choices[0].delta

            if delta.tool_calls:
                has_tool_calls = True
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_dict:
                        tool_calls_dict[idx] = {
                            "id": tc_delta.id or "",
                            "name": (tc_delta.function.name or "") if tc_delta.function else "",
                            "arguments": ""
                        }
                    if tc_delta.function and tc_delta.function.arguments:
                        tool_calls_dict[idx]["arguments"] += tc_delta.function.arguments

            elif delta.content:
                content = delta.content
                full_content += content
                
                # 仅发送字幕
                safe_delta = content.replace("\n", "\\n")
                await websocket.send_text(f"TRANSCRIPT_AI:{safe_delta}")

                if not has_tool_calls:
                    # 启动 TTS 工作流
                    if tts_worker_task is None:
                        tts_worker_task = asyncio.create_task(tts_worker())
                    
                    current_sentence_buf += content
                    parts = sentence_delimiters.split(current_sentence_buf)
                    if len(parts) > 1:
                        # 组装完整句子并送入队列
                        for i in range(0, len(parts) - 1, 2):
                            sentence = parts[i] + (parts[i+1] if i+1 < len(parts) else "")
                            if sentence.strip():
                                tts_queue.put_nowait(sentence)
                        current_sentence_buf = parts[-1]

        # ── 无工具调用：处理剩余文本并等待 TTS ──
        if not has_tool_calls:
            if current_sentence_buf.strip():
                tts_queue.put_nowait(current_sentence_buf)
            if tts_worker_task is not None:
                tts_queue.put_nowait(None)
                await tts_queue.join()
            
            if full_content and not cancel_event.is_set():
                conversation_history.append({"role": "assistant", "content": full_content})
                save_memory()

        # ── 有工具调用：执行工具 → 二次流式 → 按句 TTS ──
        else:
            tool_calls_list = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in tool_calls_dict.values()
            ]
            conversation_history.append({
                "role": "assistant",
                "content": full_content or None,
                "tool_calls": tool_calls_list
            })

            for tc in tool_calls_dict.values():
                cn_name = TOOL_NAMES_CN.get(tc["name"], tc["name"])
                await websocket.send_text(f"TOOL_CALL:{cn_name}")
                try:
                    tool_args = json.loads(tc["arguments"])
                except json.JSONDecodeError:
                    tool_args = {}
                tool_result = await execute_tool(tc["name"], tool_args)
                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result
                })

            await websocket.send_text("TOOL_DONE")

            # 准备二次回答的 TTS 流水线
            stream2 = await client.chat.completions.create(
                model="deepseek-chat",
                messages=conversation_history,
                stream=True,
                temperature=0.7
            )
            final_reply = ""
            current_sentence_buf = ""
            tts_worker_task2 = asyncio.create_task(tts_worker())

            async for chunk in stream2:
                if cancel_event.is_set():
                    break
                d2 = chunk.choices[0].delta.content
                if d2:
                    final_reply += d2
                    await websocket.send_text(f"TRANSCRIPT_AI:{d2.replace(chr(10), chr(92)+'n')}")
                    
                    current_sentence_buf += d2
                    parts = sentence_delimiters.split(current_sentence_buf)
                    if len(parts) > 1:
                        for i in range(0, len(parts) - 1, 2):
                            sentence = parts[i] + (parts[i+1] if i+1 < len(parts) else "")
                            if sentence.strip():
                                tts_queue.put_nowait(sentence)
                        current_sentence_buf = parts[-1]

            if current_sentence_buf.strip():
                tts_queue.put_nowait(current_sentence_buf)
            
            tts_queue.put_nowait(None)
            await tts_queue.join()

            if final_reply and not cancel_event.is_set():
                conversation_history.append({"role": "assistant", "content": final_reply})
                save_memory()

    except Exception as e:
        print(f"[大模型处理错误]: {e}")




@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    FRAME_SIZE = 960
    audio_buffer = bytearray()

    is_speaking = False
    silence_frames = 0
    speech_frames = 0
    # 【极限加速优化】：将判定用户闭嘴的延迟从 450ms 缩小到 300ms (10帧)
    MAX_SILENCE_FRAMES = 10

    current_speech_buffer = bytearray()
    cancel_event = asyncio.Event()

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message:
                data = message["bytes"]
                audio_buffer.extend(data)

                VOLUME_THRESHOLD = 300
                while len(audio_buffer) >= FRAME_SIZE:
                    frame = bytes(audio_buffer[:FRAME_SIZE])
                    audio_buffer = bytearray(audio_buffer[FRAME_SIZE:])

                    try:
                        is_speech_freq = vad.is_speech(frame, 16000)
                        volume = audioop.rms(frame, 2)
                        is_speech = is_speech_freq and (volume > VOLUME_THRESHOLD)
                    except Exception:
                        continue

                    if is_speech:
                        speech_frames += 1
                        silence_frames = 0
                        current_speech_buffer.extend(frame)

                        if not is_speaking and speech_frames >= 8:
                            is_speaking = True
                            await websocket.send_text("START")
                            cancel_event.set()
                    else:
                        speech_frames = 0
                        if is_speaking:
                            silence_frames += 1
                            current_speech_buffer.extend(frame)
                            if silence_frames > MAX_SILENCE_FRAMES:
                                is_speaking = False
                                await websocket.send_text("END")

                                audio_np = np.frombuffer(current_speech_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                                # transcribe 是 CPU 密集的同步调用，丢进线程池，
                                # 避免阻塞 asyncio 事件循环（否则会卡住其他连接和 TTS 推送）
                                # 根据最近对话自动推断语种，提升日语/英语识别率
                                stt_lang = detect_conversation_lang()
                                def _stt(buf):
                                    segments, _ = whisper_model.transcribe(buf, language=stt_lang, beam_size=1)
                                    text = "".join([s.text for s in segments])
                                    return {"text": text}
                                result = await asyncio.to_thread(_stt, audio_np)
                                text = result["text"].strip()
                                # 移除单纯的标点符号幻觉
                                clean_text = text.strip(' .。!！?？,，\n\t~')

                                # 常见的 Whisper 静音/呼吸声/背景噪音幻觉黑名单（转小写后匹配）
                                hallucination_blacklist = {
                                    "you", "thank you", "thanks", "谢谢", "谢谢大家", "大家", "收看", "谢谢收看",
                                    "ご視聴", "ご視聴ありがとうございました", "ご視聴ありがとうございます",
                                    "視聴ありがとうございました", "チャンネル登録"
                                }
                                is_hallucination = clean_text.lower() in hallucination_blacklist

                                if len(clean_text) > 0 and not is_hallucination:
                                    cancel_event.clear()
                                    asyncio.create_task(process_llm_and_tts(text, websocket, cancel_event, is_typed=False))
                                else:
                                    if len(clean_text) > 0:
                                        print(f"[静音/幻觉过滤] 过滤掉疑似 Whisper 幻觉文本: '{text}'")
                                    await websocket.send_text("RESET")
                                    await websocket.send_text("RESET")

                                current_speech_buffer.clear()
            elif "text" in message:
                # 处理前端发来的打字消息或控制指令
                try:
                    payload = json.loads(message["text"])
                    if payload.get("type") == "chat":
                        user_text = payload.get("text")
                        # 如果 AI 正在说话，直接打断
                        cancel_event.set()
                        await asyncio.sleep(0.05)
                        cancel_event.clear()
                        # 开始新的回答
                        asyncio.create_task(process_llm_and_tts(user_text, websocket, cancel_event, is_typed=True))
                    elif payload.get("type") == "interrupt":
                        # 仅打断当前的语音和生成
                        cancel_event.set()
                except Exception as e:
                    pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="warning")
