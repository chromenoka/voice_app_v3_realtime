"""
voice_app_v3_realtime — Phase 1 MVP
实时音频流 + VAD 断句检测（不接入 LLM / TTS）

后端：FastAPI + WebSocket + webrtcvad
前端：static/index.html（Web Audio API + 原生 WebSocket）
"""

import os
import sys
import struct
import time
import logging
import math
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ────────────────────── 配置 ──────────────────────
HOST = "127.0.0.1"
PORT = 8000

# VAD 参数
VAD_FRAME_MS = 30               # 每帧 30ms
VAD_SAMPLE_RATE = 16000         # 16kHz
VAD_FRAME_SAMPLES = int(VAD_SAMPLE_RATE * VAD_FRAME_MS / 1000)  # 480 samples
VAD_FRAME_BYTES = VAD_FRAME_SAMPLES * 2                          # 960 bytes

# 能量 VAD 阈值
ENERGY_THRESHOLD = 500          # RMS 能量阈值，低于此值视为静音
ADAPTIVE_FLOOR = 100            # 自适应底噪最低值，防止静音环境阈值过低

# 断句阈值
SPEECH_START_FRAMES = 3         # 连续 3 帧有声 → 认为开始说话 (~90ms)
SILENCE_END_FRAMES = 17         # 连续 17 帧静音 → 认为一句话结束 (~500ms)

# ────────────────────── 日志 ──────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("realtime_vad")


# ────────────────────── 能量 VAD（纯 Python，无需编译） ──────────────────────
class EnergyVAD:
    """
    基于 RMS 能量检测的 VAD，接口兼容 webrtcvad.Vad。
    纯 Python 实现，零外部依赖，无需 C 编译器。

    优势：跨平台无编译痛点，自适应底噪
    劣势：比 webrtcvad 的频谱模型准确度略低（对 MVP 足够）
    后续可无缝切换为 silero-vad 或 webrtcvad
    """

    def __init__(self, threshold: int = ENERGY_THRESHOLD):
        self._threshold = threshold
        self._noise_floor = 0.0      # 自适应底噪估计
        self._noise_samples = 0       # 已采样的噪音帧数
        self._calibrated = False

    def is_speech(self, frame_bytes: bytes, sample_rate: int) -> bool:
        """
        判定一帧 PCM 数据是否为语音。
        参数：
            frame_bytes: 16-bit little-endian PCM 帧（原始字节）
            sample_rate: 采样率（保留兼容 webrtcvad 接口）
        返回：True=有声音, False=静音
        """
        # 将字节帧解析为 int16 样本
        num_samples = len(frame_bytes) // 2
        fmt = f"<{num_samples}h"  # little-endian signed short
        try:
            samples = struct.unpack(fmt, frame_bytes)
        except struct.error:
            return False

        # 计算 RMS 能量
        sum_sq = 0.0
        for s in samples:
            sum_sq += float(s) * float(s)
        rms = math.sqrt(sum_sq / num_samples)

        # 前 50 帧用于校准底噪（假设初期为静音环境）
        if not self._calibrated:
            self._noise_samples += 1
            # 指数移动平均估计底噪
            self._noise_floor = 0.9 * self._noise_floor + 0.1 * rms
            if self._noise_samples >= 50:
                self._noise_floor = max(self._noise_floor, ADAPTIVE_FLOOR)
                self._calibrated = True
                logger.info(
                    f"VAD 底噪校准完成（{self._noise_samples} 帧），"
                    f"底噪 RMS={self._noise_floor:.1f}，"
                    f"语音阈值 RMS={self._threshold:.1f}"
                )
            return False

        # 判定：RMS 超过阈值即为语音
        return rms > self._threshold


# ────────────────────── FastAPI ──────────────────────
app = FastAPI(title="Voice Chat Realtime MVP")

# 静态文件服务
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """返回前端页面"""
    return FileResponse(str(static_dir / "index.html"))


# ────────────────────── VAD 状态机 ──────────────────────
class VADStateMachine:
    """维护 listening / speaking 状态，检测断句边界"""

    def __init__(self):
        self.state = "listening"
        self.silence_counter = 0
        self.speech_counter = 0
        self.utterance_count = 0  # 已检测到的句子数（仅日志用）

    def process(self, is_speech: bool) -> str | None:
        """
        每收到一帧 VAD 判定结果就调用一次。
        返回：状态变化时返回新状态字符串，无变化返回 None
        """
        if self.state == "listening":
            if is_speech:
                self.speech_counter += 1
                if self.speech_counter >= SPEECH_START_FRAMES:
                    self.state = "speaking"
                    self.speech_counter = 0
                    self.silence_counter = 0
                    return "speaking"
            else:
                self.speech_counter = 0

        elif self.state == "speaking":
            if not is_speech:
                self.silence_counter += 1
                if self.silence_counter >= SILENCE_END_FRAMES:
                    # 一句话结束
                    self.utterance_count += 1
                    logger.info(
                        f"检测到用户说完一句话了！"
                        f"（第 {self.utterance_count} 句，"
                        f"静音 {self.silence_counter * VAD_FRAME_MS}ms）"
                    )
                    self.state = "listening"
                    self.silence_counter = 0
                    return "silence"
            else:
                self.silence_counter = 0

        return None


# ────────────────────── WebSocket 端点 ──────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("客户端已连接")

    # 每个连接独立的 VAD 实例和状态机
    vad = webrtcvad.Vad(VAD_MODE)
    state_machine = VADStateMachine()

    # 统计
    frame_count = 0
    speech_frame_count = 0

    try:
        while True:
            # 接收二进制 PCM 帧
            data = await ws.receive_bytes()

            # 验帧
            if len(data) != VAD_FRAME_BYTES:
                logger.warning(
                    f"收到异常帧长 {len(data)} bytes（期望 {VAD_FRAME_BYTES}），已丢弃"
                )
                continue

            frame_count += 1

            # VAD 判定
            is_speech = vad.is_speech(data, VAD_SAMPLE_RATE)
            if is_speech:
                speech_frame_count += 1

            # 送入状态机
            new_state = state_machine.process(is_speech)

            if new_state:
                timestamp = time.time()
                # 发送状态变更给前端
                await ws.send_json({
                    "type": "status",
                    "state": new_state,
                    "timestamp": timestamp,
                })

    except WebSocketDisconnect:
        logger.info(
            f"客户端断开连接。共收到 {frame_count} 帧，"
            f"其中 {speech_frame_count} 帧有声音，"
            f"检测到 {state_machine.utterance_count} 句话。"
        )
    except Exception as e:
        logger.error(f"WebSocket 异常: {e}")
        try:
            await ws.close()
        except Exception:
            pass


# ────────────────────── 健康检查 ──────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "vad_mode": VAD_MODE, "frame_ms": VAD_FRAME_MS}


# ────────────────────── 启动 ──────────────────────
if __name__ == "__main__":
    import uvicorn

    # 启动前验证 VAD 可用性
    try:
        test_vad = webrtcvad.Vad(VAD_MODE)
        # 用一段静音帧测试 VAD 是否正常工作
        test_frame = b"\x00" * VAD_FRAME_BYTES
        test_vad.is_speech(test_frame, VAD_SAMPLE_RATE)
        logger.info("webrtcvad 初始化正常")
    except Exception as e:
        logger.error(f"VAD 初始化失败: {e}")
        sys.exit(1)

    logger.info("=" * 50)
    logger.info(f"实时语音通话 MVP 后端启动")
    logger.info(f"   地址: http://{HOST}:{PORT}")
    logger.info(f"   WebSocket: ws://{HOST}:{PORT}/ws")
    logger.info(f"   VAD 模式: {VAD_MODE}，帧长 {VAD_FRAME_MS}ms")
    logger.info(f"   断句阈值: 开始={SPEECH_START_FRAMES}帧 结束={SILENCE_END_FRAMES}帧")
    logger.info("=" * 50)

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
    )
