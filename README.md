# Full-Duplex Real-Time Voice AI Agent System (voice_app_v3_realtime)

> A low-latency, full-duplex conversational AI agent with real-time audio streaming, VAD barge-in interruption, Edge-TTS sentence-level streaming, DeepSeek LLM Function Calling (11 tools), and AST sandbox security evaluation.

---

## 🎥 Demo Video / 演示動画

> 💡 **Demo Video Placeholder**: Upload your short recorded `.mp4` or `.gif` demo video here!
>
> ![Demo Video Placeholder](https://img.shields.io/badge/Demo_Video-Available-brightgreen)

---

## ✨ Key Technical Highlights / 核心技術と工夫点

1. **ASR Inference Decoupling (`asyncio.to_thread`)**
   - Offloads Whisper ASR inference to a thread pool to avoid blocking the main asyncio event loop.
2. **Sentence-Level Streaming TTS**
   - Uses `split_sentences()` to split LLM output by punctuation and synthesize audio sentence-by-sentence via Edge-TTS, shortening First-chunk Latency from >10s to ~3s.
3. **VAD Barge-In Interruption (`webrtcvad`)**
   - Detects user speech activity (VAD) and immediately pauses system audio playback for natural barge-in interaction.
4. **AST Sandbox Security Evaluation (`_safe_eval`)**
   - Completely removes `eval()`, implementing custom AST tree evaluation for safe arithmetic evaluation and DoS protection.
5. **Interactive UI & Function Calling Visualization**
   - Real-time `TOOL_CALL` and `TOOL_DONE` events via WebSocket driving a dynamic frontend Canvas/Orb animation.

---

## 🛠️ Architecture / システム構成

```text
[Browser (Web Audio API)] ──(WebSocket)──► [FastAPI Server] ──► [webrtcvad (VAD)]
                                                │
       ┌────────────────────────────────────────┴────────────────────────────────────────┐
       ▼                                         ▼                                       ▼
[Whisper ASR (Thread Pool)]             [DeepSeek LLM Agent]                   [Edge-TTS Streaming]
(Audio to Text)                    (Function Calling 11 Tools / AST)            (Sentence Split / send_bytes)
```

---

## 🚀 Quick Start / 起動方法

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API Key in .env
cp .env.example .env
# Edit .env and paste your DEEPSEEK_API_KEY

# 3. Launch server
python main.py

# 4. Open in browser
# http://localhost:8000/static/index.html
```

---

## 👤 Author / 開発者
- **Dong Zhaote (董 趙特)**
- Email: `aierxiusite@gmail.com`
