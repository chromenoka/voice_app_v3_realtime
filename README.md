# Full-Duplex Real-Time Voice AI Agent System (voice_app_v3_realtime)

> A low-latency, full-duplex conversational AI agent with real-time audio streaming, VAD barge-in interruption, single-request Edge-TTS synthesis, DeepSeek LLM Function Calling (11 tools), and AST sandbox security evaluation.

---

## 🎥 Demo Video / 演示動画

> 💡 **Demo Video Placeholder**: Upload your short recorded `.mp4` or `.gif` demo video here!
>
> ![Demo Video Placeholder](https://img.shields.io/badge/Demo_Video-Available-brightgreen)

---

## ✨ Key Technical Highlights / 核心技術と工夫点

1. **ASR Inference Decoupling (`asyncio.to_thread`)**
   - Offloads Whisper ASR inference to a thread pool to avoid blocking the main asyncio event loop.

2. **Single-Request TTS Architecture (v0.5.0+)**
   - Collects the complete LLM response first, then issues exactly **one** Edge-TTS HTTP request for the whole reply — eliminating the inter-sentence pauses caused by N sequential requests in the original sentence-split approach.
   - Includes automatic retry logic for Microsoft TTS server connection timeouts.

3. **True VAD Barge-In Interruption (v0.6.0+)**
   - When user speech is detected (volume > 700 RMS, sustained ≥ 450 ms), the backend sends a `START` signal.
   - The frontend's unified `stopAudio()` immediately unbinds the `onended` callback, pauses and revokes the current audio, and clears the queue — achieving genuine real-time interruption.

4. **AST Sandbox Security Evaluation (`_safe_eval`)**
   - Completely removes `eval()`, implementing custom AST tree evaluation for safe arithmetic and DoS protection.

5. **Multi-Language TTS Voice Routing**
   - Text-content-only detection: hiragana/katakana → Japanese voice, pure Latin → English voice, otherwise Chinese voice.
   - No historical context contamination (prior Japanese conversation no longer forces Chinese/English into Japanese voice).

6. **Interactive UI & Function Calling Visualization**
   - Real-time `TOOL_CALL` and `TOOL_DONE` events via WebSocket driving a dynamic frontend orb animation.

---

## 🛠️ Architecture / システム構成

```text
[Browser (Web Audio API)] ──(WebSocket / PCM stream)──► [FastAPI Server]
                                                              │
              ┌───────────────────────────────────────────────┴──────────────────────────────┐
              ▼                                               ▼                              ▼
   [webrtcvad (VAD)]                            [Whisper ASR (Thread Pool)]      [DeepSeek LLM Agent]
   Volume + Freq gate                            Audio → Text (asyncio.to_thread)  Function Calling (11 Tools)
   700 RMS / 15 frames                                                                      │
              │                                                                             ▼
              │ START signal ──────────────────────────────────────────────► [Edge-TTS (single request)]
              ▼                                                                  1 HTTP conn / full reply
   [Frontend stopAudio()]                                                                   │
   Unbind onended → pause → revoke URL                                                     ▼
                                                                               [WebSocket send_bytes]
                                                                               Frontend Audio() plays
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

## 📋 Changelog

See [CHANGELOG.md](./CHANGELOG.md) for full version history.

