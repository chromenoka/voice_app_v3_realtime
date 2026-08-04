# Browser-Based Interruptible Voice Dialogue System (voice_app_v3_realtime)

> A browser-based voice dialogue system featuring WebSocket audio exchange, VAD-based user interruption, faster-whisper ASR, DeepSeek Function Calling, and response-path-aware Edge-TTS synthesis.

---

## 🎥 Demo Video / 演示動画

![Real-Time Voice AI Agent Demo](./demo.gif)

> 📥 **[演示動画 1080P  (demo.mp4)](https://github.com/chromenoka/voice_app_v3_realtime/raw/master/demo.mp4)**

---

## ✨ Key Technical Highlights / 核心技術と工夫点

1. **ASR Inference Decoupling (`asyncio.to_thread`)**
   - Offloads Whisper ASR inference to a thread pool to avoid blocking the main asyncio event loop.

2. **Path-Aware TTS Pipeline**
   - For ordinary responses, streamed LLM output is divided at safe sentence or clause boundaries.
   - TTS tasks are started asynchronously in advance, and completed audio segments are sent in the original text order.
   - After Function Calling, the complete final response is collected and synthesized as one segment.

3. **VAD-Triggered Barge-In Playback Control**
   - Sustained user speech detected by WebRTC VAD plus an RMS gate sends a `START` signal and cancels the current LLM/TTS task.
   - The frontend stops the active audio element, revokes its URL, and clears queued audio. The project does not claim a formally measured full-duplex system.

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
[Browser (Web Audio API)] -- WebSocket / PCM --> [FastAPI Server]
       |                                              |
       +--> [webrtcvad + RMS gate] --> turn detection |
       |                                              +--> [faster-whisper / Thread Pool] --> text
       |                                              +--> [DeepSeek LLM + 11 tools]
       |                                              +--> [Edge-TTS]
       |                                                   ordinary: segment pre-synthesis
       |                                                   after tool call: full-reply synthesis
       +<-- START: stop active audio and clear queue <-- [WebSocket events]
```

`main.py` initializes `faster-whisper` as `tiny / CPU / int8`. The active CTranslate2 path does not use Intel Arc or DirectML acceleration.

## VAD defaults

| Parameter | Current default |
| --- | ---: |
| `VAD_VOLUME_THRESHOLD` | 450 RMS |
| `VAD_MIN_SPEECH_FRAMES` | 12 frames (approximately 360 ms) |
| `VAD_MAX_SILENCE_FRAMES` | 25 frames (approximately 750 ms) |

## Voice-turn latency experiment markers

For accepted voice turns, the server writes one JSON line prefixed with `[Latency]`.
The record uses a monotonic clock and includes VAD turn-end, ASR-done,
first-LLM-text, first-TTS-segment-ready, first-audio-sent, and terminal-response
milestones. `boundary` is always `server_websocket_send`: it does not measure
browser queueing, decode, or audible playback time.

Records distinguish the `direct` and `tool` reply paths, and report `completed`,
`cancelled`, or `error`. They are instrumentation for controlled experiments,
not a claim of end-to-end playback latency or a formally measured full-duplex system.
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

