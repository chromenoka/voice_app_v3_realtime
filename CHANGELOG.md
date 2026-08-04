# 更新日志

本项目所有重要变更均记录于此。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [0.7.0] - 2026-08-05

### 实验平台对齐

- 普通无工具语音回复：LLM 流式文本按安全停顿切分，异步预合成 TTS，并按原始文本顺序发送。
- 工具调用后的最终回复：收集完整文本后使用一次 Edge-TTS 合成。
- VAD 当前默认值集中为 `450 RMS`、`12` 个连续语音帧（约 `360 ms`）和 `25` 个静音帧（约 `750 ms`）。
- 删除未参与 faster-whisper 推理的 DirectML 检测与误导性 GPU 加速表述；当前 ASR 为 `tiny / CPU / int8`。
- 增加服务端语音轮次延迟标记；该记录止于 WebSocket 发送边界，不代表浏览器实际播放时间。
- README 改为“可打断的浏览器端语音对话系统”表述，不宣称已证明的全双工或端到端低延迟能力。

---

## [0.6.0] - 2026-07-31

本轮针对「语音打断」与「噪音误触发」做集中修复，实现了 VAD 触发的播放打断控制机制。

### 🐛 问题修复

#### 1. 语音打断机制失效（核心 Bug）
- **问题**：用户说话时，AI 正在播放的音频不会停止。
- **根因 ①**：前端 WebSocket 收到 `START` 信号后，代码只清了 `audioQueue = []`，但从未调用 `currentAudio.pause()`，正在播放的那条音频持续播完。
- **根因 ②**：`playNextInQueue()` 函数开头用 `if (currentAudio && !currentAudio.ended && !currentAudio.paused) return` 作为卫兵——被 `pause()` 打断后，`currentAudio` 仍指向旧对象，导致函数永远提前 `return`，队列推进死锁。
- **修复**：
  - 新增统一的 `stopAudio()` 函数，打断前先解绑 `onended` 回调（防止 `pause()` 后意外触发 `playNextInQueue`），再暂停并释放 `ObjectURL`。
  - 所有打断场景（用户说话、发消息、点按钮）全部走 `stopAudio()`，确保一致性。
  - `playNextInQueue()` 改为通过 `onended` 回调链式推进，不再用卫兵条件导致状态死锁。

#### 2. VAD 对键盘声/呼气/哈气过于敏感
- **问题**：打字声、呼气声也会触发语音识别，导致大量空转。
- **修复**：
  - `VOLUME_THRESHOLD`：`300` → `700`（音量门槛翻倍，过滤环境噪音）
  - `MIN_SPEECH_FRAMES`：`8帧 ≈ 240ms` → `15帧 ≈ 450ms`（必须连续发声 450ms 才触发，过滤短促点击声与呼吸声）

---

## [0.5.0] - 2026-07-31

本轮针对 TTS 架构做根本性重设计，彻底消灭逐句停顿问题，并修复了 5 个系统性代码 Bug。

### 🔧 架构重设计

#### 1. TTS 从「N 句 × N 次请求」改为「整段 × 1 次请求」
- **问题**：之前的逐句切片方案会在每一句之间单独发起一次 Edge-TTS HTTP 连接，每次连接有 500ms–2000ms 的网络往返延迟，6 句话就会产生 3–12 秒的累积停顿。
- **改动**：
  - 删除 `tts_queue` / `tts_worker` 任务队列机制（约 30 行代码）。
  - LLM 流式输出期间只实时刷新字幕，LLM 结束后整段文本一次性调用 `text_to_speech_stream()`。
  - 工具调用后的二次回答也采用相同的单次合成策略。
- **影响**：N 次网络往返 → 1 次，句间停顿彻底消失。

#### 2. Edge-TTS 超时自动重试机制
- **问题**：日志中频繁出现 `Connection timeout to wss://speech.platform.bing.com`，是偶发性 2 秒卡顿的根本来源。
- **改动**：
  - `text_to_speech_stream()` 加入 2 次重试循环。
  - 首次超时/连接错误时，等待 1 秒后自动重试。
  - `Cannot call "send" once a close message has been sent`、`No audio was received` 等 WebSocket 关闭后的噪音报错全部静默屏蔽。

#### 3. System Prompt 收紧语音回答长度限制
- **改动**：追加硬性约束「每次回答必须严格控制在 2 句话以内，绝对不允许超过 50 个字」，避免 AI 在语音模式生成超长回答（过长的单次 TTS 合成延迟也会增加）。

### 🐛 系统性 Bug 修复（5 处）

#### 4. `import os` 重复
- 文件顶部 `import os` 写了两次，删除重复项。

#### 5. `detect_tts_voice()` 与 `detect_conversation_lang()` 函数调用顺序颠倒
- `detect_tts_voice` 中调用了 `detect_conversation_lang()`，但后者定义在它之后，违反 Python 函数前向引用规范（虽然在运行时不报错，但存在隐式依赖）。已将 `detect_conversation_lang` 移至 `detect_tts_voice` 之前定义。

#### 6. `detect_tts_voice()` 存在 `return` 后的死代码
- 函数在 `return TTS_VOICE_ZH` 之后还有 4 行 `if` 语句，永远不会被执行，已彻底删除。

#### 7. Smart Clause Buffer 正则逻辑错误
- 原正则 `(?<=[，,、])(?=.{2,})` 用零宽断言后跟前瞻计算 `match.end()` 切片位置，会因为前瞻宽度为 0 导致切片位置偏移。
- 重构为独立的 `smart_split_tts(buf)` 函数：先查句末标点，再查逗号位置（要求逗号前已有 ≥10 个字），逻辑清晰且正确。

#### 8. 工具调用二次 TTS 流逻辑不一致
- Stream 1（普通回答）已使用新的 `smart_split_tts`，但 Stream 2（工具调用后的回答）还在使用旧的 `sentence_delimiters.split()` 方案，行为不一致。已统一为同一策略。

---

## [0.4.1] - 2026-07-31

### 🐛 问题修复

#### TTS 发音人语言判断错误——中英文被强制用日语播报
- **问题**：`detect_tts_voice()` 中存在一段基于对话历史的回退逻辑：只要最近对话中出现过日语假名，`detect_conversation_lang()` 就会把当前会话标记为 `"ja"`，导致后续所有中文、英文回答也被强制用日文发音人播报。
- **修复**：删除 `detect_tts_voice()` 中的 `detect_conversation_lang()` 调用。发音人判断仅依赖当前文本本身（平假名/片假名 → 日语，纯拉丁 → 英语，其余 → 中文），不再受历史对话语言影响。

---

## [0.4.0] - 2026-06-24

本轮针对语音通话的「响应延迟」与「安全性」做了一次集中优化（P0 + P1）。

### 🔧 性能优化

#### 1. Whisper 语音识别不再阻塞事件循环
- **问题**：`whisper_model.transcribe(...)` 是 CPU 密集的同步调用，直接跑在 asyncio 协程里。
  识别期间会卡死整个事件循环——导致同一时刻其它 WebSocket 连接假死、
  正在推送的 TTS 音频卡顿。
- **改动**：用 `asyncio.to_thread()` 把 transcribe 丢进线程池执行。
- **影响**：识别过程不再阻塞事件循环，多用户 / 语音+文字并发时不再互相卡顿。

#### 2. TTS 改为按句流式合成与推送，大幅降低首字延迟
- **问题**：原 `text_to_speech_stream` 会把整段文字的音频 chunk 全部攒进 `audio_data`，
  **等整段合成完才一次性 `send_bytes`**。回答越长，用户等首字的时间越久。
- **改动**：
  - 新增 `split_sentences()` 按中英文句号/问号/感叹号/换行切句。
  - 逐句调用 edge-tts 合成，每个 audio chunk **立即 `send_bytes`**，不再攒整段。
- **影响**：首字延迟从「等整段合成」降到「等第一句合成」，
  配合前端边收边播，体感响应明显变快。

### 🔒 安全修复

#### 3. `calculate` 工具用 AST 白名单求值彻底替换 `eval`
- **问题**：原实现虽用字符白名单挡住了字母，但 `eval` 仍可执行
  `9**9**9**9` 这类合法但指数爆炸的表达式，瞬间吃光内存 / CPU 卡死服务。
- **改动**：
  - 新增 `_safe_eval()` 基于 `ast` 模块在语法树上做白名单递归求值。
  - 只允许数字字面量与 `+ - * / // % **` 及一元正负号。
  - 对 `**` 加指数上限校验，拒绝超大指数。
  - 任何名字（Name）、函数调用（Call）、属性访问（Attribute）一律拒绝。
- **影响**：从根上杜绝代码注入与指数爆炸 DoS。

#### 4. `open_application` 去掉 `shell=True`
- **问题**：`subprocess.Popen([target], shell=True)` 多余地启了一层 shell，
  存在 shell 注入面。
- **改动**：`target` 已是固定白名单可执行文件名，直接以列表形式 `Popen([target])` 启动，
  不再经过 shell。
- **影响**：消除 shell 注入风险，且少一层进程开销。

---

## [0.3.0] 及更早

- V3 多模态版本：Whisper + DeepSeek（含 11 个 Function Calling 工具）+ edge-TTS。
- 持续音频采集与 VAD 端点检测：Web Audio API + WebSocket + webrtcvad。
- 工具调用可视化：前端通过 `TOOL_CALL` / `TOOL_DONE` 消息驱动光球变色与齿轮动画。
- 对话记忆持久化：`chat_memory.json`，超 100 条自动滚动截断。

---

## 待办（已识别、未实现的进阶项）

- **会话隔离**：`conversation_history` 当前是全局共享，多连接会互相串话。
  建议改为 `dict[session_id, list]` 按连接隔离。
- **记忆滚动摘要**：当前仅截断历史，每轮仍全量发送，长对话 token 成本线性增长。
  可让 LLM 把旧对话压成摘要塞进 system message。
- **`save_memory` 异步化 + 防抖**：每轮同步写盘阻塞事件循环，建议 `to_thread` + debounce。
- **前端迁移 AudioWorklet**：`ScriptProcessorNode` 已废弃，会卡主线程。
- **断线重连指数退避**：当前固定 1 秒无限重连。
