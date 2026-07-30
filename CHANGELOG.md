# 更新日志

本项目所有重要变更均记录于此。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

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
- 全双工实时语音：Web Audio API + WebSocket + webrtcvad 端点检测。
- 工具调用可视化：前端通过 `TOOL_CALL` / `TOOL_DONE` 消息驱动光球变色与齿轮动画。
- 对话记忆持久化：`chat_memory.json`，超 100 条自动滚动截断。

---

## 待办（后续优化方向，未实现）

> 以下为已识别但本轮未动的进阶项，留作后续：

- **会话隔离**：`conversation_history` 当前是全局共享，多连接会互相串话。
  建议改为 `dict[session_id, list]` 按连接隔离。
- **打断机制完善**：`cancel_event` 仅在生成阶段检查，已推送的音频前端仍会播完；
  文字打断分支的 `sleep(0.05)` 不能可靠停止旧任务。建议引入 task_id + 过期即返回。
- **记忆滚动摘要**：当前仅截断历史，每轮仍全量发送，长对话 token 成本线性增长。
  可让 LLM 把旧对话压成摘要塞进 system message。
- **`save_memory` 异步化 + 防抖**：每轮同步写盘阻塞事件循环，建议 `to_thread` + debounce。
- **前端迁移 AudioWorklet**：`ScriptProcessorNode` 已废弃，会卡主线程。
- **前端释放 ObjectURL**：`URL.createObjectURL` 未 revoke，长时通话内存泄漏。
- **断线重连指数退避**：当前固定 1 秒无限重连。
- **Whisper 升级**：可换 `faster-whisper`（CTranslate2 后端，更快）+ `medium` 模型。
- **清理调试残留**：`_patch.py` / `_fix_model.py` / `_fix_punct.py` / `_rewrite_func.py` 建议移除或归档。
