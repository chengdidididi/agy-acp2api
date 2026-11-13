<div align="center">
  <img src="./assets/icon.svg" width="96" height="96" alt="agy-acp2api icon" />
  <h1>agy-acp2api</h1>
  <p><b>基于原生 ACP 协议的极简 Antigravity CLI 转 OpenAI v1 API 网关</b></p>
  <p>零逆向破解 · 原生协议合规中转 · 支持 Agent 工具调用 / 会话缓存复用 / 思考流 / 子代理并发</p>
</div>

---

> [!WARNING]
> **项目停止维护声明**
>
> 因谷歌已收紧风控，没有任何证据能证明通过 ACP 协议转发会比 OAuth 逆向路径的风控风险更小，故本项目**不再维护**。

## 待改进部分与已知问题（供 Fork 参考）

目前本项目已探明但仍需改进的部分主要包括以下几个方面：

1. **CPA 消息分块转发问题**：一些 `message chunk` 在 CPA (CLIProxyAPI) 中不会被正常转发。
2. **新模型 Token 累计错误**：在新的模型中会出现 Token 累计计算错误的问题。
3. **对话结构的不稳定问题**：包括上下文压缩后对话结构消失，以及 `agy` CLI 在出现 Error 后无法判断并过滤异常内容，导致对话结构污染。
4. **Tool Call 重复执行**：`tool call` 在一些情况下会出现重复执行的问题。

以上问题本质上都是 **JSON-RPC 与 OpenAI 兼容 API 的格式对齐问题**。在这方面本项目还需要更多的改进与维护，目前已探明的问题也只是其中的一部分。但鉴于 ACP 协议转发相比 OAuth 逆向路径并不存在风控风险上的优势，故不再继续投入维护，仅将这些信息暴露给想要 Fork 继续改进的人参考。

### CPA (CLIProxyAPI) 插件现状与接入建议

另外，本项目已经部分实现了 **CPA 插件显示管理界面**的功能（可在 CPA 管理后台直接展示网关控制台界面），但该插件目前还不算完成品：转发上仍存在一些问题（特别是新模型的 `xhigh` 思考档位），且其他路径尚未测试。

* **推荐接入方式**：建议直接将本网关暴露的端点（默认 `http://127.0.0.1:8787/v1`）填到 **CPA 提供商**里管理 API Key，该路径经实测可以正常联通。
* **插件安装方式**：想要使用或体验管理界面功能的人，直接在 **CPA 插件商店**中输入本项目地址即可安装：

```text
https://github.com/chengdidididi/agy-acp2api
```

---

## 项目初衷

市面上常见的客户端转 API 方案，基本都是基于Oauth的逆向。有一段时间算是相安无事，但是最近封号很多，特别是这段时间的时局因素+gemini ultra 20x极大概率是漏洞，所以需要一种更安全的办法做中转

linux.do里的一个项目——[dsh-plugin-bundle](https://github.com/lili-luo/dsh-plugin-bundle) 非常符合当下的需要，但是我不使用dsh，所以就想着搓一个简单的替代品，直接将agy cli的acp协议转成openai标准格式，也当作抛砖引玉、集思广益

**`agy-acp2api` 的核心目标是：用最简单、最干净的方式解决这个问题。**

* **基于官方原生 ACP (Agent Client Protocol) 协议**：直接通过官方 CLI 原生提供的 `agy acp` 标准协议管道进行通信，将标准 OpenAI `/v1/chat/completions` 请求无损翻译并中转给本地 `agy` CLI。

* **完整支持 AI Agent 的绝大多数原生能力**：通过增量提示词编译器与流式状态机，已测试支持外部 Agent 框架所需的以下功能：
  *  Streaming Tool Calls
  *  缓存命中
  *  Sub-Agent 派发 （**注意⚠️，谷歌文件禁止相关自动化流程，故谨慎使用子代理功能**）

> [!NOTE]
> **纯 AI 产出声明 (Vibe Coding Disclaimer)**
> 
> 本项目当前版本的代码（从核心状态机、进程池沙箱到内嵌的 Web 控制台 UI） **100% 由 AI 生成**。虽然已构建了一些场景的测试，但未经过人工 Code Review 与重构精简，仅供学习参考或自行 Fork 改进。

---

## 主要特性

1. **原生 ACP 协议中转+极简隔离沙箱**
   * 每个常驻 Worker 进程通过独立 `--gemini-dir`（`~/.gemini/agy2api-worker-{id}`）与自定义极简 Agent（`excludeDefaultComponents: true`）启动，彻底剥离 `agy` 默认人设与内置文件/终端/浏览器工具 Schema（单轮基础上下文从 `~7,000` tokens 精简至 `~400` tokens），同时隔离宿主全局 `~/.gemini/config` 干扰，并将工作目录锁定在只读空沙箱（Jail Directory）。
   * 底层通过 `PreToolUse` 生命周期钩子与 `session/request_permission` 双重硬阻断，**杜绝触发任何 `agy` CLI 内置工具（含残留控制面工具 `manage_task`）**，强制模型仅使用外部客户端传入的 `tools`（包括外部 MCP、Web Search、Playwright 等）。
2. **会话缓存复用**
   * 自动对 OpenAI `messages` 历史前缀计算哈希链。在多轮对话或工具结果（`role: "tool"`）回传时自动命中已有 `sessionId`（`HASH HIT`），仅编译并发送新增的增量消息（Delta Prompt），使其可以一定程度不受agy cli+acp协议的影响，照常实现一定程度的缓存命中。
3. **滑动窗口流式解析器**
   * 内置 [`ToolCallStreamParser`](./src/parser/stream-parser.ts) 滑动窗口状态机，实时拦截模型流式输出中的工具调用块并转化为标准 OpenAI `delta.tool_calls` SSE 事件，同时内置 `jsonrepair` 自动修复模型偶发的非标 JSON 参数。
4. **常驻热进程池与并发防风控**
   * 维护常驻热进程池（默认 `minWorkers=1`，`maxWorkers=2`），消除每次请求冷启动 `agy` 的耗时。
   * 当主 Agent 派生多个并行 Sub-Agent 时，自动在 `maxWorkers` 安全上限内动态扩容独立沙箱副进程，空闲超时（TTL）后自动销毁回收，超出并发上限的请求自动进入内存队列排队，控制因自动化和大数量并发带来的风控风险。
5. **Web 监控与热配置控制台**
   * 项目附带可视化仪表盘（默认 `http://127.0.0.1:8787/`），基于 Node.js 原生 `node:sqlite` 持久化统计请求延迟（TTFT / 总耗时）、输入/输出 Token、缓存读取 Token 与缓存命中率、Worker 进程状态，并支持在网页上直接热更新并发上限、API Key 与默认模型。

---

## 部署与使用指南

### 1. 环境准备

在运行本项目前，请确保本机已满足以下两个基础条件：

1. **Node.js $\ge$ 22.5.0**（本项目使用 Node.js 22 原生内置的 `node:sqlite` 模块存储监控指标，无需编译 C++ 原生依赖）。
   ```bash
   node -v   # 需 >= v22.5.0
   ```
2. **已安装并登录 Antigravity CLI (`agy`)**：
   确保在终端可以直接运行 `agy` 命令，且已完成账号登录（`~/.gemini/antigravity-cli/jetski_state.pbtxt` 存在）。
   ```bash
   agy --version
   ```

### 2. 获取项目

```bash
git clone https://github.com/chengdidididi/agy-acp2api.git
cd agy-acp2api
```

### 3. 三种启动方式

为了方便不同使用习惯，项目提供了 **一键脚本**、**全局命令** 和 **源码/生产命令** 三种打开方式：

#### 方式 1：双击 / 一键脚本启动（推荐本地日常使用）

项目根目录内置了跨平台一键启动脚本，会自动检查环境、首次运行自动 `npm install` 安装依赖，并在启动后**自动在默认浏览器打开监控控制台**：

* **Windows**：直接双击运行 [`start.bat`](./start.bat)，或在终端执行：
  ```powershell
  .\start.bat
  ```
* **macOS / Linux**：
  ```bash
  chmod +x start.sh
  ./start.sh
  ```

#### 方式 2：注册为系统全局 CLI 命令

如果你想在任意终端里直接输入 `agy-acp2api` 启动网关，可以执行一次构建与全局链接：

```bash
npm install
npm run build
npm link
```

链接完成后，在系统任意位置均可直接运行：

```bash
# 直接启动并自动打开浏览器控制台
agy-acp2api --open

# 或者指定自定义端口启动
agy-acp2api --port 8787 --open
```

#### 方式 3：使用 npm 脚本启动

```bash
npm install

# 开发模式启动（直接运行 TypeScript 源码）
npm start

# 启动并自动打开浏览器面板
npm run start:open

# 编译为纯 JS 生产模式运行（零 tsx 开销）
npm run build
npm run start:prod
```

---

## 客户端接入配置

网关启动后，默认监听地址如下：

| 服务入口 | 默认地址 | 说明 |
| :--- | :--- | :--- |
| **Web 监控与配置控制台** | `http://127.0.0.1:8787/` | 实时查看 Worker 池、Session 缓存命中率、Token 统计与热更新配置 |
| **OpenAI v1 Base URL** | `http://127.0.0.1:8787/v1` | 填入各 AI 客户端 |
| **默认 API Key** | `sk-agy2api-local-key` | 可在 Web 控制台「核心配置热更新」或环境变量中修改 |

### 模型与思考强度（Reasoning Effort）调用方式

网关支持 `/v1/models` 自动枚举当前 `agy` 账号支持的所有模型，并提供了灵活的方式控制思考强度（`low` / `medium` / `high` / `xhigh`，其中 `gemini-4-hydrogen` 原生支持 `xhigh` 档位，其余模型传入 `xhigh` 时会自动平滑降级至 `high`）：

1. **标准参数**：使用基础模型名（如 `gemini-4-hydrogen`、`gemini-4-carbon`、`gemini-4-argon`、`gemini-3.8-flash`、`claude-sonnet-5-5`），并在请求体中传入 OpenAI 标准字段 `reasoning_effort: "low" | "medium" | "high" | "xhigh"`。
2. **模型名后缀**：直接在客户端模型名中追加后缀，网关会自动解析并映射：
   * 连字符写法：`gemini-4-hydrogen-xhigh`、`gemini-4-carbon-high`、`gemini-4-argon-high`、`gemini-3.8-flash-low`
   * 括号写法：`gemini-4-hydrogen(xhigh)`、`gemini-4-carbon(high)`、`gemini-4-argon (low)`
   * 冒号写法：`gemini-4-hydrogen:xhigh`、`gemini-4-carbon:high`、`gemini-4-argon:high`

### 环境变量与配置文件参考

你可以通过环境变量或项目根目录下的 `settings.json`（可参考 [`settings.example.json`](./settings.example.json)）自定义初始配置：

| 环境变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `AGY2API_HOST` | `127.0.0.1` | 监听地址（如需局域网访问可设为 `0.0.0.0`） |
| `AGY2API_PORT` | `8787` | 监听端口 |
| `AGY2API_KEY` | `sk-agy2api-local-key` | Bearer 鉴权密钥 |
| `AGY_BIN` | `agy` | `agy` 可执行文件路径 |
| `AGY_DEFAULT_MODEL` | `gemini-3.8-flash` | 未指定或模型不存在时的回退默认模型 |
| `AGY_DEFAULT_THINKING_LEVEL` | `medium` | 默认思考强度 (`low` / `medium` / `high` / `xhigh`) |
| `AGY_MIN_WORKERS` | `1` | 常驻热进程池最小 Worker 数量 |
| `AGY_MAX_WORKERS` | `2` | 最大并发 Worker 数量（建议保持 $\le 2$ 以防上游并发风控） |

---

## 特别鸣谢
- 感谢 [LINUX DO](https://linux.do/) 社区提供交流与开源推广平台
- [dsh-plugin-bundle](https://github.com/lili-luo/dsh-plugin-bundle)给我带来的启发

## License

MIT License
