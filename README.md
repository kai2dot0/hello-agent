# hello-agents

一个通过 Python 示例学习 AI Agent 的项目：先手写工具调用循环，理解模型如何使用外部能力；再使用 OpenAI Agents SDK，学习工具注册、角色分工和工作流编排。

两个示例都通过阿里云百炼的 OpenAI 兼容接口调用 Qwen，共用根目录的依赖配置，各自拥有入口脚本和学习文档。

## 推荐学习路线

| 顺序 | 示例与学习文档 | 实际任务 | 学习目标 |
| --- | --- | --- | --- |
| 1 | [手写 Function Calling 循环](demo-function-loop/README.md) | 交互式计算、天气查询、文本文件读取 | 理解工具说明、函数执行、消息历史、结果回传和停止条件 |
| 2 | [OpenAI Agents SDK](demo-openai-sdk/README.md) | 先生成研究报告，再交给质量检查员审核 | 对照手写步骤理解 SDK 封装，学习角色配置、上下文传递和流式事件 |

建议先用计算工具追踪一次完整调用，再进入 SDK 示例。两个目录的 README 都包含概念说明、代码阅读顺序、动手练习和学习检查题。

## 通过这两个示例理解 Agent

在本项目中，Agent 的执行过程可以沿着一条数据流来理解：用户提出任务，模型根据指令和可用工具决定下一步，程序执行工具，把结果加入上下文，再请求模型继续处理，直到得到最终答复或触及执行上限。

模型提出工具调用，真正的计算、网络访问和文件读取由 Python 函数完成。工具结果回到后续模型请求，才让程序能够根据新信息继续推进任务。

| 需要理解的问题 | 手写循环中观察什么 | SDK 示例中观察什么 |
| --- | --- | --- |
| 如何约定行为 | `SYSTEM` | 各角色的 `instructions` |
| 如何描述和连接工具 | `TOOLS`、`TOOLS_IMPL` | `@function_tool`、`Agent.tools` |
| 谁推进模型与工具的交互 | `run_agent()` 内部循环 | `Runner.run()` / `Runner.run_streamed()` |
| 上下文如何传递 | 每轮累积并重新发送的 `messages` | 单次运行的历史，以及应用明确传给检查员的任务和报告 |
| 多个阶段如何连接 | 先理解单个任务的循环 | `main()` 顺序安排研究与审核 |

两个示例使用的任务不同，适合对照它们的执行机制。SDK 示例中的多角色流程由代码安排；审核未通过后，当前实现只输出建议，没有自动返工。

## 项目结构

```text
.
├── README.md                         # 学习路线与运行入口
├── pyproject.toml                    # 共用依赖，要求 Python >= 3.13
├── uv.lock                           # 锁定依赖版本
├── .env.example                      # 可复制的配置模板
├── demo-function-loop/
│   ├── README.md                     # 手写循环学习教程
│   └── demo_function_loop.py         # 交互式工具调用示例
├── demo-openai-sdk/
│   ├── README.md                     # SDK 与多角色工作流教程
│   └── demo_openai_agent_sdk.py       # 研究助手与质量检查员
└── tools/
    ├── tools_manual.py               # 手写示例的工具说明与函数映射
    ├── calculator.py                 # 数学表达式计算
    ├── query_weather.py              # 天气查询
    └── read_file.py                  # 文本文件读取
```

## 快速开始

准备 Python 3.13 或更高版本，以及 `uv`。以下命令均在项目根目录执行。

### 1. 安装依赖

```bash
uv sync
```

两个示例使用同一套虚拟环境，无需分别安装依赖。

### 2. 配置要运行的示例

首次配置时，将模板复制到对应目录；只学习一个示例时，只需配置该目录。已有 `.env` 时直接编辑现有文件。

```bash
cp .env.example demo-function-loop/.env
cp .env.example demo-openai-sdk/.env
```

在对应的 `.env` 中填写 API Key，并按需调整模型和接口地址：

```dotenv
DASHSCOPE_API_KEY="你的百炼 API Key"
QWEN_MODEL="qwen3.8-omni-flash"
DASHSCOPE_BASE_URL="https://maas.qianwenaiapi.com/compatible-mode/v1"
```

两个入口都读取**各自脚本同目录下的 `.env`**，不会自动加载根目录的 `.env`。已有系统环境变量优先于文件配置，因此同名环境变量可能覆盖你刚修改的文件值。

| 配置项 | 说明 |
| --- | --- |
| `DASHSCOPE_API_KEY` | 必填，需适用于所配置接口的地域和业务空间 |
| `QWEN_MODEL` | 默认 `qwen3.8-omni-flash`，需支持工具调用；SDK 示例还使用 `reasoning.effort="low"` 设置 |
| `DASHSCOPE_BASE_URL` | 手写示例需配置；SDK 示例默认使用模板中的地址 |

无需额外设置 `OPENAI_API_KEY`。`.env` 已加入 Git 忽略规则，分享配置格式时使用不含真实 Key 的 `.env.example`。

### 3. 运行手写循环

```bash
PYTHONPATH=. uv run demo-function-loop/demo_function_loop.py
```

`PYTHONPATH=.` 让脚本能够导入根目录的 `tools`。程序先输出一次自我介绍，然后进入交互；可以输入 `计算 (328+457)*12`，观察工具调用记录与最终答复，输入 `q` 退出。

任务内的消息历史会累积，每次交互输入则开始一个新任务，默认最多请求模型 6 轮。工具定义和练习详见 [手写循环教程](demo-function-loop/README.md)。

### 4. 运行 SDK 工作流

```bash
uv run demo-openai-sdk/demo_openai_agent_sdk.py
```

添加 `--stream` 查看逐段正文、工具参数和结果摘要：

```bash
uv run demo-openai-sdk/demo_openai_agent_sdk.py --stream
```

当前研究问题在入口脚本 `main()` 的 `user_input` 中设置。程序先运行研究助手，再将原始问题和完整报告交给质量检查员；两者共用模型配置。SDK 执行追踪上传已关闭，本地流式日志仍可展示执行过程。角色配置和练习详见 [SDK 教程](demo-openai-sdk/README.md)。

## 两个 Python 包各自负责什么

| 安装包 | 常见导入 | 在本项目中的作用 |
| --- | --- | --- |
| `openai` | `from openai import OpenAI, AsyncOpenAI` | 配置客户端，发送模型 API 请求并接收响应 |
| `openai-agents` | `from agents import Agent, Runner, function_tool` | 定义角色和工具，管理 Agent 运行中的模型与工具交互 |

手写示例自行管理调用循环，SDK 示例将这部分交给 `Runner`，研究与审核之间的阶段连接仍由应用代码完成。

## 如何检查自己学会了

1. 用一条计算任务说明工具名称、JSON 参数、函数执行和结果回传的完整路径。
2. 能解释工具执行成功后，为什么通常还需要下一轮模型请求。
3. 在两个示例中找到同一项职责的不同实现，例如消息历史或工具分发。
4. 修改一个工具或角色指令，通过实际调用记录确认行为变化。
5. 区分已有能力与后续扩展：当前示例没有跨任务持久化记忆，也没有自动审核返工流程。

进一步学习时，可以逐项加入结构化审核结果、有限次数的返工、跨任务会话和运行评估。每次先明确输入、输出、状态和停止条件，再选择手写实现或框架能力。工具调用记录用于确认执行过程，报告中的关键结论仍需对照来源核实。
