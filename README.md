# ck-agents

基于 OpenAI Agents SDK 的研究助手，通过阿里云百炼的 OpenAI 兼容接口调用 Qwen。

## 运行

```bash
uv sync
cp .env.example .env
```

在项目根目录的 `.env` 中填写 API Key，并按需修改模型和接口地址（只需配置一次）：

```dotenv
DASHSCOPE_API_KEY="你的百炼 API Key"
QWEN_MODEL="qwen3.8-omni-flash"
DASHSCOPE_BASE_URL="https://maas.qianwenaiapi.com/compatible-mode/v1"
```

之后直接运行，无需每次执行 `export`：

```bash
uv run main.py
```

程序先运行研究助手，展示研究报告，再将原始问题和完整报告交给质量检查员。检查员可调用搜索工具核实信息，并输出审核结论、逐项检查结果和修改建议。两个 Agent 使用相同的 Qwen 模型配置。

默认每个阶段完成后一次性输出该阶段的完整结果。添加 `--stream` 可逐段输出两个阶段的正文和简短的工作进度说明，并显示工具名称、调用 ID、参数和返回结果摘要：

```bash
uv run main.py --stream
```

工具参数和返回结果的日志最多显示 300 个字符，完整结果仍会交给模型处理。进度说明是面向用户的分析概述，不展示模型内部的逐步推理。

流式输出期间，模型思考或搜索工具执行仍可能需要等待。使用 `uv run main.py --help` 查看参数说明。

`QWEN_MODEL` 可省略，默认使用 `qwen3.8-omni-flash`。无需设置 `OPENAI_API_KEY`。

程序自动读取 `main.py` 同目录下的 `.env`，已有系统环境变量优先于文件配置。`.env` 已加入 Git 忽略，`.env.example` 用于分享配置格式，不填写真实 Key。

API Key 必须适用于所配置的接口地址和地域，模型需支持工具调用。

官方接入说明：https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope
