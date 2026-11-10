"""
Author: kaic
Date: 2026-11-09 15:02
LastEditors: kylechandev kylechan47@gmail.com
LastEditTime: 2026-11-09 15:15
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

"""
Author: kaic
Date: 2026-11-09 15:02
LastEditors: kylechandev kylechan47@gmail.com
LastEditTime: 2026-11-09 15:04
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# 导入模型使用的工具说明，以及实际执行工具的函数映射。
from tools.tools_manual import TOOLS, TOOLS_IMPL

# 通过 openai 依赖，直接发送请求、接收模型结果

# 从脚本同目录的 .env 加载配置，已设置的系统环境变量优先。
load_dotenv(Path(__file__).with_name(".env"))

# 定义模型 API Key
api_key = os.getenv("LLM_API_KEY", "").strip()
if not api_key:
    raise ValueError(
        "请在项目根目录的 .env 中配置 LLM_API_KEY，或设置同名环境变量"
    )

# 定义模型
client = OpenAI(
    api_key=api_key,
    base_url=os.getenv(
        "LLM_BASE_URL",
        "",
    ),
)
MODEL = os.getenv("LLM_MODEL_ID", "qwen3.8-omni-flash")

# 测试直接通过调用 API 来访问模型
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "你好，用一句话介绍你自己"}],
)
print(resp.choices[0].message.content)


# 提示词
SYSTEM = """你是一个能干活的助手。你有三件工具：calculator（计算）、query_weather（查天气）、read_file（读文件）。
遇到算数、天气、读文件的问题，必须调用工具拿真实结果，不许凭记忆编造。
工具结果出来之后，用简洁的中文给出最终答复。"""


# 执行 Agent
def run_agent(task: str, max_steps: int = 6) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task},
    ]

    for step in range(max_steps):  # 设置一个最大步数，防止模型停不下来
        resp = client.chat.completions.create(
            # 实际调用模型
            model=MODEL,
            # messages 作为此 Agent 的全部记忆（直接塞进上下文窗口的 短期记忆）
            # 包含系统提示词、用户问题、每一步的回复和每个工具的结果
            # 对话能记住上下文，就是靠每次请求都把这个 messages 列表整个发过去（模型本身是没有记忆的）
            messages=messages,
            # 告诉 AI 有哪些 TOOLS 可以使用
            tools=TOOLS,
        )
        # 拿到模型返回的结果
        msg = resp.choices[0].message
        # 并累加到 messages 中
        messages.append(msg)
        if not msg.tool_calls:
            # tool_calls 为空，说明模型认为可以直接解答，直接返回结果即可
            return msg.content
        for call in msg.tool_calls:
            # tool_calls 不为空，则说明模型认为需要调用 tools
            # 通过定义的 TOOLS_IMPL 手册，找到实际对应的 tool 并调用
            fn = TOOLS_IMPL[call.function.name]
            args = json.loads(call.function.arguments)
            print(f"  [第{step+1}步] 调用 {call.function.name}({args})")
            # 调用 tool，并把结果返回给模型继续处理
            result = fn(**args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                }
            )
    return "（已达到最大步数，任务未完成）"


if __name__ == "__main__":
    while True:
        task = input("\n你想让它做什么（输入 q 退出）：").strip()
        if task.lower() == "q":
            break
        print("答复：", run_agent(task))
