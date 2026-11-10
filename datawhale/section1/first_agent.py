"""
Author: kaic
Date: 2026-11-10 13:53
LastEditors: kai2dot0 41042242+kai2dot0@users.noreply.github.com
LastEditTime: 2026-11-10 13:58
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# 直接运行脚本时，将项目根目录加入模块搜索路径，以支持 datawhale 包导入。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datawhale.section1.openai_compatible_client import OpenAICompatibleClient
from datawhale.section1.tools import TOOLS, available_tools

AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手，根据用户需求使用工具查询真实信息。
涉及当前天气时，先查询天气，再根据天气推荐景点。
需要工具时通过工具调用发起请求，不要在正文中编写调用指令。
工具返回的信息不足或调用失败时明确说明，不要编造结果。
收集到足够信息后，直接用中文回答用户。
"""

# --- 1. 配置LLM客户端 ---
# 从项目根目录加载配置，不依赖运行目录；已有系统环境变量优先。
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

API_KEY = os.getenv("LLM_API_KEY", "").strip()
BASE_URL = os.getenv("LLM_BASE_URL", "").strip()
MODEL_ID = os.getenv("LLM_MODEL_ID", "qwen3.8-omni-flash")

if not API_KEY:
    raise ValueError(f"请在 {env_path} 中配置 LLM_API_KEY，或设置同名环境变量")
if not BASE_URL:
    raise ValueError(f"请在 {env_path} 中配置 LLM_BASE_URL，或设置同名环境变量")

# 定义模型
llm = OpenAICompatibleClient(
    model=MODEL_ID,
    api_key=API_KEY,
    base_url=BASE_URL,
)


def run_agent(task: str, max_steps: int = 5) -> str:
    """执行工具调用循环，将每次工具结果回传给模型，直到生成最终答复。"""
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    for step in range(max_steps):
        print(f"--- 循环 {step+1} ---\n")
        try:
            message = llm.generate(messages, tools=TOOLS)
        except Exception as e:
            return f"错误:调用语言模型服务时出错。{e}"

        # 保留完整的 assistant 消息，后续工具结果通过调用 ID 与其关联。
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            return message.content or "模型未返回有效答复。"

        for call in message.tool_calls:
            tool_name = call.function.name
            try:
                if tool_name not in available_tools:
                    raise ValueError(f"未定义的工具 '{tool_name}'")
                kwargs = json.loads(call.function.arguments)
                parameters = next(
                    tool["function"]["parameters"]
                    for tool in TOOLS
                    if tool["function"]["name"] == tool_name
                )
                # 当前工具的参数均为必填字符串，执行前校验模型生成的参数。
                if not isinstance(kwargs, dict) or set(kwargs) != set(
                    parameters["required"]
                ):
                    raise ValueError(f"参数必须包含且仅包含：{parameters['required']}")
                if any(not isinstance(value, str) for value in kwargs.values()):
                    raise ValueError("工具参数必须为字符串")
                print(f"调用工具: {tool_name}({kwargs})")
                observation = available_tools[tool_name](**kwargs)
            except Exception as e:
                # 失败也回传对应结果，让模型有机会修正参数或说明失败原因。
                observation = f"错误:工具 {tool_name} 调用失败 - {e}"

            print(f"工具返回: {observation}\n")
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": observation}
            )

    return "已达到最大循环次数，任务未完成。"


if __name__ == "__main__":
    user_prompt = (
        "你好，请帮我查询一下今天南京的天气，然后根据天气推荐一个合适的旅游景点。"
    )
    print(f"用户输入: {user_prompt}\n" + "=" * 40)
    print(f"任务结果: {run_agent(user_prompt)}")
