"""
Author: kaic
Date: 2026-11-09 17:56
LastEditors: kai2dot0 41042242+kai2dot0@users.noreply.github.com
LastEditTime: 2026-11-09 17:58
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

"""
Author: kaic
Date: 2026-11-09 17:56
LastEditors: kai2dot0 41042242+kai2dot0@users.noreply.github.com
LastEditTime: 2026-11-09 17:56
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# 长期记忆：记住跨会话偏好，将长期记忆持久化存储到磁盘/数据库。

MEMORY_FILE = "memory.json"


def load_memory() -> dict:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"preferences": []}


def save_memory(memory: dict):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


memory = load_memory()


# 优先读取示例自己的配置，文件不存在时复用项目根目录配置；系统环境变量优先。
env_path = Path(__file__).resolve().with_name(".env")
if not env_path.is_file():
    env_path = env_path.parent.parent / ".env"
load_dotenv(env_path)

# 定义模型 API Key
api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
if not api_key:
    raise ValueError(
        f"请在 {env_path} 中配置 DASHSCOPE_API_KEY，或设置同名环境变量"
    )

# 定义模型
client = OpenAI(
    api_key=api_key,
    base_url=os.getenv(
        "DASHSCOPE_BASE_URL",
        "",
    ),
)
MODEL = os.getenv("QWEN_MODEL", "qwen3.8-omni-flash")


def extract_and_save(messages):
    """让模型把对话中的用户偏好抽取成结构化记忆"""
    extract_prompt = (
        "从对话中提取用户表达的持久偏好，输出 JSON 数组。"
        '没有就输出 []。格式：[{"key": "称呼", "value": "老王"}]'
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages + [{"role": "user", "content": extract_prompt}],
    )
    try:
        new_prefs = json.loads(resp.choices[0].message.content)
        for p in new_prefs:
            if p not in memory["preferences"]:
                memory["preferences"].append(p)
        save_memory(memory)
    except Exception:
        pass  # 抽取失败不影响主流程


def build_system_with_memory() -> str:
    prefs = memory["preferences"]
    if not prefs:
        return "你是一个助手，回答要简洁。"
    pref_text = "；".join(f"{p['key']}: {p['value']}" for p in prefs)
    return f"你是一个助手。用户偏好如下，回答务必符合：{pref_text}"


def chat(user_text: str) -> str:
    messages = [
        {"role": "system", "content": build_system_with_memory()},
        {"role": "user", "content": user_text},
    ]
    resp = client.chat.completions.create(model=MODEL, messages=messages)
    reply = resp.choices[0].message.content
    extract_and_save(messages + [{"role": "assistant", "content": reply}])
    return reply


if __name__ == "__main__":
    print(chat("我叫小王，以后就这么称呼我。"))
    print(chat("今天天气怎么样？"))  # 重启程序后它仍记得你叫小王
