"""
Author: kaic
Date: 2026-11-10 12:56
LastEditors: kai2dot0 41042242+kai2dot0@users.noreply.github.com
LastEditTime: 2026-11-10 12:56
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 优先读取示例自己的配置，文件不存在时复用项目根目录配置；系统环境变量优先。
env_path = Path(__file__).resolve().with_name(".env")
if not env_path.is_file():
    env_path = env_path.parent.parent / ".env"
load_dotenv(env_path)

# 定义模型 API Key
api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
if not api_key:
    raise ValueError(f"请在 {env_path} 中配置 DASHSCOPE_API_KEY，或设置同名环境变量")

# 定义模型
client = OpenAI(
    api_key=api_key,
    base_url=os.getenv(
        "DASHSCOPE_BASE_URL",
        "",
    ),
)
MODEL = os.getenv("QWEN_MODEL", "qwen3.8-omni-flash")


def llm(system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content


# ---- 工人 1：研究员 ----
def researcher(task: str) -> str:
    system = (
        "你是一个研究员 Agent。基于你的知识回答，并明确标注哪些是推测。"
        "输出结构：\n- 核心结论\n- 支持依据\n- 风险/不确定性"
    )
    return llm(system, f"研究任务：{task}")


# ---- 工人 2：写手 ----
def writer(material: str, task: str) -> str:
    system = (
        "你是一个技术写手 Agent，把素材改写成结构清晰、通俗易懂的中文文章，约 300 字。"
    )
    return llm(system, f"原始任务：{task}\n\n研究素材：\n{material}")


def reviewer(artical: str, task: str) -> str:
    system = "你是一个文章评审 Agent，专门对文章进行审查并打分"
    return llm(system, f"原始任务：{task}\n\n文章：\n{artical}")


# ---- 主管：拆解与编排 ----
SUPERVISOR_SYSTEM = """
你是一个主管 Agent。你需要把用户任务拆解为三个子任务：
1. research：交给研究员
2. write：把研究员成果改写成文章
3. review: 对文章进行评审

输入是用户任务。请直接输出 JSON：
{"task": "用户原始任务"}
不要额外解释。
"""


def parse_json(text: str) -> dict:
    # 去掉可能的 ```json 包裹
    text = text.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(text)


def supervisor(user_task: str):
    print("=== 阶段 1: 主管理解任务 ===")
    plan = llm(SUPERVISOR_SYSTEM, f"用户任务：{user_task}")
    plan = parse_json(plan)
    print(f"确认任务: {plan['task']}")

    print("\n=== 阶段 2: 研究员调研 ===")
    material = researcher(plan["task"])
    print(f"调研产出({len(material)} 字):\n{material[:200]}...\n")

    print("\n=== 阶段 3: 写手撰写 ===")
    article = writer(material, plan["task"])
    print(f"成稿:\n{article}")

    print("\n=== 阶段 4: 稿件评分 ===")
    result = reviewer(article, plan["task"])
    print(f"评分:\n{result}")

    return article


if __name__ == "__main__":
    supervisor("写一篇介绍 AI Agent 未来发展趋势的短文")
