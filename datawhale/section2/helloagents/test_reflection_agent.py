"""
Author: kaic
Date: 2026-11-13 15:14
LastEditors: kai2dot0 41042242+kai2dot0@users.noreply.github.com
LastEditTime: 2026-11-13 15:14
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

# test_reflection_agent.py
from pathlib import Path

from dotenv import load_dotenv
from my_llm import MyLLM
from my_reflection_agent import MyReflectionAgent


def main():
    """加载项目配置并运行反思 Agent 示例。"""
    # 根据脚本位置加载配置，避免受启动目录影响。
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    llm = MyLLM(provider="qwen")

    # 使用默认通用提示词。
    general_agent = MyReflectionAgent(name="我的反思助手", llm=llm)

    # 使用自定义代码生成提示词（类似第四章）。
    code_prompts = {
        "initial": "你是Python专家，请编写函数:{task}",
        "reflect": "请审查代码的算法效率:\n任务:{task}\n代码:{content}",
        "refine": "请根据反馈优化代码:\n任务:{task}\n上一轮代码:{last_attempt}\n反馈:{feedback}",
    }
    code_agent = MyReflectionAgent(
        name="我的代码生成助手", llm=llm, custom_prompts=code_prompts
    )

    # 测试使用；如需代码生成，可调用 code_agent.run("你的代码任务")。
    result = general_agent.run("写一篇关于人工智能发展历程的简短文章")
    print(f"最终结果: {result}")


if __name__ == "__main__":
    main()
