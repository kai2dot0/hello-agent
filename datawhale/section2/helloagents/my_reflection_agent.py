"""
Author: kaic
Date: 2026-11-13 15:14
LastEditors: kai2dot0 41042242+kai2dot0@users.noreply.github.com
LastEditTime: 2026-11-13 15:14
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

from typing import Optional

from hello_agents import Config, HelloAgentsLLM, ReflectionAgent


DEFAULT_PROMPTS = {
    "initial": """
请根据以下要求完成任务:

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的问题或改进空间:

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"无需改进"。
""",
    "refine": """
请根据反馈意见改进你的回答:

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
""",
}


class MyReflectionAgent(ReflectionAgent):
    """复用框架的反思循环、短期记忆和对话历史，支持自定义提示词。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        custom_prompts: Optional[dict[str, str]] = None,
    ):
        """合并提示词配置；max_iterations 为初稿之后的反思改进轮数。"""
        if not isinstance(max_iterations, int) or max_iterations < 0:
            raise ValueError("max_iterations 必须是非负整数")

        # 为每个实例复制模板，允许只覆盖某个阶段而不影响其他 Agent。
        prompts = DEFAULT_PROMPTS.copy()
        if custom_prompts:
            prompts.update(custom_prompts)

        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            config=config,
            max_iterations=max_iterations,
            custom_prompts=prompts,
        )
        print(f"✅ {name} 初始化完成，最大反思轮数: {max_iterations}")

    def _get_llm_response(self, prompt: str, **kwargs) -> str:
        """在初稿、反思和改进阶段统一传递系统提示词与调用参数。"""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.llm.invoke(messages, **kwargs) or ""
