"""
Author: kaic
Date: 2026-11-13 15:42
LastEditors: kai2dot0 41042242+kai2dot0@users.noreply.github.com
LastEditTime: 2026-11-13 15:42
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

import ast
import re
from typing import Optional

from hello_agents import Config, HelloAgentsException, HelloAgentsLLM, Message
from hello_agents.core.agent import Agent


# 默认规划器提示词模板
DEFAULT_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

# 默认执行器提示词模板
DEFAULT_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""


class MyPlanAndSolveAgent(Agent):
    """基于统一 Agent 接口，显式实现先规划、再逐步求解的流程。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[dict[str, str]] = None,
    ):
        """初始化框架配置，允许分别覆盖 planner 与 executor 模板。"""
        super().__init__(name, llm, system_prompt, config)
        self.prompts = {
            "planner": DEFAULT_PLANNER_PROMPT,
            "executor": DEFAULT_EXECUTOR_PROMPT,
        }
        if custom_prompts:
            self.prompts.update(custom_prompts)
        print(f"✅ {name} 初始化完成")

    def run(self, input_text: str, **kwargs) -> str:
        """生成有效计划后按顺序执行，并将最终结果加入对话历史。"""
        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")
        plan = self._plan(input_text, **kwargs)
        if not plan:
            final_answer = "无法生成有效的行动计划，任务终止。"
            print(f"\n--- 任务终止 ---\n{final_answer}")
        else:
            final_answer = self._execute_plan(input_text, plan, **kwargs)
            print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        return final_answer

    def _plan(self, question: str, **kwargs) -> list[str]:
        """生成计划，将列表正文或代码块安全解析为非空字符串步骤。"""
        print("\n--- 正在生成计划 ---")
        prompt = self.prompts["planner"].format(question=question)
        response = self._get_llm_response(prompt, **kwargs)
        print(f"✅ 规划器响应:\n{response}")

        code_block = re.search(
            r"```(?:python)?\s*(.*?)```", response, re.DOTALL | re.IGNORECASE
        )
        plan_text = code_block.group(1).strip() if code_block else response.strip()
        try:
            # 模型输出是不可信文本，literal_eval 不会执行其中的函数调用。
            plan = ast.literal_eval(plan_text)
        except (ValueError, SyntaxError) as e:
            print(f"❌ 解析计划时出错: {e}")
            return []

        if not isinstance(plan, list) or not plan or any(
            not isinstance(step, str) or not step.strip() for step in plan
        ):
            print("❌ 计划必须是由非空字符串组成的非空列表。")
            return []
        return [step.strip() for step in plan]

    def _execute_plan(self, question: str, plan: list[str], **kwargs) -> str:
        """把已完成步骤及其结果传入下一步，返回最后一步的答案。"""
        history = ""
        final_answer = ""
        print("\n--- 正在执行计划 ---")
        for index, step in enumerate(plan, 1):
            print(f"\n-> 正在执行步骤 {index}/{len(plan)}: {step}")
            prompt = self.prompts["executor"].format(
                question=question,
                plan=plan,
                history=history or "无",
                current_step=step,
            )
            result = self._get_llm_response(prompt, **kwargs)
            if not result.strip():
                raise HelloAgentsException(f"步骤 {index} 未返回有效结果，任务终止。")

            # 执行历史只属于本次任务，不混入其他任务的对话记录。
            history += f"步骤 {index}: {step}\n结果: {result}\n\n"
            final_answer = result
            print(f"✅ 步骤 {index} 已完成，结果: {result}")
        return final_answer

    def _get_llm_response(self, prompt: str, **kwargs) -> str:
        """在规划与执行阶段统一传递系统提示词和模型调用参数。"""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.llm.invoke(messages, **kwargs) or ""
