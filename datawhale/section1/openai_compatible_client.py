"""
Author: kaic
Date: 2026-11-10 14:03
LastEditors: kai2dot0 41042242+kai2dot0@users.noreply.github.com
LastEditTime: 2026-11-10 14:03
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolUnionParam,
)


class OpenAICompatibleClient:
    """
    一个用于调用任何兼容OpenAI接口的LLM服务的客户端。
    """

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolUnionParam],
    ) -> ChatCompletionMessage:
        """发送对话历史和工具说明，返回包含正文或 tool_calls 的完整消息。"""
        print("正在调用大语言模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                stream=False,
            )
            print("大语言模型响应成功。")
            return response.choices[0].message
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            # 调用方统一结束当前任务，避免将错误文本当成模型消息继续处理。
            raise
