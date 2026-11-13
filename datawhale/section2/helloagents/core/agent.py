"""
Author: kaic
Date: 2026-11-12 16:35
LastEditors: kai2dot0 41042242+kai2dot0@users.noreply.github.com
LastEditTime: 2026-11-12 16:41
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

"""
Agent基类

Agent 类是整个框架的顶层抽象。
它定义了一个智能体应该具备的通用行为和属性，但并不关心具体的实现方式。

我们通过 Python 的 abc (Abstract Base Classes) 模块来实现它，这强制所有具体的智能体实现（如后续章节的 SimpleAgent, ReActAgent 等）都必须遵循同一个“接口”。
"""

from abc import ABC, abstractmethod
from typing import Optional, Any
from .message import Message
from .llm import HelloAgentsLLM
from .config import Config


class Agent(ABC):
    """Agent基类"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行Agent"""
        pass

    def add_message(self, message: Message):
        """添加消息到历史记录"""
        self._history.append(message)

    def clear_history(self):
        """清空历史记录"""
        self._history.clear()

    def get_history(self) -> list[Message]:
        """获取历史记录"""
        return self._history.copy()

    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"
