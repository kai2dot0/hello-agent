"""
Author: kaic
Date: 2026-11-12 15:17
LastEditors: kai2dot0 41042242+kai2dot0@users.noreply.github.com
LastEditTime: 2026-11-13 13:47
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

# my_llm.py
import os
from typing import Iterator, Optional
from openai import OpenAI
from hello_agents import HelloAgentsException, HelloAgentsLLM


class MyLLM(HelloAgentsLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = "auto",
        **kwargs,
    ):
        # 自定义 HelloAgentsLLM，增加对 ModelScope 平台的支持
        # 检查provider是否为我们想处理的'modelscope'
        if provider == "modelscope":
            print("正在使用自定义的 ModelScope Provider")
            self.provider = "modelscope"

            # 解析 ModelScope 的凭证
            self.api_key = api_key or os.getenv("MODELSCOPE_API_KEY")
            self.base_url = base_url or "https://api-inference.modelscope.cn/v1/"

            # 验证凭证是否存在
            if not self.api_key:
                raise ValueError(
                    "ModelScope API key not found. Please set MODELSCOPE_API_KEY environment variable."
                )

            # 设置默认模型和其他参数
            self.model = model or os.getenv("LLM_MODEL_ID") or "Qwen/Qwen3.5-35B-A3B"
            self.temperature = kwargs.get("temperature", 0.7)
            self.max_tokens = kwargs.get("max_tokens")
            self.timeout = kwargs.get("timeout", 60)

            # 使用获取的参数创建OpenAI客户端实例
            self._client = OpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            )

        else:
            # 如果不是 modelscope, 则完全使用父类的原始逻辑来处理
            super().__init__(
                model=model,
                api_key=api_key,
                base_url=base_url,
                provider=provider,
                **kwargs,
            )

    def think(
        self, messages: list[dict[str, str]], temperature: Optional[float] = None
    ) -> Iterator[str]:
        """读取流式文本，跳过没有候选内容的统计数据块。"""
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            with self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            ) as response:
                print("✅ 大语言模型响应成功:")
                for chunk in response:
                    # 百炼可能返回 choices 为空的统计块，不能按文本块访问。
                    if not chunk.choices:
                        continue
                    content = chunk.choices[0].delta.content or ""
                    if content:
                        print(content, end="", flush=True)
                        yield content
                print()
        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            raise HelloAgentsException(f"LLM调用失败: {str(e)}") from e
