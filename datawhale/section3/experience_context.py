"""
Author: kaic
Date: 2026-11-16 17:15
LastEditors: kai2dot0 41042242+kai2dot0@users.noreply.github.com
LastEditTime: 2026-11-16 17:15
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "hello-agents[memory,rag]==0.2.9",
#     "qdrant-client==1.15.1",
# ]
# ///

# 运行：uv run datawhale/section3/experience_context.py
# uv 根据上面的声明创建独立环境，保留项目使用的 hello-agents 0.2.0。

from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from hello_agents.context import ContextBuilder, ContextConfig
from hello_agents.memory import MemoryConfig
from hello_agents.tools import MemoryTool, RAGTool
from hello_agents.core.message import Message

APP_DIR = Path(__file__).resolve().parent
# 按脚本位置加载根目录配置，避免依赖启动命令所在的目录。
load_dotenv(APP_DIR.parents[1] / ".env")


class ContextMemoryTool(MemoryTool):
    """适配 0.2.9 中 ContextBuilder 仍使用旧 execute 接口的问题。"""

    def execute(self, action: str, **kwargs) -> str:
        """将构建器的旧接口调用转给当前 MemoryTool 的 run 接口。"""
        return self.run({"action": action, **kwargs})


def main():
    """加载记忆和知识库，演示官方 ContextBuilder 的上下文构建流程。"""
    # 1. 初始化工具
    memory_tool = ContextMemoryTool(
        user_id="context_demo_user",
        # 上下文演示使用工作记忆和情景记忆，无需额外连接 Neo4j 图数据库。
        memory_types=["working", "episodic"],
        memory_config=MemoryConfig(
            storage_path=str(APP_DIR / "memory_data" / "context_demo")
        ),
    )
    rag_tool = RAGTool(knowledge_base_path=str(APP_DIR / "knowledge_base"))
    if not rag_tool.initialized:
        raise RuntimeError(f"知识库初始化失败：{rag_tool.init_error}")

    # 2. 创建 ContextBuilder
    config = ContextConfig(
        # 官方版本按空白切词计算相关性，中文示例设为 0，避免历史和记忆全被过滤。
        max_tokens=3000, reserve_ratio=0.2, min_relevance=0.0, enable_compression=True
    )

    builder = ContextBuilder(memory_tool=memory_tool, rag_tool=rag_tool, config=config)

    # 3. 准备对话历史
    conversation_history = [
        Message(
            content="我正在开发一个数据分析工具", role="user", timestamp=datetime.now()
        ),
        Message(
            content="很好!数据分析工具通常需要处理大量数据。您计划使用什么技术栈?",
            role="assistant",
            timestamp=datetime.now(),
        ),
        Message(
            content="我打算使用Python和Pandas,已经完成了CSV读取模块",
            role="user",
            timestamp=datetime.now(),
        ),
        Message(
            content="不错的选择!Pandas在数据处理方面非常强大。接下来您可能需要考虑数据清洗和转换。",
            role="assistant",
            timestamp=datetime.now(),
        ),
    ]

    # 4. 添加一些记忆
    for memory_type, content, importance in [
        ("working", "用户正在开发数据分析工具,使用Python和Pandas", 0.8),
        ("episodic", "已完成CSV读取模块的开发", 0.7),
    ]:
        result = memory_tool.run(
            {
                "action": "add",
                "content": content,
                "memory_type": memory_type,
                "importance": importance,
            }
        )
        if not result.startswith("✅"):
            raise RuntimeError(result)
        print(result)

    # 5. 构建上下文
    context = builder.build(
        user_query="如何优化Pandas的内存占用?",
        conversation_history=conversation_history,
        system_instructions="你是一位资深的Python数据工程顾问。你的回答需要:1) 提供具体可行的建议 2) 解释技术原理 3) 给出代码示例",
    )

    print("=" * 80)
    print("构建的上下文:")
    print("=" * 80)
    print(context)
    print("=" * 80)


if __name__ == "__main__":
    main()
