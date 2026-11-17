"""
Author: kaic
Date: 2026-11-16 10:41
LastEditors: kai2dot0 41042242+kai2dot0@users.noreply.github.com
LastEditTime: 2026-11-16 10:41
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools import MemoryTool, RAGTool

# 根据脚本位置加载项目根目录的 .env，避免受启动目录影响。
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# 显式使用 ModelScope 配置，避免混用 LLM_* 中的百炼地址和模型名。
llm = HelloAgentsLLM(
    provider="modelscope",
    model="Qwen/Qwen3.5-35B-A3B",
    base_url=os.environ["MODELSCOPE_BASE_URL"],
    api_key=os.environ["MODELSCOPE_API_KEY"],
)

# 创建工具注册表
tool_registry = ToolRegistry()

# 添加记忆工具
memory_tool = MemoryTool(user_id="user123")
tool_registry.register_tool(memory_tool)

# 添加RAG工具
rag_tool = RAGTool(knowledge_base_path="./knowledge_base")
# RAG 问答复用同一个 ModelScope 客户端，避免内部默认配置再次混用凭据。
# RAGTool 的 llm 只负责工具内部的知识库问答。
rag_tool.llm = llm
tool_registry.register_tool(rag_tool)

# 创建Agent并配置工具
agent = SimpleAgent(
    name="智能助手",
    # SimpleAgent 的 llm 负责理解用户请求、决定调用哪个工具，以及生成最终回答
    llm=llm,
    system_prompt=(
        "你是一个有记忆和知识检索能力的AI助手。程序会在第一轮结束后记录对话。"
        "用户查询自己的姓名、职业时，必须先检索记忆，再根据工具结果回答。"
        "本示例的对话记录包含‘用户’这个关键词，请使用："
        "[TOOL_CALL:memory:action=search,query=用户,memory_type=episodic]。"
        "若检索失败或没有结果，必须如实说明，不能声称已查到记忆。"
    ),
    tool_registry=tool_registry,
)

# 第一轮结束后显式记录；SimpleAgent 不会自动调用记忆工具的对话记录方法。
user_input = "你好！请记住我叫张三，我是一名Python开发者"
response = agent.run(user_input)
memory_tool.auto_record_conversation(user_input, response)
print(response)

# 直接检查工具检索结果，避免仅凭模型口头回复判断是否存储成功。
print("\n--- 记忆工具检索验证 ---")
print(memory_tool.execute("search", query="张三", memory_type="episodic"))

# 清空 Agent 聊天历史，验证第二问是否能独立从记忆工具获取信息。
agent.clear_history()
response = agent.run("我是谁？我的职业是什么？")
print(response)
