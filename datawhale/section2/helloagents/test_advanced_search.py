"""
Author: kaic
Date: 2026-11-13 16:33
LastEditors: kai2dot0 41042242+kai2dot0@users.noreply.github.com
LastEditTime: 2026-11-13 16:33
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

# test_advanced_search.py
from pathlib import Path

from dotenv import load_dotenv
from my_advanced_search import create_advanced_search_registry, MyAdvancedSearchTool
from my_llm import MyLLM


def test_advanced_search():
    """调用已配置的搜索源，并检查各查询是否返回有效结果。"""
    # 根据脚本位置加载项目根目录配置，避免受启动目录影响。
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")

    # 创建包含高级搜索工具的注册表
    registry = create_advanced_search_registry()

    print("🔍 测试高级搜索工具\n")

    # 测试查询
    test_queries = ["Python编程语言的历史", "人工智能的最新发展", "2024年科技趋势"]

    for i, query in enumerate(test_queries, 1):
        print(f"测试 {i}: {query}")
        result = registry.execute_tool("advanced_search", query)
        assert isinstance(result, str) and result.strip(), f"查询 {query!r} 返回空结果"
        assert not result.startswith(("❌", "错误：")), f"查询 {query!r} 失败: {result}"
        print(f"结果: {result}\n")
        print("-" * 60 + "\n")
    print(f"✅ {len(test_queries)} 项搜索测试全部通过\n")


def test_api_configuration():
    """只检查本地配置和搜索源依赖，不发起真实搜索请求。"""
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    print("🔧 测试API配置检查:")

    # 直接创建搜索工具实例
    search_tool = MyAdvancedSearchTool()

    if search_tool.search_sources:
        print(f"已启用搜索源: {', '.join(search_tool.search_sources)}")
        print("实际连接和密钥有效性由后续搜索测试验证。")
    else:
        print("⚠️ 请配置 TAVILY_API_KEY 或 SERPAPI_API_KEY，并安装对应依赖。")


def test_with_agent():
    """演示手动调用搜索工具，再由模型根据搜索结果组织答案。"""
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    print("\n🤖 高级搜索工具与 LLM 配合演示:")

    # 显示工具描述
    registry = create_advanced_search_registry()
    tools_desc = registry.get_tools_description()
    print(f"工具描述:\n{tools_desc}")

    user_question = "人工智能的最新发展有哪些？请简要总结。"
    print(f"\n用户问题: {user_question}")
    search_result = registry.execute_tool("advanced_search", user_question)
    assert isinstance(search_result, str) and search_result.strip(), "搜索返回空结果"
    assert not search_result.startswith(("❌", "错误：")), f"搜索失败: {search_result}"
    print(f"搜索结果:\n{search_result}")

    final_messages = [
        {
            "role": "user",
            "content": (
                f"请根据以下搜索结果回答用户问题，信息不足时请明确说明，不要编造。\n\n"
                f"用户问题: {user_question}\n\n搜索结果:\n{search_result}"
            ),
        }
    ]
    # 显式使用百炼配置；MyLLM 负责处理流式响应中的空 choices 数据块。
    llm = MyLLM(provider="qwen")
    print("\n🎯 模型整理后的回答:")
    for _ in llm.stream_invoke(final_messages):
        pass  # MyLLM 已实时打印文本，这里只消费流，避免重复输出。
    print()


if __name__ == "__main__":
    test_api_configuration()
    test_advanced_search()
    test_with_agent()
