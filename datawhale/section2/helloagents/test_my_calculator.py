"""
Author: kaic
Date: 2026-11-13 16:17
LastEditors: kai2dot0 41042242+kai2dot0@users.noreply.github.com
LastEditTime: 2026-11-13 16:17
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

# test_my_calculator.py
from pathlib import Path

from dotenv import load_dotenv
from my_calculator_tool import create_calculator_registry
from my_llm import MyLLM


def test_calculator_tool():
    """测试自定义计算器工具"""

    # 创建包含计算器的注册表
    registry = create_calculator_registry()

    print("🧪 测试自定义计算器工具\n")

    # 校验实际结果，避免只打印输出却漏掉计算错误。
    test_cases = [
        ("2 + 3", "5"),  # 基本加法
        ("10 - 4", "6"),  # 基本减法
        ("5 * 6", "30"),  # 基本乘法
        ("15 / 3", "5.0"),  # 基本除法
        ("sqrt(16)", "4.0"),  # 平方根
        ("sqrt(16) + 2 * 3", "10.0"),  # 函数与运算优先级
    ]

    for i, (expression, expected) in enumerate(test_cases, 1):
        print(f"测试 {i}: {expression}")
        result = registry.execute_tool("my_calculator", expression)
        assert result == expected, f"{expression}: 预期 {expected}，实际 {result}"
        print(f"结果: {result}\n")
    print(f"✅ {len(test_cases)} 项计算器测试全部通过\n")


def test_with_simple_agent():
    """保留手动调用工具、再由模型组织答案的演示流程。"""

    # 创建包含计算器的注册表
    registry = create_calculator_registry()

    print("🤖 自定义计算器与 LLM 配合演示:")

    # 由代码直接调用工具，再把结果交给模型组织成自然语言。
    user_question = "请帮我计算 sqrt(16) + 2 * 3"

    print(f"用户问题: {user_question}")

    # 使用工具计算
    calc_result = registry.execute_tool(
        "my_calculator",
        "sqrt(16) + 2 * 3",
    )
    assert calc_result == "10.0", f"计算结果不符合预期: {calc_result}"
    print(f"计算结果: {calc_result}")

    # 构建最终回答
    final_messages = [
        {
            "role": "user",
            "content": f"计算结果是 {calc_result}，请用自然语言回答用户的问题:{user_question}",
        }
    ]

    # 根据脚本位置加载配置，并显式指定百炼，避免混用魔搭密钥。
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    llm = MyLLM(provider="qwen")

    print("\n🎯 模型整理后的回答:")
    for _ in llm.stream_invoke(final_messages):
        pass  # MyLLM 已实时打印文本，这里只消费流，避免重复输出。
    print()


if __name__ == "__main__":
    test_calculator_tool()
    test_with_simple_agent()
