"""
Author: kaic
Date: 2026-11-16 18:53
LastEditors: kai2dot0 41042242+kai2dot0@users.noreply.github.com
LastEditTime: 2026-11-16 18:53
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

from mcp.server import MCPServer

# 创建 MCP Server
mcp = MCPServer("Campus Assistant")

# 模拟图书馆数据库
rooms = {
    "A": ["A101", "A105", "A203"],
    "B": ["B201", "B305"],
}


# 1. Tool：执行查询
# @mcp.tool() 将 Python 函数注册为 MCP 工具。
@mcp.tool()
def search_available_rooms(building: str) -> str:
    """查询指定教学楼中空余的自习室，building 传楼栋编号 A 或 B。"""
    # 模型可能返回“A栋”或小写编号，统一成模拟数据库使用的楼栋键。
    available = rooms.get(building.strip().upper().removesuffix("栋"), [])
    if not available:
        return "没有查询到空余自习室"
    return "空余自习室：" + "、".join(available)


# 2. Resource：提供静态规则
# @mcp.resource() 注册一个能够读取的资源。
@mcp.resource("campus://library/rules")
def library_rules() -> str:
    """读取学校图书馆使用规则。"""
    return "开放时间：08:00—22:00\n" "自习室需要提前预约\n" "每次最多预约3小时"


# 3. Prompt：提供任务模板
# @mcp.prompt() 注册一个可复用提示词模板。
@mcp.prompt()
def study_plan(subject: str) -> str:
    """生成指定学科的学习规划提示词。"""
    return (
        f"请为大学生设计一份{subject}学习计划，" "包含学习目标、每天的任务和复习建议。"
    )


if __name__ == "__main__":
    # 允许 Agent Host 通过 stdio 子进程直接启动服务；日志请写 stderr。
    mcp.run(transport="stdio")
