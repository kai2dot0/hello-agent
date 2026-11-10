"""
Author: kaic
Date: 2026-11-10 14:00
LastEditors: kai2dot0 41042242+kai2dot0@users.noreply.github.com
LastEditTime: 2026-11-10 14:00
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

from datawhale.section1.tool_traval import get_attraction
from datawhale.section1.tool_weather import get_weather

# 通过 tools 参数提供给模型的工具说明，名称与下方函数映射保持一致。
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，例如北京"},
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_attraction",
            "description": "根据城市和已查询到的天气，搜索推荐的旅游景点。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，例如北京"},
                    "weather": {"type": "string", "description": "天气工具返回的天气信息"},
                },
                "required": ["city", "weather"],
                "additionalProperties": False,
            },
        },
    },
]

# 将所有工具函数放入一个字典，方便后续调用
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}
