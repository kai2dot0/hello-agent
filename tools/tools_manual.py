"""
Author: kaic
Date: 2026-11-09 15:10
LastEditors: kylechandev kylechan47@gmail.com
LastEditTime: 2026-11-09 15:11
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

"""
Author: kaic
Date: 2026-11-09 15:10
LastEditors: kylechandev kylechan47@gmail.com
LastEditTime: 2026-11-09 15:10
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

from tools.calculator import calculator
from tools.query_weather import query_weather
from tools.read_file import read_file

"""
定义工具说明书，模型会靠这份 JSON 说明书决定调用哪个工具。

说明书有三样必须写清：
1. 工具名（和函数名一字不差）
2. 它是干什么的（模型靠这句判断何时用）
3. 参数是什么类型

description 写的越详细，模型选择工具就越准确。
比如天气工具的 description 如果只写「天气」，问「明天适合跑步吗」模型不一定反应的过来；写上「查询未来三天天气预报」，它就知道该先查再答。
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式，只支持加减乘除和括号",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "如 (328+457)*12"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_weather",
            "description": "查询某个城市未来三天的天气预报",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如 北京、武汉"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取程序所在文件夹里的文本文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "文件名，如 notes.txt",
                    }
                },
                "required": ["filename"],
            },
        },
    },
]

# 工具名 → 函数本体，循环里按名字找函数就靠这张表
TOOLS_IMPL = {
    "calculator": calculator,
    "query_weather": query_weather,
    "read_file": read_file,
}
