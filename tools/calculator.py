"""
Author: kaic
Date: 2026-11-09 15:08
LastEditors: kylechandev kylechan47@gmail.com
LastEditTime: 2026-11-09 15:08
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

import ast
import operator

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.operand))
    raise ValueError("只支持加减乘除")


def calculator(expression: str) -> str:
    try:
        return str(_eval(ast.parse(expression, mode="eval")))
    except Exception as e:
        # 写工具的一条通用原则：工具永远要返回字符串，报错也要返回，不要让程序崩。
        return f"计算失败：{e}"
