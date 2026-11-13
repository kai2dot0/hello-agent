"""
Author: kaic
Date: 2026-11-12 16:42
LastEditors: kai2dot0 41042242+kai2dot0@users.noreply.github.com
LastEditTime: 2026-11-12 16:42
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

"""异常体系"""


class HelloAgentsException(Exception):
    """HelloAgents基础异常类"""

    pass


class LLMException(HelloAgentsException):
    """LLM相关异常"""

    pass


class AgentException(HelloAgentsException):
    """Agent相关异常"""

    pass


class ConfigException(HelloAgentsException):
    """配置相关异常"""

    pass


class ToolException(HelloAgentsException):
    """工具相关异常"""

    pass
