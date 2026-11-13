"""
Author: kaic
Date: 2026-11-12 15:24
LastEditors: kai2dot0 41042242+kai2dot0@users.noreply.github.com
LastEditTime: 2026-11-12 16:22
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

"""
Author: kaic
Date: 2026-11-12 15:24
LastEditors: kai2dot0 41042242+kai2dot0@users.noreply.github.com
LastEditTime: 2026-11-12 15:25
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

"""
Author: kaic
Date: 2026-11-12 15:24
LastEditors: kai2dot0 41042242+kai2dot0@users.noreply.github.com
LastEditTime: 2026-11-12 15:24
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

# my_main.py
from pathlib import Path

from dotenv import load_dotenv
from my_llm import MyLLM  # 注意:这里导入我们自己的类

# 根据脚本位置加载项目根目录的 .env，避免受启动目录影响。
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

# 实例化我们重写的客户端，并指定provider
llm = MyLLM(
    provider="modelscope",
    model="Qwen/Qwen3.5-35B-A3B",
)

"""
定义本地模型

安装 vllm：
brew install vllm-project/vllm-metal/vllm-metal

启动：
/opt/homebrew/bin/vllm serve Qwen/Qwen1.5-0.5B-Chat \
    --host 0.0.0.0 \
    --port 8000
"""
# llm = MyLLM(
#     provider="vllm",
#     model="Qwen/Qwen1.5-0.5B-Chat",  # 需与服务启动时指定的模型一致
#     base_url="http://localhost:8000/v1",
#     api_key="vllm",  # 本地服务通常不需要真实API Key，可填任意非空字符串
# )


# 准备消息
messages = [{"role": "user", "content": "你好，请介绍一下你自己。"}]

# 发起流式调用，由 MyLLM 兼容处理空 choices 数据块。
response_stream = llm.think(messages)

# 打印响应
print("ModelScope Response:")
for chunk in response_stream:
    # chunk在my_llm库中已经打印过一遍，这里只需要pass即可
    # print(chunk, end="", flush=True)
    pass
