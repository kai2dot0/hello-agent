"""
Author: kaic
Date: 2026-11-09 15:09
LastEditors: kylechandev kylechan47@gmail.com
LastEditTime: 2026-11-09 15:09
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent


def read_file(filename: str) -> str:
    try:
        path = (BASE_DIR / filename).resolve()
        if not str(path).startswith(str(BASE_DIR.resolve())):
            return "拒绝：只能读本文件夹内的文件"
        text = path.read_text(encoding="utf-8")
        return text[:3000]
    except FileNotFoundError:
        return f"文件不存在：{filename}"
    except Exception as e:
        return f"读文件失败：{e}"
