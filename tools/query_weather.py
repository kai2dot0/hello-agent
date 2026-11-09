"""
Author: kaic
Date: 2026-11-09 15:08
LastEditors: kylechandev kylechan47@gmail.com
LastEditTime: 2026-11-09 15:20
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

"""
Author: kaic
Date: 2026-11-09 15:08
LastEditors: kylechandev kylechan47@gmail.com
LastEditTime: 2026-11-09 15:08
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

import json
import urllib.parse
import urllib.request


def query_weather(city: str) -> str:
    try:
        q = urllib.parse.quote(city)
        with urllib.request.urlopen(
            f"https://geocoding-api.open-meteo.com/v1/search?name={q}&count=1&language=zh",
            timeout=15,
        ) as r:
            places = json.load(r).get("results")
        if not places:
            return f"找不到城市：{city}"
        p = places[0]
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={p['latitude']}&longitude={p['longitude']}"
            "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max"
            "&timezone=Asia%2FShanghai&forecast_days=3"
        )
        with urllib.request.urlopen(url, timeout=15) as r:
            d = json.load(r)["daily"]
        lines = [f"{p['name']}未来三天天气："]
        for i in range(len(d["time"])):
            lines.append(
                f"{d['time'][i]}：{d['temperature_2m_min'][i]}~{d['temperature_2m_max'][i]}℃，"
                f"降雨概率 {d['precipitation_probability_max'][i]}%"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"查天气失败：{e}"
