import argparse
import asyncio
import base64
import dataclasses
import json
import mimetypes
import os
import re
import time
import urllib
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple
import requests
import sseclient
import pandas as pd

import dataclasses_json


from dotenv import load_dotenv
load_dotenv()

print('ENV:', os.getenv("SILICON_API_KEY"))

SILICON_API_KEY = os.getenv("SILICON_API_KEY")  # 硅基流动
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 阿里云
ARK_API_KEY = os.getenv("ARK_API_KEY")  # 火山引擎


SILICON_MODELS = [
    {
        "provider": "硅基流动",
        "model": "Qwen/Qwen3-30B-A3B",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "api_key": os.getenv("SILICON_API_KEY"),
    },
    {
        "provider": "硅基流动",
        "model": "Qwen/Qwen3-32B",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "api_key": os.getenv("SILICON_API_KEY"),
    },
    {
        "provider": "硅基流动",
        "model": "Qwen/Qwen3-14B",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "api_key": os.getenv("SILICON_API_KEY"),
    },
    {
        "provider": "硅基流动",
        "model": "Qwen/Qwen3-8B",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "api_key": os.getenv("SILICON_API_KEY"),
    },
    {
        "provider": "硅基流动",
        "model": "Qwen/Qwen3-235B-A22B",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "api_key": os.getenv("SILICON_API_KEY"),
    },
    {
        "provider": "硅基流动",
        "model": "Qwen/Qwen2.5-VL-32B-Instruct",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "api_key": os.getenv("SILICON_API_KEY"),
    },
    {
        "provider": "硅基流动",
        "model": "deepseek-ai/DeepSeek-V3",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "api_key": os.getenv("SILICON_API_KEY"),
    },

]

DASHSCOPE_MODELS = [
    {
        "provider": "阿里云",
        "model": "qwen3-235b-a22b",
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
    },
    {
        "provider": "阿里云",
        "model": "qwen3-30b-a3b",
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
    },
    {
        "provider": "阿里云",
        "model": "qwen3-32b",
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
    },
    {
        "provider": "阿里云",
        "model": "qwen3-14b",
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
    },
    {
        "provider": "阿里云",
        "model": "qwen3-8b",
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
    },
    {
        "provider": "阿里云",
        "model": "qwen3-4b",
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
    },
    {
        "provider": "阿里云",
        "model": "qwen3-1.7b",
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
    },
    {
        "provider": "阿里云",
        "model": "qwen-turbo-2025-04-28",
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
    },
    {
        "provider": "阿里云",
        "model": "qwen-plus-2025-04-28",
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
    }

]


ARK_MODELS = [
    {
        "provider": "火山引擎",
        "model": "doubao-seed-1.6-250615",
        "url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        "api_key": os.getenv("ARK_API_KEY"),
    },
    {
        "provider": "火山引擎",
        "model": "doubao-seed-1.6-flash-250615",
        "url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        "api_key": os.getenv("ARK_API_KEY"),
    },
    {
        "provider": "火山引擎",
        "model": "deepseek-v3-250324",
        "url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        "api_key": os.getenv("ARK_API_KEY"),
    }
]


DEEPSEEK_MODELS = [
    {
        "provider": "DeepSeek",
        "model": "deepseek-chat",
        "url": "https://api.deepseek.com/v1/chat/completions",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
    },
]


# 测试地点配置 - 统一为北京地区
CURRENT_LOCATION = "北京"

# 合并所有模型
LLM_MODELS = SILICON_MODELS + DASHSCOPE_MODELS + ARK_MODELS + DEEPSEEK_MODELS


@dataclasses.dataclass
class Metrics(dataclasses_json.DataClassJsonMixin):
    provider: str
    model: str
    location: str
    ttft: Optional[float] = None  # Time to first token (ms)
    tps: Optional[float] = None   # Tokens per second
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    total_time: Optional[float] = None  # Total time (ms)
    output: Optional[str] = None
    error: Optional[str] = None
    status: str = "pending"  # pending, success, error


class LLMBenchmark:
    def __init__(self, provider: str, url: str, api_key: str, model: str, input_text: str, location: str):
        self.provider = provider
        self.url = url
        self.api_key = api_key
        self.model = model
        self.input_text = input_text
        self.location = location
        self.metric: Metrics = Metrics(
            provider=provider, model=model, location=location)

    async def run(self) -> Metrics:
        start_time = time.time()
        first_token_time = None
        total_tokens = 0
        input_tokens = 0
        output_tokens = 0
        output_content = ""

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": self.input_text}],
                "stream": True,
                "max_tokens": 1000,
                "stream_options": {
                    "include_usage": True,
                }
            }

            if self.provider == "火山引擎":
                data["thinking"] = {
                    "type": "disabled"
                }

            if self.provider == "阿里云":
                data["enable_thinking"] = False

            data = self.disable_thinking(data)

            response = requests.post(
                self.url, headers=headers, stream=True, json=data, timeout=30)

            if response.status_code != 200:
                raise Exception(
                    f"API请求失败: {response.status_code} - {response.text}")

            client = sseclient.SSEClient(response)

            for event in client.events():
                if event.data != "[DONE]":
                    try:
                        chunk = json.loads(event.data)
                        if chunk.get('choices'):
                            delta = chunk['choices'][0]['delta']
                            if 'content' in delta and delta['content']:
                                if first_token_time is None:
                                    first_token_time = time.time()
                                output_content += delta['content']

                        usage = chunk.get("usage")
                        if usage:
                            input_tokens = usage.get("prompt_tokens", 0)
                            output_tokens = usage.get("completion_tokens", 0)
                            total_tokens = usage.get("total_tokens", 0)
                    except json.JSONDecodeError:
                        print(f"无法解析事件数据: {event.data}")

            end_time = time.time()

            if first_token_time is None:
                raise Exception("未收到任何token")

            first_token_latency = (
                first_token_time - start_time) * 1000  # 转换为毫秒
            total_duration = (end_time - start_time) * 1000

            self.metric.total_time = total_duration
            self.metric.input_tokens = input_tokens
            self.metric.output_tokens = output_tokens
            self.metric.total_tokens = total_tokens
            self.metric.output = output_content
            self.metric.ttft = first_token_latency

            # 计算TPS (tokens per second)
            if total_duration > 0 and output_tokens > 0:
                self.metric.tps = (output_tokens / (total_duration / 1000))
            else:
                self.metric.tps = 0

            self.metric.status = "success"

        except Exception as e:
            self.metric.error = str(e)
            self.metric.status = "error"
            print(f"模型 {self.model} 测试失败: {e}")

        return self.metric

    def disable_thinking(self, data: dict):
        if self.provider == "火山引擎":
            data["thinking"] = {
                "type": "disabled"
            }
        elif self.provider == "阿里云":
            data["enable_thinking"] = False
        elif self.provider == "硅基流动":
            data["enable_thinking"] = False
        elif self.provider == "DeepSeek":
            data["enable_thinking"] = False
        return data


def generate_table(metrics: List[Metrics]) -> str:
    """生成评测结果表格"""
    # 准备表格数据
    table_data = []
    for metric in metrics:
        row = {
            "提供商": metric.provider,
            "模型": metric.model,
            "状态": metric.status,
            "首Token延迟(ms)": f"{metric.ttft:.1f}" if metric.ttft else "N/A",
            "生成速度(tokens/s)": f"{metric.tps:.1f}" if metric.tps else "N/A",
            "输入Token数": metric.input_tokens or "N/A",
            "输出Token数": metric.output_tokens or "N/A",
            "总Token数": metric.total_tokens or "N/A",
            "总耗时(ms)": f"{metric.total_time:.1f}" if metric.total_time else "N/A",
            "错误信息": metric.error or "N/A"
        }
        table_data.append(row)

    # 使用pandas生成表格
    df = pd.DataFrame(table_data)

    # 生成markdown表格
    markdown_table = df.to_markdown(index=False, tablefmt="grid")

    # 生成HTML表格
    html_table = df.to_html(
        index=False, classes="table table-striped table-bordered")

    return markdown_table, html_table


def generate_summary_stats(metrics: List[Metrics]) -> str:
    """生成统计摘要"""
    successful_metrics = [m for m in metrics if m.status == "success"]

    if not successful_metrics:
        return "没有成功的测试结果"

    # 计算统计信息
    ttft_values = [m.ttft for m in successful_metrics if m.ttft]
    tps_values = [m.tps for m in successful_metrics if m.tps]

    summary = f"""
## 评测统计摘要

### 总体统计
- 总测试模型数: {len(metrics)}
- 成功测试数: {len(successful_metrics)}
- 失败测试数: {len(metrics) - len(successful_metrics)}

### 性能指标统计
"""

    if ttft_values:
        summary += f"""
**首Token延迟 (TTFT)**
- 平均值: {sum(ttft_values)/len(ttft_values):.1f} ms
- 最小值: {min(ttft_values):.1f} ms
- 最大值: {max(ttft_values):.1f} ms
"""

    if tps_values:
        summary += f"""
**生成速度 (TPS)**
- 平均值: {sum(tps_values)/len(tps_values):.1f} tokens/s
- 最小值: {min(tps_values):.1f} tokens/s
- 最大值: {max(tps_values):.1f} tokens/s
"""

    return summary


async def run_benchmark() -> List[Metrics]:
    metrics = []
    print(f"开始评测 {len(LLM_MODELS)} 个模型，测试地区: {CURRENT_LOCATION} (每个模型测试2次取平均)")

    for i, model in enumerate(LLM_MODELS, 1):
        print(
            f"[{i}/{len(LLM_MODELS)}] 测试模型: {model['provider']} - {model['model']} (地区: {CURRENT_LOCATION})")

        # 每个模型运行2次
        run_metrics = []
        for run_num in range(2):
            print(f"  第 {run_num + 1} 次测试...")

            benchmark = LLMBenchmark(
                provider=model["provider"],
                url=model["url"],
                api_key=model["api_key"],
                model=model["model"],
                input_text="给我背一首李白的诗词，背两次",
                location=CURRENT_LOCATION
            )
            metric = await benchmark.run()
            run_metrics.append(metric)

            # 打印当前测试结果
            if metric.status == "success":
                print(
                    f"    ✓ 成功 - TTFT: {metric.ttft:.1f}ms, TPS: {metric.tps:.1f}")
            else:
                print(f"    ✗ 失败 - {metric.error}")

            # 避免请求过于频繁
            await asyncio.sleep(1.0)

        # 计算平均值
        successful_runs = [m for m in run_metrics if m.status == "success"]

        if successful_runs:
            # 计算平均指标
            avg_metric = Metrics(
                provider=model["provider"],
                model=model["model"],
                location=CURRENT_LOCATION,
                status="success"
            )

            # 计算平均值
            avg_metric.ttft = sum(
                m.ttft for m in successful_runs if m.ttft) / len(successful_runs)
            avg_metric.tps = sum(
                m.tps for m in successful_runs if m.tps) / len(successful_runs)
            avg_metric.total_time = sum(
                m.total_time for m in successful_runs if m.total_time) / len(successful_runs)
            avg_metric.input_tokens = round(sum(
                m.input_tokens for m in successful_runs if m.input_tokens) / len(successful_runs))
            avg_metric.output_tokens = round(sum(
                m.output_tokens for m in successful_runs if m.output_tokens) / len(successful_runs))
            avg_metric.total_tokens = round(sum(
                m.total_tokens for m in successful_runs if m.total_tokens) / len(successful_runs))

            # 合并输出内容（使用第一次成功的输出）
            avg_metric.output = successful_runs[0].output

            metrics.append(avg_metric)
            print(
                f"  📊 平均结果 - TTFT: {avg_metric.ttft:.1f}ms, TPS: {avg_metric.tps:.1f} (基于 {len(successful_runs)} 次成功测试)")
        else:
            # 如果都失败了，使用最后一次的错误结果
            failed_metric = run_metrics[-1]
            metrics.append(failed_metric)
            print(f"  ❌ 所有测试都失败")

        print()  # 空行分隔

    return metrics


def generate_gradio_leaderboard_data(metrics: List[Metrics], location: str) -> Dict:
    """
    生成指定地区的Gradio Leaderboard组件所需的JSON数据格式
    """
    # 过滤出指定地区和成功的测试结果
    successful_metrics = [m for m in metrics if m.status ==
                          "success" and m.location == location]

    if not successful_metrics:
        return {"data": [], "columns": []}

    # 准备数据行
    data_rows = []
    for metric in successful_metrics:
        ttft_val = round(metric.ttft, 1) if metric.ttft else 0
        tps_val = round(metric.tps, 1) if metric.tps else 0

        # 添加性能等级
        ttft_performance = '🟢 优秀' if ttft_val < 200 else (
            '🟡 良好' if ttft_val < 400 else '🔴 一般')
        tps_performance = '🟢 优秀' if tps_val > 40 else (
            '🟡 良好' if tps_val > 20 else '🔴 一般')

        row = {
            "model": metric.model,
            "provider": metric.provider,
            "location": metric.location,
            "ttft": ttft_val,
            "ttft_performance": ttft_performance,
            "tps": tps_val,
            "tps_performance": tps_performance,
            "total_time": round(metric.total_time, 1) if metric.total_time else 0,
            "output_tokens": metric.output_tokens or 0,
            "status": metric.status
        }
        data_rows.append(row)

    # 按首Token延迟排序（越低越好）
    data_rows.sort(key=lambda x: x["ttft"])

    # 定义列配置
    columns = [
        {"name": "model", "label": "模型", "type": "str"},
        {"name": "provider", "label": "提供商", "type": "str"},
        {"name": "location", "label": "测试地点", "type": "str"},
        {"name": "ttft", "label": "首Token延迟(ms)", "type": "number"},
        {"name": "ttft_performance", "label": "延迟等级", "type": "str"},
        {"name": "tps", "label": "生成速度(tokens/s)", "type": "number"},
        {"name": "tps_performance", "label": "速度等级", "type": "str"},
        {"name": "total_time", "label": "总耗时(ms)", "type": "number"},
        {"name": "output_tokens", "label": "输出Token数", "type": "number"},
        {"name": "status", "label": "状态", "type": "str"}
    ]

    return {
        "data": data_rows,
        "columns": columns
    }


def save_leaderboard_data_by_location(metrics: List[Metrics], date_str: str):
    """
    保存当前测试地区的leaderboard数据到data目录
    """
    # 确保data目录存在
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # 生成当前地区的数据
    location_data = generate_gradio_leaderboard_data(metrics, CURRENT_LOCATION)

    # 文件名格式: leaderboard_{location}_{date}.json
    filename = f"leaderboard_{CURRENT_LOCATION}_{date_str}.json"
    filepath = data_dir / filename

    # 保存数据
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(location_data, f, ensure_ascii=False, indent=2)

    print(f"已保存 {CURRENT_LOCATION} 地区数据到: {filepath}")


if __name__ == "__main__":
    print(f"开始LLM模型性能评测... (测试地区: {CURRENT_LOCATION})")

    metrics = asyncio.run(run_benchmark())

    # 生成表格
    markdown_table, html_table = generate_table(metrics)

    # 生成统计摘要
    summary = generate_summary_stats(metrics)

    # 获取当前日期
    current_date = datetime.now().strftime("%Y-%m-%d")

    # 保存当前地区的Leaderboard数据
    save_leaderboard_data_by_location(metrics, current_date)

    print("\n" + "="*50)
    print("评测完成！")
    print("="*50)
    print(summary)
    print(f"\n详细结果已保存到 data/ 目录，地区: {CURRENT_LOCATION}，日期: {current_date}")
    print("\n详细结果表格:")
    print(markdown_table)
