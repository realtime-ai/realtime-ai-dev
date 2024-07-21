
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
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple
import requests
import sseclient

import dataclasses_json


from dotenv import load_dotenv
load_dotenv()

print('ENV:', os.getenv("SILICON_API_KEY"))

LLM_MODELS = [
    {
        "provider": "硅基流动",
        "model": "deepseek-ai/DeepSeek-V2-Chat",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "api_key": os.getenv("SILICON_API_KEY"),
    },
    {
        "provider": "硅基流动",
        "model": "deepseek-ai/DeepSeek-Coder-V2-Instruct",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "api_key": os.getenv("SILICON_API_KEY"),
    },
    {
        "provider": "硅基流动",
        "model": "Qwen/Qwen2-72B-Instruct",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "api_key": os.getenv("SILICON_API_KEY"),
    },
    {
        "provider": "硅基流动",
        "model": "Qwen/Qwen2-7B-Instruct",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "api_key": os.getenv("SILICON_API_KEY"),
    },
    {
        "provider": "硅基流动",
        "model": "Qwen/Qwen2-1.5B-Instruct",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "api_key": os.getenv("SILICON_API_KEY"),
    },
    {
        "provider": "硅基流动",
        "model": "01-ai/Yi-1.5-34B-Chat-16K",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "api_key": os.getenv("SILICON_API_KEY"),
    },
    {
        "provider": "硅基流动",
        "model": "01-ai/Yi-1.5-9B-Chat-16K",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "api_key": os.getenv("SILICON_API_KEY"),
    },
    {
        "provider": "硅基流动",
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "api_key": os.getenv("SILICON_API_KEY"),
    },
    {
        "provider": "硅基流动",
        "model": "meta-llama/Meta-Llama-3-70B-Instruct",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "api_key": os.getenv("SILICON_API_KEY"),
    },
    {
        "provider": "硅基流动",
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "api_key": os.getenv("SILICON_API_KEY"),
    },
    {
        "provider": "硅基流动",
        "model": "meta-llama/Meta-Llama-3-70B-Instruct",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "api_key": os.getenv("SILICON_API_KEY"),
    },
    {
        "provider": "硅基流动",
        "model": "google/gemma-2-9b-it",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "api_key": os.getenv("SILICON_API_KEY"),
    },
    {
        "provider": "硅基流动",
        "model": "google/gemma-2-27b-it",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "api_key": os.getenv("SILICON_API_KEY"),
    },
]


@dataclasses.dataclass
class Metrics(dataclasses_json.DataClassJsonMixin):
    model: str
    ttft: Optional[float] = None
    tps: Optional[float] = None
    input_tokens: Optional[int] = None
    num_tokens: Optional[int] = None
    total_time: Optional[float] = None
    output: Optional[str] = None
    error: Optional[str] = None


class LLMBenchmark:
    def __init__(self, url: str, api_key: str, model: str, input_text: str):
        self.url = url
        self.api_key = api_key
        self.model = model
        self.input_text = input_text
        self.metric: Metrics = Metrics(model=model)

    async def run(self) -> Metrics:

        start_time = time.time()
        first_token_time = None
        total_tokens = 0
        input_tokens = 0

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Tell me a short joke."}],
                "stream": True
            }
            response = requests.post(
                self.url, headers=headers, stream=True, json=data)
            client = sseclient.SSEClient(response)

            for event in client.events():
                if event.data != "[DONE]":
                    try:
                        chunk = json.loads(event.data)
                        if chunk.get('choices'):
                            delta = chunk['choices'][0]['delta']
                            if 'content' in delta:
                                if first_token_time is None:
                                    first_token_time = time.time()
                        usage = chunk.get("usage")
                        if usage:
                            input_tokens = usage.get("prompt_tokens")
                            num_output_tokens = usage.get("completion_tokens")
                            total_tokens = usage.get("total_tokens")
                    except json.JSONDecodeError:
                        print(f"无法解析事件数据: {event.data}")

            end_time = time.time()

            first_token_latency = (
                first_token_time - start_time) * 1000  # 转换为毫秒
            total_duration = (end_time - start_time) * 1000

            self.metric.total_time = total_duration
            self.metric.input_tokens = input_tokens
            self.metric.num_tokens = total_tokens
            self.metric.output = ""
            self.metric.ttft = first_token_latency
            self.metric.tps = min(
                (self.metric.num_tokens - 1) / self.metric.total_time, 999) * 1000

        except Exception as e:
            self.metric.error = str(e)

        return self.metric


async def run_benchmark() -> List[Metrics]:
    metrics = []
    for model in LLM_MODELS:
        benchmark = LLMBenchmark(
            url=model["url"],
            api_key=model["api_key"],
            model=model["model"],
            input_text="Tell me a short joke."
        )
        metric = await benchmark.run()
        metrics.append(metric)
        print(metric.to_json(indent=2))
        time.sleep(1.0)
    return metrics


if __name__ == "__main__":
    metrics = asyncio.run(run_benchmark())
    for metric in metrics:
        print(metric.to_json(indent=2))
        print()
        print()

    json.dump([metric.to_dict() for metric in metrics],
              open("metrics.json", "w"), indent=2)
