import requests
import json
import time
import os

from sseclient import SSEClient

from dotenv import load_dotenv
load_dotenv()


url = "https://api.siliconflow.cn/v1/chat/completions"


API_KEY = os.getenv("SILICON_API_KEY")

payload = {
    "model": "deepseek-ai/DeepSeek-V2-Chat",
    "messages": [
        {
            "role": "user",
            "content": "你是谁? 请说中文"
        }
    ],
    "stream": True
}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": f'Bearer {API_KEY}'
}


start = time.perf_counter()
first_chunk = False
response = requests.post(url, json=payload, headers=headers, stream=True)


# 使用 SSEClient 处理流式响应
client = SSEClient(response)

for event in client.events():
    if event.data != "[DONE]":
        data = json.loads(event.data)
        usage = data.get("usage")
        if usage:
            num_input_tokens = usage.get("prompt_tokens")
            num_output_tokens = usage.get("completion_tokens")
            print('num_output_tokens: ', num_output_tokens)
    else:
        break
