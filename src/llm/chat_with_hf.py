import os

import requests
from dotenv import load_dotenv


def chat_with_hf_inference(
        message: str,
        system_message: str,
        stream: bool = False
):
    load_dotenv()
    hf_api_key = os.getenv("HF_API_KEY")
    hf_model = os.getenv("HF_MODEL")

    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }

    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": message
        }
    ]

    payload = {
        "max_tokens": 500,
        "messages": messages,
        "stream": stream
    }

    url = f"https://api-inference.huggingface.co/models/{hf_model}/v1/chat/completions"

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        data = response.json()
        return data['choices'][0]['message']['content']
    else:
        raise Exception(f"Error {response.status_code}: {response.reason}")
