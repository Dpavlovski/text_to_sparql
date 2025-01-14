import json
import os

import requests


def chat_with_ollama(message: str, system_message: str, temperature: float = 1.0, top_p: float = 1.0) -> str:
    url = os.getenv("OLLAMA_API_URL")
    headers = {
        'Content-Type': 'application/json',
    }

    data = {
        'model': os.getenv("OLLAMA_MODEL"),
        'messages': [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": message
            }
        ],
        'temperature': temperature,
        'top_p': top_p,
        'stream': False
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        response_data = response.json()

        return response_data.get('message', {}).get('content', 'No content returned.')
    except requests.exceptions.RequestException as e:
        return f"HTTP error occurred: {e}"
    except json.JSONDecodeError:
        return "Error decoding JSON response."
    except Exception as e:
        return f"An unexpected error occurred: {e}"
