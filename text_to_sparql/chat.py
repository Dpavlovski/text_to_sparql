import json

import requests


def chat_with_ollama(message: str, system_message: str) -> str:
    url = 'https://llama3.finki.ukim.mk/api/chat'
    headers = {
        'Content-Type': 'application/json',
    }

    data = {
        'model': 'llama3.1:70b',
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
