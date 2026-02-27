import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


class LLMProvider:
    """
    A centralized provider for creating language model instances from different services.
    """

    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.ollama_api_url = os.getenv("OLLAMA_API_URL")
        self.ollama_api_key = os.getenv("OLLAMA_API_KEY")
        self.zhipu_api_key = os.getenv("ZHIPU_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")


    def get_model(self, model_identifier: str) -> BaseChatModel:
        # --- Ollama Models ---
        if model_identifier.startswith("ollama/"):
            model_name = model_identifier.split("/", 1)[1]
            return ChatOllama(
                model=model_name,
                base_url=self.ollama_api_url,
                client_kwargs={
                    "headers": {"Authorization": f"Bearer {self.ollama_api_key}"},
                    "timeout": 60,
                },
            )

        # --- Zhipu AI (GLM) Models ---
        elif model_identifier.startswith("glm-"):
            if not self.zhipu_api_key:
                raise ValueError("ZHIPU_API_KEY not found.")

            return ChatOpenAI(
                api_key=self.zhipu_api_key,
                base_url="https://api.z.ai/api/coding/paas/v4",
                model=model_identifier,
                max_tokens=4096,
            )


        # --- OpenAI Models ---
        elif model_identifier.startswith("gpt-"):
            return ChatOpenAI(
                api_key=self.openai_api_key,
                model=model_identifier,
            )

        elif model_identifier.startswith("gemini-"):
            if not self.google_api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables.")

            return ChatGoogleGenerativeAI(
                model=model_identifier,
                google_api_key=self.google_api_key,
                temperature=0,
                max_retries=2,
            )

        # --- OpenRouter Models ---
        elif "/" in model_identifier:
            return ChatOpenAI(
                api_key=self.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                model=model_identifier,
                max_tokens=100000
            )

        else:
            raise ValueError(
                f"Unknown model identifier format: '{model_identifier}'."
            )


# Create a single, reusable instance
llm_provider = LLMProvider()
