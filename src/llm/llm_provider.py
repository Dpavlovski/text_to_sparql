import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


class LLMProvider:
    """
    A centralized provider for creating language model instances from different services
    (OpenAI, OpenRouter, Ollama) based on a model identifier string.
    """

    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.ollama_api_url = os.getenv("OLLAMA_API_URL")
        self.ollama_api_key = os.getenv("OLLAMA_API_KEY")

        if not self.openrouter_api_key:
            print("Warning: OPENROUTER_API_KEY environment variable not set.")

    def get_model(self, model_identifier: str) -> BaseChatModel:
        """
        Gets a chat model instance based on a string identifier.

        Args:
            model_identifier (str): The identifier for the model.
        Returns:
            An instance of a BaseChatModel.

        Raises:
            ValueError: If the model identifier is not recognized.
        """
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

        # --- OpenAI Models ---
        elif model_identifier.startswith("gpt-"):
            return ChatOpenAI(
                api_key=self.openai_api_key,
                model=model_identifier,
            )

        # --- OpenRouter Models ---
        elif "/" in model_identifier:
            return ChatOpenAI(
                api_key=self.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                model=model_identifier,
                max_tokens=1024
            )

        else:
            raise ValueError(
                f"Unknown model identifier format: '{model_identifier}'. "
                "Expected formats: 'gpt-...', 'ollama/...', or 'vendor/model' for OpenRouter."
            )


# Create a single, reusable instance
llm_provider = LLMProvider()
