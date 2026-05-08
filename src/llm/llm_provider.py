from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from loguru import logger

from src.config.settings import settings


class LLMProvider:
    """
    A centralized factory for creating language model instances from different services.
    """

    @staticmethod
    def get_model(model_identifier: str) -> BaseChatModel:
        """
        Returns a configured LangChain ChatModel based on the model identifier.
        """
        logger.info(f"Initializing LLM model: '{model_identifier}'")

        # 1. OpenAI Models
        if model_identifier.startswith("gpt-") or model_identifier.startswith("o1-"):
            if not settings.openai_api_key:
                logger.error("OPENAI_API_KEY not found in environment.")
                raise ValueError("OPENAI_API_KEY is required for OpenAI models.")

            return ChatOpenAI(
                api_key=settings.openai_api_key,
                model=model_identifier,
                temperature=0.0,
            )

        # 2. Anthropic Models (Claude)
        elif model_identifier.startswith("claude-"):
            if not settings.anthropic_api_key:
                logger.error("ANTHROPIC_API_KEY not found in environment.")
                raise ValueError("ANTHROPIC_API_KEY is required for Anthropic models.")

            return ChatAnthropic(
                api_key=settings.anthropic_api_key,
                model_name=model_identifier,
                timeout=None,
                stop=None
            )

        # 3. OpenRouter Models (Local, Llama, Nemotron, etc.)
        elif "/" in model_identifier:
            if not settings.openrouter_api_key:
                logger.error("OPENROUTER_API_KEY not found in environment.")
                raise ValueError("OPENROUTER_API_KEY is required for OpenRouter models.")

            return ChatOpenAI(
                api_key=settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                model=model_identifier,
                max_tokens=8192,
                temperature=0.0,
            )

        else:
            logger.error(f"Unknown model identifier format: '{model_identifier}'")
            raise ValueError(f"Unknown model identifier format: '{model_identifier}'.")
