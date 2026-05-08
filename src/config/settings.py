from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    default_llm_model: str = Field(default="claude-opus-4-7")
    openai_api_key: str | None = None
    openrouter_api_key: str | None = None
    anthropic_api_key: str | None = None

    qdrant_host: str = Field(default="localhost")
    qdrant_port: int | None = Field(default=6333)

    qald_json_path: str = Field(default="data/qald_10_with_mk.json")

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH),
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
