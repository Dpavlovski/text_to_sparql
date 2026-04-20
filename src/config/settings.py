from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# This dynamically finds your project root, assuming the structure:
# Project_Root/
# ├── .env
# └── src/
#     └── config/
#         └── settings.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    default_llm_model: str = Field(default="gpt-4.1-mini")
    openai_api_key: str | None = None
    openrouter_api_key: str | None = None

    qdrant_host: str = Field(default="localhost")
    qdrant_port: int | None = Field(default=6333)

    qald_json_path: str = Field(default="data/qald_10_with_mk.json")

    # Pass the exact absolute path to the .env file
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH),
        env_file_encoding="utf-8",
        extra="ignore"  # Good practice: ignores extra variables in .env without crashing
    )


settings = Settings()
