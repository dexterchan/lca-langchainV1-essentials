"""Create OpenRouter models with shared configuration for all Python lessons."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

LESSON_DIR = Path(__file__).resolve().parent
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"


def create_model(*, env_path: str | Path | None = None, **model_options) -> ChatOpenAI:
    """Load credentials and create a model without making an API request."""
    load_dotenv(env_path if env_path is not None else LESSON_DIR / ".env")
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("Set OPENROUTER_API_KEY in the environment or python/.env.")
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", DEFAULT_MODEL),
        base_url=OPENROUTER_BASE_URL,
        api_key=key,
        **model_options,
    )


def create_workflow_model(*, env_path: str | Path | None = None) -> ChatOpenAI:
    """Create a model with the fixed workflows' existing client limits."""
    return create_model(env_path=env_path, temperature=0, max_retries=1, timeout=45)
