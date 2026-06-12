# src/llm/factory.py

import os

from src.llm.ollama_llm import OllamaLLM
from src.llm.openai_llm import OpenAILLM


def get_llm_provider():
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        return OllamaLLM()

    if provider == "openai":
        return OpenAILLM()

    raise ValueError(f"Unsupported LLM provider: {provider}")