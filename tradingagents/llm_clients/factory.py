from typing import Optional

from .base_client import BaseLLMClient


_OPENAI_COMPATIBLE = (
    "openai",
    "local_openai",
    "xai",
    "deepseek",
    "qwen",
    "glm",
    "ollama",
    "openrouter",
)


def create_llm_client(provider: str, model: str, base_url: Optional[str] = None, **kwargs) -> BaseLLMClient:
    provider_lower = (provider or "openai").lower()

    if provider_lower in _OPENAI_COMPATIBLE:
        from .openai_client import OpenAIClient

        return OpenAIClient(model, base_url, provider=provider_lower, **kwargs)

    if provider_lower == "google":
        from .google_client import GoogleClient

        return GoogleClient(model, base_url, **kwargs)

    if provider_lower == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(model, base_url, **kwargs)

    if provider_lower == "azure":
        from .azure_client import AzureOpenAIClient

        return AzureOpenAIClient(model, base_url, **kwargs)

    raise ValueError(f"Unsupported LLM provider: {provider}")
