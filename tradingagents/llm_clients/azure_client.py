import os
from typing import Any

from langchain_openai import AzureChatOpenAI

from .base_client import BaseLLMClient, normalize_content


class NormalizedAzureChatOpenAI(AzureChatOpenAI):
    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


class AzureOpenAIClient(BaseLLMClient):
    def get_llm(self) -> Any:
        api_key = self.kwargs.get("api_key") or os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = self.base_url or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not api_key:
            raise ValueError("Provider 'azure' requires AZURE_OPENAI_API_KEY.")
        if not endpoint:
            raise ValueError("Provider 'azure' requires AZURE_OPENAI_ENDPOINT or backend_url.")

        llm_kwargs = {
            "model": self.model,
            "azure_deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", self.model),
            "azure_endpoint": endpoint,
            "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            "api_key": api_key,
        }
        for key in ("timeout", "max_retries", "reasoning_effort", "callbacks", "http_client", "http_async_client"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        return NormalizedAzureChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        return True
