import os
from typing import Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model


def _normalize_thinking_level(model: str, thinking_level: Any) -> str:
    """Map the UI's common levels onto the selected Gemini family's API."""
    level = str(thinking_level or "").strip().lower()
    model_lower = str(model or "").lower()
    no_minimal_families = ("gemini-3.8", "gemini-3.7")
    if level == "minimal" and (
        any(family in model_lower for family in no_minimal_families)
        or ("gemini-3.1" in model_lower and "pro" in model_lower)
    ):
        return "low"
    return level


class NormalizedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


class GoogleClient(BaseLLMClient):
    def get_llm(self) -> Any:
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}
        if self.base_url:
            llm_kwargs["base_url"] = self.base_url
        api_key = (
            self.kwargs.get("api_key")
            or self.kwargs.get("google_api_key")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not api_key:
            raise ValueError("Provider 'google' requires GOOGLE_API_KEY.")
        llm_kwargs["google_api_key"] = api_key
        thinking_level = self.kwargs.get("thinking_level")
        if thinking_level:
            model_lower = self.model.lower()
            if "gemini-3" in model_lower:
                llm_kwargs["thinking_level"] = _normalize_thinking_level(
                    self.model, thinking_level
                )
            else:
                # Gemini 2.5 uses a thinking budget rather than thinking_level.
                level = _normalize_thinking_level(self.model, thinking_level)
                llm_kwargs["thinking_budget"] = -1 if level == "high" else 0
        for key in ("timeout", "max_retries", "callbacks", "http_client", "http_async_client"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        return NormalizedChatGoogleGenerativeAI(**llm_kwargs)

    def validate_model(self) -> bool:
        return validate_model("google", self.model)
