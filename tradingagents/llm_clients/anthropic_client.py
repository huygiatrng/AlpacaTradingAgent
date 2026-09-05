import os
from typing import Any

from langchain_anthropic import ChatAnthropic

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model


class NormalizedChatAnthropic(ChatAnthropic):
    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))

    def with_structured_output(self, schema, *, method=None, **kwargs):
        # Fable 5.1 rejects forced tool_choice values. Its native JSON Schema
        # output is the supported way to guarantee structured responses.
        if method is None and str(self.model).startswith("claude-fable-5-1"):
            method = "json_schema"
        return super().with_structured_output(
            schema,
            method=method or "function_calling",
            **kwargs,
        )


class AnthropicClient(BaseLLMClient):
    def get_llm(self) -> Any:
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}
        if self.base_url:
            llm_kwargs["base_url"] = self.base_url
        api_key = self.kwargs.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Provider 'anthropic' requires ANTHROPIC_API_KEY.")
        llm_kwargs["api_key"] = api_key
        for key in ("timeout", "max_retries", "max_tokens", "callbacks", "http_client", "http_async_client", "effort"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        return NormalizedChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        return validate_model("anthropic", self.model)
