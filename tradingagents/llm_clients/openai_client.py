import os
from typing import Any, Optional

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model


class NormalizedChatOpenAI(ChatOpenAI):
    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))

    def with_structured_output(self, schema, *, method=None, **kwargs):
        return super().with_structured_output(schema, method=method or "function_calling", **kwargs)


def _input_to_messages(input_: Any) -> list:
    if isinstance(input_, list):
        return input_
    if hasattr(input_, "to_messages"):
        return input_.to_messages()
    return []


class DeepSeekChatOpenAI(NormalizedChatOpenAI):
    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        for message_dict, message in zip(payload.get("messages", []), _input_to_messages(input_)):
            if isinstance(message, AIMessage):
                reasoning = message.additional_kwargs.get("reasoning_content")
                if reasoning is not None:
                    message_dict["reasoning_content"] = reasoning
        return payload

    def _create_chat_result(self, response, generation_info=None):
        chat_result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(exclude={"choices": {"__all__": {"message": {"parsed"}}}})
        )
        for generation, choice in zip(chat_result.generations, response_dict.get("choices", [])):
            reasoning = choice.get("message", {}).get("reasoning_content")
            if reasoning is not None:
                generation.message.additional_kwargs["reasoning_content"] = reasoning
        return chat_result

    def with_structured_output(self, schema, *, method=None, **kwargs):
        if self.model_name == "deepseek-reasoner":
            raise NotImplementedError("deepseek-reasoner does not support structured output")
        return super().with_structured_output(schema, method=method, **kwargs)


_PROVIDER_CONFIG = {
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "glm": ("https://open.bigmodel.cn/api/paas/v4/", "ZHIPU_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}


class OpenAIClient(BaseLLMClient):
    def __init__(self, model: str, base_url: Optional[str] = None, provider: str = "openai", **kwargs):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()

        if self.provider in ("openai", "local_openai"):
            from tradingagents.agents.utils.gpt5_llm import get_chat_model

            api_key = self.kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")
            if self.provider == "openai" and not api_key:
                raise ValueError("Provider 'openai' requires OPENAI_API_KEY.")
            if self.provider == "local_openai":
                api_key = api_key or "local-llm"

            return get_chat_model(
                self.model,
                api_key=api_key,
                base_url=self.base_url,
                model_role=self.kwargs.get("model_role", "deep"),
                **{k: v for k, v in self.kwargs.items() if k not in ("api_key", "model_role")},
            )

        llm_kwargs = {"model": self.model}
        default_base, api_key_env = _PROVIDER_CONFIG[self.provider]
        llm_kwargs["base_url"] = self.base_url or default_base
        if api_key_env:
            api_key = self.kwargs.get("api_key") or os.environ.get(api_key_env)
            if not api_key:
                raise ValueError(f"Provider '{self.provider}' requires {api_key_env}.")
            llm_kwargs["api_key"] = api_key
        else:
            llm_kwargs["api_key"] = self.kwargs.get("api_key") or "ollama"

        for key in ("timeout", "max_retries", "reasoning_effort", "callbacks", "http_client", "http_async_client"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        chat_cls = DeepSeekChatOpenAI if self.provider == "deepseek" else NormalizedChatOpenAI
        return chat_cls(**llm_kwargs)

    def validate_model(self) -> bool:
        if self.provider in ("openai", "local_openai"):
            return True
        return validate_model(self.provider, self.model)
