import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from tradingagents.llm_clients import create_llm_client
from tradingagents.llm_clients.google_client import GoogleClient
from tradingagents.llm_clients.openai_client import DeepSeekChatOpenAI


class LLMClientFactoryTests(unittest.TestCase):
    def test_factory_supports_all_configured_providers(self):
        provider_models = {
            "openai": "gpt-4.1",
            "local_openai": "gpt-4.1",
            "google": "gemini-2.5-flash",
            "anthropic": "claude-sonnet-4-6",
            "xai": "grok-4-0709",
            "deepseek": "deepseek-chat",
            "qwen": "qwen-plus",
            "glm": "glm-5",
            "openrouter": "custom/openrouter-model",
            "ollama": "qwen3:latest",
            "azure": "deployment-name",
        }

        for provider, model in provider_models.items():
            with self.subTest(provider=provider):
                client = create_llm_client(provider, model, api_key="test-key")
                self.assertEqual(client.model, model)

    def test_missing_api_keys_raise_clear_errors(self):
        required_key_cases = {
            "openai": ("gpt-4.1", "OPENAI_API_KEY"),
            "google": ("gemini-2.5-flash", "GOOGLE_API_KEY"),
            "anthropic": ("claude-sonnet-4-6", "ANTHROPIC_API_KEY"),
            "xai": ("grok-4-0709", "XAI_API_KEY"),
            "deepseek": ("deepseek-chat", "DEEPSEEK_API_KEY"),
            "qwen": ("qwen-plus", "DASHSCOPE_API_KEY"),
            "glm": ("glm-5", "ZHIPU_API_KEY"),
            "openrouter": ("custom/openrouter-model", "OPENROUTER_API_KEY"),
            "azure": ("deployment-name", "AZURE_OPENAI_API_KEY"),
        }

        home_env = {
            key: value
            for key in ("HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH")
            if (value := os.environ.get(key))
        }
        home_env["PYTHON_DOTENV_DISABLED"] = "1"
        with patch.dict(os.environ, home_env, clear=True):
            for provider, (model, env_name) in required_key_cases.items():
                with self.subTest(provider=provider):
                    client = create_llm_client(provider, model)
                    with self.assertRaisesRegex(ValueError, env_name):
                        client.get_llm()

    def test_deepseek_reasoning_content_round_trip(self):
        llm = DeepSeekChatOpenAI(
            model="deepseek-chat",
            api_key="test-key",
            base_url="http://localhost/v1",
        )
        request_payload = llm._get_request_payload(
            [AIMessage(content="answer", additional_kwargs={"reasoning_content": "why"})]
        )
        self.assertEqual(request_payload["messages"][0]["reasoning_content"], "why")

        response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning_content": "why",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        result = llm._create_chat_result(response)
        self.assertEqual(
            result.generations[0].message.additional_kwargs["reasoning_content"],
            "why",
        )

    def test_google_thinking_level_maps_by_model_family(self):
        with patch("tradingagents.llm_clients.google_client.NormalizedChatGoogleGenerativeAI") as chat_cls:
            GoogleClient("gemini-2.5-flash", api_key="test-key", thinking_level="high").get_llm()
            kwargs = chat_cls.call_args.kwargs
            self.assertEqual(kwargs["thinking_budget"], -1)
            self.assertNotIn("thinking_level", kwargs)

        with patch("tradingagents.llm_clients.google_client.NormalizedChatGoogleGenerativeAI") as chat_cls:
            GoogleClient("gemini-3.1-pro-preview", api_key="test-key", thinking_level="minimal").get_llm()
            kwargs = chat_cls.call_args.kwargs
            self.assertEqual(kwargs["thinking_level"], "low")
            self.assertNotIn("thinking_budget", kwargs)


if __name__ == "__main__":
    unittest.main()
