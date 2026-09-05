import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from tradingagents.llm_clients import create_llm_client
from tradingagents.llm_clients.anthropic_client import AnthropicClient
from tradingagents.llm_clients.google_client import GoogleClient
from tradingagents.llm_clients.openai_client import DeepSeekChatOpenAI
from tradingagents.agents.utils.gpt5_llm import GPT5ChatModel


class LLMClientFactoryTests(unittest.TestCase):
    def test_factory_supports_all_configured_providers(self):
        provider_models = {
            "openai": "gpt-6-astra",
            "local_openai": "gpt-4.1",
            "google": "gemini-3.8-flash",
            "anthropic": "claude-fable-5-1",
            "xai": "grok-4.6",
            "minimax": "MiniMax-M2.7",
            "deepseek": "deepseek-v4-pro",
            "qwen": "qwen3.8-max",
            "glm": "glm-5.2",
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
            "openai": ("gpt-6-astra", "OPENAI_API_KEY"),
            "google": ("gemini-3.8-flash", "GOOGLE_API_KEY"),
            "anthropic": ("claude-fable-5-1", "ANTHROPIC_API_KEY"),
            "xai": ("grok-4.6", "XAI_API_KEY"),
            "minimax": ("MiniMax-M2.7", "MINIMAX_API_KEY"),
            "deepseek": ("deepseek-v4-pro", "DEEPSEEK_API_KEY"),
            "qwen": ("qwen3.8-max", "DASHSCOPE_API_KEY"),
            "glm": ("glm-5.2", "ZHIPU_API_KEY"),
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

        with patch("tradingagents.llm_clients.google_client.NormalizedChatGoogleGenerativeAI") as chat_cls:
            GoogleClient("gemini-3.8-flash", api_key="test-key", thinking_level="minimal").get_llm()
            self.assertEqual(chat_cls.call_args.kwargs["thinking_level"], "low")

        with patch("tradingagents.llm_clients.google_client.NormalizedChatGoogleGenerativeAI") as chat_cls:
            GoogleClient("gemini-3.6-flash", api_key="test-key", thinking_level="minimal").get_llm()
            self.assertEqual(chat_cls.call_args.kwargs["thinking_level"], "minimal")

    def test_current_provider_models_build_with_expected_native_clients(self):
        with patch("tradingagents.llm_clients.google_client.NormalizedChatGoogleGenerativeAI") as chat_cls:
            GoogleClient("gemini-3.8-flash", api_key="test-key").get_llm()
            self.assertEqual(chat_cls.call_args.kwargs["model"], "gemini-3.8-flash")

        with patch("tradingagents.llm_clients.anthropic_client.NormalizedChatAnthropic") as chat_cls:
            AnthropicClient("claude-fable-5-1", api_key="test-key", effort="max").get_llm()
            self.assertEqual(chat_cls.call_args.kwargs["model"], "claude-fable-5-1")
            self.assertEqual(chat_cls.call_args.kwargs["effort"], "max")

        with patch("tradingagents.llm_clients.openai_client.NormalizedChatOpenAI") as chat_cls:
            create_llm_client("xai", "grok-4.6", api_key="test-key").get_llm()
            self.assertEqual(chat_cls.call_args.kwargs["model"], "grok-4.6")
            self.assertEqual(chat_cls.call_args.kwargs["base_url"], "https://api.x.ai/v1")

        current_openai_compatible = {
            "qwen": ("qwen3.8-max", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "glm": ("glm-5.2", "https://open.bigmodel.cn/api/paas/v4/"),
        }
        for provider, (model, endpoint) in current_openai_compatible.items():
            with self.subTest(provider=provider):
                with patch("tradingagents.llm_clients.openai_client.NormalizedChatOpenAI") as chat_cls:
                    create_llm_client(provider, model, api_key="test-key").get_llm()
                    self.assertEqual(chat_cls.call_args.kwargs["model"], model)
                    self.assertEqual(chat_cls.call_args.kwargs["base_url"], endpoint)

        with patch("tradingagents.llm_clients.openai_client.DeepSeekChatOpenAI") as chat_cls:
            create_llm_client("deepseek", "deepseek-v4-pro", api_key="test-key").get_llm()
            self.assertEqual(chat_cls.call_args.kwargs["model"], "deepseek-v4-pro")
            self.assertEqual(chat_cls.call_args.kwargs["base_url"], "https://api.deepseek.com")

    def test_anthropic_effort_is_nested_in_output_config(self):
        llm = AnthropicClient(
            "claude-fable-5-1",
            api_key="test-key",
            effort="max",
        ).get_llm()
        payload = llm._get_request_payload([HumanMessage(content="analyze")])

        self.assertEqual(payload["output_config"], {"effort": "max"})
        self.assertNotIn("effort", payload)

    def test_fable_5_1_uses_native_json_schema_for_structured_output(self):
        class ResultSchema(BaseModel):
            answer: str

        llm = AnthropicClient("claude-fable-5-1", api_key="test-key").get_llm()
        structured = llm.with_structured_output(ResultSchema)

        output_format = structured.first.kwargs["output_config"]["format"]
        self.assertEqual(output_format["type"], "json_schema")
        self.assertNotIn("tool_choice", structured.first.kwargs)

    def test_gpt6_astra_builds_responses_payload_without_sampling_params(self):
        with patch("tradingagents.agents.utils.gpt5_llm.OpenAI") as openai_cls:
            response = SimpleNamespace(output=[], output_text="ok", usage=None)
            openai_cls.return_value.responses.create.return_value = response
            llm = create_llm_client(
                "openai",
                "gpt-6-astra",
                api_key="test-key",
                model_role="deep",
                reasoning_effort="none",
                temperature=0.8,
                top_p=0.7,
            ).get_llm()

            self.assertIsInstance(llm, GPT5ChatModel)
            result = llm.invoke("hello")
            payload = openai_cls.return_value.responses.create.call_args.kwargs
            self.assertEqual(result.content, "ok")
            self.assertEqual(payload["model"], "gpt-6-astra")
            self.assertEqual(payload["reasoning"]["effort"], "high")
            self.assertNotIn("temperature", payload)
            self.assertNotIn("top_p", payload)

    def test_minimax_uses_official_openai_compatible_endpoint_and_custom_override(self):
        with patch("tradingagents.llm_clients.openai_client.NormalizedChatOpenAI") as chat_cls:
            create_llm_client("minimax", "MiniMax-M2.7", api_key="test-key").get_llm()
            kwargs = chat_cls.call_args.kwargs
            self.assertEqual(kwargs["model"], "MiniMax-M2.7")
            self.assertEqual(kwargs["api_key"], "test-key")
            self.assertEqual(kwargs["base_url"], "https://api.minimax.io/v1")

        with patch("tradingagents.llm_clients.openai_client.NormalizedChatOpenAI") as chat_cls:
            create_llm_client(
                "minimax",
                "MiniMax-future-model",
                base_url="https://minimax-proxy.example/v1",
                api_key="test-key",
            ).get_llm()
            self.assertEqual(chat_cls.call_args.kwargs["model"], "MiniMax-future-model")
            self.assertEqual(
                chat_cls.call_args.kwargs["base_url"],
                "https://minimax-proxy.example/v1",
            )


if __name__ == "__main__":
    unittest.main()
