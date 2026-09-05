import unittest

from tradingagents.openai_model_registry import (
    apply_responses_model_params,
    get_default_model_for_provider,
    get_model_spec,
    get_model_options_for_provider,
    get_openai_model_options,
    get_provider_ui_metadata,
    is_responses_model,
    normalize_model_params,
    resolve_model_choice,
)


class OpenAIModelRegistryTests(unittest.TestCase):
    def test_model_options_remove_deprecated_choices_and_keep_low_cost_model(self):
        quick_values = {option["value"] for option in get_openai_model_options("quick")}
        deep_values = {option["value"] for option in get_openai_model_options("deep")}

        self.assertIn("gpt-6-astra", quick_values)
        self.assertIn("gpt-6-astra", deep_values)
        self.assertIn("gpt-5.6-luna", quick_values)
        self.assertIn("gpt-5.6-terra", deep_values)
        self.assertIn("gpt-5.6-sol", deep_values)
        self.assertIn("gpt-5.5", deep_values)
        self.assertIn("gpt-5.4-nano", quick_values)
        self.assertIn("gpt-5-nano", quick_values)
        self.assertIn("gpt-5.4-mini", deep_values)
        self.assertIn("gpt-5.4-pro", deep_values)

        removed_models = {"gpt-4o", "gpt-4o-mini", "o1", "o3", "o3-mini", "o4-mini"}
        self.assertFalse(removed_models & quick_values)
        self.assertFalse(removed_models & deep_values)

    def test_reasoning_model_params_are_limited_to_supported_options(self):
        params = normalize_model_params(
            "gpt-5-nano",
            {
                "reasoning_effort": "xhigh",
                "text_verbosity": "high",
                "temperature": 0.9,
            },
            role="quick",
        )

        self.assertEqual(params["reasoning_effort"], "minimal")
        self.assertEqual(params["text_verbosity"], "high")
        self.assertNotIn("temperature", params)

    def test_non_reasoning_model_exposes_sampling_params(self):
        params = normalize_model_params(
            "gpt-4.1",
            {"temperature": 2.5, "top_p": -1, "reasoning_effort": "high"},
            role="deep",
        )

        self.assertEqual(params["temperature"], 2.0)
        self.assertEqual(params["top_p"], 0.0)
        self.assertNotIn("reasoning_effort", params)

    def test_responses_payload_nests_reasoning_and_text_controls(self):
        payload = {
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "text": {"format": {"type": "text"}},
        }

        apply_responses_model_params(
            payload,
            "gpt-5.4",
            {
                "reasoning_effort": "xhigh",
                "text_verbosity": "low",
                "reasoning_summary": "concise",
                "max_output_tokens": 128,
                "store": False,
            },
            role="deep",
        )

        self.assertEqual(payload["reasoning"], {"effort": "xhigh", "summary": "concise"})
        self.assertEqual(payload["text"]["verbosity"], "low")
        self.assertEqual(payload["max_output_tokens"], 128)
        self.assertFalse(payload["store"])

    def test_provider_catalog_exposes_custom_model_paths_where_needed(self):
        for provider in (
            "openai",
            "local_openai",
            "google",
            "anthropic",
            "xai",
            "minimax",
            "deepseek",
            "qwen",
            "glm",
            "openrouter",
            "ollama",
            "azure",
        ):
            with self.subTest(provider=provider):
                values = {option["value"] for option in get_model_options_for_provider(provider, "quick")}
                self.assertIn("custom", values)

        self.assertFalse(get_provider_ui_metadata("openai")["backend_visible"])
        self.assertTrue(get_provider_ui_metadata("azure")["backend_visible"])
        self.assertEqual(
            get_provider_ui_metadata("minimax")["endpoint"],
            "https://api.minimax.io/v1",
        )

    def test_openai_provider_defaults_stay_cost_safe_after_switching(self):
        self.assertEqual(get_default_model_for_provider("openai", "quick"), "gpt-5.4-nano")
        self.assertEqual(get_default_model_for_provider("openai", "deep"), "gpt-5.4-mini")
        self.assertEqual(get_default_model_for_provider("local_openai", "quick"), "gpt-5.4-nano")

    def test_custom_model_choice_resolves_to_runtime_model_id(self):
        self.assertEqual(resolve_model_choice("custom", " openai/gpt-5.4-mini "), "openai/gpt-5.4-mini")
        self.assertIsNone(resolve_model_choice("custom", " "))
        self.assertEqual(resolve_model_choice("gpt-5.4-mini", "ignored"), "gpt-5.4-mini")

    def test_unknown_openai_compatible_model_uses_custom_chat_controls(self):
        params = normalize_model_params(
            "qwen3:latest",
            {"temperature": 0.4, "top_p": 0.7, "reasoning_effort": "high"},
            role="quick",
        )

        self.assertEqual(params["temperature"], 0.4)
        self.assertEqual(params["top_p"], 0.7)
        self.assertNotIn("reasoning_effort", params)

    def test_future_numbered_openai_model_uses_responses_controls(self):
        params = normalize_model_params(
            "gpt-5.7-terra",
            {"reasoning_effort": "max", "text_verbosity": "high", "temperature": 0.5},
            role="deep",
        )

        self.assertEqual(params["reasoning_effort"], "max")
        self.assertEqual(params["text_verbosity"], "high")
        self.assertNotIn("temperature", params)

    def test_gpt6_astra_uses_responses_and_rejects_unsupported_none_effort(self):
        spec = get_model_spec("gpt-6-astra")
        params = normalize_model_params(
            "gpt-6-astra",
            {"reasoning_effort": "none", "temperature": 0.9, "top_p": 0.5},
            role="deep",
        )

        self.assertTrue(is_responses_model("gpt-6-astra"))
        self.assertEqual(spec["reasoning_effort_options"], ["low", "medium", "high", "xhigh", "max"])
        self.assertEqual(params["reasoning_effort"], "high")
        self.assertNotIn("temperature", params)
        self.assertNotIn("top_p", params)

    def test_future_gpt6_family_uses_conservative_responses_controls(self):
        self.assertTrue(is_responses_model("gpt-6.1-astra"))
        spec = get_model_spec("gpt-6.1-astra")
        self.assertNotIn("none", spec["reasoning_effort_options"])

    def test_provider_catalogs_expose_current_model_ids(self):
        expected = {
            "google": {"gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.1-pro-preview"},
            "anthropic": {"claude-fable-5-1", "claude-opus-5", "claude-sonnet-5"},
            "xai": {"grok-4.6", "grok-4.20", "grok-4.20-non-reasoning"},
            "minimax": {"MiniMax-M2.7", "MiniMax-M2.7-highspeed"},
            "deepseek": {"deepseek-v4-pro", "deepseek-v4-flash"},
            "qwen": {"qwen3.8-max", "qwen3.8-flash"},
            "glm": {"glm-5.2"},
        }
        for provider, model_ids in expected.items():
            with self.subTest(provider=provider):
                values = {
                    option["value"]
                    for role in ("quick", "deep")
                    for option in get_model_options_for_provider(provider, role)
                }
                self.assertTrue(model_ids <= values)

        deepseek_values = {
            option["value"]
            for role in ("quick", "deep")
            for option in get_model_options_for_provider("deepseek", role)
        }
        self.assertNotIn("deepseek-chat", deepseek_values)
        self.assertNotIn("deepseek-reasoner", deepseek_values)


if __name__ == "__main__":
    unittest.main()
