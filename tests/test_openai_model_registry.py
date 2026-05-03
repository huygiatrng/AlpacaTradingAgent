import unittest

from tradingagents.openai_model_registry import (
    apply_responses_model_params,
    get_default_model_for_provider,
    get_model_options_for_provider,
    get_openai_model_options,
    get_provider_ui_metadata,
    normalize_model_params,
    resolve_model_choice,
)


class OpenAIModelRegistryTests(unittest.TestCase):
    def test_model_options_remove_deprecated_choices_and_keep_low_cost_model(self):
        quick_values = {option["value"] for option in get_openai_model_options("quick")}
        deep_values = {option["value"] for option in get_openai_model_options("deep")}

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
        for provider in ("local_openai", "deepseek", "qwen", "glm", "openrouter", "ollama", "azure"):
            with self.subTest(provider=provider):
                values = {option["value"] for option in get_model_options_for_provider(provider, "quick")}
                self.assertIn("custom", values)

        self.assertFalse(get_provider_ui_metadata("openai")["backend_visible"])
        self.assertTrue(get_provider_ui_metadata("azure")["backend_visible"])

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


if __name__ == "__main__":
    unittest.main()
