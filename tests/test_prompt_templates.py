import os
import string
import tempfile
import unittest
from pathlib import Path

from tradingagents.prompts import (
    PromptTemplateError,
    list_prompt_templates,
    load_prompt,
    render_prompt,
)


class DefaultPromptValues(dict):
    def __missing__(self, key):
        value = f"<{key}>"
        self[key] = value
        return value


class PromptTemplateTests(unittest.TestCase):
    expected_groups = {
        "analysts",
        "graph",
        "managers",
        "researchers",
        "risk",
        "shared",
        "trader",
        "trading_modes",
    }

    def test_core_prompt_templates_are_available(self):
        templates = set(list_prompt_templates())
        expected = {
            "shared/analyst_tool_system.md",
            "analysts/market_system.md",
            "managers/research_manager.md",
            "trader/trader_system.md",
            "managers/risk_manager.md",
            "graph/signal_extraction_system.md",
            "graph/reflection_system.md",
        }
        self.assertTrue(expected.issubset(templates))

    def test_templates_are_grouped_for_searchability(self):
        templates = list_prompt_templates()
        root_files = {template for template in templates if "/" not in template}
        grouped = {template.split("/", 1)[0] for template in templates if "/" in template}

        self.assertEqual(root_files, {"README.md"})
        self.assertEqual(grouped, self.expected_groups)

    def test_every_model_template_loads_and_renders_with_sample_values(self):
        values = DefaultPromptValues(
            {
                "actions": "BUY, HOLD, or SELL",
                "agent_context": "Agent context.",
                "analysis_content": "Analysis content.",
                "analysis_context": "Analysis packet.",
                "asset_context": "The company we want to look at is NVDA",
                "base_context": "Base trading context.",
                "current_date": "2026-05-03",
                "current_position": "NEUTRAL",
                "decision_format": "BUY/HOLD/SELL",
                "final_decision": "FINAL TRANSACTION PROPOSAL: **HOLD**",
                "final_format": "FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**",
                "mode_name": "SWING TRADING INVESTMENT MODE",
                "raw_return": "+1.0%",
                "ticker": "NVDA",
                "tool_names": "tool_a, tool_b",
            }
        )

        for template_name in list_prompt_templates():
            if template_name == "README.md":
                continue
            with self.subTest(template=template_name):
                template = load_prompt(template_name)
                rendered = template.format_map(values)
                self.assertIsInstance(rendered, str)
                self.assertGreater(len(rendered.strip()), 20)

    def test_template_placeholders_are_parseable(self):
        formatter = string.Formatter()
        for template_name in list_prompt_templates():
            with self.subTest(template=template_name):
                fields = [
                    field_name
                    for _, field_name, _, _ in formatter.parse(load_prompt(template_name))
                    if field_name
                ]
                self.assertTrue(all(" " not in field for field in fields))

    def test_render_prompt_substitutes_values(self):
        rendered = render_prompt(
            "shared/analyst_final_recommendation",
            analysis_label="market analysis",
            subject="NVDA",
            request="provide a final recommendation.",
            analysis_content="Technical evidence here.",
            closing_instruction="Conclude with the required final line.",
        )
        self.assertIn("NVDA", rendered)
        self.assertIn("Technical evidence here.", rendered)
        self.assertNotIn("{subject}", rendered)

    def test_missing_render_value_raises_clear_error(self):
        with self.assertRaisesRegex(PromptTemplateError, "Missing prompt value"):
            render_prompt("shared/analyst_final_recommendation", subject="NVDA")

    def test_rejects_path_traversal(self):
        with self.assertRaises(PromptTemplateError):
            load_prompt("../secrets")

    def test_prompt_dir_override(self):
        original = os.environ.get("TRADINGAGENTS_PROMPT_DIR")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                Path(temp_dir, "custom.md").write_text("Ticker: {ticker}", encoding="utf-8")
                Path(temp_dir, "analysts").mkdir()
                Path(temp_dir, "analysts", "market_system.md").write_text(
                    "Custom analyst prompt for {ticker}",
                    encoding="utf-8",
                )
                os.environ["TRADINGAGENTS_PROMPT_DIR"] = temp_dir
                self.assertEqual(render_prompt("custom", ticker="BTC/USD"), "Ticker: BTC/USD")
                self.assertEqual(
                    render_prompt("analysts/market_system", ticker="NVDA"),
                    "Custom analyst prompt for NVDA",
                )
                self.assertIn("investment decision", load_prompt("graph/signal_extraction_system"))
        finally:
            if original is None:
                os.environ.pop("TRADINGAGENTS_PROMPT_DIR", None)
            else:
                os.environ["TRADINGAGENTS_PROMPT_DIR"] = original


if __name__ == "__main__":
    unittest.main()
