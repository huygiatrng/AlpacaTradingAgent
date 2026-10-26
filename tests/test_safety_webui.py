"""Wiring tests for the safety guardrails WebUI panel and callbacks."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tradingagents.safety import DEFAULT_SAFETY_CONFIG, SafetyGuard


def _guard(tmp, **overrides):
    config = dict(DEFAULT_SAFETY_CONFIG)
    config.update(overrides)
    return SafetyGuard(
        config=config,
        state_path=Path(tmp) / "state.json",
        kill_switch_path=Path(tmp) / "KILL_SWITCH",
    )


class SafetyPanelWiringTests(unittest.TestCase):
    def test_panel_component_builds(self):
        from webui.components.safety_panel import create_safety_panel

        rendered = str(create_safety_panel())
        for component_id in (
            "safety-status-container",
            "safety-kill-switch-btn",
            "safety-release-btn",
            "safety-refresh-interval",
            "safety-action-status",
        ):
            self.assertIn(component_id, rendered)

    def test_callbacks_register_on_fresh_app(self):
        import dash

        from webui.callbacks.safety_callbacks import register_safety_callbacks

        app = dash.Dash(__name__, suppress_callback_exceptions=True)
        register_safety_callbacks(app)
        self.assertTrue(
            any("safety-status-container" in key for key in app.callback_map),
            f"safety callback missing from callback map: {list(app.callback_map)}",
        )

    def test_status_cards_render_green_and_red(self):
        from webui.callbacks.safety_callbacks import _status_cards, _safety_alerts, _safety_summary

        with tempfile.TemporaryDirectory() as tmp:
            guard = _guard(tmp)
            healthy = str(
                _status_cards(
                    guard.status(account={"equity": 100_000.0, "last_equity": 100_000.0})
                )
            )
            self.assertIn("Kill Switch", healthy)
            self.assertIn("LLM Budget", healthy)
            self.assertIn("fa-check-circle", healthy)

            guard.engage_kill_switch("test halt")
            status = guard.status(account={"equity": 100_000.0, "last_equity": 100_000.0})
            tripped = str(_status_cards(status))
            self.assertIn("fa-exclamation-triangle", tripped)
            # Blocking reasons stay outside the collapsed detail cards.
            self.assertIn("test halt", str(_safety_alerts(status)))
            self.assertIn("Orders halted", str(_safety_summary(status)))

    def test_analysis_start_is_gated_by_llm_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard = _guard(tmp, daily_llm_token_budget=100)
            guard.record_llm_tokens(500)
            with patch("tradingagents.safety.get_safety_guard", return_value=guard):
                from webui.components.analysis import start_analysis

                message = start_analysis(
                    ticker="AAPL",
                    analysts_market=True,
                    analysts_social=False,
                    analysts_news=False,
                    analysts_fundamentals=False,
                    analysts_macro=False,
                    research_depth="Shallow",
                    allow_shorts=False,
                    quick_llm="gpt-test",
                    deep_llm="gpt-test",
                )
            self.assertIsInstance(message, str)
            self.assertIn("budget", message.lower())


if __name__ == "__main__":
    unittest.main()
