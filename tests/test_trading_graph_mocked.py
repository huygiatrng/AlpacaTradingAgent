import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


class FakeLLM:
    def invoke(self, _prompt):
        raise AssertionError("Mocked graph should not call an LLM")


class FakeClient:
    def get_llm(self):
        return FakeLLM()


class FakeCompiledGraph:
    def __init__(self, final_state):
        self.final_state = final_state

    def invoke(self, _state, **_kwargs):
        return self.final_state


class FakeWorkflow:
    def __init__(self, final_state):
        self.final_state = final_state
        self.compile_calls = []

    def compile(self, checkpointer=None):
        self.compile_calls.append(checkpointer)
        return FakeCompiledGraph(self.final_state)


def _final_state(ticker, trade_date):
    return {
        "company_of_interest": ticker,
        "trade_date": trade_date,
        "market_report": "market",
        "sentiment_report": "sentiment",
        "news_report": "news",
        "fundamentals_report": "fundamentals",
        "macro_report": "macro",
        "report_context": {"stats": {"macro_report": 1}},
        "investment_debate_state": {
            "bull_history": "bull",
            "bear_history": "bear",
            "history": "debate",
            "current_response": "manager",
            "judge_decision": "research decision",
        },
        "trader_investment_plan": "trader plan\nFINAL TRANSACTION PROPOSAL: **BUY**",
        "risk_debate_state": {
            "risky_history": "risky",
            "safe_history": "safe",
            "neutral_history": "neutral",
            "history": "risk debate",
            "judge_decision": "risk decision",
        },
        "investment_plan": "investment plan",
        "final_trade_decision": "risk decision\nFINAL TRANSACTION PROPOSAL: **BUY**",
    }


class MockedTradingGraphTests(unittest.TestCase):
    def test_propagate_preserves_macro_and_safe_paths_with_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for ticker, safe_ticker in (("AAPL", "AAPL"), ("BTC/USD", "BTC_USD")):
                with self.subTest(ticker=ticker):
                    config = DEFAULT_CONFIG.copy()
                    config.update(
                        {
                            "llm_provider": "local_openai",
                            "backend_url": "http://localhost:11434/v1",
                            "quick_think_llm": "gpt-4.1",
                            "deep_think_llm": "gpt-4.1",
                            "data_cache_dir": str(tmp_path / f"cache-{safe_ticker}"),
                            "results_dir": str(tmp_path / "results"),
                            "memory_log_path": str(tmp_path / f"memory-{safe_ticker}.md"),
                            "checkpoint_enabled": True,
                        }
                    )
                    final_state = _final_state(ticker, "2026-01-02")
                    workflow = FakeWorkflow(final_state)

                    with patch("tradingagents.graph.trading_graph.create_llm_client", return_value=FakeClient()), patch(
                        "tradingagents.graph.trading_graph.GraphSetup.setup_graph",
                        return_value=workflow,
                    ):
                        graph = TradingAgentsGraph(
                            selected_analysts=["market", "news", "macro"],
                            config=config,
                            debug=False,
                        )
                        state, signal = graph.propagate(ticker, "2026-01-02")

                    self.assertEqual(signal, "BUY")
                    self.assertEqual(state["macro_report"], "macro")
                    self.assertTrue(
                        (tmp_path / "results" / safe_ticker / "TradingAgentsStrategy_logs" / "full_states_log.json").exists()
                    )
                    self.assertIn(ticker, Path(config["memory_log_path"]).read_text(encoding="utf-8"))
                    self.assertGreaterEqual(len(workflow.compile_calls), 2)

    def test_provider_specific_runtime_options_reach_llm_factory(self):
        cases = [
            ("google", {"google_thinking_level": "high"}, "thinking_level", "high"),
            ("anthropic", {"anthropic_effort": "medium"}, "effort", "medium"),
        ]

        for provider, provider_config, expected_key, expected_value in cases:
            with self.subTest(provider=provider), tempfile.TemporaryDirectory() as tmp:
                calls = []

                def fake_create_llm_client(**kwargs):
                    calls.append(kwargs)
                    return FakeClient()

                config = DEFAULT_CONFIG.copy()
                config.update(
                    {
                        "llm_provider": provider,
                        "quick_think_llm": "custom-quick",
                        "deep_think_llm": "custom-deep",
                        "data_cache_dir": str(Path(tmp) / "cache"),
                        "results_dir": str(Path(tmp) / "results"),
                        "memory_log_path": str(Path(tmp) / "memory.md"),
                        **provider_config,
                    }
                )
                workflow = FakeWorkflow(_final_state("AAPL", "2026-01-02"))

                with patch("tradingagents.graph.trading_graph.create_llm_client", side_effect=fake_create_llm_client), patch(
                    "tradingagents.graph.trading_graph.GraphSetup.setup_graph",
                    return_value=workflow,
                ):
                    TradingAgentsGraph(
                        selected_analysts=["market", "macro"],
                        config=config,
                        debug=False,
                    )

                self.assertEqual(len(calls), 2)
                self.assertTrue(all(call.get(expected_key) == expected_value for call in calls))


if __name__ == "__main__":
    unittest.main()
