"""Tests for LLM cost accounting: pricing resolution, run-log scanning,
aggregation per day/symbol/model, realized-return join, and WebUI wiring."""

import json
import tempfile
import unittest
from pathlib import Path

from tradingagents.llm_cost import (
    aggregate_costs,
    estimate_cost_usd,
    parse_return_pct,
    realized_returns_by_symbol,
    resolve_pricing,
    scan_run_costs,
)


def _write_run(
    root,
    symbol,
    trade_date,
    llm_calls=(),
    summary_tokens=None,
    status="completed",
    started_at=None,
):
    runs_dir = Path(root) / symbol / "TradingAgentsStrategy_logs" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    started = started_at or f"{trade_date}T10:00:00+00:00"
    events = [
        {
            "type": "llm_call",
            "timestamp": started,
            "payload": {"model": model, "usage": usage},
        }
        for model, usage in llm_calls
    ]
    payload = {
        "symbol": symbol,
        "trade_date": trade_date,
        "started_at": started,
        "status": status,
        "events": events,
        "summary": {"total_llm_tokens": summary_tokens or 0},
    }
    name = f"{trade_date}_{started.replace(':', '')}.json"
    (runs_dir / name).write_text(json.dumps(payload), encoding="utf-8")


class PricingTests(unittest.TestCase):
    def test_longest_prefix_wins(self):
        generic = resolve_pricing("gpt-5-preview")
        mini = resolve_pricing("gpt-5-mini-2025")
        self.assertIsNotNone(generic)
        self.assertIsNotNone(mini)
        self.assertLess(mini["input"], generic["input"])

    def test_unknown_model_is_unpriced(self):
        self.assertIsNone(resolve_pricing("totally-unknown-llm-9000"))

    def test_latest_flagship_prices_use_exact_provider_rates(self):
        self.assertEqual(resolve_pricing("gpt-6-astra"), {"input": 10.0, "output": 50.0})
        self.assertEqual(
            resolve_pricing("claude-fable-5-1"),
            {"input": 10.0, "output": 50.0},
        )
        self.assertEqual(resolve_pricing("grok-4.6"), {"input": 2.0, "output": 6.0})
        self.assertEqual(resolve_pricing("grok-4.20"), {"input": 1.25, "output": 2.5})
        self.assertEqual(resolve_pricing("MiniMax-M2.7"), {"input": 0.3, "output": 1.2})
        self.assertEqual(
            resolve_pricing("deepseek-v4-pro"),
            {"input": 0.435, "output": 0.87},
        )

    def test_config_override_beats_defaults(self):
        overrides = {"gpt-5-mini": {"input": 1.0, "output": 2.0}}
        priced = resolve_pricing("gpt-5-mini", overrides=overrides)
        self.assertEqual(priced["input"], 1.0)

    def test_estimate_cost_arithmetic(self):
        overrides = {"fake-model": {"input": 1.0, "output": 10.0}}
        # 1M input @ $1 + 0.5M output @ $10 = $6.
        cost = estimate_cost_usd(
            "fake-model", 1_000_000, 500_000, overrides=overrides
        )
        self.assertAlmostEqual(cost, 6.0)

    def test_estimate_cost_unknown_model_returns_none(self):
        self.assertIsNone(estimate_cost_usd("mystery", 1000, 1000))


class ScanRunCostsTests(unittest.TestCase):
    def test_scans_runs_and_attributes_models(self):
        overrides = {"fake-deep": {"input": 10.0, "output": 30.0}}
        with tempfile.TemporaryDirectory() as root:
            _write_run(
                root,
                "AAPL",
                "2026-07-01",
                llm_calls=[
                    ("fake-deep", {"input_tokens": 100_000, "output_tokens": 10_000}),
                    ("fake-deep", {"input_tokens": 50_000, "output_tokens": 5_000}),
                    ("mystery-model", {"input_tokens": 1_000, "output_tokens": 100}),
                ],
            )
            records = scan_run_costs(eval_results_dir=root, overrides=overrides)

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["symbol"], "AAPL")
        self.assertEqual(record["trade_date"], "2026-07-01")
        self.assertEqual(record["input_tokens"], 151_000)
        self.assertEqual(record["output_tokens"], 15_100)
        # fake-deep: 150k in @ $10/M + 15k out @ $30/M = 1.5 + 0.45 = 1.95
        self.assertAlmostEqual(record["cost_usd"], 1.95)
        self.assertEqual(record["unpriced_tokens"], 1_100)
        self.assertIn("fake-deep", record["models"])

    def test_run_without_events_falls_back_to_summary(self):
        with tempfile.TemporaryDirectory() as root:
            _write_run(root, "NVDA", "2026-07-02", summary_tokens=42_000)
            records = scan_run_costs(eval_results_dir=root)
        self.assertEqual(records[0]["total_tokens"], 42_000)
        self.assertIsNone(records[0]["cost_usd"])
        self.assertEqual(records[0]["unpriced_tokens"], 42_000)

    def test_missing_dir_returns_empty(self):
        self.assertEqual(scan_run_costs(eval_results_dir="does-not-exist-xyz"), [])


class AggregationTests(unittest.TestCase):
    def _records(self):
        overrides = {"fake-deep": {"input": 10.0, "output": 30.0}}
        with tempfile.TemporaryDirectory() as root:
            _write_run(
                root,
                "AAPL",
                "2026-07-01",
                llm_calls=[("fake-deep", {"input_tokens": 100_000, "output_tokens": 10_000})],
            )
            _write_run(
                root,
                "AAPL",
                "2026-07-02",
                llm_calls=[("fake-deep", {"input_tokens": 200_000, "output_tokens": 20_000})],
                started_at="2026-07-02T09:00:00+00:00",
            )
            _write_run(
                root,
                "NVDA",
                "2026-07-02",
                llm_calls=[("fake-deep", {"input_tokens": 300_000, "output_tokens": 30_000})],
                started_at="2026-07-02T11:00:00+00:00",
            )
            return scan_run_costs(eval_results_dir=root, overrides=overrides)

    def test_aggregates_by_day_symbol_and_model(self):
        agg = aggregate_costs(self._records())
        self.assertEqual(agg["totals"]["runs"], 3)
        self.assertEqual(set(agg["per_day"]), {"2026-07-01", "2026-07-02"})
        self.assertEqual(agg["per_day"]["2026-07-02"]["runs"], 2)
        self.assertEqual(set(agg["per_symbol"]), {"AAPL", "NVDA"})
        self.assertEqual(agg["per_symbol"]["AAPL"]["runs"], 2)
        self.assertIn("fake-deep", agg["per_model"])
        # Total cost: (600k in @ $10/M) + (60k out @ $30/M) = 6.0 + 1.8
        self.assertAlmostEqual(agg["totals"]["cost_usd"], 7.8)


class RealizedReturnJoinTests(unittest.TestCase):
    def test_parse_return_pct(self):
        self.assertAlmostEqual(parse_return_pct("+3.2%"), 0.032)
        self.assertAlmostEqual(parse_return_pct("-1.5%"), -0.015)
        self.assertIsNone(parse_return_pct("n/a"))
        self.assertIsNone(parse_return_pct(None))

    def test_returns_joined_from_memory_log(self):
        from tradingagents.agents.utils.memory import TradingMemoryLog

        with tempfile.TemporaryDirectory() as tmp:
            log_path = str(Path(tmp) / "trading_memory.md")
            log = TradingMemoryLog({"memory_log_path": log_path})
            log.store_decision("AAPL", "2026-07-01", "FINAL DECISION: BUY")
            log.batch_update_with_outcomes(
                [
                    {
                        "ticker": "AAPL",
                        "trade_date": "2026-07-01",
                        "raw_return": 0.04,
                        "alpha_return": None,
                        "holding_days": 5,
                        "reflection": "went well",
                    }
                ]
            )
            returns = realized_returns_by_symbol({"memory_log_path": log_path})

        self.assertIn("AAPL", returns)
        self.assertEqual(returns["AAPL"]["resolved"], 1)
        self.assertAlmostEqual(returns["AAPL"]["avg_return"], 0.04, places=3)


class CostWebUIWiringTests(unittest.TestCase):
    def test_panel_component_builds(self):
        from webui.components.cost_panel import create_cost_panel

        rendered = str(create_cost_panel())
        for component_id in (
            "cost-refresh-btn",
            "cost-summary-cards",
            "cost-daily-graph",
            "cost-symbol-table",
        ):
            self.assertIn(component_id, rendered)

    def test_callbacks_register_on_fresh_app(self):
        import dash

        from webui.callbacks.cost_callbacks import register_cost_callbacks

        app = dash.Dash(__name__, suppress_callback_exceptions=True)
        register_cost_callbacks(app)
        self.assertTrue(
            any("cost-summary-cards" in key for key in app.callback_map),
            f"cost callback missing: {list(app.callback_map)}",
        )


if __name__ == "__main__":
    unittest.main()
