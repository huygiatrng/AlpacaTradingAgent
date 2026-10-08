import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from tradingagents.agents.analysts.macro_analyst import create_macro_analyst
from tradingagents.dataflows import fxmacrodata_utils
from tradingagents.dataflows.config import is_fxmacrodata_enabled
from tradingagents.default_config import DEFAULT_CONFIG


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.reason = "OK" if status_code < 400 else "Error"

    def json(self):
        return self._payload


def _announcement_payload(rows, unit="%", name="Policy Rate", as_of="2026-10-01T23:59:59Z"):
    return {
        "name": name,
        "value_metadata": {"source_unit": unit},
        "pagination": {"has_more": False},
        "replay": {"as_of": as_of},
        "data": [{"publication_time_status": "confirmed", **row} for row in rows],
    }


class FXMacroDataUtilsTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        historical = patch.object(fxmacrodata_utils, "_is_historical_date", return_value=True)
        historical.start()
        self.addCleanup(historical.stop)

    def _router(self, routes):
        def fake_get(url, params=None, headers=None, timeout=None, allow_redirects=True):
            self.assertFalse(allow_redirects)
            self.calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {})})
            path = url.replace(fxmacrodata_utils.BASE_URL, "")
            handler = routes.get(path)
            if handler is None:
                return FakeResponse(404, {"detail": "not found"})
            return handler(dict(params or {})) if callable(handler) else handler

        return fake_get

    def test_keyless_requests_usd_only_and_sends_no_key_header(self):
        routes = {
            "/announcements/usd/policy_rate": FakeResponse(
                200,
                _announcement_payload(
                    [
                        {"date": "2026-09-16", "val": 4.0, "announcement_datetime": 1789581600},
                        {"date": "2026-07-29", "val": 3.75, "announcement_datetime": 1785348000},
                    ]
                ),
            ),
        }
        with patch.object(fxmacrodata_utils, "get_fxmacrodata_api_key", return_value=None), patch(
            "tradingagents.dataflows.fxmacrodata_utils.requests.get", side_effect=self._router(routes)
        ):
            report = fxmacrodata_utils.get_global_macro_indicators("2026-10-01", "USD,EUR")

        self.assertTrue(self.calls)
        self.assertTrue(all("/usd/" in call["url"] for call in self.calls))
        self.assertTrue(all("X-API-Key" not in call["headers"] for call in self.calls))
        self.assertIn("| Policy Rate | 4.00% | 2026-09-16 | 3.75% (2026-07-29) | +0.25 |", report)
        self.assertIn("EUR need FXMACRODATA_API_KEY", report)

    def test_null_values_and_unreleased_rows_are_skipped(self):
        routes = {
            "/announcements/eur/inflation": FakeResponse(
                200,
                _announcement_payload(
                    [
                        # announced after the backtest date
                        {"date": "2026-08-01", "val": 2.4, "announcement_datetime": 1789000000},
                        {"date": "2026-07-01", "val": None, "announcement_datetime": 1786000000},
                        {"date": "2026-06-01", "val": 2.1, "announcement_datetime": 1783000000},
                        {"date": "2026-05-01", "val": 1.9, "announcement_datetime": 1780000000},
                    ],
                    unit="%YoY",
                    name="Inflation (HICP)",
                    as_of="2026-08-20T23:59:59Z",
                ),
            ),
        }
        with patch.object(fxmacrodata_utils, "get_fxmacrodata_api_key", return_value="test-key"), patch(
            "tradingagents.dataflows.fxmacrodata_utils.requests.get", side_effect=self._router(routes)
        ):
            report = fxmacrodata_utils.get_global_macro_indicators("2026-08-20", ["EUR"])

        self.assertEqual(self.calls[0]["headers"].get("X-API-Key"), "test-key")
        self.assertEqual(self.calls[0]["params"].get("end_date"), "2026-08-20")
        self.assertIn("| Inflation (HICP) | 2.10 %YoY | 2026-06-01 | 1.90 %YoY (2026-05-01) | +0.20 |", report)
        self.assertNotIn("2.40", report)

    def test_calendar_merges_currencies_in_time_order(self):
        routes = {
            "/calendar/usd": FakeResponse(
                200,
                {
                    "data": [
                        {
                            "announcement_datetime": 1790944200,
                            "announcement_datetime_utc": "2026-10-02T12:30:00+00:00",
                            "release": "non_farm_payrolls",
                            "name": "Nonfarm Payrolls",
                            "event_importance": "high",
                            "reference_period": "September 2026",
                        }
                    ]
                },
            ),
            "/calendar/eur": FakeResponse(
                200,
                {
                    "data": [
                        {
                            "announcement_datetime": 1790846000,
                            "release": "inflation",
                            "name": "Inflation (HICP)",
                            "event_importance": "high",
                        }
                    ]
                },
            ),
        }
        with patch.object(fxmacrodata_utils, "get_fxmacrodata_api_key", return_value="test-key"), patch(
            "tradingagents.dataflows.fxmacrodata_utils.requests.get", side_effect=self._router(routes)
        ):
            report = fxmacrodata_utils.get_economic_release_calendar("2026-10-01", "USD,EUR", days_ahead=7)

        self.assertEqual(self.calls[0]["params"], {"start_date": "2026-10-01", "end_date": "2026-10-08"})
        self.assertLess(report.index("Inflation (HICP)"), report.index("Nonfarm Payrolls"))
        self.assertIn("| 2026-10-02 12:30 | USD | Nonfarm Payrolls | high | September 2026 |", report)

    def test_invalid_key_is_reported_not_raised(self):
        routes = {"/calendar/gbp": FakeResponse(401, {"detail": "invalid key"})}
        with patch.object(fxmacrodata_utils, "get_fxmacrodata_api_key", return_value="bad-key"), patch(
            "tradingagents.dataflows.fxmacrodata_utils.requests.get", side_effect=self._router(routes)
        ):
            report = fxmacrodata_utils.get_economic_release_calendar("2026-10-01", "GBP")

        self.assertIn("GBP need FXMACRODATA_API_KEY", report)

    def test_fx_rates_follow_pagination(self):
        def forex_pages(params):
            if params["offset"] == 0:
                return FakeResponse(
                    200,
                    {
                        "pagination": {"has_more": True, "next_offset": 2},
                        "data": [{"date": "2026-09-30", "val": 1.10}, {"date": "2026-09-29", "val": None}],
                    },
                )
            return FakeResponse(
                200,
                {"pagination": {"has_more": False}, "data": [{"date": "2026-09-01", "val": 1.00}]},
            )

        routes = {"/forex/eur/usd": forex_pages}
        with patch.object(fxmacrodata_utils, "get_fxmacrodata_api_key", return_value="test-key"), patch(
            "tradingagents.dataflows.fxmacrodata_utils.requests.get", side_effect=self._router(routes)
        ):
            report = fxmacrodata_utils.get_fx_rates_report("2026-09-30", "EUR/USD", lookback_days=30)

        self.assertEqual([call["params"]["offset"] for call in self.calls], [0, 2])
        self.assertEqual(self.calls[0]["params"]["limit"], 100)
        self.assertIn("| EUR/USD | 1.1000 | 2026-09-30 | +10.00% | 1.0000 - 1.1000 |", report)

    def test_fx_rates_need_a_key(self):
        with patch.object(fxmacrodata_utils, "get_fxmacrodata_api_key", return_value=None), patch(
            "tradingagents.dataflows.fxmacrodata_utils.requests.get"
        ) as fake_get:
            report = fxmacrodata_utils.get_fx_rates_report("2026-09-30", "EUR/USD")

        fake_get.assert_not_called()
        self.assertTrue(report.startswith("Error"))


class FXMacroDataOptInTests(unittest.TestCase):
    def test_disabled_by_default(self):
        with patch.dict(os.environ, {"FXMACRODATA_API_KEY": ""}):
            os.environ.pop("FXMACRODATA_ENABLED", None)
            self.assertFalse(is_fxmacrodata_enabled())

    def test_enabled_by_key_or_flag(self):
        with patch.dict(os.environ, {"FXMACRODATA_API_KEY": "test-key"}):
            self.assertTrue(is_fxmacrodata_enabled())
        with patch.dict(os.environ, {"FXMACRODATA_API_KEY": "", "FXMACRODATA_ENABLED": "true"}):
            self.assertTrue(is_fxmacrodata_enabled())


class FakeTool:
    def __init__(self, name):
        self.name = name


class FakeMacroToolkit:
    def __init__(self, fxmacrodata=False, fxmacrodata_key=False):
        self.config = {**DEFAULT_CONFIG, "online_tools": False}
        self._fxmacrodata = fxmacrodata
        self._fxmacrodata_key = fxmacrodata_key
        for name in (
            "get_macro_analysis",
            "get_economic_indicators",
            "get_yield_curve_analysis",
            "get_macro_news_openai",
            "get_global_macro_indicators",
            "get_economic_release_calendar",
            "get_fx_rates",
        ):
            setattr(self, name, FakeTool(name))

    def has_fred(self):
        return True

    def has_openai_web_search(self):
        return False

    def has_fxmacrodata(self):
        return self._fxmacrodata

    def has_fxmacrodata_key(self):
        return self._fxmacrodata_key


class CapturingLLM:
    def __init__(self):
        self.bound_tool_names = []

    def bind_tools(self, tools):
        self.bound_tool_names.append([tool.name for tool in tools])
        return RunnableLambda(lambda _messages: AIMessage(content="Macro analysis."))


class MacroAnalystFXMacroDataToolTests(unittest.TestCase):
    def _bound_tools(self, toolkit):
        llm = CapturingLLM()
        node = create_macro_analyst(llm, toolkit)
        with patch("tradingagents.agents.analysts.macro_analyst.capture_agent_prompt"):
            node({"trade_date": "2026-10-01", "company_of_interest": "SPY", "messages": []})
        return llm.bound_tool_names[-1]

    def test_tools_absent_when_not_enabled(self):
        tools = self._bound_tools(FakeMacroToolkit())
        self.assertIn("get_macro_analysis", tools)
        self.assertNotIn("get_global_macro_indicators", tools)
        self.assertNotIn("get_economic_release_calendar", tools)

    def test_keyless_mode_adds_macro_and_calendar_but_not_fx(self):
        tools = self._bound_tools(FakeMacroToolkit(fxmacrodata=True))
        self.assertIn("get_global_macro_indicators", tools)
        self.assertIn("get_economic_release_calendar", tools)
        self.assertNotIn("get_fx_rates", tools)
        # FRED tools stay first
        self.assertEqual(tools[0], "get_macro_analysis")

    def test_key_adds_fx_rates(self):
        tools = self._bound_tools(FakeMacroToolkit(fxmacrodata=True, fxmacrodata_key=True))
        self.assertIn("get_fx_rates", tools)


if __name__ == "__main__":
    unittest.main()
