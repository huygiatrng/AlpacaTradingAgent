"""Research settings reach tool requests, and default risk limits fail closed."""

from copy import deepcopy

import pytest

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.safety import SafetyGuard


@pytest.mark.parametrize("depth,context", [("Shallow", "low"), ("Medium", "medium"), ("Deep", "high")])
def test_web_tools_follow_depth_without_changing_selected_model(monkeypatch, depth, context):
    from tradingagents.dataflows import interface

    config = deepcopy(DEFAULT_CONFIG)
    config.update(research_depth=depth, quick_think_llm="gpt-5.6-terra")
    requests = []
    monkeypatch.setattr(interface, "get_config", lambda: config)
    monkeypatch.setattr("tradingagents.dataflows.interface_utils.get_config", lambda: config)
    monkeypatch.setattr(interface, "get_api_key", lambda *_: "test-only")
    monkeypatch.setattr(interface, "get_openai_client_with_timeout", lambda *a, **kw: object())

    def capture(_client, params):
        requests.append(params)
        return type("Response", (), {"output_text": "Evidence with a dated source."})()

    monkeypatch.setattr(interface, "_create_response_with_output_cap_fallback", capture)
    interface.get_stock_news_openai("TSLA", "2026-10-26")
    interface.get_fundamentals_openai("TSLA", "2026-10-26")
    interface.get_global_news_openai("2026-10-26", ticker_context="TSLA")
    assert len(requests) == 3
    for request in requests:
        assert request["tools"][0]["search_context_size"] == context
        assert request["model"] == "gpt-5.6-terra"
        assert request["max_output_tokens"] == 4096
        assert request["store"] is False


@pytest.mark.parametrize("equity,allowed", [(97001, True), (97000, False)])
def test_default_daily_loss_boundary(tmp_path, equity, allowed):
    guard = SafetyGuard(state_path=tmp_path / "state.json", kill_switch_path=tmp_path / "kill")
    result = guard.check_order("AAPL", 100, account={"equity": equity, "last_equity": 100000})
    assert result.allowed is allowed


@pytest.mark.parametrize("equity,allowed", [(90001, True), (90000, False)])
def test_default_drawdown_boundary_survives_restart(tmp_path, equity, allowed):
    paths = dict(state_path=tmp_path / "state.json", kill_switch_path=tmp_path / "kill")
    guard = SafetyGuard(**paths)
    guard.check_order("AAPL", 100, account={"equity": 100000, "last_equity": 100000})
    guard = SafetyGuard(**paths)
    result = guard.check_order("AAPL", 100, account={"equity": equity, "last_equity": equity})
    assert result.allowed is allowed
