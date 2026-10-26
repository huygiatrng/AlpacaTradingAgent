"""Opening the workspace must not wait for broker data or the analysis engine."""

import os
from pathlib import Path
import subprocess
import sys

from dash import Dash, html


def test_initial_layout_does_not_contact_services_or_load_analysis_engine():
    script = """
import socket
import sys
attempts = []
def blocked(*args, **kwargs):
    attempts.append(True)
    raise RuntimeError('Network is unavailable during startup')
socket.socket.connect = blocked
socket.create_connection = blocked
socket.getaddrinfo = blocked
from webui.app_dash import create_app
app = create_app()
response = app.server.test_client().get('/_dash-layout')
assert response.status_code == 200
assert not attempts, 'Initial layout attempted network access'
assert 'tradingagents.graph.trading_graph' not in sys.modules
assert 'chromadb' not in sys.modules
assert b'account-summary-container' in response.data
assert b'orders-table-body-container' in response.data
assert b'"setup-api-btn"' in response.data
assert b'"open-api-config-btn"' not in response.data
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, PYTHON_DOTENV_DISABLED="1"),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_chart_collapses_again_when_data_is_cleared():
    from webui.callbacks.chart_callbacks import register_chart_callbacks
    app = Dash(__name__)
    register_chart_callbacks(app)
    toggle = app.callback_map['..chart-content.style...chart-empty-state.style..']['callback'].__wrapped__
    assert toggle(None) == ({"display": "none"}, {})
    assert toggle({"data": [{"x": ["2026-01-02"], "close": [100], "type": "candlestick"}]}) == ({}, {"display": "none"})
    assert toggle({"data": []}) == ({"display": "none"}, {})


def test_account_summary_refreshes_after_opening_and_saving_keys(monkeypatch):
    from webui.callbacks import trading_callbacks
    calls = []
    def render():
        calls.append(True)
        return html.P("Loaded account")
    monkeypatch.setattr(trading_callbacks, "render_account_summary", render)
    app = Dash(__name__)
    trading_callbacks.register_trading_callbacks(app)
    entry = app.callback_map['account-summary-container.children']
    callback = entry['callback'].__wrapped__
    assert callback(0, None, None, {}).children == "Loaded account"
    assert callback(0, None, None, {"alpaca-key": "changed"}).children == "Loaded account"
    assert len(calls) == 2
    assert 'api-keys-store' in {item['id'] for item in entry['inputs']}
