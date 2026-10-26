"""
webui/components/backtest_panel.py - Walk-forward backtest dashboard panel

Replays the decisions already recorded under eval_results/ against
historical prices (zero LLM cost) and reports the TradingAgents-paper
metric set: cumulative/annualized return, Sharpe ratio, max drawdown,
plus win rate, per walk-forward window and for the full period.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html


def create_backtest_panel():
    """Create the backtest panel card for the web UI."""
    controls = dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Label("Symbol", html_for="backtest-symbol-input", className="small"),
                    dbc.Input(
                        id="backtest-symbol-input",
                        type="text",
                        placeholder="e.g. AAPL or BTC/USD",
                        debounce=True,
                    ),
                ],
                xs=12, sm=6, lg=4,
            ),
            dbc.Col(
                [
                    dbc.Label("Start date (optional)", html_for="backtest-start-date", className="small"),
                    dbc.Input(id="backtest-start-date", type="date"),
                ],
                xs=12, sm=6, lg=4,
            ),
            dbc.Col(
                [
                    dbc.Label("End date (optional)", html_for="backtest-end-date", className="small"),
                    dbc.Input(id="backtest-end-date", type="date"),
                ],
                xs=12, sm=6, lg=4,
            ),
            dbc.Col(
                [
                    dbc.Label("Window (bars)", html_for="backtest-window-bars", className="small"),
                    dbc.Input(
                        id="backtest-window-bars",
                        type="number",
                        value=63,
                        min=2,
                        step=1,
                    ),
                ],
                xs=12, sm=6, lg=4,
            ),
            dbc.Col(
                [
                    dbc.Label("Slippage", html_for="backtest-slippage-model", className="small"),
                    dbc.Select(
                        id="backtest-slippage-model",
                        options=[
                            {"label": "Fixed (5 bps)", "value": "fixed"},
                            {"label": "Volatility-scaled", "value": "volatility"},
                            {"label": "None (frictionless)", "value": "none"},
                        ],
                        value="fixed",
                    ),
                ],
                xs=12, sm=6, lg=4,
            ),
            dbc.Col(
                [
                    dbc.Label(" ", className="small d-block"),
                    dbc.Switch(
                        id="backtest-allow-shorts",
                        label="Allow shorts",
                        value=False,
                    ),
                ],
                xs=12, sm=6, lg=4,
            ),
            dbc.Col(
                [
                    dbc.Label(" ", className="small d-block"),
                    dbc.Button(
                        [html.I(className="fas fa-flask me-2"), "Run Backtest"],
                        id="backtest-run-btn",
                        color="primary",
                        className="w-100",
                    ),
                ],
                xs=12, sm=6, lg=4,
            ),
        ],
        className="g-2 align-items-end mb-3",
    )

    teach_row = dbc.Row(
        [
            dbc.Col(
                dbc.Button(
                    [html.I(className="fas fa-graduation-cap me-2"), "Teach Memory"],
                    id="backtest-teach-btn",
                    color="secondary",
                    outline=True,
                    size="sm",
                ),
                width="auto",
            ),
            dbc.Col(
                html.Div(
                    "Adds dated lessons to agent memory. BUY and SHORT lessons "
                    "use hypothetical fixed-horizon returns, not realized account "
                    "profits. Duplicate lessons are skipped; embedding API costs may apply.",
                    className="text-muted small",
                ),
            ),
        ],
        className="g-2 align-items-center mb-2",
    )

    return dbc.Card(
        dbc.CardBody(
            [
                html.H4("Walk-Forward Backtest", className="mb-1"),
                html.Div(
                    "Replays this deployment's recorded agent decisions on historical "
                    "prices — signals execute at the next bar's open (no lookahead).",
                    className="text-muted small mb-3",
                ),
                controls,
                teach_row,
                dcc.Loading(
                    id="backtest-teach-loading",
                    type="default",
                    children=html.Div(id="backtest-teach-status", className="mb-2"),
                ),
                dcc.Loading(
                    id="backtest-loading",
                    type="default",
                    children=html.Div(
                        [
                            html.Div(id="backtest-status", className="mb-2"),
                            html.Div(id="backtest-metrics", className="mb-3"),
                            html.Div(
                                dcc.Graph(
                                    id="backtest-equity-graph",
                                    config={"displayModeBar": False, "responsive": True},
                                    style={"height": "320px", "width": "100%"},
                                ),
                                id="backtest-graph-container",
                                style={"display": "none"},
                            ),
                            html.Div(id="backtest-windows-table"),
                        ]
                    ),
                ),
            ]
        ),
        className="mb-4",
    )
