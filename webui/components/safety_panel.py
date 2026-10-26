"""
webui/components/safety_panel.py - Production safety layer status panel

Shows a green/red badge for every deterministic guard (kill switch,
pre-trade checks, circuit breakers, LLM budget) plus a kill-switch toggle
that halts all order flow immediately, regardless of agent decisions.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html


def create_safety_panel():
    """Create the safety guardrails card for the web UI."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.H6("Safety Guardrails", className="compact-panel-title"),
                        html.Span("Checking…", id="safety-summary", className="compact-panel-summary", role="status"),
                        html.Div(
                            [
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button(
                                            [
                                                html.I(className="fas fa-hand-paper me-2"),
                                                "Engage Kill Switch",
                                            ],
                                            id="safety-kill-switch-btn",
                                            color="danger",
                                            size="sm",
                                        ),
                                        dbc.Button(
                                            "Release",
                                            id="safety-release-btn",
                                            color="secondary",
                                            outline=True,
                                            size="sm",
                                        ),
                                    ],
                                ),
                                dbc.Button("Details", id="safety-details-toggle", color="outline-secondary", size="sm"),
                            ],
                            className="compact-panel-actions",
                        ),
                    ],
                    className="compact-panel-toolbar",
                ),
                html.Div(id="safety-action-status"),
                html.Div(id="safety-alerts"),
                dbc.Collapse(html.Div(id="safety-status-container"), id="safety-details", is_open=False,
                             className="compact-panel-details"),
                dcc.Interval(
                    id="safety-refresh-interval",
                    interval=30_000,  # 30s
                    n_intervals=0,
                ),
            ]
        ),
        className="mb-2 safety-panel compact-panel",
    )
