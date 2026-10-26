"""
webui/components/status_panel.py - Status panel for the web UI.
"""

import dash_bootstrap_components as dbc
from dash import html

def create_status_panel():
    """Create the status panel for the web UI."""
    return dbc.Card(
        dbc.CardBody([
            html.H4("Analysis Status", className="mb-3"),
            html.Hr(),
            html.Div(id="status-table"),
            dbc.Row([
                dbc.Col([
                    html.Div(id="tool-calls-text", children="🧰 Tool Calls: 0"),
                ], xs=12, sm=4),
                dbc.Col([
                    html.Div(id="llm-calls-text", children="🤖 LLM Calls: 0"),
                ], xs=12, sm=4),
                dbc.Col([
                    html.Div(id="reports-text", children="📊 Generated Reports: 0"),
                ], xs=12, sm=4),
            ], className="mt-3"),
            html.Div(id="refresh-status", children="⏸️ Updates paused until analysis starts", className="text-secondary mt-2")
        ]),
        className="mb-4"
    )
