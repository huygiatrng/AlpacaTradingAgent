"""
webui/components/header.py - Header component for the web UI.
"""

import dash_bootstrap_components as dbc
from dash import html

def create_header():
    """Create the header component for the web UI."""
    return dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H1("AlpacaTradingAgent",
                            className="text-center mb-0 mobile-header-title")
                ], xs=9, sm=10),
                dbc.Col([
                    # Debug panel toggle button
                    dbc.Button(
                        [
                            html.I(className="fas fa-bug me-2"),
                            "Debug"
                        ],
                        id="toggle-debug-panel",
                        color="outline-light",
                        size="sm",
                        className="float-end",
                        title="Open Debug Tools panel to view tool calls and prompts"
                    )
                ], xs=3, sm=2, className="d-flex align-items-center justify-content-end")
            ], align="center", className="mobile-header-row")
        ]),
        className="mb-3 mb-md-4"
    ) 