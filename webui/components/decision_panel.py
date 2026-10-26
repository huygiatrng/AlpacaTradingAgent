"""
webui/components/decision_panel.py - Decision summary panel for the web UI.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

def create_decision_panel():
    """Create the decision summary panel for the web UI."""
    return dbc.Card(
        dbc.CardBody([
            html.H4("Decision Summary", className="mb-3"),
            html.Hr(),
            html.Div(
                dcc.Markdown(
                    id="decision-summary",
                    children="Run analysis to see the final decision summary",
                    className="dash-markdown"
                ),
                style={
                    "maxHeight": "400px",
                    "overflowY": "auto",
                    "overflowX": "hidden",
                    "border": "1px solid #334155",
                    "borderRadius": "5px",
                    "padding": "15px",
                    "backgroundColor": "#1E293B"
                }
            )
        ]),
        className="mb-4"
    )
