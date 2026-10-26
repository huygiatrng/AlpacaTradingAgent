"""
webui/components/header.py - Header component for the web UI.
"""

import dash_bootstrap_components as dbc
from dash import html

from webui.components.api_config_modal import create_config_button


def create_header():
    """Create the header component for the web UI."""
    return dbc.Card(
        dbc.CardBody([
            dbc.Row([
                # Left side spacer for balance
                dbc.Col(width=2),
                
                # Center title
                dbc.Col([
                    html.H1(
                        "AlpacaTradingAgent",
                        className="text-center mb-0"
                    )
                ], width=8, className="d-flex align-items-center justify-content-center"),
                
                # Right side with Config APIs button
                dbc.Col([
                    create_config_button()
                ], width=2, className="d-flex align-items-center justify-content-end"),
            ], className="align-items-center")
        ]),
        className="mb-4"
    ) 
