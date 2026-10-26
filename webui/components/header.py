"""
webui/components/header.py - Header component for the web UI.
"""

from dash import html

def create_header():
    """Create the header component for the web UI."""
    return html.Header(
        [
            html.Div(
                [
                    html.Img(src="/assets/app.png", alt="", className="app-brand-icon"),
                    html.Div([
                        html.H1("AlpacaTradingAgent", className="app-title"),
                        html.P("Research, portfolio and performance", className="app-subtitle"),
                    ]),
                ],
                className="app-brand",
            ),
        ],
        className="app-header",
    )
