"""
webui/components/batch_overview_panel.py - Batch overview panel for parallel batch analysis
"""

import dash_bootstrap_components as dbc
from dash import html, dcc

def create_batch_overview_panel():
    """Create the batch overview panel component with interactive ticker navigation"""
    return dbc.Card(
        dbc.CardBody([
            html.H4([
                html.I(className="fas fa-layer-group me-2"),
                "Batch Overview"
            ]),
            html.Hr(),

            # Symbol pagination buttons container
            html.Div(
                id="batch-pagination-container",
                children=[],
                className="mb-3"
            ),

            html.Div(id="batch-summary-header", children=[
                html.P("No batch analysis running", className="text-muted text-center")
            ]),
            html.Div(id="batch-ticker-table", children=[]),
            html.Div([
                html.Small([
                    html.Span("✅ Complete", className="me-3"),
                    html.Span("🔄 In Progress", className="me-3"),
                    html.Span("⏸️ Queued")
                ], className="text-muted")
            ], className="mt-2 text-center"),
            # Hidden store for triggering navigation
            dcc.Store(id="batch-ticker-click-store")
        ]),
        className="mb-4"
    )
