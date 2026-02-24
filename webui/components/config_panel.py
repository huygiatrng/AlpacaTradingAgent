"""
webui/components/config_panel.py - Configuration panel for the web UI.
"""

import dash_bootstrap_components as dbc
from dash import html
from datetime import datetime

def create_config_panel():
    """Create the configuration panel for the web UI."""
    return dbc.Card(
        dbc.CardBody([
            html.H4("Analysis Configuration", className="mb-3"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    dbc.Input(
                        id="ticker-input",
                        type="text",
                        placeholder="Enter stock symbols (e.g., AAPL,NVDA)",
                        value="NVDA, AMD, TSLA",
                        className="mb-2"
                    ),
                ], width=12),
            ]),
            html.H5("Select Analysts:", className="mt-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Checkbox(id="analyst-market", label="Market Analyst", value=True, className="mb-2"),
                ], width=6),
                dbc.Col([
                    dbc.Checkbox(id="analyst-social", label="Social Media Analyst", value=True, className="mb-2"),
                ], width=6),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Checkbox(id="analyst-news", label="News Analyst", value=True, className="mb-2"),
                ], width=6),
                dbc.Col([
                    dbc.Checkbox(id="analyst-fundamentals", label="Fundamentals Analyst", value=True, className="mb-2"),
                ], width=6),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Checkbox(id="analyst-macro", label="Macro Analyst", value=True, className="mb-2"),
                ], width=6),
                dbc.Col([
                    # Empty column for alignment
                ], width=6),
            ]),
            html.H5("Research Depth:", className="mt-3"),
            # 50/50 split for research depth selection and information
            dbc.Row([
                dbc.Col([
                    dbc.RadioItems(
                        id="research-depth",
                        options=[
                            {"label": "Shallow", "value": "Shallow"},
                            {"label": "Medium", "value": "Medium"},
                            {"label": "Deep", "value": "Deep"},
                        ],
                        value="Shallow",
                        inline=False,
                        className="mb-3"
                    ),
                ], width=6),
                dbc.Col([
                    # Interactive research depth information
                    html.Div(id="research-depth-info", className="mb-3"),
                ], width=6),
            ]),
            # Execution Mode section removed - always use sequential execution
            html.H5("Trading Mode:", className="mt-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Switch(
                        id="allow-shorts",
                        label="Allow Shorts (Trading Mode)",
                        value=False,
                        className="mb-2"
                    ),
                ], width=6),
                dbc.Col([
                    html.Div(id="trading-mode-info", className="mb-3"),
                ], width=6),
            ]),
            html.H5("Execution Mode:", className="mt-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Switch(
                        id="parallel-execution",
                        label="Enable Parallel Analyst Execution",
                        value=False,  # Default to disabled for more reliable execution
                        className="mb-2"
                    ),
                ], width=6),
                dbc.Col([
                    html.Div(id="parallel-execution-info", className="mb-3"),
                ], width=6),
            ]),
            html.H5("Parallel Batch Configuration:", className="mt-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Batch Size (symbols per batch)", className="mb-1"),
                    dbc.Input(
                        id="batch-size",
                        type="number",
                        placeholder="5",
                        value=5,
                        min=1,
                        max=20,
                        className="mb-2"
                    ),
                ], width=6),
                dbc.Col([
                    dbc.Label("Batch Delay (seconds between batches)", className="mb-1"),
                    dbc.Input(
                        id="batch-delay",
                        type="number",
                        placeholder="5",
                        value=5,
                        min=0,
                        max=60,
                        className="mb-2"
                    ),
                ], width=6),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(id="batch-config-info", className="mb-3"),
                ], width=12),
            ]),
            html.H5("Scheduling Configuration:", className="mt-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Switch(
                        id="loop-enabled",
                        label="Enable Loop Mode",
                        value=False,
                        className="mb-2"
                    ),
                ], width=6),
                dbc.Col([
                    dbc.Label("Loop Interval (minutes)", className="mb-1"),
                    dbc.Input(
                        id="loop-interval",
                        type="number",
                        placeholder="60",
                        value=60,
                        min=1,
                        max=1440,  # Max 24 hours
                        className="mb-2"
                    ),
                ], width=6),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Switch(
                        id="market-hour-enabled",
                        label="Trade at Market Hour",
                        value=False,
                        className="mb-2"
                    ),
                ], width=6),
                dbc.Col([
                    dbc.Label("Trading Hours (e.g., 10,15 for 10AM & 3PM)", className="mb-1"),
                    dbc.Input(
                        id="market-hours-input",
                        type="text",
                        placeholder="e.g., 11,13",
                        value="",
                        className="mb-2"
                    ),
                ], width=6),
            ]),
            # Market hours validation message
            html.Div(id="market-hours-validation", className="mb-2"),
            # Add scheduling mode information display
            html.Div(id="scheduling-mode-info", className="mb-3"),
            html.H5("Automated Trading:", className="mt-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Switch(
                        id="trade-after-analyze",
                        label="Trade After Analyze",
                        value=False,
                        className="mb-2"
                    ),
                ], width=6),
                dbc.Col([
                    dbc.Switch(
                        id="ai-position-sizing",
                        label="AI-Determined Position Sizing",
                        value=True,  # Default enabled
                        className="mb-2"
                    ),
                ], width=6),
            ]),
            html.Small("AI agents determine trade size based on risk analysis", className="text-muted mb-2"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Max Order Amount ($)", className="mb-1"),
                    html.Small(
                        "When AI sizing enabled: acts as maximum cap. When disabled: fixed trade amount.",
                        className="text-muted d-block mb-1"
                    ),
                    dbc.Input(
                        id="trade-dollar-amount",
                        type="number",
                        placeholder="4500",
                        value=4500,
                        min=1,
                        max=10000000,
                        className="mb-2"
                    ),
                ], width=12),
            ]),
            # Add trading mode information display
            html.Div(id="trade-after-analyze-info", className="mb-3"),

            # Order Protection Settings
            html.H5("Order Protection:", className="mt-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Switch(
                        id="use-stop-loss",
                        label="Use Stop Loss Orders",
                        value=True,  # Default enabled
                        className="mb-2"
                    ),
                    html.Small("Place automatic stop loss orders based on AI analysis", className="text-muted d-block mb-2"),
                ], width=6),
                dbc.Col([
                    dbc.Switch(
                        id="use-take-profit",
                        label="Use Take Profit Orders",
                        value=True,  # Default enabled
                        className="mb-2"
                    ),
                    html.Small("Place automatic take profit orders based on AI analysis", className="text-muted d-block mb-2"),
                ], width=6),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Switch(
                        id="use-bracket-orders",
                        label="Use Native Bracket Orders",
                        value=False,
                        className="mb-2"
                    ),
                    html.Small(
                        "Atomic entry + stop + target in one order (uses T1 only, no T2 scale-out)",
                        className="text-muted d-block mb-3"
                    ),
                ], width=12),
            ]),

            html.H5("LLM Quick Thinker Model:", className="mt-3"),
            dbc.Select(
                id="quick-llm",
                options=[
                    {"label": "gpt-5.2-2025-12-11", "value": "gpt-5.2-2025-12-11"},
                    {"label": "gpt-5-mini-2025-08-07", "value": "gpt-5-mini-2025-08-07"},
                    {"label": "gpt-5", "value": "gpt-5"},
                    {"label": "gpt-5-mini", "value": "gpt-5-mini"},
                    {"label": "gpt-5-nano", "value": "gpt-5-nano"},
                    {"label": "gpt-4.1", "value": "gpt-4.1"},
                    {"label": "gpt-4.1-nano", "value": "gpt-4.1-nano"},
                    {"label": "gpt-4.1-mini", "value": "gpt-4.1-mini"},
                    {"label": "gpt-4o", "value": "gpt-4o"},
                    {"label": "gpt-4o-mini", "value": "gpt-4o-mini"},
                    {"label": "o3-mini", "value": "o3-mini"},
                    {"label": "o3", "value": "o3"},
                    {"label": "o1", "value": "o1"},
                ],
                value="gpt-5-mini-2025-08-07",
                className="mb-2"
            ),
            html.H5("LLM Deep Thinker Model:", className="mt-3"),
            dbc.Select(
                id="deep-llm",
                options=[
                    {"label": "gpt-5.2-2025-12-11", "value": "gpt-5.2-2025-12-11"},
                    {"label": "gpt-5-mini-2025-08-07", "value": "gpt-5-mini-2025-08-07"},
                    {"label": "gpt-5", "value": "gpt-5"},
                    {"label": "gpt-5-mini", "value": "gpt-5-mini"},
                    {"label": "gpt-5-nano", "value": "gpt-5-nano"},
                    {"label": "gpt-4.1", "value": "gpt-4.1"},
                    {"label": "gpt-4.1-nano", "value": "gpt-4.1-nano"},
                    {"label": "gpt-4.1-mini", "value": "gpt-4.1-mini"},
                    {"label": "gpt-4o", "value": "gpt-4o"},
                    {"label": "gpt-4o-mini", "value": "gpt-4o-mini"},
                    {"label": "o3-mini", "value": "o3-mini"},
                    {"label": "o3", "value": "o3"},
                    {"label": "o1", "value": "o1"},
                ],
                value="gpt-5.2-2025-12-11",
                className="mb-3"
            ),
            # Dynamic Start/Stop button
            html.Div(id="control-button-container", children=[
                dbc.Button(
                    "Start Analysis",
                    id="control-btn",
                    color="primary",
                    size="lg",
                    className="w-100 mt-2"
                )
            ]),
            html.Div(id="result-text", className="mt-3")
        ]),
        className="mb-4",
    ) 
