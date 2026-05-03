"""
API Configuration Modal Component for TradingAgents WebUI

This component creates a modal dialog for configuring API keys.
Keys are hidden by default (password style) but can be toggled to show.
Supports both localStorage persistence and .env file fallback.
"""

import dash_bootstrap_components as dbc
from dash import html, dcc


# Define API configuration structure
API_CONFIGS = [
    {
        "id": "openai",
        "name": "OpenAI API Key",
        "env_var": "OPENAI_API_KEY",
        "placeholder": "sk-...",
        "help_url": "https://platform.openai.com/api-keys",
        "help_text": "Required for LLM functionality",
        "icon": "fas fa-brain"
    },
    {
        "id": "google",
        "name": "Google API Key",
        "env_var": "GOOGLE_API_KEY",
        "placeholder": "Your Google AI Studio API key",
        "help_url": "https://aistudio.google.com/app/apikey",
        "help_text": "Required when using Gemini models",
        "icon": "fab fa-google"
    },
    {
        "id": "anthropic",
        "name": "Anthropic API Key",
        "env_var": "ANTHROPIC_API_KEY",
        "placeholder": "sk-ant-...",
        "help_url": "https://console.anthropic.com/settings/keys",
        "help_text": "Required when using Claude models",
        "icon": "fas fa-robot"
    },
    {
        "id": "xai",
        "name": "xAI API Key",
        "env_var": "XAI_API_KEY",
        "placeholder": "xai-...",
        "help_url": "https://console.x.ai/",
        "help_text": "Required when using Grok models",
        "icon": "fas fa-x"
    },
    {
        "id": "deepseek",
        "name": "DeepSeek API Key",
        "env_var": "DEEPSEEK_API_KEY",
        "placeholder": "sk-...",
        "help_url": "https://platform.deepseek.com/api_keys",
        "help_text": "Required when using DeepSeek models",
        "icon": "fas fa-water"
    },
    {
        "id": "dashscope",
        "name": "DashScope API Key",
        "env_var": "DASHSCOPE_API_KEY",
        "placeholder": "sk-...",
        "help_url": "https://dashscope.console.aliyun.com/apiKey",
        "help_text": "Required when using Qwen models",
        "icon": "fas fa-cloud"
    },
    {
        "id": "zhipu",
        "name": "Zhipu API Key",
        "env_var": "ZHIPU_API_KEY",
        "placeholder": "Your Zhipu API key",
        "help_url": "https://bigmodel.cn/usercenter/proj-mgmt/apikeys",
        "help_text": "Required when using GLM models",
        "icon": "fas fa-cube"
    },
    {
        "id": "openrouter",
        "name": "OpenRouter API Key",
        "env_var": "OPENROUTER_API_KEY",
        "placeholder": "sk-or-...",
        "help_url": "https://openrouter.ai/settings/keys",
        "help_text": "Required when using OpenRouter",
        "icon": "fas fa-route"
    },
    {
        "id": "azure-openai",
        "name": "Azure OpenAI API Key",
        "env_var": "AZURE_OPENAI_API_KEY",
        "placeholder": "Your Azure OpenAI key",
        "help_url": "https://portal.azure.com/",
        "help_text": "Required when using Azure OpenAI",
        "icon": "fab fa-microsoft"
    },
    {
        "id": "alpaca-key",
        "name": "Alpaca API Key",
        "env_var": "ALPACA_API_KEY",
        "placeholder": "PK...",
        "help_url": "https://app.alpaca.markets/signup",
        "help_text": "Required for real stock data and trading",
        "icon": "fas fa-chart-line"
    },
    {
        "id": "alpaca-secret",
        "name": "Alpaca Secret Key",
        "env_var": "ALPACA_SECRET_KEY",
        "placeholder": "Your Alpaca secret key",
        "help_url": "https://app.alpaca.markets/signup",
        "help_text": "Required for Alpaca authentication",
        "icon": "fas fa-key"
    },
    {
        "id": "finnhub",
        "name": "Finnhub API Key",
        "env_var": "FINNHUB_API_KEY",
        "placeholder": "Your Finnhub API key",
        "help_url": "https://finnhub.io/register",
        "help_text": "Required for financial news and data",
        "icon": "fas fa-newspaper"
    },
    {
        "id": "fred",
        "name": "FRED API Key",
        "env_var": "FRED_API_KEY",
        "placeholder": "Your FRED API key",
        "help_url": "https://fred.stlouisfed.org/docs/api/api_key.html",
        "help_text": "Required for macro economic analysis",
        "icon": "fas fa-university"
    },
    {
        "id": "coindesk",
        "name": "CryptoCompare API Key",
        "env_var": "COINDESK_API_KEY",
        "placeholder": "Your CryptoCompare API key",
        "help_url": "https://www.cryptocompare.com/cryptopian/api-keys",
        "help_text": "Required for cryptocurrency news",
        "icon": "fab fa-bitcoin"
    },
    {
        "id": "alpha-vantage",
        "name": "Alpha Vantage API Key",
        "env_var": "ALPHA_VANTAGE_API_KEY",
        "placeholder": "Your Alpha Vantage key",
        "help_url": "https://www.alphavantage.co/support/#api-key",
        "help_text": "Optional fallback market data source",
        "icon": "fas fa-chart-area"
    },
]


def create_api_input_row(api_config):
    """Create a single API key input row with show/hide toggle"""
    api_id = api_config["id"]
    
    return dbc.Row([
        dbc.Col([
            dbc.Label([
                html.I(className=f"{api_config['icon']} me-2"),
                api_config["name"]
            ], className="fw-bold"),
            html.Small(
                [
                    api_config["help_text"],
                    " - ",
                    html.A("Get API Key", href=api_config["help_url"], target="_blank", className="text-info")
                ],
                className="text-muted d-block mb-1"
            ),
        ], width=12),
        dbc.Col([
            dbc.InputGroup([
                dbc.Input(
                    id=f"api-input-{api_id}",
                    type="password",
                    placeholder=api_config["placeholder"],
                    className="api-key-input",
                    style={
                        "background": "#1E293B",
                        "border": "1px solid #334155",
                        "color": "#E2E8F0"
                    }
                ),
                dbc.Button(
                    html.I(id=f"api-toggle-icon-{api_id}", className="fas fa-eye"),
                    id=f"api-toggle-{api_id}",
                    color="outline-secondary",
                    className="api-toggle-btn",
                    title="Show/Hide API Key"
                ),
                dbc.Button(
                    html.I(className="fas fa-check"),
                    id=f"api-status-{api_id}",
                    color="outline-success",
                    disabled=True,
                    className="api-status-indicator",
                    title="API Key Status"
                ),
            ]),
        ], width=12),
    ], className="mb-3")


def create_api_config_modal():
    """Create the API configuration modal"""
    
    # Build API input rows
    api_inputs = [create_api_input_row(api) for api in API_CONFIGS]
    
    # Add Alpaca paper trading toggle
    alpaca_paper_toggle = dbc.Row([
        dbc.Col([
            dbc.Label([
                html.I(className="fas fa-flask me-2"),
                "Alpaca Paper Trading"
            ], className="fw-bold"),
            html.Small(
                "Enable paper trading mode (recommended for testing)",
                className="text-muted d-block mb-1"
            ),
        ], width=8),
        dbc.Col([
            dbc.Switch(
                id="api-alpaca-paper",
                label="",
                value=True,
                className="mt-2"
            ),
        ], width=4, className="d-flex align-items-center justify-content-end"),
    ], className="mb-3")
    
    modal = dbc.Modal(
        [
            dbc.ModalHeader(
                [
                    html.H4([
                        html.I(className="fas fa-cog me-2"),
                        "API Configuration"
                    ], className="mb-0"),
                ],
                close_button=True,
                className="api-config-modal-header"
            ),
            dbc.ModalBody(
                [
                    # Info alert
                    html.Div([
                        html.I(className="fas fa-info-circle me-2"),
                        "Configure your API keys below. Keys are stored in your browser's local storage and take precedence over .env file settings. ",
                        html.Strong("Your keys never leave your browser."),
                    ], className="alert alert-info mb-4"),
                    
                    # .env file status
                    html.Div(
                        id="env-file-status",
                        className="mb-3"
                    ),
                    
                    html.Hr(),
                    
                    # API key inputs
                    html.Div(api_inputs),
                    
                    html.Hr(),
                    
                    # Alpaca paper trading toggle
                    alpaca_paper_toggle,
                    
                    # Hidden store for tracking visibility states
                    dcc.Store(id="api-visibility-store", data={api["id"]: False for api in API_CONFIGS}),
                ],
                className="api-config-modal-body",
                style={"max-height": "70vh", "overflow-y": "auto"}
            ),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        [
                            html.I(className="fas fa-trash me-2"),
                            "Clear All"
                        ],
                        id="clear-api-keys-btn",
                        color="outline-danger",
                        size="sm",
                        className="me-auto"
                    ),
                    dbc.Button(
                        [
                            html.I(className="fas fa-sync me-2"),
                            "Load from .env"
                        ],
                        id="load-env-btn",
                        color="outline-info",
                        size="sm",
                        className="me-2"
                    ),
                    dbc.Button(
                        [
                            html.I(className="fas fa-save me-2"),
                            "Save & Apply"
                        ],
                        id="save-api-keys-btn",
                        color="primary",
                        size="sm",
                        className="me-2"
                    ),
                    dbc.Button(
                        [
                            html.I(className="fas fa-times me-2"),
                            "Close"
                        ],
                        id="close-api-config-btn",
                        color="secondary",
                        size="sm"
                    )
                ]
            )
        ],
        id="api-config-modal",
        is_open=False,
        size="lg",
        backdrop=True,
        scrollable=True,
        className="api-config-modal",
        style={"z-index": "9999"}
    )
    
    return modal


def create_config_button():
    """Create the Config APIs button for the header"""
    return dbc.Button(
        [
            html.I(className="fas fa-key me-2"),
            "Config APIs"
        ],
        id="open-api-config-btn",
        color="outline-warning",
        size="sm",
        className="config-apis-btn",
        title="Configure API Keys"
    )


def get_api_configs():
    """Return the API configuration list for use in callbacks"""
    return API_CONFIGS
