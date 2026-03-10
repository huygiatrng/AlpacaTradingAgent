"""
API Configuration Callbacks for TradingAgents WebUI

Handles:
- Opening/closing the API config modal
- Toggle password visibility for each API key
- Saving API keys to localStorage
- Loading API keys from localStorage or .env file
- Applying API keys to the runtime configuration
"""

import os
from dash import Input, Output, State, callback_context as ctx, no_update, ALL
from dash.exceptions import PreventUpdate
from dotenv import load_dotenv

from webui.components.api_config_modal import get_api_configs
from webui.utils.storage import get_default_api_keys


def register_api_config_callbacks(app):
    """Register API configuration callbacks"""
    
    api_configs = get_api_configs()
    api_ids = [api["id"] for api in api_configs]
    
    # Callback to open/close the API config modal
    @app.callback(
        Output("api-config-modal", "is_open"),
        [
            Input("open-api-config-btn", "n_clicks"),
            Input("close-api-config-btn", "n_clicks"),
            Input("save-api-keys-btn", "n_clicks")
        ],
        State("api-config-modal", "is_open"),
        prevent_initial_call=True
    )
    def toggle_api_config_modal(open_clicks, close_clicks, save_clicks, is_open):
        """Toggle the API config modal open/close state"""
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if trigger_id == "open-api-config-btn":
            return True
        elif trigger_id in ["close-api-config-btn", "save-api-keys-btn"]:
            return False
        
        return is_open
    
    # Create individual toggle callbacks for each API key visibility
    # Using pattern matching callbacks for cleaner code
    for api_config in api_configs:
        api_id = api_config["id"]
        
        @app.callback(
            [
                Output(f"api-input-{api_id}", "type"),
                Output(f"api-toggle-icon-{api_id}", "className")
            ],
            Input(f"api-toggle-{api_id}", "n_clicks"),
            State(f"api-input-{api_id}", "type"),
            prevent_initial_call=True
        )
        def toggle_password_visibility(n_clicks, current_type, _api_id=api_id):
            """Toggle password visibility for an API key input"""
            if not n_clicks:
                raise PreventUpdate
            
            if current_type == "password":
                return "text", "fas fa-eye-slash"
            else:
                return "password", "fas fa-eye"
    
    # Callback to load API keys from localStorage on page load
    @app.callback(
        [
            Output("api-input-openai", "value"),
            Output("api-input-alpaca-key", "value"),
            Output("api-input-alpaca-secret", "value"),
            Output("api-input-finnhub", "value"),
            Output("api-input-fred", "value"),
            Output("api-input-coindesk", "value"),
            Output("api-alpaca-paper", "value"),
            Output("env-file-status", "children")
        ],
        Input("api-keys-store", "data")
    )
    def load_api_keys(stored_keys):
        """Load API keys from localStorage, falling back to .env on first load"""
        import dash_bootstrap_components as dbc
        from dash import html
        
        # Load .env file
        load_dotenv()
        
        # Get values from .env (filter out placeholder values)
        def get_env_value(key):
            val = os.getenv(key, "")
            if val and not val.startswith("your_"):
                return val
            return ""
        
        env_vars = {
            "openai": get_env_value("OPENAI_API_KEY"),
            "alpaca-key": get_env_value("ALPACA_API_KEY"),
            "alpaca-secret": get_env_value("ALPACA_SECRET_KEY"),
            "finnhub": get_env_value("FINNHUB_API_KEY"),
            "fred": get_env_value("FRED_API_KEY"),
            "coindesk": get_env_value("COINDESK_API_KEY"),
        }
        
        alpaca_paper_str = os.getenv("ALPACA_USE_PAPER", "True")
        env_alpaca_paper = alpaca_paper_str.lower() in ("true", "1", "yes")
        
        # Count how many .env keys are set
        env_keys_set = sum(1 for v in env_vars.values() if v)
        
        if env_keys_set > 0:
            env_status = dbc.Alert([
                html.I(className="fas fa-file-alt me-2"),
                f".env file detected with {env_keys_set} API key(s) configured. ",
                "LocalStorage keys will take precedence."
            ], color="success", className="mb-0 py-2")
        else:
            env_status = dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "No .env file detected or no keys configured. Please enter your API keys below."
            ], color="warning", className="mb-0 py-2")
        
        # Check if localStorage has any real keys set
        has_stored_keys = stored_keys and any(
            stored_keys.get(key) for key in ["openai", "alpaca-key", "alpaca-secret", "finnhub", "fred", "coindesk"]
        )
        
        if not has_stored_keys:
            # No localStorage keys - load from .env as default and apply to runtime
            keys_to_apply = {
                "openai": env_vars.get("openai", ""),
                "alpaca-key": env_vars.get("alpaca-key", ""),
                "alpaca-secret": env_vars.get("alpaca-secret", ""),
                "finnhub": env_vars.get("finnhub", ""),
                "fred": env_vars.get("fred", ""),
                "coindesk": env_vars.get("coindesk", ""),
                "alpaca-paper": env_alpaca_paper
            }
            apply_api_keys_to_config(keys_to_apply)
            
            return (
                env_vars.get("openai", ""),
                env_vars.get("alpaca-key", ""),
                env_vars.get("alpaca-secret", ""),
                env_vars.get("finnhub", ""),
                env_vars.get("fred", ""),
                env_vars.get("coindesk", ""),
                env_alpaca_paper,
                env_status
            )
        
        # localStorage has keys - use those and apply to runtime
        apply_api_keys_to_config(stored_keys)
        
        return (
            stored_keys.get("openai", ""),
            stored_keys.get("alpaca-key", ""),
            stored_keys.get("alpaca-secret", ""),
            stored_keys.get("finnhub", ""),
            stored_keys.get("fred", ""),
            stored_keys.get("coindesk", ""),
            stored_keys.get("alpaca-paper", True),
            env_status
        )
    
    # Callback to save API keys to localStorage
    @app.callback(
        Output("api-keys-store", "data"),
        Input("save-api-keys-btn", "n_clicks"),
        [
            State("api-input-openai", "value"),
            State("api-input-alpaca-key", "value"),
            State("api-input-alpaca-secret", "value"),
            State("api-input-finnhub", "value"),
            State("api-input-fred", "value"),
            State("api-input-coindesk", "value"),
            State("api-alpaca-paper", "value"),
            State("api-keys-store", "data")
        ],
        prevent_initial_call=True
    )
    def save_api_keys(n_clicks, openai, alpaca_key, alpaca_secret, finnhub, fred, coindesk, alpaca_paper, current_data):
        """Save API keys to localStorage and apply to runtime config"""
        if not n_clicks:
            raise PreventUpdate
        
        new_keys = {
            "openai": openai or "",
            "alpaca-key": alpaca_key or "",
            "alpaca-secret": alpaca_secret or "",
            "finnhub": finnhub or "",
            "fred": fred or "",
            "coindesk": coindesk or "",
            "alpaca-paper": alpaca_paper if alpaca_paper is not None else True
        }
        
        # Apply API keys to runtime configuration
        apply_api_keys_to_config(new_keys)
        
        return new_keys
    
    # Callback to clear all API keys
    @app.callback(
        [
            Output("api-input-openai", "value", allow_duplicate=True),
            Output("api-input-alpaca-key", "value", allow_duplicate=True),
            Output("api-input-alpaca-secret", "value", allow_duplicate=True),
            Output("api-input-finnhub", "value", allow_duplicate=True),
            Output("api-input-fred", "value", allow_duplicate=True),
            Output("api-input-coindesk", "value", allow_duplicate=True),
            Output("api-alpaca-paper", "value", allow_duplicate=True),
            Output("api-keys-store", "data", allow_duplicate=True)
        ],
        Input("clear-api-keys-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def clear_api_keys(n_clicks):
        """Clear all API keys from inputs and localStorage"""
        if not n_clicks:
            raise PreventUpdate
        
        defaults = get_default_api_keys()
        return ("", "", "", "", "", "", True, defaults)
    
    # Callback to load API keys from .env file
    @app.callback(
        [
            Output("api-input-openai", "value", allow_duplicate=True),
            Output("api-input-alpaca-key", "value", allow_duplicate=True),
            Output("api-input-alpaca-secret", "value", allow_duplicate=True),
            Output("api-input-finnhub", "value", allow_duplicate=True),
            Output("api-input-fred", "value", allow_duplicate=True),
            Output("api-input-coindesk", "value", allow_duplicate=True),
            Output("api-alpaca-paper", "value", allow_duplicate=True)
        ],
        Input("load-env-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def load_from_env(n_clicks):
        """Load API keys from .env file into the inputs"""
        if not n_clicks:
            raise PreventUpdate
        
        load_dotenv()
        
        alpaca_paper_str = os.getenv("ALPACA_USE_PAPER", "True")
        alpaca_paper = alpaca_paper_str.lower() in ("true", "1", "yes")
        
        return (
            os.getenv("OPENAI_API_KEY", "") or "",
            os.getenv("ALPACA_API_KEY", "") or "",
            os.getenv("ALPACA_SECRET_KEY", "") or "",
            os.getenv("FINNHUB_API_KEY", "") or "",
            os.getenv("FRED_API_KEY", "") or "",
            os.getenv("COINDESK_API_KEY", "") or "",
            alpaca_paper
        )
    
    # Callback to update API key status indicators
    for api_config in api_configs:
        api_id = api_config["id"]
        
        @app.callback(
            Output(f"api-status-{api_id}", "color"),
            Input(f"api-input-{api_id}", "value"),
            prevent_initial_call=True
        )
        def update_status_indicator(value, _api_id=api_id):
            """Update the status indicator color based on whether key is set"""
            if value and len(value.strip()) > 5:
                return "success"
            else:
                return "outline-secondary"


def apply_api_keys_to_config(api_keys):
    """Apply API keys to the runtime configuration"""
    try:
        from tradingagents.dataflows.config import set_runtime_api_keys
        
        # Map storage keys to config keys
        config_keys = {
            "openai_api_key": api_keys.get("openai", ""),
            "alpaca_api_key": api_keys.get("alpaca-key", ""),
            "alpaca_secret_key": api_keys.get("alpaca-secret", ""),
            "finnhub_api_key": api_keys.get("finnhub", ""),
            "fred_api_key": api_keys.get("fred", ""),
            "coindesk_api_key": api_keys.get("coindesk", ""),
            "alpaca_use_paper": api_keys.get("alpaca-paper", True)
        }
        
        set_runtime_api_keys(config_keys)
        return True
    except Exception as e:
        print(f"Warning: Could not apply API keys to config: {e}")
        return False
