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
    
    def _env_values():
        load_dotenv()

        def get_env_value(key):
            val = os.getenv(key, "")
            if val and not val.startswith("your_"):
                return val
            return ""

        return {
            api["id"]: get_env_value(api["env_var"])
            for api in api_configs
        }

    # Callback to load API keys from localStorage on page load
    @app.callback(
        [
            *[Output(f"api-input-{api_id}", "value") for api_id in api_ids],
            Output("api-alpaca-paper", "value"),
            Output("env-file-status", "children"),
        ],
        Input("api-keys-store", "data")
    )
    def load_api_keys(stored_keys):
        """Load API keys from localStorage, falling back to .env on first load"""
        import dash_bootstrap_components as dbc
        from dash import html

        env_vars = _env_values()

        alpaca_paper_str = os.getenv("ALPACA_USE_PAPER", "True")
        env_alpaca_paper = alpaca_paper_str.lower() in ("true", "1", "yes")

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

        has_stored_keys = stored_keys and any(stored_keys.get(key) for key in api_ids)
        if not has_stored_keys:
            keys_to_apply = {**env_vars, "alpaca-paper": env_alpaca_paper}
            apply_api_keys_to_config(keys_to_apply)
            return tuple(env_vars.get(api_id, "") for api_id in api_ids) + (
                env_alpaca_paper,
                env_status,
            )

        apply_api_keys_to_config(stored_keys)
        return tuple(stored_keys.get(api_id, "") for api_id in api_ids) + (
            stored_keys.get("alpaca-paper", True),
            env_status,
        )

    # Callback to save API keys to localStorage
    @app.callback(
        Output("api-keys-store", "data"),
        Input("save-api-keys-btn", "n_clicks"),
        [
            *[State(f"api-input-{api_id}", "value") for api_id in api_ids],
            State("api-alpaca-paper", "value"),
            State("api-keys-store", "data"),
        ],
        prevent_initial_call=True
    )
    def save_api_keys(n_clicks, *values):
        """Save API keys to localStorage and apply to runtime config"""
        if not n_clicks:
            raise PreventUpdate

        key_values = values[:len(api_ids)]
        alpaca_paper = values[len(api_ids)]
        new_keys = {
            api_id: (value or "")
            for api_id, value in zip(api_ids, key_values)
        }
        new_keys["alpaca-paper"] = alpaca_paper if alpaca_paper is not None else True

        apply_api_keys_to_config(new_keys)
        return new_keys

    # Callback to clear all API keys
    @app.callback(
        [
            *[Output(f"api-input-{api_id}", "value", allow_duplicate=True) for api_id in api_ids],
            Output("api-alpaca-paper", "value", allow_duplicate=True),
            Output("api-keys-store", "data", allow_duplicate=True),
        ],
        Input("clear-api-keys-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def clear_api_keys(n_clicks):
        """Clear all API keys from inputs and localStorage"""
        if not n_clicks:
            raise PreventUpdate

        defaults = get_default_api_keys()
        return tuple("" for _ in api_ids) + (True, defaults)

    # Callback to load API keys from .env file
    @app.callback(
        [
            *[Output(f"api-input-{api_id}", "value", allow_duplicate=True) for api_id in api_ids],
            Output("api-alpaca-paper", "value", allow_duplicate=True),
        ],
        Input("load-env-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def load_from_env(n_clicks):
        """Load API keys from .env file into the inputs"""
        if not n_clicks:
            raise PreventUpdate

        env_vars = _env_values()
        alpaca_paper_str = os.getenv("ALPACA_USE_PAPER", "True")
        alpaca_paper = alpaca_paper_str.lower() in ("true", "1", "yes")

        return tuple(env_vars.get(api_id, "") for api_id in api_ids) + (alpaca_paper,)
    
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
            "google_api_key": api_keys.get("google", ""),
            "anthropic_api_key": api_keys.get("anthropic", ""),
            "xai_api_key": api_keys.get("xai", ""),
            "deepseek_api_key": api_keys.get("deepseek", ""),
            "dashscope_api_key": api_keys.get("dashscope", ""),
            "zhipu_api_key": api_keys.get("zhipu", ""),
            "openrouter_api_key": api_keys.get("openrouter", ""),
            "azure_openai_api_key": api_keys.get("azure-openai", ""),
            "alpaca_api_key": api_keys.get("alpaca-key", ""),
            "alpaca_secret_key": api_keys.get("alpaca-secret", ""),
            "finnhub_api_key": api_keys.get("finnhub", ""),
            "fred_api_key": api_keys.get("fred", ""),
            "coindesk_api_key": api_keys.get("coindesk", ""),
            "alpha_vantage_api_key": api_keys.get("alpha-vantage", ""),
            "alpaca_use_paper": api_keys.get("alpaca-paper", True)
        }
        
        set_runtime_api_keys(config_keys)
        return True
    except Exception as e:
        print(f"Warning: Could not apply API keys to config: {e}")
        return False
