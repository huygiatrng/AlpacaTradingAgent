"""
Storage utility for persisting user settings in localStorage
"""

from typing import Dict, Any

# Default settings structure
DEFAULT_SETTINGS = {
    "ticker_input": "NVDA, AMD, TSLA",
    "analyst_market": True,
    "analyst_social": True,
    "analyst_news": True,
    "analyst_fundamentals": True,
    "analyst_macro": True,
    "research_depth": "Shallow",
    "allow_shorts": False,
    "loop_enabled": False,
    "loop_interval": 60,
    "market_hour_enabled": False,
    "market_hours_input": "",
    "trade_after_analyze": False,
    "trade_dollar_amount": 4500,
    "llm_provider": "openai",
    "backend_url": "",
    "output_language": "English",
    "checkpoint_enabled": False,
    "quick_llm": "gpt-5.4-nano",
    "deep_llm": "gpt-5.4-mini",
    "quick_llm_custom_model": "",
    "deep_llm_custom_model": "",
    "google_thinking_level": "",
    "anthropic_effort": "",
}

# Default API keys structure (empty by default, loaded from localStorage or .env)
DEFAULT_API_KEYS = {
    "openai": "",
    "google": "",
    "anthropic": "",
    "xai": "",
    "deepseek": "",
    "dashscope": "",
    "zhipu": "",
    "openrouter": "",
    "azure-openai": "",
    "alpha-vantage": "",
    "alpaca-key": "",
    "alpaca-secret": "",
    "finnhub": "",
    "fred": "",
    "coindesk": "",
    "alpaca-paper": True
}


def get_default_settings() -> Dict[str, Any]:
    """Get the default settings structure"""
    return DEFAULT_SETTINGS.copy()


def get_default_api_keys() -> Dict[str, Any]:
    """Get the default API keys structure"""
    return DEFAULT_API_KEYS.copy()


def create_storage_store_component():
    """Create a dcc.Store component for localStorage persistence"""
    from dash import dcc
    return dcc.Store(id='settings-store', storage_type='local', data=DEFAULT_SETTINGS)


def create_api_keys_store_component():
    """Create a dcc.Store component for API keys localStorage persistence"""
    from dash import dcc
    return dcc.Store(id='api-keys-store', storage_type='local', data=DEFAULT_API_KEYS)
