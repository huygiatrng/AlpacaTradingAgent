# -------------------------------- config.py -----------------------
import tradingagents.default_config as default_config
from typing import Dict, Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Use default config but allow it to be overridden
_config: Optional[Dict] = None
DATA_DIR: Optional[str] = None

# Runtime API keys (set from WebUI, takes precedence over .env)
_runtime_api_keys: Dict[str, str] = {}


def initialize_config():
    """Initialize the configuration with default values."""
    global _config, DATA_DIR
    if _config is None:
        _config = default_config.DEFAULT_CONFIG.copy()
        DATA_DIR = _config["data_dir"]


def set_config(config: Dict):
    """Update the configuration with custom values."""
    global _config, DATA_DIR
    if _config is None:
        _config = default_config.DEFAULT_CONFIG.copy()
    _config.update(config)
    DATA_DIR = _config["data_dir"]


def get_config() -> Dict:
    """Get the current configuration."""
    if _config is None:
        initialize_config()
    return _config.copy()


def set_runtime_api_keys(api_keys: Dict[str, str]):
    """
    Set API keys at runtime from the WebUI.
    These take precedence over .env file values.
    """
    global _runtime_api_keys
    _runtime_api_keys.update(api_keys)


def get_runtime_api_keys() -> Dict[str, str]:
    """Get the runtime API keys set from WebUI."""
    return _runtime_api_keys.copy()


def clear_runtime_api_keys():
    """Clear all runtime API keys."""
    global _runtime_api_keys
    _runtime_api_keys = {}


def get_api_key(key_name: str, env_var_name: str) -> str:
    """
    Get API key with priority:
    1. Runtime API keys (set from WebUI)
    2. Environment variables (.env file)
    3. Config defaults
    """
    # First check runtime API keys (from WebUI localStorage)
    if key_name in _runtime_api_keys and _runtime_api_keys[key_name] is not None and _runtime_api_keys[key_name] != "":
        return _runtime_api_keys[key_name]
    
    # Then check environment variables
    api_key = os.getenv(env_var_name)
    
    # If not found, check config
    if api_key is None and _config is not None and key_name in _config:
        api_key = _config[key_name]
    
    return api_key


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def is_local_openai_enabled() -> bool:
    """Return True when LLM calls should use an OpenAI-compatible local endpoint."""
    env_value = os.getenv("OPENAI_USE_LOCAL")
    if env_value is not None:
        return _coerce_bool(env_value)
    config = get_config()
    return _coerce_bool(config.get("openai_use_local", False))


def get_openai_base_url() -> Optional[str]:
    """Get the configured OpenAI-compatible base URL, if any."""
    config = get_config()
    base_url = os.getenv("OPENAI_BASE_URL") or config.get("openai_base_url")
    return str(base_url).strip() if base_url else None


def get_openai_embedding_model() -> str:
    """Return the embedding model name used by reflection memory."""
    config = get_config()
    return (
        os.getenv("OPENAI_EMBEDDING_MODEL")
        or config.get("openai_embedding_model")
        or "text-embedding-ada-002"
    )


def get_openai_client_config() -> Dict[str, str]:
    """
    Build OpenAI SDK client kwargs.

    Local mode only applies to generic chat/embedding clients. OpenAI web-search
    tools should keep their existing cloud-only behavior unless explicitly
    reworked, because most local OpenAI-compatible servers do not support
    `responses.create(..., tools=[{"type": "web_search"}])`.
    """
    api_key = get_openai_api_key()
    use_local = is_local_openai_enabled()
    base_url = get_openai_base_url()

    client_config: Dict[str, str] = {}
    if use_local and base_url:
        client_config["base_url"] = base_url
        client_config["api_key"] = api_key or "local-llm"
        return client_config

    if api_key:
        client_config["api_key"] = api_key
    return client_config


def get_openai_api_key() -> str:
    """Get OpenAI API key from runtime, environment variables, or config."""
    return get_api_key("openai_api_key", "OPENAI_API_KEY")


def get_finnhub_api_key() -> str:
    """Get Finnhub API key from runtime, environment variables, or config."""
    return get_api_key("finnhub_api_key", "FINNHUB_API_KEY")


def get_alpaca_api_key() -> str:
    """Get Alpaca API key from runtime, environment variables, or config."""
    return get_api_key("alpaca_api_key", "ALPACA_API_KEY")


def get_alpaca_secret_key() -> str:
    """Get Alpaca secret key from runtime, environment variables, or config."""
    return get_api_key("alpaca_secret_key", "ALPACA_SECRET_KEY")


def get_alpaca_use_paper() -> str:
    """Get Alpaca paper trading flag from runtime, environment variables, or config."""
    value = get_api_key("alpaca_use_paper", "ALPACA_USE_PAPER")
    # Handle boolean values from WebUI
    if isinstance(value, bool):
        return str(value)
    return value


def get_fred_api_key() -> str:
    """Get FRED API key from runtime, environment variables, or config."""
    return get_api_key("fred_api_key", "FRED_API_KEY")


def get_coindesk_api_key() -> str:
    """Get CoinDesk/CryptoCompare API key from runtime, environment variables, or config."""
    return get_api_key("coindesk_api_key", "COINDESK_API_KEY")


# Initialize with default config
initialize_config()
