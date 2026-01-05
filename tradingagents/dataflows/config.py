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


def get_api_key(key_name: str, env_var_name: str) -> str:
    """Get API key from environment variables or config."""
    # First check environment variables
    api_key = os.getenv(env_var_name)
    
    # If not found, check config
    if api_key is None and _config is not None and key_name in _config:
        api_key = _config[key_name]
    
    return api_key


def get_openai_client_config() -> dict:
    """Get OpenAI client configuration including base URL for local LLMs."""
    config = get_config()
    client_config = {}
    
    # Get API key
    api_key = get_api_key("openai_api_key", "OPENAI_API_KEY")
    if api_key:
        client_config["api_key"] = api_key
    
    # Check for local LLM configuration
    use_local = config.get("openai_use_local", False) or os.getenv("OPENAI_USE_LOCAL", "false").lower() == "true"
    base_url = config.get("openai_base_url") or os.getenv("OPENAI_BASE_URL")
    
    if use_local and base_url:
        client_config["base_url"] = base_url
        # For local LLMs, use a default API key if none provided
        if not client_config.get("api_key"):
            client_config["api_key"] = "local-llm"
    
    return client_config


def get_openai_api_key() -> str:
    """Get OpenAI API key from environment variables or config."""
    return get_api_key("openai_api_key", "OPENAI_API_KEY")


def get_finnhub_api_key() -> str:
    """Get Finnhub API key from environment variables or config."""
    return get_api_key("finnhub_api_key", "FINNHUB_API_KEY")


def get_alpaca_api_key() -> str:
    """Get Alpaca API key from environment variables or config."""
    return get_api_key("alpaca_api_key", "ALPACA_API_KEY")


def get_alpaca_secret_key() -> str:
    """Get Alpaca secret key from environment variables or config."""
    return get_api_key("alpaca_secret_key", "ALPACA_SECRET_KEY")


def get_alpaca_use_paper() -> str:
    """Get Alpaca paper trading flag from environment variables or config."""
    return get_api_key("alpaca_use_paper", "ALPACA_USE_PAPER")


def get_fred_api_key() -> str:
    """Get FRED API key from environment variables or config."""
    return get_api_key("fred_api_key", "FRED_API_KEY")


# Initialize with default config
initialize_config()
