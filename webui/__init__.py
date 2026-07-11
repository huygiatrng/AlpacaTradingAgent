"""
Trading Agents Framework - Web UI Package
"""

# Import lazily (PEP 562): eagerly importing webui.app_dash builds the whole
# Dash layout, which hits the Alpaca API.  Core tradingagents modules import
# webui.utils.* helpers and must not pay that cost (or need network access).


def __getattr__(name):
    if name == "run_app":
        from webui.app_dash import run_app

        return run_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
