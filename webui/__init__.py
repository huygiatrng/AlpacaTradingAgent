"""
Trading Agents Framework - Web UI Package
"""


def __getattr__(name):
    # Importing utilities from an analysis worker does not need the dashboard.
    if name == "run_app":
        from webui.app_dash import run_app

        return run_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
