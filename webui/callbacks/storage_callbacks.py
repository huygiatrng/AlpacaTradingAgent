"""
Storage callbacks for persisting user settings in localStorage
"""

from dash import Input, Output, State, callback_context as ctx
from webui.utils.storage import get_default_settings


def _parse_symbols(value):
    if isinstance(value, list):
        return [str(symbol).strip().upper() for symbol in value if str(symbol).strip()]
    return [symbol.strip().upper() for symbol in (value or "").split(",") if symbol.strip()]


def register_storage_callbacks(app):
    """Register storage-related callbacks"""

    # Callback to save settings to localStorage when they change
    @app.callback(
        Output("settings-store", "data"),
        [
            Input("ticker-picker", "value"),
            Input("analyst-market", "value"),
            Input("analyst-social", "value"),
            Input("analyst-news", "value"),
            Input("analyst-fundamentals", "value"),
            Input("analyst-macro", "value"),
            Input("research-depth", "value"),
            Input("allow-shorts", "value"),
            Input("loop-interval", "value"),
            Input("market-hours-input", "value"),
            Input("trade-after-analyze", "value"),
            Input("trade-dollar-amount", "value"),
            Input("llm-provider", "value"),
            Input("backend-url", "value"),
            Input("output-language", "value"),
            Input("checkpoint-enabled", "value"),
            Input("quick-llm", "value"),
            Input("deep-llm", "value"),
            Input("quick-llm-custom-model", "value"),
            Input("deep-llm-custom-model", "value"),
            Input("google-thinking-level", "value"),
            Input("anthropic-effort", "value"),
        ],
        [
            State("settings-store", "data"),
            State("loop-enabled", "value"),
            State("market-hour-enabled", "value")
        ],
        prevent_initial_call=True
    )
    def save_settings(ticker_symbols, analyst_market, analyst_social, analyst_news,
                     analyst_fundamentals, analyst_macro, research_depth, allow_shorts,
                     loop_interval, market_hours_input,
                     trade_after_analyze, trade_dollar_amount,
                     llm_provider, backend_url, output_language, checkpoint_enabled,
                     quick_llm, deep_llm, quick_llm_custom_model, deep_llm_custom_model,
                     google_thinking_level, anthropic_effort,
                     current_settings, loop_enabled, market_hour_enabled):
        """Save settings to localStorage store"""
        
        # Don't save if triggered by initial load
        if not ctx.triggered:
            return current_settings or get_default_settings()
        
        new_settings = {
            "ticker_input": ", ".join(_parse_symbols(ticker_symbols)),
            "analyst_market": analyst_market,
            "analyst_social": analyst_social,
            "analyst_news": analyst_news,
            "analyst_fundamentals": analyst_fundamentals,
            "analyst_macro": analyst_macro,
            "research_depth": research_depth,
            "allow_shorts": allow_shorts,
            "loop_enabled": loop_enabled,
            "loop_interval": loop_interval,
            "market_hour_enabled": market_hour_enabled,
            "market_hours_input": market_hours_input,
            "trade_after_analyze": trade_after_analyze,
            "trade_dollar_amount": trade_dollar_amount,
            "llm_provider": llm_provider,
            "backend_url": backend_url,
            "output_language": output_language,
            "checkpoint_enabled": checkpoint_enabled,
            "quick_llm": quick_llm,
            "deep_llm": deep_llm,
            "quick_llm_custom_model": quick_llm_custom_model or "",
            "deep_llm_custom_model": deep_llm_custom_model or "",
            "google_thinking_level": google_thinking_level or "",
            "anthropic_effort": anthropic_effort or "",
        }
        
        # Check if settings actually changed to prevent circular updates
        if current_settings:
            settings_changed = False
            for key, value in new_settings.items():
                if current_settings.get(key) != value:
                    settings_changed = True
                    break
            
            # If no changes, don't update the store to prevent circular callback
            if not settings_changed:
                return current_settings
        
        return new_settings
