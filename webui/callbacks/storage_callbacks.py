"""
Storage callbacks for persisting user settings in localStorage
"""

from dash import Input, Output, State, callback_context as ctx
from webui.utils.storage import get_default_settings

def register_storage_callbacks(app):
    """Register storage-related callbacks"""
    
    # Callback to load settings from localStorage on page load
    @app.callback(
        [
            Output("ticker-input", "value"),
            Output("analyst-market", "value"),
            Output("analyst-social", "value"),
            Output("analyst-news", "value"),
            Output("analyst-fundamentals", "value"),
            Output("analyst-macro", "value"),
            Output("parallel-execution", "value"),
            Output("research-depth", "value"),
            Output("allow-shorts", "value"),
            Output("loop-interval", "value"),
            Output("market-hours-input", "value"),
            Output("trade-after-analyze", "value"),
            Output("trade-dollar-amount", "value"),
            Output("ai-position-sizing", "value"),
            Output("quick-llm", "value"),
            Output("deep-llm", "value")
        ],
        Input("settings-store", "data")
    )
    def load_settings(stored_settings):
        """Load settings from localStorage store"""
        if not stored_settings:
            # Return default settings if nothing stored
            defaults = get_default_settings()
            return [
                defaults["ticker_input"],
                defaults["analyst_market"],
                defaults["analyst_social"],
                defaults["analyst_news"],
                defaults["analyst_fundamentals"],
                defaults["analyst_macro"],
                defaults["parallel_analysts"],
                defaults["research_depth"],
                defaults["allow_shorts"],
                defaults["loop_interval"],
                defaults["market_hours_input"],
                defaults["trade_after_analyze"],
                defaults["trade_dollar_amount"],
                defaults.get("ai_position_sizing", True),
                defaults["quick_llm"],
                defaults["deep_llm"]
            ]

        return [
            stored_settings.get("ticker_input", "NVDA, AMD, TSLA"),
            stored_settings.get("analyst_market", True),
            stored_settings.get("analyst_social", True),
            stored_settings.get("analyst_news", True),
            stored_settings.get("analyst_fundamentals", True),
            stored_settings.get("analyst_macro", True),
            stored_settings.get("parallel_analysts", False),
            stored_settings.get("research_depth", "Shallow"),
            stored_settings.get("allow_shorts", False),
            stored_settings.get("loop_interval", 60),
            stored_settings.get("market_hours_input", ""),
            stored_settings.get("trade_after_analyze", False),
            stored_settings.get("trade_dollar_amount", 4500),
            stored_settings.get("ai_position_sizing", True),
            stored_settings.get("quick_llm", "gpt-5-nano"),
            stored_settings.get("deep_llm", "gpt-5-nano")
        ]
    
    # Callback to save settings to localStorage when they change
    @app.callback(
        Output("settings-store", "data"),
        [
            Input("ticker-input", "value"),
            Input("analyst-market", "value"),
            Input("analyst-social", "value"),
            Input("analyst-news", "value"),
            Input("analyst-fundamentals", "value"),
            Input("analyst-macro", "value"),
            Input("parallel-execution", "value"),
            Input("research-depth", "value"),
            Input("allow-shorts", "value"),
            Input("loop-interval", "value"),
            Input("market-hours-input", "value"),
            Input("trade-after-analyze", "value"),
            Input("trade-dollar-amount", "value"),
            Input("ai-position-sizing", "value"),
            Input("quick-llm", "value"),
            Input("deep-llm", "value")
        ],
        [
            State("settings-store", "data"),
            State("loop-enabled", "value"),
            State("market-hour-enabled", "value")
        ],
        prevent_initial_call=True
    )
    def save_settings(ticker_input, analyst_market, analyst_social, analyst_news,
                     analyst_fundamentals, analyst_macro, parallel_analysts, research_depth, allow_shorts,
                     loop_interval, market_hours_input,
                     trade_after_analyze, trade_dollar_amount, ai_position_sizing, quick_llm, deep_llm,
                     current_settings, loop_enabled, market_hour_enabled):
        """Save settings to localStorage store"""
        
        # Don't save if triggered by initial load
        if not ctx.triggered:
            return current_settings or get_default_settings()
        
        new_settings = {
            "ticker_input": ticker_input,
            "analyst_market": analyst_market,
            "analyst_social": analyst_social,
            "analyst_news": analyst_news,
            "analyst_fundamentals": analyst_fundamentals,
            "analyst_macro": analyst_macro,
            "parallel_analysts": parallel_analysts,
            "research_depth": research_depth,
            "allow_shorts": allow_shorts,
            "loop_enabled": loop_enabled,
            "loop_interval": loop_interval,
            "market_hour_enabled": market_hour_enabled,
            "market_hours_input": market_hours_input,
            "trade_after_analyze": trade_after_analyze,
            "trade_dollar_amount": trade_dollar_amount,
            "ai_position_sizing": ai_position_sizing,
            "quick_llm": quick_llm,
            "deep_llm": deep_llm
        }
        
        return new_settings
