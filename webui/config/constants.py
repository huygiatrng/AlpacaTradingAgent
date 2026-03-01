"""
Constants and configuration for TradingAgents WebUI
"""

# Define colors for consistency
COLORS = {
    "primary": "#3B82F6",         # Bright blue
    "secondary": "#10B981",       # Green
    "background": "#0F172A",      # Dark blue background
    "card": "#1E293B",            # Slightly lighter card background
    "text": "#F1F5F9",            # Light text
    "pending": "#94A3B8",         # Slate gray
    "in_progress": "#F59E0B",     # Amber
    "completed": "#10B981",       # Green
    "error": "#EF4444",           # Red
    "nav_active": "#F1F5F9",      # White for active nav
    "nav_inactive": "#64748B",    # Slate for inactive nav
    "border": "#334155",          # Border color
    "hover": "#2563EB",           # Hover color
}

# Refresh intervals (in milliseconds)
REFRESH_INTERVALS = {
    "fast": 2000,      # 2 seconds for critical updates during analysis
    "medium": 10000,   # 5 seconds for reports (reduced frequency for less interference)
    "slow": 60000,    # 1 minutes for account data (was 30 seconds)
}

# Debounce duration to prevent auto-refresh from overriding user symbol selection
SYMBOL_CLICK_DEBOUNCE_SECONDS = 0  # No debounce — update immediately

# App configuration
APP_CONFIG = {
    "title": "TradingAgents - Multi-Agent Financial Analysis",
    "external_stylesheets": [
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ],
    "suppress_callback_exceptions": True,
    "update_title": None,
} 