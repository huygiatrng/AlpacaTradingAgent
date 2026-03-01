"""
app_dash.py - Simplified Dash-based web UI for TradingAgents

This is the refactored version of app_dash.py that uses organized modules
for better code structure and maintainability.

RECENT FIX: Multiple Symbol Page Refresh Issue
- Fixed issue where only the first symbol would show after page refresh when analyzing multiple symbols
- The app now stores symbols list in browser storage and restores all symbol pages correctly
- Added safeguards to prevent index out of range errors during pagination
- Users can now refresh the page while analyzing multiple symbols without losing access to all symbol pages
"""

import dash
import dash_bootstrap_components as dbc
from flask import Flask
import logging

from webui.config.constants import APP_CONFIG, COLORS
from webui.layout import create_main_layout
from webui.callbacks import register_all_callbacks
from webui.watchdog import start_watchdog


def apply_sequential_mode_fix():
    """Apply fix for sequential execution mode report mapping bug"""
    try:
        from webui.utils.state import AppState
        
        # Check if fix is already applied
        if hasattr(AppState, '_mapping_fix_applied'):
            return True
            
        # Patch the process_chunk_updates method to fix report mapping
        original_process_chunk_updates = AppState.process_chunk_updates
        
        def fixed_process_chunk_updates(self, chunk, ticker=None):
            """Fixed version that correctly maps social analyst reports"""
            
            # 🔍 DEBUG: Log what we're receiving
            current_symbol = getattr(self, 'current_symbol', '')
            if current_symbol:
                state = self.get_state(current_symbol)
                if state:
                    social_status = state["agent_statuses"].get("Social Analyst")
                    if social_status == "in_progress":
                        chunk_fields = list(chunk.keys())
                        # print(f"[DEBUG] Social Analyst chunk received: {chunk_fields}")
                        
                        # Check for the ACTUAL bug: Social Analyst writing to market_report
                        if "market_report" in chunk and "sentiment_report" not in chunk:
                            # print(f"[FIX] 🛑 Detected Social Analyst incorrectly updating market_report - fixing...")
                            # Move the content to the correct field
                            chunk["sentiment_report"] = chunk["market_report"]
                            del chunk["market_report"]
                            # print(f"[FIX] ✅ Corrected: market_report -> sentiment_report")
                        elif "sentiment_report" in chunk:
                            # This is correct - Social Analyst updating sentiment_report
                            sentiment_length = len(chunk["sentiment_report"])
                            # print(f"[DEBUG] ✅ Social Analyst correctly updating sentiment_report ({sentiment_length} chars)")
            
            # Call the original method with the fixed chunk
            return original_process_chunk_updates(self, chunk, ticker=ticker)
        
        # Apply the patch
        AppState.process_chunk_updates = fixed_process_chunk_updates
        AppState._mapping_fix_applied = True
        return True
        
    except Exception as e:
        print(f"⚠️ Could not apply sequential mode fix: {e}")
        return False


def create_app():
    """Create and configure the Dash application"""

    # Apply the sequential mode fix first
    apply_sequential_mode_fix()

    # Install stdout log interceptor so [TAG] lines are captured for the debug Logs tab
    from webui.utils.log_interceptor import install as install_log_interceptor
    from webui.utils.state import app_state as _app_state
    install_log_interceptor(_app_state)
    
    # Initialize Flask server
    server = Flask(__name__)

    # Initialize Dash app with Bootstrap
    app = dash.Dash(
        __name__,
        server=server,
        external_stylesheets=[
            dbc.themes.DARKLY,
            *APP_CONFIG["external_stylesheets"]
        ],
        suppress_callback_exceptions=APP_CONFIG["suppress_callback_exceptions"],
        update_title=APP_CONFIG["update_title"],
    )

    # Set app title
    app.title = APP_CONFIG["title"]

    # Set the layout
    app.layout = create_main_layout()

    # Register all callbacks
    register_all_callbacks(app)

    return app


def run_app(port=7860, share=False, server_name="127.0.0.1", debug=False, max_threads=1):
    """Run the TradingAgents Dash Web UI"""
    
    # Create the app
    app = create_app()
    
    if debug:
        print(f"Starting TradingAgents Dash Web UI on port {port}...")
    else:
        print("Starting TradingAgents Web UI...")
    
    # Suppress verbose HTTP request logs from Werkzeug
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    # Optionally also silence Dash's callback exceptions logger
    logging.getLogger("dash.callback").setLevel(logging.ERROR)

    # Start background watchdog to detect stuck analyses
    start_watchdog()

    # Run the app
    app.run(
        port=port,
        host=server_name,
        debug=debug,
        dev_tools_hot_reload=debug
    )
    
    return 0


# Create the app instance for use by other modules
app = create_app()

if __name__ == "__main__":
    run_app(debug=True) 