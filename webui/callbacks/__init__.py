"""
Callbacks package for TradingAgents WebUI
Contains organized callback functions grouped by functionality
"""

from .status_callbacks import register_status_callbacks
from .chart_callbacks import register_chart_callbacks
from .report_callbacks import register_report_callbacks
from .control_callbacks import register_control_callbacks
from .trading_callbacks import register_trading_callbacks
from .storage_callbacks import register_storage_callbacks
from .batch_callbacks import register_batch_callbacks
from .debug_callbacks import register_debug_callbacks

def register_all_callbacks(app):
    """Register all callback functions with the Dash app"""
    register_status_callbacks(app)
    register_chart_callbacks(app)
    register_report_callbacks(app)
    register_control_callbacks(app)
    register_trading_callbacks(app)
    register_storage_callbacks(app)
    register_batch_callbacks(app)
    register_debug_callbacks(app) 