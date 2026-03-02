"""
Trading and Alpaca-related callbacks for TradingAgents WebUI
"""

from dash import Input, Output, State, html
import dash_bootstrap_components as dbc
import dash.dependencies
import json

from webui.components.alpaca_account import render_positions_table, render_orders_table, render_account_summary
from tradingagents.dataflows.alpaca_utils import AlpacaUtils


def register_trading_callbacks(app):
    """Register all trading and Alpaca-related callbacks"""

    @app.callback(
        [Output("positions-table-container", "children"),
         Output("orders-table-container", "children"),
         Output("account-summary-container", "children")],
        [Input("slow-refresh-interval", "n_intervals"),
         Input("refresh-btn", "n_clicks"),
         Input("refresh-alpaca-btn", "n_clicks")]
    )
    def update_enhanced_alpaca_tables(n_intervals, n_clicks, alpaca_refresh):
        """Update the enhanced positions and orders tables"""
        try:
            orders_data = AlpacaUtils.get_recent_orders(limit=100)
        except Exception:
            orders_data = None

        positions_table = render_positions_table()
        orders_table = render_orders_table(orders_data=orders_data)

        return positions_table, orders_table, render_account_summary()

    @app.callback(
        [Output('liquidate-confirm', 'displayed'),
         Output('liquidate-confirm', 'message')],
        [Input({'type': 'liquidate-btn', 'index': dash.dependencies.ALL}, 'n_clicks')],
        prevent_initial_call=True
    )
    def show_liquidate_confirmation(n_clicks_list):
        """Show confirmation dialog for liquidation"""
        if not any(n_clicks_list) or not any(n_clicks_list):
            return False, ""
        
        # Get the button that was clicked
        from dash import ctx
        if not ctx.triggered:
            return False, ""
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        # Safely parse the JSON string
        try:
            button_data = json.loads(button_id)
            symbol = button_data['index']
        except (json.JSONDecodeError, KeyError):
            return False, ""
        
        message = f"Are you sure you want to liquidate your entire position in {symbol}? This action cannot be undone."
        return True, message

    @app.callback(
        Output('liquidation-status', 'children'),
        [Input('liquidate-confirm', 'submit_n_clicks')],
        [State('liquidate-confirm', 'message')],
        prevent_initial_call=True
    )
    def handle_liquidation(submit_n_clicks, message):
        """Handle the actual liquidation when confirmed"""
        if not submit_n_clicks:
            return ""
        
        try:
            # Extract symbol from confirmation message
            symbol = message.split(" in ")[1].split("?")[0]
            
            # Import AlpacaUtils for liquidation
            from tradingagents.dataflows.alpaca_utils import AlpacaUtils
            
            # Cancel any open bracket orders (stop-loss / take-profit legs) before closing.
            # Alpaca returns 403 "held_for_orders" if open orders still hold the shares.
            cancel_result = AlpacaUtils.cancel_open_orders_for_symbol(symbol)
            if not cancel_result.get("success") or cancel_result.get("failed", 0) > 0:
                import logging
                logging.getLogger(__name__).warning(
                    "Some orders could not be cancelled for %s before liquidation: %s",
                    symbol, cancel_result.get("errors", [])
                )

            # Execute liquidation
            result = AlpacaUtils.close_position(symbol)

            if result.get("success"):
                return dbc.Alert([
                    html.I(className="fas fa-check-circle me-2"),
                    f"Successfully liquidated position in {symbol}. Order ID: {result.get('order_id', 'N/A')}"
                ], color="success", duration=5000, className="mt-3")
            else:
                return dbc.Alert([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    f"Failed to liquidate position in {symbol}: {result.get('error', 'Unknown error')}"
                ], color="danger", duration=8000, className="mt-3")
                
        except Exception as e:
            return dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                f"Error during liquidation: {str(e)}"
            ], color="danger", duration=8000, className="mt-3")
