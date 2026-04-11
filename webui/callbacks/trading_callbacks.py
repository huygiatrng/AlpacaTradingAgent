"""
Trading and Alpaca-related callbacks for TradingAgents WebUI
"""

from dash import Input, Output, State, html
import dash_bootstrap_components as dbc
import dash.dependencies
import json

from webui.components.alpaca_account import render_positions_table, render_orders_table


def register_trading_callbacks(app):
    """Register all trading and Alpaca-related callbacks"""

    @app.callback(
        Output("alpaca-account-title", "children"),
        Input("api-keys-store", "data")
    )
    def update_account_title(stored_keys):
        """Update account section title to reflect current paper/live trading mode"""
        if isinstance(stored_keys, dict) and "alpaca-paper" in stored_keys:
            use_paper_val = stored_keys["alpaca-paper"]
        else:
            from tradingagents.dataflows.config import get_alpaca_use_paper
            use_paper_val = get_alpaca_use_paper()
        is_paper = str(use_paper_val).strip().lower() not in ("false", "0", "no")
        return f"Alpaca {'Paper' if is_paper else 'Live'} Trading Account"

    @app.callback(
        [Output("positions-table-container", "children"),
         Output("orders-table-container", "children")],
        [Input("slow-refresh-interval", "n_intervals"),
         Input("refresh-btn", "n_clicks"),
         Input("refresh-alpaca-btn", "n_clicks"),
         Input("orders-pagination", "active_page")]
    )
    def update_enhanced_alpaca_tables(n_intervals, n_clicks, alpaca_refresh, orders_page):
        """Update the enhanced positions and orders tables"""
        
        page = orders_page if orders_page is not None else 1
        
        positions_table = render_positions_table()
        orders_table = render_orders_table(page=page)
        
        return positions_table, orders_table

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
