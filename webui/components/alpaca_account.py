"""
webui/components/alpaca_account.py - Alpaca account information components
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
import pandas as pd
from datetime import datetime
import pytz
from tradingagents.dataflows.alpaca_utils import AlpacaUtils
from tradingagents.dataflows.alpaca_exceptions import AlpacaAuthError

def render_positions_table():
    """Render the enhanced positions table with liquidate buttons"""
    try:
        positions_data = AlpacaUtils.get_positions_data()

        if not positions_data:
            return html.Div([
                html.Div([
                    html.I(className="fas fa-chart-line fa-2x mb-3"),
                    html.H5("No Open Positions", className="text-muted"),
                    html.P("Your portfolio is currently empty", className="text-muted small")
                ], className="text-center p-5")
            ], className="enhanced-table-container")

        # Helper to decide colour based on the numeric value (sign) rather than the raw string.
        def _get_pl_color(pl_str: str) -> str:
            """Return the appropriate Bootstrap text class for a P/L value string."""
            try:
                # Remove $ signs and commas then convert to float
                value = float(pl_str.replace("$", "").replace(",", ""))
            except ValueError:
                # Fallback to neutral colour if parsing fails
                return "text-muted"

            if value > 0:
                return "text-success"
            elif value < 0:
                return "text-danger"
            else:
                return "text-muted"

        def make_position_row(position):
            today_pl_color = _get_pl_color(position["Today's P/L ($)"])
            total_pl_color = _get_pl_color(position["Total P/L ($)"])
            return html.Tr([
                html.Td([
                    html.Div([
                        html.Strong(position["Symbol"], className="symbol-text"),
                        html.Br(),
                        html.Small(f"{abs(position['Qty'])} shares", className="text-muted")
                    ])
                ], className="symbol-cell"),
                html.Td([
                    html.Div([
                        html.Div(position["Market Value"], className="fw-bold"),
                        html.Small(f"Entry: {position['Avg Entry']}", className="text-muted")
                    ])
                ], className="value-cell"),
                html.Td([
                    html.Div([
                        html.Div(position["Today's P/L ($)"], className=f"fw-bold {today_pl_color}"),
                        html.Small(position["Today's P/L (%)"], className=f"{today_pl_color}")
                    ])
                ], className="pnl-cell"),
                html.Td([
                    html.Div([
                        html.Div(position["Total P/L ($)"], className=f"fw-bold {total_pl_color}"),
                        html.Small(position["Total P/L (%)"], className=f"{total_pl_color}")
                    ])
                ], className="pnl-cell"),
                html.Td([
                    dbc.Button([
                        html.I(className="fas fa-times-circle me-1"),
                        "Liquidate"
                    ],
                    id={"type": "liquidate-btn", "index": position["Symbol"]},
                    color="danger",
                    size="sm",
                    outline=True,
                    className="liquidate-btn"
                    )
                ], className="action-cell")
            ], className="table-row-hover", id=f"position-row-{position['Symbol']}")

        # Split into long and short positions
        long_positions = [p for p in positions_data if p["Qty"] >= 0]
        short_positions = [p for p in positions_data if p["Qty"] < 0]
        has_both = len(long_positions) > 0 and len(short_positions) > 0

        table_rows = []

        if has_both:
            table_rows.append(html.Tr([
                html.Td([
                    html.Span([
                        html.I(className="fas fa-arrow-up me-2"),
                        "LONG POSITIONS",
                        html.Span(f" ({len(long_positions)})", className="opacity-75")
                    ], className="fw-semibold")
                ], colSpan=5)
            ], className="position-section-header position-section-long"))

        for position in long_positions:
            table_rows.append(make_position_row(position))

        if has_both:
            table_rows.append(html.Tr([
                html.Td([
                    html.Span([
                        html.I(className="fas fa-arrow-down me-2"),
                        "SHORT POSITIONS",
                        html.Span(f" ({len(short_positions)})", className="opacity-75")
                    ], className="fw-semibold")
                ], colSpan=5)
            ], className="position-section-header position-section-short"))

        for position in short_positions:
            table_rows.append(make_position_row(position))

        # Create enhanced table
        table = html.Div([
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Position", className="table-header"),
                        html.Th("Market Value", className="table-header"),
                        html.Th("Today's P/L", className="table-header"),
                        html.Th("Total P/L", className="table-header"),
                        html.Th("Actions", className="table-header text-center")
                    ])
                ]),
                html.Tbody(table_rows)
            ], className="enhanced-table")
        ], className="enhanced-table-container")

        return table

    except AlpacaAuthError as e:
        # Show specific auth error message
        return html.Div([
            html.Div([
                html.I(className="fas fa-key fa-2x mb-3 text-danger"),
                html.H5("Authentication Error", className="text-danger"),
                html.P("Cannot fetch positions - API key authentication failed", className="text-muted mb-2"),
                html.Small("Please regenerate your Alpaca API keys", className="text-muted")
            ], className="text-center p-4")
        ], className="enhanced-table-container error-state")
    except Exception as e:
        print(f"Error rendering positions table: {e}")
        return html.Div([
            html.Div([
                html.I(className="fas fa-exclamation-triangle fa-2x mb-3 text-warning"),
                html.H5("Unable to Load Positions", className="text-warning"),
                html.P("Check your Alpaca API keys", className="text-muted"),
                html.Small(f"Error: {str(e)}", className="text-muted")
            ], className="text-center p-4")
        ], className="enhanced-table-container error-state")

def render_orders_table(page=1, page_size=7):
    """Render the enhanced recent orders table"""
    try:
        orders_data = AlpacaUtils.get_recent_orders(page=page, page_size=page_size)

        if not orders_data:
            return html.Div([
                html.Div([
                    html.I(className="fas fa-history fa-2x mb-3"),
                    html.H5("No Recent Orders", className="text-muted"),
                    html.P("No trading activity found", className="text-muted small")
                ], className="text-center p-5")
            ], className="enhanced-table-container")

        # Group orders by Asset (preserve insertion order)
        groups = {}
        for order in orders_data:
            asset = order.get("Asset", "")
            if asset not in groups:
                groups[asset] = []
            groups[asset].append(order)

        # Create enhanced table rows
        table_rows = []
        row_counter = 0
        for asset, asset_orders in groups.items():
            is_group = len(asset_orders) > 1

            if is_group:
                # Determine position direction from dominant side
                sides = [str(o.get("Side", "")).lower() for o in asset_orders]
                buy_count = sum(1 for s in sides if "buy" in s)
                sell_count = sum(1 for s in sides if "sell" in s)
                if sell_count > buy_count:
                    group_label = "SHORT"
                    label_class = "text-danger"
                    icon_class = "fas fa-arrow-down me-1"
                elif buy_count > sell_count:
                    group_label = "LONG"
                    label_class = "text-success"
                    icon_class = "fas fa-arrow-up me-1"
                else:
                    group_label = "MIXED"
                    label_class = "text-warning"
                    icon_class = "fas fa-arrows-alt-v me-1"

                header_row = html.Tr([
                    html.Td([
                        html.Div([
                            html.Span(asset, className="fw-bold symbol-text me-2"),
                            html.Span([
                                html.I(className=icon_class),
                                group_label
                            ], className=f"order-group-badge {label_class} me-2"),
                            html.Span(f"{len(asset_orders)} orders", className="order-group-count"),
                        ], className="d-flex align-items-center gap-1")
                    ], colSpan=5)
                ], className="order-group-header")
                table_rows.append(header_row)

            for idx, order in enumerate(asset_orders):
                # Status color coding
                status_color = {
                    "filled": "text-success",
                    "canceled": "text-danger",
                    "pending_new": "text-warning",
                    "accepted": "text-info",
                    "rejected": "text-danger"
                }.get(str(order.get("Status", "")).lower(), "text-muted")

                # Side color coding
                side_str = str(order.get("Side", "")).lower()
                side_color = "text-success" if "buy" in side_str else "text-danger"

                row_class = "table-row-hover order-group-child" if is_group else "table-row-hover"
                row = html.Tr([
                    html.Td([
                        html.Div([
                            html.Strong(order["Asset"], className="symbol-text"),
                            html.Br(),
                            html.Small(order["Order Type"], className="text-muted")
                        ])
                    ], className="symbol-cell"),
                    html.Td([
                        html.Div([
                            html.Span(str(order["Side"]), className=f"fw-bold {side_color}"),
                            html.Br(),
                            html.Small(f"{order['Qty']} shares", className="text-muted")
                        ])
                    ], className="side-cell"),
                    html.Td([
                        html.Div([
                            html.Div(f"{order['Filled Qty']}", className="fw-bold"),
                            html.Small("filled", className="text-muted")
                        ])
                    ], className="filled-cell"),
                    html.Td([
                        html.Div([
                            html.Div(order["Avg. Fill Price"], className="fw-bold"),
                            html.Small("avg price", className="text-muted")
                        ])
                    ], className="price-cell"),
                    html.Td([
                        html.Span([
                            html.I(className=f"fas fa-circle me-1 {status_color}"),
                            str(order["Status"])
                        ], className=f"status-badge {status_color}")
                    ], className="status-cell")
                ], className=row_class, id=f"order-row-{order.get('Asset', '')}-{page}-{row_counter}")

                table_rows.append(row)
                row_counter += 1

        # Create enhanced table with pagination
        table = html.Div([
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Asset", className="table-header"),
                        html.Th("Side & Qty", className="table-header"),
                        html.Th("Filled", className="table-header"),
                        html.Th("Avg Price", className="table-header"),
                        html.Th("Status", className="table-header")
                    ])
                ]),
                html.Tbody(table_rows)
            ], className="enhanced-table"),
            html.Div([
                dbc.Pagination(
                    id="orders-pagination",
                    max_value=10,
                    active_page=page,
                    size="sm",
                    className="mt-3"
                )
            ], className="d-flex justify-content-end")
        ], className="enhanced-table-container")

        return table

    except AlpacaAuthError as e:
        # Show specific auth error message
        return html.Div([
            html.Div([
                html.I(className="fas fa-key fa-2x mb-3 text-danger"),
                html.H5("Authentication Error", className="text-danger"),
                html.P("Cannot fetch orders - API key authentication failed", className="text-muted mb-2"),
                html.Small("Please regenerate your Alpaca API keys", className="text-muted")
            ], className="text-center p-4")
        ], className="enhanced-table-container error-state")
    except Exception as e:
        print(f"Error rendering orders table: {e}")
        return html.Div([
            html.Div([
                html.I(className="fas fa-exclamation-triangle fa-2x mb-3 text-warning"),
                html.H5("Unable to Load Orders", className="text-warning"),
                html.P("Check your Alpaca API keys", className="text-muted"),
                html.Small(f"Error: {str(e)}", className="text-muted")
            ], className="text-center p-4")
        ], className="enhanced-table-container error-state")

def render_account_summary():
    """Render account summary information"""
    try:
        account_info = AlpacaUtils.get_account_info()

        buying_power = account_info["buying_power"]
        cash = account_info["cash"]
        daily_change_dollars = account_info["daily_change_dollars"]
        daily_change_percent = account_info["daily_change_percent"]

        # Determine value class for daily change based on whether it's positive or negative
        daily_change_class = "positive" if daily_change_dollars >= 0 else "negative"
        change_icon = "fas fa-arrow-up" if daily_change_dollars >= 0 else "fas fa-arrow-down"

        summary = html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-wallet me-2"),
                            "Buying Power"
                        ], className="summary-label"),
                        html.Div(f"${buying_power:.2f}", className="summary-value")
                    ], className="summary-item enhanced-summary-item")
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-dollar-sign me-2"),
                            "Cash"
                        ], className="summary-label"),
                        html.Div(f"${cash:.2f}", className="summary-value")
                    ], className="summary-item enhanced-summary-item")
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className=f"{change_icon} me-2"),
                            "Daily Change"
                        ], className="summary-label"),
                        html.Div([
                            f"${daily_change_dollars:.2f} ",
                            html.Span(f"({daily_change_percent:.2f}%)")
                        ], className=f"summary-value {daily_change_class}")
                    ], className="summary-item enhanced-summary-item")
                ], width=4)
            ])
        ], className="account-summary enhanced-account-summary")

        return summary

    except AlpacaAuthError as e:
        # Show specific auth error message with actionable steps
        return html.Div([
            html.Div([
                html.I(className="fas fa-key fa-2x mb-3 text-danger"),
                html.H5("Alpaca Authentication Failed", className="text-danger"),
                html.P([
                    "Your Alpaca API keys are not working for account data retrieval. ",
                    "This may happen if you reset your paper trading account."
                ], className="mb-3"),
                html.Div([
                    html.Strong("To fix this:"),
                    html.Ol([
                        html.Li("Go to Alpaca Dashboard: https://app.alpaca.markets/paper/dashboard"),
                        html.Li("Navigate to 'API Keys' section"),
                        html.Li("Click 'Regenerate Keys'"),
                        html.Li("Update your .env file with new keys"),
                        html.Li("Restart the application")
                    ], className="text-start")
                ], className="alert alert-info text-start"),
                html.Small([
                    html.Strong("Note: "),
                    "Trading still works, only dashboard refresh is affected."
                ], className="text-muted")
            ], className="text-center p-4")
        ], className="enhanced-account-summary error-state")
    except Exception as e:
        print(f"Error rendering account summary: {e}")
        return html.Div([
            html.Div([
                html.I(className="fas fa-exclamation-triangle fa-2x mb-3 text-warning"),
                html.H5("Unable to Load Account Summary", className="text-warning"),
                html.P("Check your Alpaca API keys", className="text-muted"),
                html.Small(f"Error: {str(e)}", className="text-muted")
            ], className="text-center p-4")
        ], className="enhanced-account-summary error-state")

def get_positions_data():
    """Get positions data for table callback"""
    try:
        return AlpacaUtils.get_positions_data()
    except Exception as e:
        print(f"Error getting positions data: {e}")
        return []

def get_recent_orders(page=1, page_size=7):
    """Get recent orders data for table callback"""
    try:
        return AlpacaUtils.get_recent_orders(page=page, page_size=page_size)
    except Exception as e:
        print(f"Error getting orders data: {e}")
        return []

def render_alpaca_account_section():
    """Render the complete Alpaca account section"""
    return html.Div([
        html.H4([
            html.I(className="fas fa-chart-line me-2"),
            "Alpaca Paper Trading Account", 
            html.Button([
                html.I(className="fas fa-sync-alt")
            ], 
            id="refresh-alpaca-btn",
            className="btn btn-sm btn-outline-primary ms-auto",
            title="Refresh Alpaca account data"
            )
        ], className="mb-3 d-flex align-items-center"),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.H5([
                    html.I(className="fas fa-briefcase me-2"),
                    "Open Positions"
                ], className="mb-3"),
                html.Div(id="positions-table-container", children=render_positions_table())
            ], md=7),
            dbc.Col([
                html.H5([
                    html.I(className="fas fa-history me-2"),
                    "Recent Orders"
                ], className="mb-3"),
                html.Div(id="orders-table-container", children=render_orders_table())
            ], md=5)
        ]),
        html.Div(id="account-summary-container", children=render_account_summary()),
        # Hidden div for liquidation confirmations
        dcc.ConfirmDialog(
            id='liquidate-confirm',
            message='',
        ),
        html.Div(id="liquidation-status", className="mt-3")
    ], className="mb-4 alpaca-account-section enhanced-alpaca-section") 