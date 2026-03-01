"""
webui/components/alpaca_account.py - Alpaca account information components
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
import pandas as pd
from datetime import datetime, timedelta, timezone
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

def render_orders_table(orders_data=None):
    """Render the enhanced recent orders with card-based timeline view"""
    try:
        if orders_data is None:
            orders_data = AlpacaUtils.get_recent_orders(limit=100)

        if not orders_data:
            return html.Div([
                html.Div([
                    html.I(className="fas fa-history fa-2x mb-3"),
                    html.H5("No Recent Orders", className="text-muted"),
                    html.P("No trading activity found", className="text-muted small")
                ], className="text-center p-5")
            ], className="enhanced-table-container")

        # --- timestamp helpers ---
        _EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)

        def _parse_dt(iso_str):
            if not iso_str:
                return None
            try:
                dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                return None

        now = datetime.now(timezone.utc)
        today = now.date()
        yesterday = (now - timedelta(days=1)).date()
        week_start = today - timedelta(days=today.weekday())

        # Attach parsed datetimes
        for order in orders_data:
            order["_submitted_dt"] = _parse_dt(order.get("submitted_at"))

        # Sort newest-first
        orders_sorted = sorted(
            orders_data,
            key=lambda o: o["_submitted_dt"] or _EPOCH,
            reverse=True,
        )

        # --- bracket grouping: same symbol, submitted within 60 seconds ---
        def _build_groups(orders_list):
            groups = []
            used = set()
            for i, order in enumerate(orders_list):
                if i in used:
                    continue
                used.add(i)
                group = [order]
                if order["_submitted_dt"] is not None:
                    for j, other in enumerate(orders_list):
                        if j in used:
                            continue
                        if (other["Asset"] == order["Asset"]
                                and other["_submitted_dt"] is not None):
                            diff = abs((other["_submitted_dt"] - order["_submitted_dt"]).total_seconds())
                            if diff <= 60:
                                group.append(other)
                                used.add(j)
                groups.append(group)
            return groups

        def _get_bucket(order):
            dt = order.get("_submitted_dt")
            if dt is None:
                return "Older"
            d = dt.date()
            if d == today:
                return "Today"
            elif d == yesterday:
                return "Yesterday"
            elif d >= week_start:
                return "This Week"
            return "Older"

        all_groups = _build_groups(orders_sorted)

        bucket_order = ["Today", "Yesterday", "This Week", "Older"]
        buckets = {b: [] for b in bucket_order}
        for group in all_groups:
            buckets[_get_bucket(group[0])].append(group)

        # --- rendering helpers ---
        def _side_display(side_str):
            """Return (label, css_color_class, italic)"""
            s = side_str.lower()
            if s == "sell_short":
                return "SELL SHORT", "text-danger", True
            elif s == "buy_to_cover":
                return "BUY TO COVER", "text-success", True
            elif "sell" in s:
                return "SELL", "text-danger", False
            return "BUY", "text-success", False

        def _render_order_card(order, is_child=False):
            side_str = str(order.get("Side", "")).lower()
            side_label, side_color, is_italic = _side_display(side_str)

            status_str = str(order.get("Status", "")).lower()
            status_map = {
                "filled": ("success", "Filled"),
                "canceled": ("danger", "Canceled"),
                "cancelled": ("danger", "Canceled"),
                "pending_new": ("warning", "Pending"),
                "new": ("info", "New"),
                "accepted": ("info", "Accepted"),
                "rejected": ("danger", "Rejected"),
                "partially_filled": ("warning", "Partial"),
                "held": ("warning", "Held"),
                "done_for_day": ("secondary", "Done"),
                "expired": ("secondary", "Expired"),
            }
            status_color, status_text = status_map.get(status_str, ("secondary", str(order.get("Status", "-"))))

            qty = order.get("Qty", 0)
            filled_qty = order.get("Filled Qty", 0)
            avg_price = order.get("Avg. Fill Price", "-")
            order_type_raw = str(order.get("Order Type", ""))
            order_type = order_type_raw.replace("_", " ").title()

            dt = order.get("_submitted_dt")
            if dt:
                ts_str = dt.strftime("%H:%M") if dt.date() == today else dt.strftime("%b %d")
            else:
                ts_str = "-"

            side_elem = html.Span(
                html.I(side_label) if is_italic else side_label,
                className=f"order-side-pill {side_color}",
            )

            qty_text = f"{qty}"
            if filled_qty and float(filled_qty) != float(qty):
                qty_text = f"{filled_qty}/{qty}"

            card_class = "order-card bracket-child" if is_child else "order-card"
            return html.Div([
                html.Div([
                    html.Span(order["Asset"], className="order-symbol"),
                    html.Br(),
                    html.Small(order_type, className="text-muted"),
                ], className="order-card-left"),
                html.Div([
                    side_elem,
                    html.Span(f" {qty_text} shs", className="text-muted ms-1 small"),
                ], className="order-card-middle"),
                html.Div([
                    html.Span(status_text, className=f"order-status-badge badge bg-{status_color}"),
                    html.Br(),
                    html.Small(avg_price, className="fw-bold"),
                    html.Span(" · ", className="text-muted"),
                    html.Small(ts_str, className="text-muted"),
                ], className="order-card-right"),
            ], className=card_class)

        def _render_bracket_header(group):
            sides_lower = [str(o.get("Side", "")).lower() for o in group]
            sells = [s for s in sides_lower if "sell" in s]  # captures sell, sell_short
            buys  = [s for s in sides_lower if "buy"  in s]  # captures buy, buy_to_cover

            if any("sell_short" in s for s in sides_lower):
                group_label, label_class, icon_class = "SHORT", "text-danger", "fas fa-arrow-down me-1"
            elif len(sells) > 0 and len(sells) < len(buys):
                # 1 sell entry + N buy legs → short bracket
                group_label, label_class, icon_class = "SHORT", "text-danger", "fas fa-arrow-down me-1"
            elif len(buys) > 0 and len(buys) < len(sells):
                # 1 buy entry + N sell legs → long bracket
                group_label, label_class, icon_class = "LONG", "text-success", "fas fa-arrow-up me-1"
            elif len(buys) >= len(sells):
                group_label, label_class, icon_class = "LONG", "text-success", "fas fa-arrow-up me-1"
            else:
                group_label, label_class, icon_class = "SHORT", "text-danger", "fas fa-arrow-down me-1"

            return html.Div([
                html.Span(group[0]["Asset"], className="order-symbol me-2"),
                html.Span([
                    html.I(className=icon_class),
                    group_label,
                ], className=f"order-group-badge {label_class} me-2"),
                html.Span(f"{len(group)} legs", className="order-group-count"),
            ], className="order-bracket-header")

        # --- assemble timeline content ---
        total_orders = len(orders_data)
        content_items = [
            html.Div(
                html.Span(f"{total_orders} orders", className="badge bg-secondary"),
                className="mb-2",
            )
        ]

        for bucket_name in bucket_order:
            groups = buckets[bucket_name]
            if not groups:
                continue
            content_items.append(html.Div(bucket_name, className="orders-date-header"))
            for group in groups:
                if len(group) > 1:
                    content_items.append(_render_bracket_header(group))
                    for order in group:
                        content_items.append(_render_order_card(order, is_child=True))
                else:
                    content_items.append(_render_order_card(group[0], is_child=False))

        return html.Div([
            html.Div(content_items, className="orders-timeline"),
        ], className="enhanced-table-container")

    except AlpacaAuthError as e:
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


def get_recent_orders(limit=100):
    """Get recent orders data"""
    try:
        return AlpacaUtils.get_recent_orders(limit=limit)
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