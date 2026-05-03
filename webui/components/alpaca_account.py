"""
webui/components/alpaca_account.py - Alpaca account information components
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
import pandas as pd
from datetime import datetime
import pytz
from tradingagents.dataflows.alpaca_utils import AlpacaUtils
from tradingagents.dataflows.config import get_alpaca_use_paper


ORDERS_PAGE_SIZE = 10


def _visible_order_pages(active_page, total_pages):
    """Show latest five pages, oldest five pages, and the active page if it sits between them."""
    if total_pages <= 11:
        return list(range(1, total_pages + 1))

    latest_pages = set(range(1, min(5, total_pages) + 1))
    oldest_pages = set(range(max(1, total_pages - 4), total_pages + 1))
    pages = sorted(latest_pages | oldest_pages | {active_page})

    visible = []
    previous = None
    for page in pages:
        if previous is not None and page - previous > 1:
            visible.append("gap")
        visible.append(page)
        previous = page
    return visible


def render_orders_pagination(active_page, total_pages, total_orders=0, has_more=False):
    """Render compact Recent Orders pagination with latest/oldest page windows."""
    total_pages = max(1, int(total_pages or 1))
    active_page = max(1, min(int(active_page or 1), total_pages))
    page_items = _visible_order_pages(active_page, total_pages)

    buttons = [
        dbc.Button(
            html.I(className="fas fa-chevron-left"),
            id={"type": "orders-page-btn", "page": f"prev-{max(1, active_page - 1)}"},
            size="sm",
            color="secondary",
            outline=True,
            disabled=active_page <= 1,
            className="orders-page-btn",
            title="Previous page",
        )
    ]

    for item in page_items:
        if item == "gap":
            buttons.append(html.Span("...", className="orders-page-gap"))
            continue

        buttons.append(
            dbc.Button(
                str(item),
                id={"type": "orders-page-btn", "page": f"page-{item}"},
                size="sm",
                color="primary" if item == active_page else "secondary",
                outline=item != active_page,
                disabled=item == active_page,
                className=f"orders-page-btn {'active' if item == active_page else ''}",
                title=f"Page {item}",
            )
        )

    buttons.append(
        dbc.Button(
            html.I(className="fas fa-chevron-right"),
            id={"type": "orders-page-btn", "page": f"next-{min(total_pages, active_page + 1)}"},
            size="sm",
            color="secondary",
            outline=True,
            disabled=active_page >= total_pages,
            className="orders-page-btn",
            title="Next page",
        )
    )

    total_text = f"{total_orders} orders"
    if has_more:
        total_text = f"{total_orders}+ orders"

    return html.Div(
        [
            html.Div(buttons, className="orders-page-buttons"),
            html.Div(
                f"Page {active_page} of {total_pages} | {total_text}",
                className="orders-page-meta",
            ),
        ],
        className="orders-pagination",
    )

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

        # Create enhanced table rows with liquidate buttons
        table_rows = []
        for position in positions_data:
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

            today_pl_color = _get_pl_color(position["Today's P/L ($)"])
            total_pl_color = _get_pl_color(position["Total P/L ($)"])

            row = html.Tr([
                html.Td([
                    html.Div([
                        html.Strong(position["Symbol"], className="symbol-text"),
                        html.Br(),
                        html.Small(f"{position['Qty']} shares", className="text-muted")
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

            table_rows.append(row)

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

def render_orders_table_body(orders_data, page=1):
    """Render only the Recent Orders table body so the loading spinner stays off pagination."""
    if not orders_data:
        return html.Div([
            html.Div([
                html.I(className="fas fa-history fa-2x mb-3"),
                html.H5("No Recent Orders", className="text-muted"),
                html.P("No trading activity found", className="text-muted small")
            ], className="text-center p-4")
        ], className="orders-empty-state")

    table_rows = []
    for idx, order in enumerate(orders_data):
        status_color = {
            "filled": "text-success",
            "canceled": "text-danger",
            "pending_new": "text-warning",
            "accepted": "text-info",
            "rejected": "text-danger"
        }.get(order.get("Status", "").lower(), "text-muted")

        side_color = "text-success" if order.get("Side", "").lower() == "buy" else "text-danger"

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
                    html.Span(order["Side"], className=f"fw-bold {side_color}"),
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
                    order["Status"]
                ], className=f"status-badge {status_color}")
            ], className="status-cell")
        ], className="table-row-hover order-row", id=f"order-row-{order.get('Asset', '')}-{page}-{idx}")

        table_rows.append(row)

    return html.Table([
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
    ], className="enhanced-table orders-table")


def render_orders_table_error(error):
    return html.Div([
        html.Div([
            html.I(className="fas fa-exclamation-triangle fa-2x mb-3 text-warning"),
            html.H5("Unable to Load Orders", className="text-warning"),
            html.P("Check your Alpaca API keys", className="text-muted"),
            html.Small(f"Error: {str(error)}", className="text-muted")
        ], className="text-center p-4")
    ], className="orders-empty-state error-state")


def render_orders_table(page=1, page_size=ORDERS_PAGE_SIZE):
    """Render the enhanced Recent Orders surface with table-only loading."""
    try:
        page_data = AlpacaUtils.get_recent_orders_page(page=page, page_size=page_size)
        active_page = page_data.get("page", page)
        total_pages = page_data.get("total_pages", 1)
        total_orders = page_data.get("total_orders", 0)
        has_more = page_data.get("has_more", False)

        return html.Div([
            dcc.Loading(
                html.Div(
                    id="orders-table-body-container",
                    children=render_orders_table_body(page_data.get("orders", []), active_page),
                    className="orders-table-body",
                ),
                type="circle",
                className="orders-loading",
            ),
            html.Div(
                id="orders-pagination-container",
                children=render_orders_pagination(active_page, total_pages, total_orders, has_more),
            ),
        ], className="enhanced-table-container orders-table-container")

    except Exception as e:
        print(f"Error rendering orders table: {e}")
        return html.Div([
            dcc.Loading(
                html.Div(id="orders-table-body-container", children=render_orders_table_error(e)),
                type="circle",
                className="orders-loading",
            ),
            html.Div(id="orders-pagination-container", children=render_orders_pagination(1, 1, 0, False)),
        ], className="enhanced-table-container orders-table-container error-state")

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

def get_recent_orders(page=1, page_size=ORDERS_PAGE_SIZE):
    """Get recent orders data for table callback"""
    try:
        return AlpacaUtils.get_recent_orders(page=page, page_size=page_size)
    except Exception as e:
        print(f"Error getting orders data: {e}")
        return []

def render_alpaca_account_section():
    """Render the complete Alpaca account section"""
    use_paper_str = get_alpaca_use_paper()
    is_paper = str(use_paper_str).strip().lower() not in ("false", "0", "no")
    account_mode_label = "Paper Trading" if is_paper else "Live Trading"
    return html.Div([
        html.H4([
            html.I(className="fas fa-chart-line me-2"),
            html.Span(f"Alpaca {account_mode_label} Account", id="alpaca-account-title"),
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
                dcc.Store(id="orders-page-store", data=1),
                html.Div(id="orders-table-container", children=render_orders_table())
            ], md=5)
        ]),
        render_account_summary(),
        # Hidden div for liquidation confirmations
        dcc.ConfirmDialog(
            id='liquidate-confirm',
            message='',
        ),
        html.Div(id="liquidation-status", className="mt-3")
    ], className="mb-4 alpaca-account-section enhanced-alpaca-section")
