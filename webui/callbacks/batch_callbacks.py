"""
Batch overview callbacks for TradingAgents WebUI
Handles batch analysis progress display and ticker navigation
"""

from dash import Input, Output, State, html, ctx, callback_context, ALL
import dash_bootstrap_components as dbc
import dash
import time
import math

from webui.utils.state import app_state


def create_symbol_button(symbol, index, is_active=False):
    """Create a symbol button for batch overview pagination"""
    return dbc.Button(
        symbol,
        id={"type": "symbol-btn", "index": index, "component": "batch"},
        color="primary" if is_active else "outline-primary",
        size="sm",
        className=f"symbol-btn {'active' if is_active else ''}",
    )


def calculate_ticker_progress(symbol, state):
    """
    Calculate progress percentage by counting completed/in-progress agents

    13 agents total, each worth ~7.7% (100/13)
    - Completed agent = full weight
    - In-progress agent = half weight

    Returns: 0-100
    """
    if not state or "agent_statuses" not in state:
        return 0

    # Check if analysis is complete
    if state.get("analysis_complete", False):
        return 100

    agent_statuses = state["agent_statuses"]
    total_agents = len(agent_statuses)

    if total_agents == 0:
        return 0

    completed_count = sum(1 for status in agent_statuses.values() if status == "completed")
    in_progress_count = sum(1 for status in agent_statuses.values() if status == "in_progress")

    # Each completed agent is worth full weight, in-progress is worth half
    progress = (completed_count * 100 / total_agents) + (in_progress_count * 50 / total_agents)

    return min(100, int(progress))


def get_ticker_phase(symbol, state):
    """
    Determine current phase by checking agent statuses in reverse order

    Returns: Phase string (Complete, Risk Mgmt, Trading, Research, Analysts, Queued)
    """
    if not state or "agent_statuses" not in state:
        return "Queued"

    # Check if analysis is complete
    if state.get("analysis_complete", False):
        return "Complete"

    agent_statuses = state["agent_statuses"]

    # Check phases in reverse order (most recent first)
    # Risk Management Phase
    risk_agents = ["Portfolio Manager", "Risky Analyst", "Safe Analyst", "Neutral Analyst"]
    if any(agent_statuses.get(agent) in ["in_progress", "completed"] for agent in risk_agents):
        return "Risk Mgmt"

    # Trading Phase
    if agent_statuses.get("Trader") in ["in_progress", "completed"]:
        return "Trading"

    # Research Phase
    research_agents = ["Bull Researcher", "Bear Researcher", "Research Manager"]
    if any(agent_statuses.get(agent) in ["in_progress", "completed"] for agent in research_agents):
        return "Research"

    # Analysts Phase
    analyst_agents = ["Market Analyst", "Social Analyst", "News Analyst", "Fundamentals Analyst", "Macro Analyst"]
    if any(agent_statuses.get(agent) in ["in_progress", "completed"] for agent in analyst_agents):
        return "Analysts"

    # Default to Queued if no agents have started
    return "Queued"


def format_elapsed_time(start_time):
    """Format elapsed time from start_time timestamp to human-readable string"""
    if not start_time:
        return "0m 00s"

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    return f"{minutes}m {seconds:02d}s"


def register_batch_callbacks(app):
    """Register all batch overview callbacks"""

    @app.callback(
        [Output("batch-summary-header", "children"),
         Output("batch-ticker-table", "children")],
        [Input("refresh-interval", "n_intervals")]
    )
    def update_batch_overview(n_intervals):
        """Update batch overview panel during analysis"""

        # CHECK: Has user clicked recently? If so, don't override their choice
        from webui.config.constants import SYMBOL_CLICK_DEBOUNCE_SECONDS
        time_since_last_click = time.time() - app_state.last_symbol_click_time
        if time_since_last_click < SYMBOL_CLICK_DEBOUNCE_SECONDS:
            # User clicked recently - don't update to prevent override
            print(f"[BATCH_OVERVIEW] Skipping update (user clicked {time_since_last_click:.1f}s ago, debounce={SYMBOL_CLICK_DEBOUNCE_SECONDS}s)")
            import dash
            return dash.no_update, dash.no_update

        # Check if analysis is running or if there are any symbols
        if not app_state.analysis_running and not app_state.symbol_states:
            return (
                html.P("No batch analysis running", className="text-muted text-center"),
                []
            )

        # Get all symbols from symbol_states
        symbols = list(app_state.symbol_states.keys())

        if not symbols:
            return (
                html.P("No symbols to display", className="text-muted text-center"),
                []
            )

        total_symbols = len(symbols)

        # Calculate batch metadata
        batch_size = getattr(app_state, 'batch_size', 5)  # Default to 5 if not set
        total_batches = math.ceil(total_symbols / batch_size)

        # Count completed and analyzing symbols
        completed_count = 0
        analyzing_count = 0

        for symbol in symbols:
            state = app_state.get_state(symbol)
            if state:
                if state.get("analysis_complete", False):
                    completed_count += 1
                else:
                    # Check if any agent is in progress
                    agent_statuses = state.get("agent_statuses", {})
                    if any(status == "in_progress" for status in agent_statuses.values()):
                        analyzing_count += 1

        # Calculate current batch number
        current_batch = (completed_count // batch_size) + 1 if completed_count < total_symbols else total_batches

        # Create summary header
        summary_header = dbc.Alert([
            html.Strong(f"Batch {current_batch}/{total_batches}"),
            html.Span(" | ", className="mx-2"),
            html.Span(f"{total_symbols} tickers", className="me-2"),
            html.Span(" | ", className="mx-2"),
            html.Span(f"{completed_count} complete", className="text-success me-2"),
            html.Span(" | ", className="mx-2"),
            html.Span(f"{analyzing_count} analyzing", className="text-info")
        ], color="info", className="mb-3")

        # Create ticker table rows
        table_rows = []

        for symbol in symbols:
            state = app_state.get_state(symbol)

            if not state:
                continue

            # Calculate progress and phase
            progress = calculate_ticker_progress(symbol, state)
            phase = get_ticker_phase(symbol, state)
            elapsed = format_elapsed_time(state.get("session_start_time"))

            # Determine status icon
            if phase == "Complete":
                icon = "✅"
                row_color = None
            elif phase == "Queued":
                icon = "⏸️"
                row_color = None
            else:
                icon = "🔄"
                row_color = None

            # No row highlighting - all rows use neutral styling
            row_style = {}

            # Create progress bar
            progress_bar = dbc.Progress(
                value=progress,
                color="success" if progress == 100 else "info",
                className="mb-0",
                style={"height": "20px"}
            )

            # Create table row (clickable to navigate to symbol details)
            row = html.Tr(
                [
                    html.Td([icon, " ", html.Strong(symbol)], style={"width": "15%"}),
                    html.Td([
                        progress_bar,
                        html.Small(f"{progress}%", className="text-muted ms-2")
                    ], style={"width": "35%"}),
                    html.Td(phase, style={"width": "25%"}),
                    html.Td(elapsed, style={"width": "25%"})
                ],
                id={"type": "batch-row", "symbol": symbol},
                style={**row_style, "cursor": "pointer"},  # Add pointer cursor
                className="ticker-row-clickable"
            )

            table_rows.append(row)

        # Create table
        if table_rows:
            ticker_table = dbc.Table(
                [
                    html.Thead(html.Tr([
                        html.Th("Ticker"),
                        html.Th("Progress"),
                        html.Th("Phase"),
                        html.Th("Time Elapsed")
                    ])),
                    html.Tbody(table_rows)
                ],
                bordered=True,
                hover=True,
                responsive=True,
                striped=True,
                className="mb-0"
            )
        else:
            ticker_table = html.P("No tickers to display", className="text-muted text-center")

        return summary_header, ticker_table


    @app.callback(
        Output("batch-pagination-container", "children"),
        [Input("app-store", "data"),
         Input("refresh-interval", "n_intervals")]
    )
    def update_batch_symbol_pagination(store_data, n_intervals):
        """Update the symbol pagination buttons for batch overview"""

        # CHECK: Has user clicked recently? If so, don't override their choice
        from webui.config.constants import SYMBOL_CLICK_DEBOUNCE_SECONDS
        time_since_last_click = time.time() - app_state.last_symbol_click_time
        if time_since_last_click < SYMBOL_CLICK_DEBOUNCE_SECONDS:
            # User clicked recently - don't update to prevent override
            return dash.no_update

        if not app_state.symbol_states:
            return html.Div("No symbols available",
                          className="text-muted text-center",
                          style={"padding": "10px"})

        symbols = list(app_state.symbol_states.keys())

        # Use batch-specific index instead of current_symbol
        active_index = getattr(app_state, 'active_batch_symbol_index', 0)

        buttons = []
        for i, symbol in enumerate(symbols):
            is_active = i == active_index
            buttons.append(create_symbol_button(symbol, i, is_active))

        if len(symbols) > 1:
            # Add navigation info
            nav_info = html.Div([
                html.I(className="fas fa-layer-group me-2"),
                f"Viewing batch of {len(symbols)} symbols"
            ], className="text-muted small text-center mt-2")

            return html.Div([
                dbc.ButtonGroup(buttons, className="d-flex flex-wrap justify-content-center"),
                nav_info
            ], className="symbol-pagination-wrapper")
        else:
            return dbc.ButtonGroup(buttons, className="d-flex justify-content-center")


    @app.callback(
        [Output("chart-pagination", "active_page", allow_duplicate=True),
         Output("report-pagination", "active_page", allow_duplicate=True),
         Output("batch-pagination-container", "children", allow_duplicate=True)],
        [Input({"type": "batch-row", "symbol": ALL}, "n_clicks")],
        prevent_initial_call=True
    )
    def handle_batch_row_click(row_clicks):
        """Handle clicks on batch table rows - navigate Chart AND Report to selected symbol"""
        if not any(row_clicks) or not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update

        # Find which row was clicked
        trigger_id = ctx.triggered[0]["prop_id"]
        if "batch-row" in trigger_id:
            # Extract symbol from the row ID
            import json
            row_data = json.loads(trigger_id.split('.')[0])
            clicked_symbol = row_data["symbol"]

            symbols = list(app_state.symbol_states.keys())
            if clicked_symbol in symbols:
                clicked_index = symbols.index(clicked_symbol)

                # CRITICAL: Record timestamp BEFORE updating state
                app_state.last_symbol_click_time = time.time()

                # Update section-specific states
                app_state.active_batch_symbol_index = clicked_index
                app_state.active_chart_symbol = clicked_symbol
                app_state.active_report_symbol = clicked_symbol

                # Also update current_symbol for backward compatibility with other UI components
                app_state.current_symbol = clicked_symbol

                page_number = clicked_index + 1

                # Update batch button container to reflect selection
                buttons = []
                for i, symbol in enumerate(symbols):
                    is_active = i == clicked_index
                    buttons.append(create_symbol_button(symbol, i, is_active))

                if len(symbols) > 1:
                    nav_info = html.Div([
                        html.I(className="fas fa-layer-group me-2"),
                        f"Viewing batch of {len(symbols)} symbols"
                    ], className="text-muted small text-center mt-2")

                    button_container = html.Div([
                        dbc.ButtonGroup(buttons, className="d-flex flex-wrap justify-content-center"),
                        nav_info
                    ], className="symbol-pagination-wrapper")
                else:
                    button_container = dbc.ButtonGroup(buttons, className="d-flex justify-content-center")

                return page_number, page_number, button_container

        return dash.no_update, dash.no_update, dash.no_update


    @app.callback(
        Output("batch-pagination-container", "children", allow_duplicate=True),
        [Input({"type": "symbol-btn", "index": dash.dependencies.ALL, "component": "batch"}, "n_clicks")],
        prevent_initial_call=True
    )
    def handle_batch_symbol_click(symbol_clicks):
        """Handle symbol button clicks in batch overview with immediate visual feedback"""
        if not any(symbol_clicks) or not ctx.triggered:
            return dash.no_update

        # Find which button was clicked
        button_id = ctx.triggered[0]["prop_id"]
        if "symbol-btn" in button_id:
            # Extract index from the button ID
            import json
            button_data = json.loads(button_id.split('.')[0])
            clicked_index = button_data["index"]

            # Update batch-specific state
            symbols = list(app_state.symbol_states.keys())
            if 0 <= clicked_index < len(symbols):
                # CRITICAL: Record timestamp BEFORE updating state
                app_state.last_symbol_click_time = time.time()

                # Track batch-specific index (for button highlighting)
                app_state.active_batch_symbol_index = clicked_index

                # ⚡ IMMEDIATE BUTTON UPDATE - No waiting for refresh!
                buttons = []
                for i, symbol in enumerate(symbols):
                    is_active = i == clicked_index  # Active state based on click
                    buttons.append(create_symbol_button(symbol, i, is_active))

                if len(symbols) > 1:
                    # Add navigation info
                    nav_info = html.Div([
                        html.I(className="fas fa-layer-group me-2"),
                        f"Viewing batch of {len(symbols)} symbols"
                    ], className="text-muted small text-center mt-2")

                    button_container = html.Div([
                        dbc.ButtonGroup(buttons, className="d-flex flex-wrap justify-content-center"),
                        nav_info
                    ], className="symbol-pagination-wrapper")
                else:
                    button_container = dbc.ButtonGroup(buttons, className="d-flex justify-content-center")

                return button_container

        return dash.no_update
