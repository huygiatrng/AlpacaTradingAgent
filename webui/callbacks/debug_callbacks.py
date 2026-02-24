"""
Debug Panel Callbacks for TradingAgents WebUI

Callbacks for handling the debug panel functionality including:
- Toggle panel open/close
- Filter tool calls by symbol and agent
- Display prompts for selected symbol and report type
- Auto-refresh during analysis
"""

from dash import Input, Output, State, html, ctx, no_update
from dash.exceptions import PreventUpdate
import dash

from webui.utils.state import app_state
from webui.components.debug_panel import (
    format_tool_calls_for_debug,
    format_prompt_for_debug,
    format_tool_calls_stats,
    format_logs_for_debug,
    get_available_symbols_from_tool_calls,
    get_available_agents_from_tool_calls
)


def register_debug_callbacks(app):
    """Register all debug panel callbacks"""

    @app.callback(
        Output("debug-panel", "is_open"),
        Input("toggle-debug-panel", "n_clicks"),
        State("debug-panel", "is_open"),
        prevent_initial_call=True
    )
    def toggle_debug_panel(n_clicks, is_open):
        """Toggle the debug panel open/close"""
        if n_clicks:
            new_state = not is_open
            print(f"[DEBUG PANEL] Toggled: {'OPEN' if new_state else 'CLOSED'}")
            print(f"[DEBUG PANEL] Total tool calls in log: {len(app_state.tool_calls_log)}")
            return new_state
        return is_open

    @app.callback(
        [
            Output("debug-symbol-filter", "options"),
            Output("debug-agent-filter", "options")
        ],
        [
            Input("refresh-interval", "n_intervals"),
            Input("medium-refresh-interval", "n_intervals"),
            Input("slow-refresh-interval", "n_intervals"),
            Input("debug-panel", "is_open")
        ],
        prevent_initial_call=False
    )
    def update_tool_calls_filter_options(fast_n, medium_n, slow_n, is_open):
        """Update filter dropdown options based on available tool calls"""
        # Get all tool calls
        all_tool_calls = app_state.get_tool_calls_for_display()

        # Extract unique symbols and agents
        symbols = get_available_symbols_from_tool_calls(all_tool_calls)
        agents = get_available_agents_from_tool_calls(all_tool_calls)

        # Build options
        symbol_options = [{"label": "All Symbols", "value": "ALL"}] + [
            {"label": symbol, "value": symbol} for symbol in symbols
        ]

        agent_options = [{"label": "All Agents", "value": "ALL"}] + [
            {"label": agent, "value": agent} for agent in agents
        ]

        return symbol_options, agent_options

    @app.callback(
        [
            Output("debug-tool-calls-content", "children"),
            Output("debug-tool-calls-stats", "children")
        ],
        [
            Input("debug-symbol-filter", "value"),
            Input("debug-agent-filter", "value"),
            Input("refresh-interval", "n_intervals"),
            Input("medium-refresh-interval", "n_intervals"),
            Input("slow-refresh-interval", "n_intervals"),
            Input("debug-panel", "is_open")
        ],
        prevent_initial_call=False
    )
    def update_tool_calls_content(symbol_filter, agent_filter, fast_refresh, medium_refresh, slow_refresh, is_open):
        """Update tool calls display based on filters"""

        total_calls = len(app_state.tool_calls_log)

        # Apply filters
        agent_filter_value = None if agent_filter == "ALL" else agent_filter
        symbol_filter_value = None if symbol_filter == "ALL" else symbol_filter

        # Get filtered tool calls
        filtered_calls = app_state.get_tool_calls_for_display(
            agent_filter=agent_filter_value,
            symbol_filter=symbol_filter_value
        )

        # Format content
        try:
            content = format_tool_calls_for_debug(filtered_calls)
            stats = format_tool_calls_stats(filtered_calls)
        except Exception as e:
            import traceback
            print(f"[DEBUG PANEL] ❌ Render error: {e}")
            print(traceback.format_exc())
            content = html.Div(f"Render error: {str(e)}", className="text-danger p-3")
            stats = html.Div("Error")

        return content, stats

    @app.callback(
        Output("debug-prompt-symbol-filter", "options"),
        [
            Input("refresh-interval", "n_intervals"),
            Input("medium-refresh-interval", "n_intervals"),
            Input("slow-refresh-interval", "n_intervals"),
            Input("debug-panel", "is_open")
        ],
        prevent_initial_call=False
    )
    def update_prompt_symbol_filter_options(fast_n, medium_n, slow_n, is_open):
        """Update symbol filter options for prompts tab"""
        # Get symbols that have states (and therefore prompts)
        symbols = []
        for symbol in app_state.symbol_states.keys():
            if symbol:  # Only include non-empty symbols
                symbols.append(symbol)

        return [{"label": symbol, "value": symbol} for symbol in sorted(symbols)]

    @app.callback(
        Output("debug-prompts-content", "children"),
        [
            Input("debug-prompt-symbol-filter", "value"),
            Input("debug-prompt-type-filter", "value"),
            Input("medium-refresh-interval", "n_intervals"),
            Input("slow-refresh-interval", "n_intervals")
        ],
        prevent_initial_call=False
    )
    def update_prompts_content(symbol, report_type, medium_n, slow_n):
        """Update prompts display based on selected symbol and report type"""

        if not symbol or not report_type:
            return html.Div(
                [
                    html.I(className="fas fa-hand-pointer me-2"),
                    "Select a symbol and report type to view the prompt"
                ],
                className="text-muted text-center p-4"
            )

        # Get the prompt from app_state
        prompt = app_state.get_agent_prompt(report_type, symbol)

        if not prompt:
            return html.Div(
                [
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    f"No prompt available for {report_type.replace('_', ' ').title()} - {symbol}"
                ],
                className="text-warning text-center p-4"
            )

        return format_prompt_for_debug(prompt, report_type)

    @app.callback(
        Output("debug-copy-prompt-btn", "n_clicks"),
        Input("debug-copy-prompt-btn", "n_clicks"),
        [
            State("debug-prompt-symbol-filter", "value"),
            State("debug-prompt-type-filter", "value")
        ],
        prevent_initial_call=True
    )
    def copy_prompt_to_clipboard(n_clicks, symbol, report_type):
        """Copy prompt to clipboard (client-side handled via dcc.Clipboard)"""
        if n_clicks and symbol and report_type:
            # The actual copy operation would be handled client-side
            # This callback just triggers the action
            pass
        return None

    @app.callback(
        Output("debug-export-prompt-btn", "n_clicks"),
        Input("debug-export-prompt-btn", "n_clicks"),
        [
            State("debug-prompt-symbol-filter", "value"),
            State("debug-prompt-type-filter", "value")
        ],
        prevent_initial_call=True
    )
    def export_prompt(n_clicks, symbol, report_type):
        """Export prompt as a file (would trigger download)"""
        if n_clicks and symbol and report_type:
            # Export functionality could be implemented with dcc.Download
            pass
        return None

    # Add a simple callback to check panel state periodically
    @app.callback(
        Output("debug-tool-calls-stats", "children", allow_duplicate=True),
        Input("slow-refresh-interval", "n_intervals"),
        prevent_initial_call=True
    )
    def debug_state_check(n):
        """Periodic debug state check"""
        if n % 10 == 0:  # Every 10 intervals
            print(f"[DEBUG PANEL] Periodic check - Tool calls: {len(app_state.tool_calls_log)}")
            print(f"[DEBUG PANEL] Sample data: {app_state.tool_calls_log[:1] if app_state.tool_calls_log else 'No data'}")
        return no_update

    # ── Logs tab: filter options ────────────────────────────────────────────
    @app.callback(
        [
            Output("debug-logs-symbol-filter", "options"),
            Output("debug-logs-tag-filter", "options"),
        ],
        [
            Input("refresh-interval", "n_intervals"),
            Input("medium-refresh-interval", "n_intervals"),
            Input("slow-refresh-interval", "n_intervals"),
            Input("debug-panel", "is_open"),
        ],
        prevent_initial_call=False,
    )
    def update_logs_filter_options(fast_n, medium_n, slow_n, is_open):
        """Populate symbol and tag dropdowns from captured log entries."""
        symbols = app_state.get_unique_log_symbols()
        tags = app_state.get_unique_log_tags()

        symbol_opts = [{"label": "All Symbols", "value": "ALL"}] + [
            {"label": s, "value": s} for s in symbols
        ]
        tag_opts = [{"label": "All Types", "value": "ALL"}] + [
            {"label": t, "value": t} for t in tags
        ]
        return symbol_opts, tag_opts

    # ── Logs tab: content display ───────────────────────────────────────────
    @app.callback(
        Output("debug-logs-content", "children"),
        [
            Input("debug-logs-symbol-filter", "value"),
            Input("debug-logs-tag-filter", "value"),
            Input("refresh-interval", "n_intervals"),
            Input("medium-refresh-interval", "n_intervals"),
            Input("slow-refresh-interval", "n_intervals"),
            Input("debug-panel", "is_open"),
        ],
        prevent_initial_call=False,
    )
    def update_logs_content(symbol_filter, tag_filter, fast_n, medium_n, slow_n, is_open):
        """Render system log entries applying current filter selections."""
        logs = app_state.get_system_logs_for_display(
            symbol_filter=symbol_filter,
            tag_filter=tag_filter,
        )
        try:
            return format_logs_for_debug(logs)
        except Exception as e:
            import traceback
            print(f"[DEBUG PANEL] ❌ Logs render error: {e}")
            print(traceback.format_exc())
            return html.Div(f"Render error: {str(e)}", className="text-danger p-3")

    # ── Logs tab: clear button ──────────────────────────────────────────────
    @app.callback(
        Output("debug-logs-clear-btn", "n_clicks"),
        Input("debug-logs-clear-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_logs(n_clicks):
        """Clear all captured system log entries."""
        if n_clicks:
            app_state.system_logs = []
            print("[DEBUG PANEL] System logs cleared by user")
        return None

    # ── Clientside tab switching (3 tabs) ───────────────────────────────────
    app.clientside_callback(
        """
        function(tool_n, prompts_n, logs_n) {
            var ctx = dash_clientside.callback_context;
            var triggered = ctx.triggered && ctx.triggered.length > 0
                            ? ctx.triggered[0].prop_id.split('.')[0]
                            : 'debug-tab-btn-tool-calls';

            var showTool   = {"display": "none"};
            var showPrompt = {"display": "none"};
            var showLogs   = {"display": "none"};
            var clsTool    = "debug-tab-btn";
            var clsPrompt  = "debug-tab-btn";
            var clsLogs    = "debug-tab-btn";

            if (triggered === 'debug-tab-btn-prompts') {
                showPrompt = {"display": "block"};
                clsPrompt  = "debug-tab-btn debug-tab-btn-active";
            } else if (triggered === 'debug-tab-btn-logs') {
                showLogs  = {"display": "block"};
                clsLogs   = "debug-tab-btn debug-tab-btn-active";
            } else {
                showTool  = {"display": "block"};
                clsTool   = "debug-tab-btn debug-tab-btn-active";
            }

            return [showTool, showPrompt, showLogs, clsTool, clsPrompt, clsLogs];
        }
        """,
        Output("debug-pane-tool-calls", "style"),
        Output("debug-pane-prompts", "style"),
        Output("debug-pane-logs", "style"),
        Output("debug-tab-btn-tool-calls", "className"),
        Output("debug-tab-btn-prompts", "className"),
        Output("debug-tab-btn-logs", "className"),
        Input("debug-tab-btn-tool-calls", "n_clicks"),
        Input("debug-tab-btn-prompts", "n_clicks"),
        Input("debug-tab-btn-logs", "n_clicks"),
        prevent_initial_call=True,
    )

    print("✓ Debug callbacks registered")
