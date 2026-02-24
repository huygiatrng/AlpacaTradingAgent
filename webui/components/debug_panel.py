"""
Debug Panel Component for TradingAgents WebUI

This component creates a side panel for displaying tool calls and prompts,
providing centralized access to debugging information without depending on
symbol selection or report navigation.
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
import json


def create_debug_panel():
    """Create the debug panel - a collapsible side panel with tool calls and prompts"""

    panel = dbc.Offcanvas(
        [
            html.Div([
                # ── Custom tab nav ──────────────────────────────────────────
                html.Div([
                    html.Button(
                        [html.I(className="fas fa-tools me-2"), "Tool Calls"],
                        id="debug-tab-btn-tool-calls",
                        n_clicks=0,
                        className="debug-tab-btn debug-tab-btn-active"
                    ),
                    html.Button(
                        [html.I(className="fas fa-file-alt me-2"), "Prompts"],
                        id="debug-tab-btn-prompts",
                        n_clicks=0,
                        className="debug-tab-btn"
                    ),
                    html.Button(
                        [html.I(className="fas fa-terminal me-2"), "Logs"],
                        id="debug-tab-btn-logs",
                        n_clicks=0,
                        className="debug-tab-btn"
                    ),
                ], className="debug-tab-nav"),

                # ── Tab 1: Tool Calls ───────────────────────────────────────
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Symbol Filter", className="text-light small mb-1"),
                            dcc.Dropdown(
                                id="debug-symbol-filter",
                                options=[{"label": "All Symbols", "value": "ALL"}],
                                value="ALL",
                                clearable=False,
                                className="debug-dropdown",
                                style={"backgroundColor": "#1a1a1a", "color": "#fff"}
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Agent Filter", className="text-light small mb-1"),
                            dcc.Dropdown(
                                id="debug-agent-filter",
                                options=[{"label": "All Agents", "value": "ALL"}],
                                value="ALL",
                                clearable=False,
                                className="debug-dropdown",
                                style={"backgroundColor": "#1a1a1a", "color": "#fff"}
                            )
                        ], width=6)
                    ], className="mb-3"),
                    html.Div(
                        id="debug-tool-calls-stats",
                        className="mb-3 p-2 rounded",
                        style={"backgroundColor": "#2a2a2a", "border": "1px solid #444"}
                    ),
                    html.Div(
                        id="debug-tool-calls-content",
                        className="debug-content-area",
                        style={
                            "maxHeight": "calc(100vh - 350px)",
                            "overflowY": "auto",
                            "overflowX": "hidden"
                        }
                    )
                ], id="debug-pane-tool-calls", className="p-3"),

                # ── Tab 2: Prompts ──────────────────────────────────────────
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Symbol Filter", className="text-light small mb-1"),
                            dcc.Dropdown(
                                id="debug-prompt-symbol-filter",
                                options=[],
                                placeholder="Select a symbol",
                                clearable=False,
                                className="debug-dropdown"
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Report Type", className="text-light small mb-1"),
                            dcc.Dropdown(
                                id="debug-prompt-type-filter",
                                options=[
                                    {"label": "Market Analysis",   "value": "market_report"},
                                    {"label": "Social Sentiment",  "value": "sentiment_report"},
                                    {"label": "News Analysis",     "value": "news_report"},
                                    {"label": "Fundamentals",      "value": "fundamentals_report"},
                                    {"label": "Macro Analysis",    "value": "macro_report"},
                                    {"label": "Bull Research",     "value": "bull_research"},
                                    {"label": "Bear Research",     "value": "bear_research"},
                                    {"label": "Investment Judge",  "value": "investment_judge"},
                                    {"label": "Trader Plan",       "value": "trader_investment_plan"},
                                    {"label": "Risk Manager",      "value": "risk_manager"}
                                ],
                                placeholder="Select report type",
                                clearable=False,
                                className="debug-dropdown"
                            )
                        ], width=6)
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                [html.I(className="fas fa-copy me-2"), "Copy"],
                                id="debug-copy-prompt-btn",
                                color="outline-primary",
                                size="sm",
                                className="me-2"
                            ),
                            dbc.Button(
                                [html.I(className="fas fa-download me-2"), "Export"],
                                id="debug-export-prompt-btn",
                                color="outline-success",
                                size="sm"
                            )
                        ])
                    ], className="mb-3"),
                    html.Div(
                        id="debug-prompts-content",
                        className="debug-content-area",
                        style={"maxHeight": "calc(100vh - 400px)", "overflowY": "auto"}
                    )
                ], id="debug-pane-prompts", className="p-3",
                   style={"display": "none"}),

                # ── Tab 3: System Logs ──────────────────────────────────────
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Symbol Filter", className="text-light small mb-1"),
                            dcc.Dropdown(
                                id="debug-logs-symbol-filter",
                                options=[{"label": "All Symbols", "value": "ALL"}],
                                value="ALL",
                                clearable=False,
                                className="debug-dropdown",
                                style={"backgroundColor": "#1a1a1a", "color": "#fff"}
                            )
                        ], width=5),
                        dbc.Col([
                            dbc.Label("Log Type Filter", className="text-light small mb-1"),
                            dcc.Dropdown(
                                id="debug-logs-tag-filter",
                                options=[{"label": "All Types", "value": "ALL"}],
                                value="ALL",
                                clearable=False,
                                className="debug-dropdown",
                                style={"backgroundColor": "#1a1a1a", "color": "#fff"}
                            )
                        ], width=5),
                        dbc.Col([
                            dbc.Label("\u00a0", className="text-light small mb-1 d-block"),
                            dbc.Button(
                                [html.I(className="fas fa-trash me-1"), "Clear"],
                                id="debug-logs-clear-btn",
                                color="outline-danger",
                                size="sm"
                            )
                        ], width=2)
                    ], className="mb-3"),
                    html.Div(
                        id="debug-logs-content",
                        className="debug-content-area",
                        style={
                            "maxHeight": "calc(100vh - 350px)",
                            "overflowY": "auto",
                            "overflowX": "hidden"
                        }
                    )
                ], id="debug-pane-logs", className="p-3",
                   style={"display": "none"}),

            ])
        ],
        id="debug-panel",
        title=[html.I(className="fas fa-bug me-2"), "Debug Tools"],
        placement="end",
        is_open=False,
        backdrop=False,
        scrollable=True,
        style={
            "width": "650px",
            "backgroundColor": "#1a1a1a",
            "borderLeft": "2px solid #444"
        }
    )

    return panel


def format_tool_calls_for_debug(tool_calls):
    """Format tool calls into accordion items for the debug panel"""

    if not tool_calls:
        return html.Div(
            [
                html.I(className="fas fa-info-circle me-2"),
                "No tool calls recorded yet. Tool calls will appear here as analysis runs."
            ],
            className="text-muted text-center p-4"
        )

    items = []

    for i, call in enumerate(tool_calls, 1):
        try:
            timestamp = call.get('timestamp', 'Unknown')
            tool_name = call.get('tool_name', 'Unknown')
            inputs = call.get('inputs', {})
            output = call.get('output', 'No output')
            execution_time = call.get('execution_time', 'Unknown')
            status = call.get('status', 'unknown')
            agent_type = call.get('agent_type', 'Unknown Agent')
            symbol = call.get('symbol', 'Unknown Symbol')

            status_icon = "✓" if status == "success" else ("✗" if status == "error" else "◌")
            summary_text = f"#{i}: {tool_name}  {status_icon} {status}  |  {agent_type}  |  {symbol}  |  {execution_time}"

            try:
                inputs_text = json.dumps(inputs, indent=2, ensure_ascii=False, default=repr)
            except Exception:
                inputs_text = repr(inputs)

            display_output = str(output) if not isinstance(output, str) else output
            if len(display_output) > 10000:
                display_output = display_output[:9997] + "..."

            item = html.Details([
                html.Summary(summary_text, className="tool-call-summary"),
                html.Div([
                    html.Div([
                        html.Small(f"⏰ {timestamp}", className="text-muted me-3"),
                    ], className="mb-2"),
                    html.H6("📥 Inputs:"),
                    html.Pre(html.Code(inputs_text), className="tool-call-code-block"),
                    html.H6("📤 Output:", className="mt-3"),
                    html.Pre(html.Code(display_output), className="tool-call-code-block"),
                ], className="tool-call-details-body p-3")
            ], className="tool-call-details mb-2")

            items.append(item)

        except Exception as e:
            import traceback
            print(f"[DEBUG PANEL] ❌ Error rendering tool call #{i}: {e}")
            print(traceback.format_exc())
            items.append(html.Details([
                html.Summary(f"#{i}: render error"),
                html.Div(f"Error: {str(e)}", className="text-danger p-2")
            ], className="tool-call-details mb-2"))

    return html.Div(items, className="debug-accordion")


def format_prompt_for_debug(prompt, report_type):
    """Format a prompt for display in the debug panel"""

    if not prompt:
        return html.Div(
            [
                html.I(className="fas fa-exclamation-triangle me-2"),
                "No prompt available for this selection."
            ],
            className="text-muted text-center p-4"
        )

    # Create a formatted display
    report_title = report_type.replace('_', ' ').title()

    return html.Div([
        html.Div([
            html.H5(f"📝 {report_title} Prompt", className="mb-3"),
            html.Hr(style={"borderColor": "#444"})
        ]),
        dcc.Markdown(
            f"```\n{prompt}\n```",
            className="prompt-code-block",
            style={
                "backgroundColor": "#1e1e1e",
                "border": "1px solid #444",
                "borderRadius": "4px",
                "padding": "1rem",
                "fontFamily": "'Courier New', monospace",
                "fontSize": "0.85rem",
                "whiteSpace": "pre-wrap",
                "maxHeight": "600px",
                "overflowY": "auto",
                "color": "#e2e8f0"
            }
        )
    ])


def format_tool_calls_stats(tool_calls):
    """Format statistics summary for tool calls"""

    if not tool_calls:
        return html.Div("No data", className="text-muted small")

    total = len(tool_calls)
    success_count = len([c for c in tool_calls if c.get('status') == 'success'])
    error_count = len([c for c in tool_calls if c.get('status') == 'error'])

    # Get unique symbols and agents
    symbols = set(c.get('symbol', 'Unknown') for c in tool_calls)
    agents = set(c.get('agent_type', 'Unknown') for c in tool_calls)

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div(str(total), className="h4 mb-0 text-info"),
                    html.Div("Total Calls", className="small text-muted")
                ], className="text-center")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Div(str(success_count), className="h4 mb-0 text-success"),
                    html.Div("Successful", className="small text-muted")
                ], className="text-center")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Div(str(error_count), className="h4 mb-0 text-danger"),
                    html.Div("Errors", className="small text-muted")
                ], className="text-center")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Div(str(len(symbols)), className="h4 mb-0 text-warning"),
                    html.Div("Symbols", className="small text-muted")
                ], className="text-center")
            ], width=3)
        ])
    ])


def get_available_symbols_from_tool_calls(tool_calls):
    """Extract unique symbols from tool calls"""
    symbols = set()
    for call in tool_calls:
        symbol = call.get('symbol')
        if symbol and symbol != 'Unknown Symbol':
            symbols.add(symbol)
    return sorted(list(symbols))


def get_available_agents_from_tool_calls(tool_calls):
    """Extract unique agent types from tool calls"""
    agents = set()
    for call in tool_calls:
        agent = call.get('agent_type')
        if agent and agent != 'Unknown Agent':
            agents.add(agent)
    return sorted(list(agents))


# ── Tag → color mapping for Logs tab ──────────────────────────────────────────
_TAG_COLORS = {
    "RISK MANAGER":       "#4fc3f7",  # blue
    "PRICE VALIDATION":   "#81c784",  # green
    "STOP LOSS":          "#ffb74d",  # orange
    "TAKE PROFIT":        "#ffb74d",  # orange
    "STATE":              "#b0bec5",  # grey
    "ERROR":              "#ef5350",  # red
}
_DEFAULT_TAG_COLOR = "#e2e8f0"


def _tag_color(tag: str) -> str:
    tag_up = tag.upper()
    for key, color in _TAG_COLORS.items():
        if key in tag_up:
            return color
    if "ERROR" in tag_up or "FAIL" in tag_up or "❌" in tag_up:
        return _TAG_COLORS["ERROR"]
    return _DEFAULT_TAG_COLOR


def format_logs_for_debug(logs):
    """Render system log entries as a compact monospace list with tag color-coding."""

    if not logs:
        return html.Div(
            [
                html.I(className="fas fa-info-circle me-2"),
                "No log entries yet. System logs will appear here as analysis runs."
            ],
            className="text-muted text-center p-4"
        )

    rows = []
    for entry in reversed(logs):  # newest first
        ts = entry.get("timestamp", "")
        tag = entry.get("tag", "")
        symbol = entry.get("symbol", "")
        msg = entry.get("message", "")
        color = _tag_color(tag)

        symbol_span = (
            html.Span(f" [{symbol}]", style={"color": "#ffd54f", "fontSize": "0.75rem"})
            if symbol else ""
        )

        row = html.Div(
            [
                html.Span(ts, style={"color": "#888", "fontSize": "0.72rem", "marginRight": "6px"}),
                html.Span(f"[{tag}]", style={"color": color, "fontWeight": "bold", "fontSize": "0.78rem"}),
                symbol_span,
                html.Span(f" {msg}", style={"color": "#e2e8f0", "fontSize": "0.78rem", "wordBreak": "break-word"}),
            ],
            style={
                "fontFamily": "'Courier New', monospace",
                "padding": "2px 4px",
                "borderBottom": "1px solid #2a2a2a",
            }
        )
        rows.append(row)

    return html.Div(rows, style={"lineHeight": "1.6"})
