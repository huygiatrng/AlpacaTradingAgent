"""
Report-related callbacks for TradingAgents WebUI
Enhanced with symbol-based pagination
"""

from dash import Input, Output, State, ctx, html, ALL, dash, dcc, callback_context
import dash_bootstrap_components as dbc
import re
from webui.utils.state import app_state
from webui.components.ui import render_researcher_debate, render_risk_debate
from webui.utils.report_validator import validate_reports_for_ui
from webui.utils.prompt_capture import get_agent_prompt


def _is_table_row(line):
    if not line:
        return False
    return line.count("|") >= 2


def _is_separator_row(line):
    if not line:
        return False
    stripped = line.replace("|", "").replace(" ", "")
    if not stripped:
        return False
    return all(ch in "-:" for ch in stripped)


def _normalize_table_row(line):
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    if not any(cells):
        return ""
    return "| " + " | ".join(cells) + " |"


def _normalize_table_block(lines):
    normalized_rows = []
    title_line = None

    for idx, line in enumerate(lines):
        working_line = line
        if idx == 0 and "table |" in working_line.lower() and not working_line.strip().startswith("|"):
            title_line, remainder = working_line.split("|", 1)
            title_line = title_line.strip()
            working_line = "| " + remainder.strip()

        normalized = _normalize_table_row(working_line)
        if normalized:
            normalized_rows.append(normalized)

    if title_line:
        normalized_rows.insert(0, title_line)

    # Ensure there is a separator row after the header
    if normalized_rows:
        header_index = 1 if title_line else 0
        separator_index = header_index + 1
        has_separator = len(normalized_rows) > separator_index and _is_separator_row(normalized_rows[separator_index])
        if len(normalized_rows) > header_index and not has_separator:
            header_cells = normalized_rows[header_index]
            num_cells = len(header_cells.strip().strip("|").split("|"))
            separator = "| " + " | ".join(["---"] * num_cells) + " |"
            normalized_rows.insert(separator_index, separator)

    return normalized_rows


def normalize_markdown_tables(content):
    """Convert inline pipe tables into proper markdown tables."""
    if not content:
        return content

    # Split inline table rows that are concatenated on one line
    split_lines = []
    for raw_line in content.splitlines():
        line = raw_line
        # Handle inline "Table: | a | b |" patterns
        if ":" in line and "|" in line and not line.strip().startswith("|"):
            prefix, remainder = line.split(":", 1)
            if remainder.strip().startswith("|"):
                split_lines.append(prefix.strip())
                line = remainder.strip()
        # Handle inline "TABLE | a | b |" patterns (no colon)
        if "table" in line.lower() and "|" in line and not line.strip().startswith("|"):
            prefix, remainder = line.split("|", 1)
            split_lines.append(prefix.strip())
            line = "| " + remainder.strip()
        # Split concatenated table rows on the same line
        if "|" in line and " | |" in line:
            line = re.sub(r"\s*\|\s*\|\s*", "\n| ", line)
        # Split separator rows concatenated on the same line
        line = re.sub(r"\s*\|\s*\|\s*-", "\n|-", line)
        split_lines.extend(line.splitlines())

    # Normalize table blocks
    output_lines = []
    i = 0
    while i < len(split_lines):
        line = split_lines[i]
        if _is_table_row(line):
            block = []
            while i < len(split_lines) and _is_table_row(split_lines[i]):
                block.append(split_lines[i])
                i += 1
            output_lines.extend(_normalize_table_block(block))
        else:
            output_lines.append(line)
            i += 1

    # Ensure "Notes" following a table starts on new line
    normalized = "\n".join(output_lines)
    normalized = re.sub(r"\|\s*Notes", "|\nNotes", normalized)
    return normalized


def normalize_market_markdown_sections(content):
    """Improve readability for market report outputs that come in inline section format."""
    if not content:
        return content

    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    section_map = {
        "conclusion": "Conclusion",
        "entry conditions": "Entry Conditions",
        "invalidation": "Invalidation",
        "risk sizing hint": "Risk Sizing Hint",
        "narrative": "Narrative",
        "summary table": "Summary Table",
    }

    for raw_label, display_label in section_map.items():
        pattern = rf"(?i)(?:^|\s)[a-f]\)\s*{re.escape(raw_label)}\b"
        normalized = re.sub(pattern, f"\n\n## {display_label}\n", normalized)
        inline_pattern = rf"(?i)(?:^|\s){re.escape(raw_label)}\s*:"
        normalized = re.sub(inline_pattern, f"\n\n## {display_label}\n", normalized)

    normalized = re.sub(r"(?i)\bConclude with:\s*", "\n\n", normalized)
    normalized = re.sub(
        r"(?i)\bFINAL TRANSACTION PROPOSAL\s*:",
        "\n\nFINAL TRANSACTION PROPOSAL:",
        normalized,
    )
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def create_symbol_button(symbol, index, is_active=False):
    """Create a symbol button for pagination"""
    return dbc.Button(
        symbol,
        id={"type": "symbol-btn", "index": index, "component": "reports"},
        color="primary" if is_active else "outline-primary",
        size="sm",
        className=f"symbol-btn {'active' if is_active else ''}",
    )


def create_markdown_content(content, default_message="No content available yet.", report_type=None):
    """Create a markdown component with enhanced styling and conditional prompt button"""
    has_content = content and content.strip() != "" and content != default_message
    
    # Check if this is a loading or default message
    # More precise loading detection - only flag as loading if content is clearly a status message
    is_loading_message = False
    if content:
        content_lower = content.lower().strip()
        # Only flag as loading if it's a short status message starting with these patterns
        if (
            content == default_message or
            (len(content) < 200 and (
                content_lower.startswith("loading") or
                content_lower.startswith("waiting") or
                content_lower.startswith("analysis in progress") or
                content_lower.startswith("no ") and "available yet" in content_lower or
                content_lower.startswith("⏳") or
                content_lower.startswith("🔄")
            ))
        ):
            is_loading_message = True
    else:
        is_loading_message = True
    
    if not content or content.strip() == "":
        content = default_message
    elif not is_loading_message:
        if report_type == "market_report":
            content = normalize_market_markdown_sections(content)
        content = normalize_markdown_tables(content)
    
    markdown_component = dcc.Markdown(
        content,
        mathjax=True,
        highlight_config={"theme": "dark"},
        dangerously_allow_html=False,
        className='enhanced-markdown-content',
        style={
            "background": "linear-gradient(135deg, #0F172A 0%, #1E293B 100%)",
            "border-radius": "8px",
            "padding": "1.5rem",
            "border": "1px solid rgba(51, 65, 85, 0.3)",
            "min-height": "1000px",
            "color": "#E2E8F0",
            "line-height": "1.6"
        }
    )
    
    # If we have actual content and a report type, add a prompt button
    if has_content and not is_loading_message and report_type:
        from webui.components.prompt_modal import create_show_prompt_button
        from webui.components.tool_outputs_modal import create_show_tool_outputs_button
        
        return html.Div([
            html.Div([
                html.Div([
                    create_show_prompt_button(report_type, className="me-2"),
                    create_show_tool_outputs_button(report_type)
                ], className="text-end mb-2 report-debug-buttons")
            ]),
            markdown_component
        ])
    
    return markdown_component


def register_report_callbacks(app):
    """Register all report-related callbacks including symbol pagination"""

    @app.callback(
        Output("report-pagination-container", "children"),
        [Input("app-store", "data"),
         Input("refresh-interval", "n_intervals")]
    )
    def update_report_symbol_pagination(store_data, n_intervals):
        """Update the symbol pagination buttons for reports"""
        if not app_state.symbol_states:
            return html.Div("No symbols available", 
                          className="text-muted text-center",
                          style={"padding": "10px"})
        
        symbols = list(app_state.symbol_states.keys())
        current_symbol = app_state.current_symbol
        
        # Find active symbol index
        active_index = 0
        if current_symbol and current_symbol in symbols:
            active_index = symbols.index(current_symbol)
        
        buttons = []
        for i, symbol in enumerate(symbols):
            is_active = i == active_index
            buttons.append(create_symbol_button(symbol, i, is_active))
        
        if len(symbols) > 1:
            # Add navigation info
            nav_info = html.Div([
                html.I(className="fas fa-info-circle me-2"),
                f"Showing {len(symbols)} symbols"
            ], className="text-muted small text-center mt-2")
            
            return html.Div([
                dbc.ButtonGroup(buttons, className="d-flex flex-wrap justify-content-center"),
                nav_info
            ], className="symbol-pagination-wrapper")
        else:
            return dbc.ButtonGroup(buttons, className="d-flex justify-content-center")

    @app.callback(
        [Output("report-pagination", "active_page", allow_duplicate=True),
         Output("chart-pagination", "active_page", allow_duplicate=True),
         Output("report-pagination-container", "children", allow_duplicate=True)],
        [Input({"type": "symbol-btn", "index": ALL, "component": "reports"}, "n_clicks")],
        prevent_initial_call=True
    )
    def handle_report_symbol_click(symbol_clicks):
        """Handle symbol button clicks for reports with immediate visual feedback"""
        if not any(symbol_clicks) or not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update
        
        # Find which button was clicked
        button_id = ctx.triggered[0]["prop_id"]
        if "symbol-btn" in button_id:
            # Extract index from the button ID
            import json
            button_data = json.loads(button_id.split('.')[0])
            clicked_index = button_data["index"]
            
            # Update current symbol
            symbols = list(app_state.symbol_states.keys())
            if 0 <= clicked_index < len(symbols):
                app_state.current_symbol = symbols[clicked_index]
                page_number = clicked_index + 1
                
                # ⚡ IMMEDIATE BUTTON UPDATE - No waiting for refresh!
                buttons = []
                for i, symbol in enumerate(symbols):
                    is_active = i == clicked_index  # Active state based on click
                    buttons.append(create_symbol_button(symbol, i, is_active))
                
                if len(symbols) > 1:
                    # Add navigation info
                    nav_info = html.Div([
                        html.I(className="fas fa-info-circle me-2"),
                        f"Showing {len(symbols)} symbols"
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
        Output("researcher-debate-tab-content", "children"),
        [Input("report-pagination", "active_page"),
         Input("medium-refresh-interval", "n_intervals")]
    )
    def update_researcher_debate(active_page, n_intervals):
        """Update the researcher debate tab with Dash components and prompt buttons"""
        if not app_state.symbol_states or not active_page:
            return create_markdown_content("", "No researcher debate available yet.")

        # Safeguard against accessing invalid page index (e.g., after page refresh)
        symbols_list = list(app_state.symbol_states.keys())
        if active_page > len(symbols_list):
            return create_markdown_content("", "Page index out of range. Please refresh or restart analysis.")

        symbol = symbols_list[active_page - 1]
        state = app_state.get_state(symbol)
        
        if not state:
            return create_markdown_content("", f"No active analysis for {symbol}. Researcher debate will appear here once analysis starts.")

        # Get the debate state
        debate_state = state.get("investment_debate_state")
        
        if not debate_state or not debate_state.get("history"):
            return create_markdown_content("", "Researcher debate will begin once analysis starts.")

        debate_components = []
        
        # Get message arrays for proper conversation display
        bull_messages = debate_state.get("bull_messages", [])
        bear_messages = debate_state.get("bear_messages", [])
        
        # Create conversation-style debate display
        if bull_messages or bear_messages:
            from webui.components.prompt_modal import create_show_prompt_button
            
            # Interleave messages chronologically based on debate flow
            # Usually: Bull -> Bear -> Bull -> Bear, etc.
            max_messages = max(len(bull_messages), len(bear_messages))
            
            for i in range(max_messages):
                # Add Bull message if available
                if i < len(bull_messages):
                    bull_message = bull_messages[i]
                    # Remove the "Bull Analyst: " prefix for cleaner display
                    clean_bull_message = bull_message.replace("Bull Analyst: ", "")
                    
                    bull_section = html.Div([
                        html.Div([
                            html.Div([
                                html.Span("🐂 Bull Researcher", className="me-2", style={"fontWeight": "bold", "color": "#10B981"}),
                                create_show_prompt_button("bull_report")
                            ], className="d-flex justify-content-between align-items-center mb-2")
                        ]),
                        dcc.Markdown(
                            clean_bull_message,
                            mathjax=True,
                            highlight_config={"theme": "dark"},
                            dangerously_allow_html=False,
                            className='enhanced-markdown-content',
                            style={
                                "background": "linear-gradient(135deg, #064E3B 0%, #047857 100%)",
                                "border-radius": "8px",
                                "padding": "1rem",
                                "border-left": "4px solid #10B981",
                                "color": "#E2E8F0",
                                "margin-bottom": "1rem"
                            }
                        )
                    ])
                    debate_components.append(bull_section)
                
                # Add Bear message if available
                if i < len(bear_messages):
                    bear_message = bear_messages[i]
                    # Remove the "Bear Analyst: " prefix for cleaner display
                    clean_bear_message = bear_message.replace("Bear Analyst: ", "")
                    
                    bear_section = html.Div([
                        html.Div([
                            html.Div([
                                html.Span("🐻 Bear Researcher", className="me-2", style={"fontWeight": "bold", "color": "#EF4444"}),
                                create_show_prompt_button("bear_report")
                            ], className="d-flex justify-content-between align-items-center mb-2")
                        ]),
                        dcc.Markdown(
                            clean_bear_message,
                            mathjax=True,
                            highlight_config={"theme": "dark"},
                            dangerously_allow_html=False,
                            className='enhanced-markdown-content',
                            style={
                                "background": "linear-gradient(135deg, #7F1D1D 0%, #B91C1C 100%)",
                                "border-radius": "8px",
                                "padding": "1rem",
                                "border-left": "4px solid #EF4444",
                                "color": "#E2E8F0",
                                "margin-bottom": "1rem"
                            }
                        )
                    ])
                    debate_components.append(bear_section)
        
        # Fallback to old format if new message arrays don't exist
        elif debate_state.get("bull_history") or debate_state.get("bear_history"):
            from webui.components.prompt_modal import create_show_prompt_button
            
            # Add Bull Researcher section if available
            bull_history = debate_state.get("bull_history", "")
            if bull_history and bull_history.strip():
                bull_section = html.Div([
                    html.Div([
                        html.Div([
                            html.Span("🐂 Bull Researcher", className="me-2", style={"fontWeight": "bold", "color": "#10B981"}),
                            create_show_prompt_button("bull_report")
                        ], className="d-flex justify-content-between align-items-center mb-2")
                    ]),
                    dcc.Markdown(
                        bull_history,
                        mathjax=True,
                        highlight_config={"theme": "dark"},
                        dangerously_allow_html=False,
                        className='enhanced-markdown-content',
                        style={
                            "background": "linear-gradient(135deg, #064E3B 0%, #047857 100%)",
                            "border-radius": "8px",
                            "padding": "1rem",
                            "border-left": "4px solid #10B981",
                            "color": "#E2E8F0",
                            "margin-bottom": "1rem"
                        }
                    )
                ])
                debate_components.append(bull_section)
            
            # Add Bear Researcher section if available  
            bear_history = debate_state.get("bear_history", "")
            if bear_history and bear_history.strip():
                bear_section = html.Div([
                    html.Div([
                        html.Div([
                            html.Span("🐻 Bear Researcher", className="me-2", style={"fontWeight": "bold", "color": "#EF4444"}),
                            create_show_prompt_button("bear_report")
                        ], className="d-flex justify-content-between align-items-center mb-2")
                    ]),
                    dcc.Markdown(
                        bear_history,
                        mathjax=True,
                        highlight_config={"theme": "dark"},
                        dangerously_allow_html=False,
                        className='enhanced-markdown-content',
                        style={
                            "background": "linear-gradient(135deg, #7F1D1D 0%, #B91C1C 100%)",
                            "border-radius": "8px",
                            "padding": "1rem",
                            "border-left": "4px solid #EF4444",
                            "color": "#E2E8F0",
                            "margin-bottom": "1rem"
                        }
                    )
                ])
                debate_components.append(bear_section)
        
        if not debate_components:
            return create_markdown_content("", "Researcher debate will begin once analysis starts.")
        
        return html.Div(
            debate_components,
            style={
                "background": "linear-gradient(135deg, #0F172A 0%, #1E293B 100%)",
                "border-radius": "8px",
                "padding": "1.5rem",
                "min-height": "1000px",
                "maxHeight": "600px",
                "overflowY": "auto"
            }
        )

    @app.callback(
        Output("risk-debate-tab-content", "children"),
        [Input("report-pagination", "active_page"),
         Input("medium-refresh-interval", "n_intervals")]
    )
    def update_risk_debate(active_page, n_intervals):
        """Update the risk debate tab with Dash components and prompt buttons"""
        if not app_state.symbol_states or not active_page:
            return create_markdown_content("", "No risk debate available yet.")

        # Safeguard against accessing invalid page index (e.g., after page refresh)
        symbols_list = list(app_state.symbol_states.keys())
        if active_page > len(symbols_list):
            return create_markdown_content("", "Page index out of range. Please refresh or restart analysis.")

        symbol = symbols_list[active_page - 1]
        state = app_state.get_state(symbol)
        
        if not state:
            return create_markdown_content("", f"No active analysis for {symbol}. Risk debate will appear here once analysis starts.")

        # Get the risk debate state
        risk_debate_state = state.get("risk_debate_state")
        
        if not risk_debate_state or not risk_debate_state.get("history"):
            return create_markdown_content("", "Risk debate will begin once analysis starts.")

        debate_components = []
        
        # Get message arrays for proper conversation display
        risky_messages = risk_debate_state.get("risky_messages", [])
        safe_messages = risk_debate_state.get("safe_messages", [])
        neutral_messages = risk_debate_state.get("neutral_messages", [])
        
        # Create conversation-style debate display
        if risky_messages or safe_messages or neutral_messages:
            from webui.components.prompt_modal import create_show_prompt_button
            
            # Interleave messages chronologically based on debate flow
            # Usually: Risky -> Safe -> Neutral -> Risky -> Safe -> Neutral, etc.
            max_messages = max(len(risky_messages), len(safe_messages), len(neutral_messages))
            
            for i in range(max_messages):
                # Add Risky message if available
                if i < len(risky_messages):
                    risky_message = risky_messages[i]
                    # Remove the "Risky Analyst: " prefix for cleaner display
                    clean_risky_message = risky_message.replace("Risky Analyst: ", "")
                    
                    risky_section = html.Div([
                        html.Div([
                            html.Div([
                                html.Span("⚡ Risky Analyst", className="me-2", style={"fontWeight": "bold", "color": "#EF4444"}),
                                create_show_prompt_button("aggressive_report")
                            ], className="d-flex justify-content-between align-items-center mb-2")
                        ]),
                        dcc.Markdown(
                            clean_risky_message,
                            mathjax=True,
                            highlight_config={"theme": "dark"},
                            dangerously_allow_html=False,
                            className='enhanced-markdown-content',
                            style={
                                "background": "linear-gradient(135deg, #7F1D1D 0%, #B91C1C 100%)",
                                "border-radius": "8px",
                                "padding": "1rem",
                                "border-left": "4px solid #EF4444",
                                "color": "#E2E8F0",
                                "margin-bottom": "1rem"
                            }
                        )
                    ])
                    debate_components.append(risky_section)
                
                # Add Safe message if available
                if i < len(safe_messages):
                    safe_message = safe_messages[i]
                    # Remove the "Safe Analyst: " prefix for cleaner display
                    clean_safe_message = safe_message.replace("Safe Analyst: ", "")
                    
                    safe_section = html.Div([
                        html.Div([
                            html.Div([
                                html.Span("🛡️ Safe Analyst", className="me-2", style={"fontWeight": "bold", "color": "#10B981"}),
                                create_show_prompt_button("conservative_report")
                            ], className="d-flex justify-content-between align-items-center mb-2")
                        ]),
                        dcc.Markdown(
                            clean_safe_message,
                            mathjax=True,
                            highlight_config={"theme": "dark"},
                            dangerously_allow_html=False,
                            className='enhanced-markdown-content',
                            style={
                                "background": "linear-gradient(135deg, #064E3B 0%, #047857 100%)",
                                "border-radius": "8px",
                                "padding": "1rem",
                                "border-left": "4px solid #10B981",
                                "color": "#E2E8F0",
                                "margin-bottom": "1rem"
                            }
                        )
                    ])
                    debate_components.append(safe_section)
                
                # Add Neutral message if available
                if i < len(neutral_messages):
                    neutral_message = neutral_messages[i]
                    # Remove the "Neutral Analyst: " prefix for cleaner display
                    clean_neutral_message = neutral_message.replace("Neutral Analyst: ", "")
                    
                    neutral_section = html.Div([
                        html.Div([
                            html.Div([
                                html.Span("⚖️ Neutral Analyst", className="me-2", style={"fontWeight": "bold", "color": "#3B82F6"}),
                                create_show_prompt_button("neutral_report")
                            ], className="d-flex justify-content-between align-items-center mb-2")
                        ]),
                        dcc.Markdown(
                            clean_neutral_message,
                            mathjax=True,
                            highlight_config={"theme": "dark"},
                            dangerously_allow_html=False,
                            className='enhanced-markdown-content',
                            style={
                                "background": "linear-gradient(135deg, #1E3A8A 0%, #1D4ED8 100%)",
                                "border-radius": "8px",
                                "padding": "1rem",
                                "border-left": "4px solid #3B82F6",
                                "color": "#E2E8F0",
                                "margin-bottom": "1rem"
                            }
                        )
                    ])
                    debate_components.append(neutral_section)
        
        # Fallback to old format if new message arrays don't exist
        elif risk_debate_state.get("risky_history") or risk_debate_state.get("safe_history") or risk_debate_state.get("neutral_history"):
            from webui.components.prompt_modal import create_show_prompt_button
            
            # Add Risky/Aggressive section if available
            risky_history = risk_debate_state.get("risky_history", "")
            if risky_history and risky_history.strip():
                risky_section = html.Div([
                    html.Div([
                        html.Div([
                            html.Span("⚡ Risky Analyst", className="me-2", style={"fontWeight": "bold", "color": "#EF4444"}),
                            create_show_prompt_button("aggressive_report")
                        ], className="d-flex justify-content-between align-items-center mb-2")
                    ]),
                    dcc.Markdown(
                        risky_history,
                        mathjax=True,
                        highlight_config={"theme": "dark"},
                        dangerously_allow_html=False,
                        className='enhanced-markdown-content',
                        style={
                            "background": "linear-gradient(135deg, #7F1D1D 0%, #B91C1C 100%)",
                            "border-radius": "8px",
                            "padding": "1rem",
                            "border-left": "4px solid #EF4444",
                            "color": "#E2E8F0",
                            "margin-bottom": "1rem"
                        }
                    )
                ])
                debate_components.append(risky_section)
            
            # Add Safe/Conservative section if available  
            safe_history = risk_debate_state.get("safe_history", "")
            if safe_history and safe_history.strip():
                safe_section = html.Div([
                    html.Div([
                        html.Div([
                            html.Span("🛡️ Safe Analyst", className="me-2", style={"fontWeight": "bold", "color": "#10B981"}),
                            create_show_prompt_button("conservative_report")
                        ], className="d-flex justify-content-between align-items-center mb-2")
                    ]),
                    dcc.Markdown(
                        safe_history,
                        mathjax=True,
                        highlight_config={"theme": "dark"},
                        dangerously_allow_html=False,
                        className='enhanced-markdown-content',
                        style={
                            "background": "linear-gradient(135deg, #064E3B 0%, #047857 100%)",
                            "border-radius": "8px",
                            "padding": "1rem",
                            "border-left": "4px solid #10B981",
                            "color": "#E2E8F0",
                            "margin-bottom": "1rem"
                        }
                    )
                ])
                debate_components.append(safe_section)
            
            # Add Neutral section if available
            neutral_history = risk_debate_state.get("neutral_history", "")
            if neutral_history and neutral_history.strip():
                neutral_section = html.Div([
                    html.Div([
                        html.Div([
                            html.Span("⚖️ Neutral Analyst", className="me-2", style={"fontWeight": "bold", "color": "#3B82F6"}),
                            create_show_prompt_button("neutral_report")
                        ], className="d-flex justify-content-between align-items-center mb-2")
                    ]),
                    dcc.Markdown(
                        neutral_history,
                        mathjax=True,
                        highlight_config={"theme": "dark"},
                        dangerously_allow_html=False,
                        className='enhanced-markdown-content',
                        style={
                            "background": "linear-gradient(135deg, #1E3A8A 0%, #1D4ED8 100%)",
                            "border-radius": "8px",
                            "padding": "1rem",
                            "border-left": "4px solid #3B82F6",
                            "color": "#E2E8F0",
                            "margin-bottom": "1rem"
                        }
                    )
                ])
                debate_components.append(neutral_section)
        
        if not debate_components:
            return create_markdown_content("", "Risk debate will begin once analysis starts.")
        
        return html.Div(
            debate_components,
            style={
                "background": "linear-gradient(135deg, #0F172A 0%, #1E293B 100%)",
                "border-radius": "8px",
                "padding": "1.5rem",
                "min-height": "1000px",
                "maxHeight": "600px",
                "overflowY": "auto"
            }
        )

    @app.callback(
        [Output("market-analysis-tab-content", "children"),
         Output("social-sentiment-tab-content", "children"),
         Output("news-analysis-tab-content", "children"),
         Output("fundamentals-analysis-tab-content", "children"),
         Output("macro-analysis-tab-content", "children"),
         Output("research-manager-tab-content", "children"),
         Output("trader-plan-tab-content", "children"),
         Output("final-decision-tab-content", "children")],
        [Input("report-pagination", "active_page"),
         Input("medium-refresh-interval", "n_intervals")]
    )
    def update_tabs_content(active_page, n_intervals):
        """Update the content of all tabs with validation to ensure complete reports"""
        # print(f"[REPORTS] Called with active_page={active_page}, symbol_states={list(app_state.symbol_states.keys()) if app_state.symbol_states else []}")
        
        if not app_state.symbol_states or not active_page:
            # print(f"[REPORTS] No symbol states or no active page, returning default content")
            return [create_markdown_content("", "No analysis available yet.")] * 8
        
        # Safeguard against accessing invalid page index (e.g., after page refresh)
        symbols_list = list(app_state.symbol_states.keys())
        if active_page > len(symbols_list):
            return [create_markdown_content("", "Page index out of range. Please refresh or restart analysis.")] * 8
        
        symbol = symbols_list[active_page - 1]
        # print(f"[REPORTS] Selected symbol: {symbol} (page {active_page})")
        state = app_state.get_state(symbol)
        
        if not state:
            return [create_markdown_content("", "No data for this symbol.")] * 8
            
        reports = state["current_reports"]
        agent_statuses = state["agent_statuses"]
        
        # 🛡️ VALIDATION: Only show complete reports in UI
        # For analysts marked as "completed", validate reports are actually complete
        analyst_reports = {
            "market_report": reports.get("market_report"),
            "sentiment_report": reports.get("sentiment_report"), 
            "news_report": reports.get("news_report"),
            "fundamentals_report": reports.get("fundamentals_report"),
            "macro_report": reports.get("macro_report")
        }
        
        # Check which analysts are completed
        analyst_status_map = {
            "market_report": agent_statuses.get("Market Analyst"),
            "sentiment_report": agent_statuses.get("Social Analyst"),
            "news_report": agent_statuses.get("News Analyst"), 
            "fundamentals_report": agent_statuses.get("Fundamentals Analyst"),
            "macro_report": agent_statuses.get("Macro Analyst")
        }
        
        # 🛡️ PRIORITY: Analyst status takes precedence over content validation
        # If analyst is completed, always show the report regardless of content validation
        validated_reports = {}
        
        for report_type, content in analyst_reports.items():
            status = analyst_status_map.get(report_type)
            
            if status == "completed" and content:
                # Analyst is done - show the final report
                validated_reports[report_type] = content
            elif status == "in_progress":
                validated_reports[report_type] = f"🔄 {report_type.replace('_', ' ').title()} - Analysis in progress..."
            elif status == "pending":
                validated_reports[report_type] = f"⏳ {report_type.replace('_', ' ').title()} - Waiting to start..."
            elif content:
                # Analyst status unknown but we have content - validate it
                content_validated = validate_reports_for_ui({report_type: content})
                validated_reports[report_type] = content_validated[report_type]
            else:
                validated_reports[report_type] = f"No {report_type.replace('_', ' ').title()} available yet."
        
        # Get final validated reports or defaults
        market_report = validated_reports.get("market_report", "No market analysis available yet.")
        sentiment_report = validated_reports.get("sentiment_report", "No sentiment analysis available yet.")
        news_report = validated_reports.get("news_report", "No news analysis available yet.")
        fundamentals_report = validated_reports.get("fundamentals_report", "No fundamentals analysis available yet.")
        macro_report = validated_reports.get("macro_report", "No macro analysis available yet.")
        
        # Research team reports (no validation needed - these come as complete chunks)
        research_manager_report = reports.get("research_manager_report") or "No research manager decision available yet."
        trader_report = reports.get("trader_investment_plan") or "No trader report available yet."
        
        # Final Decision tab shows the Portfolio Manager Decision
        portfolio_report = reports.get("final_trade_decision") or "No final decision available yet."
        
        return (
            create_markdown_content(market_report, "No market analysis available yet.", "market_report"),
            create_markdown_content(sentiment_report, "No sentiment analysis available yet.", "sentiment_report"),
            create_markdown_content(news_report, "No news analysis available yet.", "news_report"),
            create_markdown_content(fundamentals_report, "No fundamentals analysis available yet.", "fundamentals_report"),
            create_markdown_content(macro_report, "No macro analysis available yet.", "macro_report"),
            create_markdown_content(research_manager_report, "No research manager decision available yet.", "research_manager_report"),
            create_markdown_content(trader_report, "No trader report available yet.", "trader_investment_plan"),
            create_markdown_content(portfolio_report, "No final decision available yet.", "final_trade_decision")
        )

    @app.callback(
        Output("decision-summary", "children"),
        [Input("report-pagination", "active_page"),
         Input("medium-refresh-interval", "n_intervals")]
    )
    def update_decision_summary(active_page, n_intervals):
        """Update the decision summary"""
        if not app_state.symbol_states or not active_page:
            return "Analysis not complete yet."

        # Safeguard against accessing invalid page index (e.g., after page refresh)
        symbols_list = list(app_state.symbol_states.keys())
        if active_page > len(symbols_list):
            return "Page index out of range. Please refresh or restart analysis."

        symbol = symbols_list[active_page - 1]
        state = app_state.get_state(symbol)

        if not state:
            return "No data for this symbol."

        reports = state["current_reports"]
        final_report_content = reports.get("final_trade_decision")

        # A race condition can occur where the final report is generated but the analysis_complete flag is not yet set.
        # We should only show the final decision when the state is confirmed as complete.
        if state.get("analysis_complete") and final_report_content is not None:
            if state["analysis_results"]:
                decision_text = f"## Final Decision for {state['ticker_symbol']}\n\n"
                decision_text += f"**Trade Action:** {state['analysis_results'].get('decision', 'No decision')}\n\n"
                decision_text += "**Date:** " + state['analysis_results'].get("date", "N/A")
            else:
                # Show the recommended action if available
                decision_text = f"## Final Decision for {state['ticker_symbol']}\n\n"
                
                # Display the extracted recommendation prominently
                if "recommended_action" in state and state["recommended_action"]:
                    decision_text += f"**📈 RECOMMENDED ACTION: {state['recommended_action']}**\n\n"
                
                decision_text += "**Full Analysis:**\n"
                decision_text += final_report_content
        else:
            # Show partial decision summary based on available reports
            available_reports = []
            if state["current_reports"].get("market_report"):
                available_reports.append("Market Analysis")
            if state["current_reports"].get("sentiment_report"):
                available_reports.append("Social Media Sentiment")
            if state["current_reports"].get("news_report"):
                available_reports.append("News Analysis")
            if state["current_reports"].get("fundamentals_report"):
                available_reports.append("Fundamentals Analysis")
            if state["current_reports"].get("macro_report"):
                available_reports.append("Macro Analysis")
            if state["current_reports"].get("research_manager_report"):
                available_reports.append("Research Manager Decision")
            if state["current_reports"].get("trader_investment_plan"):
                available_reports.append("Trader Investment Plan")
            if state.get("risk_debate_state", {}).get("history"):
                available_reports.append("Risk Debate")
            if final_report_content is not None:
                available_reports.append("Portfolio Manager Final Decision")
            
            if available_reports:
                decision_text = f"## Partial Analysis for {state['ticker_symbol']}\n\n"
                decision_text += "**Completed Reports:** " + ", ".join(available_reports) + "\n\n"
                
                # Show the latest available decision as "Current Decision"
                risk_debate_latest = ""
                if state.get("risk_debate_state", {}).get("history"):
                    # Get the last message from risk debate
                    risk_history = state["risk_debate_state"]["history"]
                    if risk_history:
                        risk_debate_latest = risk_history.split('\n')[-1] if risk_history else ""
                
                current_decision = (
                    final_report_content or
                    risk_debate_latest or
                    state["current_reports"].get("trader_investment_plan") or
                    state["current_reports"].get("research_manager_report")
                )
                
                if current_decision:
                    decision_text += "**Current Decision:** Based on completed analysis\n\n"
                    decision_text += current_decision
            else:
                decision_text = "Analysis not complete yet."
        
        return decision_text

    @app.callback(
        Output("current-symbol-report-display", "children"),
        [Input("report-pagination", "active_page")]
    )
    def update_report_display_text(active_page):
        if not app_state.symbol_states or not active_page:
            return ""
        
        # Safeguard against accessing invalid page index (e.g., after page refresh)
        symbols_list = list(app_state.symbol_states.keys())
        if active_page > len(symbols_list):
            return "Invalid page"
        
        symbol = symbols_list[active_page - 1]
        return f"📊 {symbol}"

    @app.callback(
        Output("tabs", "active_tab"),
        [Input("nav-market", "n_clicks"),
         Input("nav-social", "n_clicks"),
         Input("nav-news", "n_clicks"),
         Input("nav-fundamentals", "n_clicks"),
         Input("nav-researcher", "n_clicks"),
         Input("nav-research-mgr", "n_clicks"),
         Input("nav-trader", "n_clicks"),
         Input("nav-risk-agg", "n_clicks"),
         Input("nav-risk-cons", "n_clicks"),
         Input("nav-risk-neut", "n_clicks"),
         Input("nav-final", "n_clicks")]
    )
    def switch_tab(*args):
        """Switch between tabs based on navigation clicks"""
        ctx = dash.callback_context
        if not ctx.triggered:
            return "market-analysis"
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        tab_mapping = {
            "nav-market": "market-analysis",
            "nav-social": "social-sentiment", 
            "nav-news": "news-analysis",
            "nav-fundamentals": "fundamentals-analysis",
            "nav-researcher": "researcher-debate",
            "nav-research-mgr": "research-manager",
            "nav-trader": "trader-plan",
            "nav-risk-agg": "risk-debate",
            "nav-risk-cons": "risk-debate", 
            "nav-risk-neut": "risk-debate",
            "nav-final": "final-decision"
        }
        
        return tab_mapping.get(trigger_id, "market-analysis")

    # Prompt Modal Callbacks
    @app.callback(
        [Output("prompt-modal", "is_open"),
         Output("prompt-modal-title", "children"), 
         Output("prompt-modal-content", "children"),
         Output("global-prompt-modal-state", "data")],
        [Input({"type": "show-prompt-btn", "report": ALL}, "n_clicks"),
         Input("close-prompt-modal-btn", "n_clicks")],
        [State("global-prompt-modal-state", "data")],
        prevent_initial_call=True
    )
    def handle_prompt_modal(show_clicks, close_clicks, modal_state):
        """Handle opening and closing the prompt modal - returns no_update when nothing to do"""
        if not ctx.triggered:
            # Nothing triggered → keep existing state/content untouched
            return dash.no_update, dash.no_update, dash.no_update, modal_state
        
        trigger_id = ctx.triggered[0]['prop_id']
        
        # Close modal
        if "close-prompt-modal-btn" in trigger_id:
            updated_state = {
                "is_open": False,
                "report_type": None,
                "title": "Agent Prompt"
            }
            return False, "Agent Prompt", "No prompt selected", updated_state
        
        # Open modal with specific prompt
        if "show-prompt-btn" in trigger_id and any(show_clicks):
            # Find which button was clicked
            import json
            
            # Extract the report type from the button that was clicked
            for i, clicks in enumerate(show_clicks):
                if clicks:
                    # Get the report type from the pattern match
                    button_data = json.loads(trigger_id.split('.')[0])
                    report_type = button_data.get("report")
                    
                    if report_type:
                        # Get current symbol
                        current_symbol = app_state.current_symbol
                        
                        # Get the prompt for this report type
                        prompt_content = get_agent_prompt(report_type, current_symbol)
                        
                        # Create a nice title
                        report_titles = {
                            "market_report": "Market Analyst Prompt",
                            "sentiment_report": "Social Media Analyst Prompt", 
                            "news_report": "News Analyst Prompt",
                            "fundamentals_report": "Fundamentals Analyst Prompt",
                            "macro_report": "Macro Analyst Prompt",
                            "bull_report": "Bull Researcher Prompt",
                            "bear_report": "Bear Researcher Prompt",
                            "research_manager_report": "Research Manager Prompt",
                            "trader_investment_plan": "Trader Prompt"
                        }
                        
                        title = report_titles.get(report_type, f"{report_type.replace('_', ' ').title()} Prompt")
                        
                        # Update modal state
                        updated_state = {
                            "is_open": True,
                            "report_type": report_type,
                            "title": title
                        }
                        
                        return True, title, f"```\n{prompt_content}\n```", updated_state
        
        # Fallback - keep everything unchanged
        return dash.no_update, dash.no_update, dash.no_update, modal_state or {"is_open": False, "report_type": None, "title": "Agent Prompt"}

    @app.callback(
        Output("copy-prompt-btn", "children"),
        [Input("copy-prompt-btn", "n_clicks")],
        prevent_initial_call=True
    )
    def copy_prompt_to_clipboard(n_clicks):
        """Handle copying prompt to clipboard (visual feedback only)"""
        if n_clicks:
            # In a real implementation, you'd use clientside callback or JavaScript
            # For now, just provide visual feedback
            return [
                html.I(className="fas fa-check me-2"),
                "Copied!"
            ]
        return [
            html.I(className="fas fa-copy me-2"),
            "Copy to Clipboard"
        ]

    # Tool Outputs Modal Callbacks
    @app.callback(
        [Output("tool-outputs-modal", "is_open"),
         Output("tool-outputs-modal-title", "children"),
         Output("tool-outputs-modal-content", "children"),
         Output("global-tool-outputs-modal-state", "data")],
        [Input({"type": "show-tool-outputs-btn", "report": ALL}, "n_clicks"),
         Input("close-tool-outputs-modal-btn", "n_clicks")],
        [State("global-tool-outputs-modal-state", "data")],
        prevent_initial_call=True
    )
    def handle_tool_outputs_modal(show_clicks, close_clicks, modal_state):
        """Handle opening and closing the tool outputs modal - returns no_update when nothing to do"""
        if not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update, modal_state
        
        trigger_id = ctx.triggered[0]['prop_id']
        
        # Close modal
        if "close-tool-outputs-modal-btn" in trigger_id:
            updated_state = {
                "is_open": False,
                "report_type": None,
                "title": "Tool Outputs"
            }
            return False, "Tool Outputs", "No tool outputs available", updated_state
        
        # Open modal with tool outputs for specific report
        if "show-tool-outputs-btn" in trigger_id and any(show_clicks):
            # Find which button was clicked
            import json
            
            # Extract the report type from the button that was clicked
            for i, clicks in enumerate(show_clicks):
                if clicks:
                    # Get the report type from the pattern match
                    button_data = json.loads(trigger_id.split('.')[0])
                    report_type = button_data.get("report")
                    
                    if report_type:
                        from webui.components.tool_outputs_modal import format_tool_outputs_content
                        
                        # Get current symbol for filtering
                        current_symbol = app_state.current_symbol
                        
                        # Get tool calls from app state filtered by agent type and symbol
                        tool_calls = app_state.get_tool_calls_for_display(agent_filter=report_type, symbol_filter=current_symbol)
                        formatted_content = format_tool_outputs_content(tool_calls, report_type)
                        
                        # Create a nice title
                        report_titles = {
                            "market_report": "Market Analyst Tool Outputs",
                            "sentiment_report": "Social Media Analyst Tool Outputs", 
                            "news_report": "News Analyst Tool Outputs",
                            "fundamentals_report": "Fundamentals Analyst Tool Outputs",
                            "macro_report": "Macro Analyst Tool Outputs",
                            "bull_report": "Bull Researcher Tool Outputs",
                            "bear_report": "Bear Researcher Tool Outputs",
                            "research_manager_report": "Research Manager Tool Outputs",
                            "trader_investment_plan": "Trader Tool Outputs",
                            "final_trade_decision": "Portfolio Manager Tool Outputs"
                        }
                        
                        title = report_titles.get(report_type, f"{report_type.replace('_', ' ').title()} Tool Outputs")
                        
                        # Create the content with markdown rendering
                        content = dcc.Markdown(
                            formatted_content,
                            highlight_config={"theme": "dark"},
                            style={
                                "white-space": "pre-wrap",
                                "color": "#F8FAFC"  # Lighter text for better readability
                            }
                        )
                        
                        # Update modal state
                        updated_state = {
                            "is_open": True,
                            "report_type": report_type,
                            "title": title
                        }
                        
                        return True, title, content, updated_state
        
        # Fallback - keep everything unchanged
        return dash.no_update, dash.no_update, dash.no_update, modal_state or {"is_open": False, "report_type": None, "title": "Tool Outputs"}

    @app.callback(
        Output("copy-tool-outputs-btn", "children"),
        [Input("copy-tool-outputs-btn", "n_clicks")],
        prevent_initial_call=True
    )
    def copy_tool_outputs_to_clipboard(n_clicks):
        """Handle copying tool outputs to clipboard (visual feedback only)"""
        if n_clicks:
            return [
                html.I(className="fas fa-check me-2"),
                "Copied!"
            ]
        return [
            html.I(className="fas fa-copy me-2"),
            "Copy All"
        ]

    @app.callback(
        Output("export-tool-outputs-btn", "children"),
        [Input("export-tool-outputs-btn", "n_clicks")],
        prevent_initial_call=True
    )
    def export_tool_outputs_as_json(n_clicks):
        """Handle exporting tool outputs as JSON (visual feedback only)"""
        if n_clicks:
            # In a real implementation, this would trigger a download
            return [
                html.I(className="fas fa-check me-2"),
                "Exported!"
            ]
        return [
            html.I(className="fas fa-download me-2"),
            "Export JSON"
        ] 
