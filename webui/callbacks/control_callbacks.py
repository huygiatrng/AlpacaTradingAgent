"""
Control and configuration callbacks for TradingAgents WebUI
"""

from dash import Input, Output, State, ctx, html
import dash_bootstrap_components as dbc
import dash
import threading
import time

from webui.utils.state import app_state
from webui.components.analysis import start_analysis


def register_control_callbacks(app):
    """Register all control and configuration callbacks"""

    @app.callback(
        Output("research-depth-info", "children"),
        [Input("research-depth", "value")]
    )
    def update_research_depth_info(selected_depth):
        """Update the research depth information display based on selection"""
        if not selected_depth:
            return ""
        
        research_depth_info = {
            "Shallow": {
                "description": "Quick research, few debate and strategy discussion rounds",
                "settings": [
                    "max_debate_rounds: 1",
                    "max_risk_discuss_rounds: 1"
                ],
                "use_case": "Fast analysis when you need quick results and don't require extensive deliberation between agents",
                "header_color": "#17a2b8",  # info blue
                "bg_color": "#d1ecf1"
            },
            "Medium": {
                "description": "Middle ground, moderate debate rounds and strategy discussion",
                "settings": [
                    "max_debate_rounds: 3",
                    "max_risk_discuss_rounds: 3"
                ],
                "use_case": "Balanced approach providing reasonable depth while maintaining efficiency",
                "header_color": "#ffc107",  # warning yellow
                "bg_color": "#fff3cd"
            },
            "Deep": {
                "description": "Comprehensive research, in depth debate and strategy discussion",
                "settings": [
                    "max_debate_rounds: 5",
                    "max_risk_discuss_rounds: 5"
                ],
                "use_case": "Most thorough analysis with extensive agent debates and risk discussions",
                "header_color": "#28a745",  # success green
                "bg_color": "#d4edda"
            }
        }
        
        info = research_depth_info.get(selected_depth, {})
        if not info:
            return ""
        
        return dbc.Card([
            dbc.CardHeader([
                html.H6(f"{selected_depth} Mode", 
                       className="mb-0", 
                       style={"fontWeight": "bold", "color": "white"})
            ], style={"backgroundColor": info["header_color"], "border": "none"}),
            dbc.CardBody([
                html.P([
                    html.Strong("Description: ", style={"color": "black"}), 
                    html.Span(info["description"], style={"color": "black"})
                ], className="mb-2"),
                html.P([
                    html.Strong("Settings:", style={"color": "black"}),
                    html.Ul([
                        html.Li(setting, style={"color": "black"}) for setting in info["settings"]
                    ], className="mb-1")
                ], className="mb-2"),
                html.P([
                    html.Strong("Use Case: ", style={"color": "black"}),
                    html.Span(info["use_case"], style={"color": "black"})
                ], className="mb-0")
            ], style={"backgroundColor": info["bg_color"]})
        ])

    @app.callback(
        Output("market-hours-validation", "children"),
        [Input("market-hours-input", "value")]
    )
    def validate_market_hours_input(hours_input):
        """Validate market hours input and show validation message"""
        if not hours_input or not hours_input.strip():
            return ""
        
        from webui.utils.market_hours import validate_market_hours
        
        is_valid, hours, error_msg = validate_market_hours(hours_input)
        
        if not is_valid:
            return dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                error_msg
            ], color="danger", className="mb-2")
        else:
            # Format hours for display
            formatted_hours = []
            for hour in hours:
                if hour < 12:
                    formatted_hours.append(f"{hour}:00 AM")
                else:
                    formatted_hours.append(f"{hour-12}:00 PM" if hour > 12 else "12:00 PM")
            
            hours_str = " and ".join(formatted_hours)
            return dbc.Alert([
                html.I(className="fas fa-check-circle me-2"),
                f"Valid trading hours: {hours_str} EST/EDT"
            ], color="success", className="mb-2")

    @app.callback(
        [Output("loop-enabled", "value"),
         Output("market-hour-enabled", "value"),
         Output("loop-interval", "disabled"),
         Output("market-hours-input", "disabled")],
        [Input("loop-enabled", "value"),
         Input("market-hour-enabled", "value")],
        prevent_initial_call=True
    )
    def mutual_exclusive_scheduling_modes(loop_enabled, market_hour_enabled):
        """Ensure only one scheduling mode can be enabled at a time"""
        ctx = dash.callback_context
        if not ctx.triggered:
            return loop_enabled, market_hour_enabled, False, False
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == "loop-enabled" and loop_enabled:
            # Loop mode was enabled, disable market hour mode
            return True, False, False, True
        elif trigger_id == "market-hour-enabled" and market_hour_enabled:
            # Market hour mode was enabled, disable loop mode
            return False, True, True, False
        else:
            # Either mode was disabled, enable both inputs
            return loop_enabled, market_hour_enabled, not loop_enabled, not market_hour_enabled

    @app.callback(
        Output("scheduling-mode-info", "children"),
        [Input("loop-enabled", "value"),
         Input("loop-interval", "value"),
         Input("market-hour-enabled", "value"),
         Input("market-hours-input", "value")]
    )
    def update_scheduling_mode_info(loop_enabled, loop_interval, market_hour_enabled, market_hours_input):
        """Update the scheduling mode information display based on settings"""
        if market_hour_enabled:
            # Market Hour Mode
            from webui.utils.market_hours import validate_market_hours, format_market_hours_info
            
            if not market_hours_input or not market_hours_input.strip():
                return dbc.Card([
                    dbc.CardHeader([
                        html.H6("Market Hour Mode - Incomplete", 
                               className="mb-0", 
                               style={"fontWeight": "bold", "color": "white"})
                    ], style={"backgroundColor": "#dc3545", "border": "none"}),
                    dbc.CardBody([
                        html.P([
                            html.Strong("Status: ", style={"color": "black"}), 
                            html.Span("Please enter trading hours to activate", style={"color": "black"})
                        ], className="mb-0")
                    ], style={"backgroundColor": "#f8d7da"})
                ])
            
            is_valid, hours, error_msg = validate_market_hours(market_hours_input)
            
            if not is_valid:
                return dbc.Card([
                    dbc.CardHeader([
                        html.H6("Market Hour Mode - Invalid Hours", 
                               className="mb-0", 
                               style={"fontWeight": "bold", "color": "white"})
                    ], style={"backgroundColor": "#dc3545", "border": "none"}),
                    dbc.CardBody([
                        html.P([
                            html.Strong("Error: ", style={"color": "black"}), 
                            html.Span(error_msg, style={"color": "black"})
                        ], className="mb-0")
                    ], style={"backgroundColor": "#f8d7da"})
                ])
            
            # Valid market hours
            hours_info = format_market_hours_info(hours)
            
            next_executions = []
            for exec_info in hours_info["next_executions"]:
                next_executions.append(f"Next {exec_info['formatted_hour']}: {exec_info['next_formatted']}")
            
            return dbc.Card([
                dbc.CardHeader([
                    html.H6("Market Hour Mode Enabled", 
                           className="mb-0", 
                           style={"fontWeight": "bold", "color": "white"})
                ], style={"backgroundColor": "#28a745", "border": "none"}),
                dbc.CardBody([
                    html.P([
                        html.Strong("Trading Hours: ", style={"color": "black"}), 
                        html.Span(hours_info["formatted_hours"], style={"color": "black"})
                    ], className="mb-2"),
                    html.P([
                        html.Strong("Behavior:", style={"color": "black"}),
                        html.Ul([
                            html.Li("Wait for market hours and market open status", style={"color": "black"}),
                            html.Li("Check holidays and weekends automatically", style={"color": "black"}),
                            html.Li("Run analysis at specified times daily", style={"color": "black"}),
                            html.Li("Continue until manually stopped", style={"color": "black"})
                        ], className="mb-1")
                    ], className="mb-2"),
                    html.P([
                        html.Strong("Next Executions:", style={"color": "black"}),
                        html.Ul([
                            html.Li(exec_str, style={"color": "black"}) for exec_str in next_executions
                        ], className="mb-1")
                    ], className="mb-0")
                ], style={"backgroundColor": "#d4edda"})
            ])
        
        elif loop_enabled:
            # Loop Mode
            interval = loop_interval if loop_interval and loop_interval > 0 else 60
            
            return dbc.Card([
                dbc.CardHeader([
                    html.H6("Loop Mode Enabled", 
                           className="mb-0", 
                           style={"fontWeight": "bold", "color": "white"})
                ], style={"backgroundColor": "#fd7e14", "border": "none"}),
                dbc.CardBody([
                    html.P([
                        html.Strong("Description: ", style={"color": "black"}), 
                        html.Span(f"Analysis will run continuously, restarting every {interval} minutes", style={"color": "black"})
                    ], className="mb-2"),
                    html.P([
                        html.Strong("Behavior:", style={"color": "black"}),
                        html.Ul([
                            html.Li("Process all symbols sequentially", style={"color": "black"}),
                            html.Li(f"Wait {interval} minutes after completion", style={"color": "black"}),
                            html.Li("Clear previous results and restart analysis", style={"color": "black"}),
                            html.Li("Continue until manually stopped", style={"color": "black"})
                        ], className="mb-1")
                    ], className="mb-2"),
                    html.P([
                        html.Strong("Note: ", style={"color": "black"}),
                        html.Span("Use 'Stop Analysis' button to terminate the loop", style={"color": "black"})
                    ], className="mb-0")
                ], style={"backgroundColor": "#fff3cd"})
            ])
        
        else:
            # Single Run Mode
            return dbc.Card([
                dbc.CardHeader([
                    html.H6("Single Run Mode", 
                           className="mb-0", 
                           style={"fontWeight": "bold", "color": "white"})
                ], style={"backgroundColor": "#6c757d", "border": "none"}),
                dbc.CardBody([
                    html.P([
                        html.Strong("Description: ", style={"color": "black"}), 
                        html.Span("Analysis will run once for all symbols and then stop", style={"color": "black"})
                    ], className="mb-2"),
                    html.P([
                        html.Strong("Behavior:", style={"color": "black"}),
                        html.Ul([
                            html.Li("Process all symbols sequentially", style={"color": "black"}),
                            html.Li("Stop after completion", style={"color": "black"}),
                            html.Li("Manual restart required for new analysis", style={"color": "black"})
                        ], className="mb-1")
                    ], className="mb-0")
                ], style={"backgroundColor": "#f8f9fa"})
            ])

    @app.callback(
        Output("control-button-container", "children"),
        [Input("refresh-interval", "n_intervals")]
    )
    def update_control_button(n_intervals):
        """Update the control button (Start/Stop) based on current state"""
        if app_state.analysis_running or app_state.loop_enabled or app_state.market_hour_enabled:
            return dbc.Button(
                "Stop Analysis",
                id="control-btn",
                color="danger",
                size="lg",
                className="w-100 mt-2"
            )
        else:
            return dbc.Button(
                "Start Analysis",
                id="control-btn",
                color="primary",
                size="lg",
                className="w-100 mt-2"
            )

    @app.callback(
        Output("trading-mode-info", "children"),
        [Input("allow-shorts", "value")]
    )
    def update_trading_mode_info(allow_shorts):
        """Update the trading mode information display based on allow shorts selection"""
        if allow_shorts is None:
            return ""
        
        if allow_shorts:
            # Short selling enabled
            info = {
                "title": "Short Trading Enabled",
                "description": "The system can recommend both long and short positions",
                "details": [
                    "Can profit from both rising and falling markets",
                    "Increased trading opportunities",
                    "Higher risk and complexity",
                    "Requires margin account with broker"
                ],
                "note": "Short selling involves borrowing shares to sell, hoping to buy back at lower prices",
                "header_color": "#dc3545",  # danger red
                "bg_color": "#f8d7da"
            }
        else:
            # Long-only mode
            info = {
                "title": "Long-Only Mode",
                "description": "The system will only recommend long (buy) positions",
                "details": [
                    "Only profits from rising markets",
                    "Lower complexity and risk",
                    "No margin requirements",
                    "Suitable for conservative investors"
                ],
                "note": "Traditional buy-and-hold approach focusing on asset appreciation",
                "header_color": "#28a745",  # success green
                "bg_color": "#d4edda"
            }
        
        return dbc.Card([
            dbc.CardHeader([
                html.H6(info["title"], 
                       className="mb-0", 
                       style={"fontWeight": "bold", "color": "white"})
            ], style={"backgroundColor": info["header_color"], "border": "none"}),
            dbc.CardBody([
                html.P([
                    html.Strong("Description: ", style={"color": "black"}), 
                    html.Span(info["description"], style={"color": "black"})
                ], className="mb-2"),
                html.P([
                    html.Strong("Features:", style={"color": "black"}),
                    html.Ul([
                        html.Li(detail, style={"color": "black"}) for detail in info["details"]
                    ], className="mb-1")
                ], className="mb-2"),
                html.P([
                    html.Strong("Note: ", style={"color": "black"}),
                    html.Span(info["note"], style={"color": "black"})
                ], className="mb-0")
            ], style={"backgroundColor": info["bg_color"]})
        ])

    @app.callback(
        Output("trade-after-analyze-info", "children"),
        [Input("trade-after-analyze", "value"),
         Input("trade-dollar-amount", "value")]
    )
    def update_trade_after_analyze_info(trade_enabled, dollar_amount):
        """Update the trade after analyze information display"""
        if not trade_enabled:
            return dbc.Card([
                dbc.CardHeader([
                    html.H6("Manual Trading Mode", 
                           className="mb-0", 
                           style={"fontWeight": "bold", "color": "white"})
                ], style={"backgroundColor": "#6c757d", "border": "none"}),
                dbc.CardBody([
                    html.P([
                        html.Strong("Description: ", style={"color": "black"}), 
                        html.Span("Analysis results will be shown for manual review and trading decisions", style={"color": "black"})
                    ], className="mb-2"),
                    html.P([
                        html.Strong("Behavior:", style={"color": "black"}),
                        html.Ul([
                            html.Li("No automatic orders will be placed", style={"color": "black"}),
                            html.Li("Review analysis results manually", style={"color": "black"}),
                            html.Li("Execute trades manually through broker", style={"color": "black"})
                        ], className="mb-1")
                    ], className="mb-0")
                ], style={"backgroundColor": "#f8f9fa"})
            ])
        
        amount = dollar_amount if dollar_amount and dollar_amount > 0 else 1000
        
        return dbc.Card([
            dbc.CardHeader([
                html.H6("Automated Trading Enabled", 
                       className="mb-0", 
                       style={"fontWeight": "bold", "color": "white"})
            ], style={"backgroundColor": "#fd7e14", "border": "none"}),
            dbc.CardBody([
                html.P([
                    html.Strong("Description: ", style={"color": "black"}), 
                    html.Span(f"System will automatically execute trades with ${amount:.2f} per order", style={"color": "black"})
                ], className="mb-2"),
                html.P([
                    html.Strong("Behavior:", style={"color": "black"}),
                    html.Ul([
                        html.Li("Execute trades automatically after analysis", style={"color": "black"}),
                        html.Li("Use fractional shares based on dollar amount", style={"color": "black"}),
                        html.Li("Follow position management rules", style={"color": "black"}),
                        html.Li("All trades execute via your configured Alpaca account", style={"color": "black"})
                    ], className="mb-1")
                ], className="mb-2"),
                html.P([
                    html.Strong("Warning: ", style={"color": "black"}),
                    html.Span("Ensure Alpaca API keys are configured (paper or live trading mode)", style={"color": "black"})
                ], className="mb-0")
            ], style={"backgroundColor": "#fff3cd"})
        ])

    # Major callback for analysis control
    @app.callback(
        [Output("result-text", "children"),
         Output("app-store", "data"),
         Output("chart-pagination", "max_value"),
         Output("chart-pagination", "active_page"),
         Output("report-pagination", "max_value"),
         Output("report-pagination", "active_page")],
        [Input("control-btn", "n_clicks"),
         Input("control-btn", "children")],
        [State("ticker-input", "value"),
         State("analyst-market", "value"),
         State("analyst-social", "value"),
         State("analyst-news", "value"),
         State("analyst-fundamentals", "value"),
         State("analyst-macro", "value"),
         State("research-depth", "value"),
         State("quick-llm", "value"),
         State("deep-llm", "value"),
         State("allow-shorts", "value"),
         State("loop-enabled", "value"),
         State("loop-interval", "value"),
         State("trade-after-analyze", "value"),
         State("trade-dollar-amount", "value"),
         State("market-hour-enabled", "value"),
         State("market-hours-input", "value")]
    )
    def on_control_button_click(n_clicks, button_children, tickers, analysts_market, analysts_social, analysts_news, 
                               analysts_fundamentals, analysts_macro, research_depth, quick_llm, deep_llm, 
                               allow_shorts, loop_enabled, loop_interval, trade_enabled, trade_amount,
                               market_hour_enabled, market_hours_input):
        """Handle control button clicks"""
        # Detect which property triggered this callback
        triggered_prop = None
        if dash.callback_context.triggered:
            triggered_prop = dash.callback_context.triggered[0]['prop_id']

        # If the callback was invoked solely because the button *label* changed, ignore it
        if triggered_prop == "control-btn.children":
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        # Ignore callbacks caused by the periodic re-rendering of the button itself
        if triggered_prop == "control-btn.n_clicks" and (n_clicks is None or n_clicks == 0):
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        # Real user click handling begins here
        if n_clicks is None:
            return "", {}, 1, 1, 1, 1
        
        # Always use current/real-time data for analysis
        from datetime import datetime
        
        # Determine action based on current state
        is_stop_action = app_state.analysis_running or app_state.loop_enabled or app_state.market_hour_enabled
        
        # Handle stop action
        if is_stop_action:
            if app_state.loop_enabled:
                app_state.stop_loop_mode()
                return "Loop analysis stopped.", dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            elif app_state.market_hour_enabled:
                app_state.stop_market_hour_mode()
                return "Market hour analysis stopped.", dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            else:
                app_state.analysis_running = False
                return "Analysis stopped.", dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        # Handle start action
        if app_state.analysis_running:
            return "Analysis already in progress. Please wait.", dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        symbols = [s.strip().upper() for s in tickers.split(',') if s.strip()]
        if not symbols:
            return "Please enter at least one stock symbol.", {}, 1, 1, 1, 1

        if not app_state.analysis_running:
            app_state.reset()

        # Store selected analysts for the status table
        app_state.active_analysts = []
        if analysts_market: app_state.active_analysts.append("Market Analyst")
        if analysts_social: app_state.active_analysts.append("Social Analyst")
        if analysts_news: app_state.active_analysts.append("News Analyst")
        if analysts_fundamentals: app_state.active_analysts.append("Fundamentals Analyst")
        if analysts_macro: app_state.active_analysts.append("Macro Analyst")

        # Set loop configuration
        app_state.loop_interval_minutes = loop_interval if loop_interval and loop_interval > 0 else 60
        
        # Store trading configuration
        app_state.trade_enabled = trade_enabled
        app_state.trade_amount = trade_amount if trade_amount and trade_amount > 0 else 1000
        
        # Validate market hour configuration if enabled
        if market_hour_enabled:
            from webui.utils.market_hours import validate_market_hours
            is_valid, market_hours_list, error_msg = validate_market_hours(market_hours_input)
            if not is_valid:
                return f"Invalid market hours: {error_msg}", {}, 1, 1, 1, 1
        
        num_symbols = len(symbols)

        # Initialize symbol states IMMEDIATELY so pagination works right away
        for symbol in symbols:
            app_state.init_symbol_state(symbol)

        def analysis_thread():
            if market_hour_enabled:
                # Start market hour mode with scheduling logic
                market_hour_config = {
                    'analysts_market': analysts_market,
                    'analysts_social': analysts_social,
                    'analysts_news': analysts_news,
                    'analysts_fundamentals': analysts_fundamentals,
                    'analysts_macro': analysts_macro,
                    'research_depth': research_depth,
                    'allow_shorts': allow_shorts,
                    'quick_llm': quick_llm,
                    'deep_llm': deep_llm,
                    'trade_enabled': trade_enabled,
                    'trade_amount': trade_amount
                }
                app_state.start_market_hour_mode(symbols, market_hour_config, market_hours_list)
                
                # Market hour scheduling loop
                import datetime
                import pytz
                from webui.utils.market_hours import get_next_market_datetime, is_market_open

                eastern = pytz.timezone('US/Eastern')
                utc = pytz.utc
                
                while not app_state.stop_market_hour:
                    # Compute the schedule from current UTC time projected into Eastern.
                    now = datetime.datetime.now(utc).astimezone(eastern)
                    next_execution_times = []
                    
                    for hour in app_state.market_hours:
                        next_dt = get_next_market_datetime(hour, now)
                        next_execution_times.append((hour, next_dt))
                    
                    # Sort by next execution time
                    next_execution_times.sort(key=lambda x: x[1])
                    next_hour, next_dt = next_execution_times[0]
                    
                    print(f"[MARKET_HOUR] Next execution: {next_dt.strftime('%A, %B %d at %I:%M %p %Z')} (Hour {next_hour})")
                    
                    # Wait until next execution time
                    while datetime.datetime.now(utc).astimezone(eastern) < next_dt and not app_state.stop_market_hour:
                        time.sleep(60)  # Check every minute
                    
                    if app_state.stop_market_hour:
                        break
                    
                    # Check if market is actually open
                    is_open, reason = is_market_open()
                    if not is_open:
                        print(f"[MARKET_HOUR] Market is closed: {reason}. Waiting for next execution time.")
                        continue
                    
                    print(f"[MARKET_HOUR] Market is open, starting analysis at {next_hour}:00")
                    
                    # Reset states for new analysis
                    app_state.reset_for_loop()
                    
                    # Initialize symbol states
                    for symbol in symbols:
                        app_state.init_symbol_state(symbol)
                    
                    # Add symbols to queue and run analysis
                    app_state.add_symbols_to_queue(symbols)
                    
                    while app_state.analysis_queue and not app_state.stop_market_hour:
                        symbol = app_state.get_next_symbol()
                        if symbol:
                            print(f"[MARKET_HOUR] Analyzing {symbol} at {next_hour}:00 with current market data...")
                            start_analysis(
                                symbol,
                                analysts_market, analysts_social, analysts_news, analysts_fundamentals, analysts_macro,
                                research_depth, allow_shorts, quick_llm, deep_llm
                            )
                            
                            if app_state.stop_market_hour:
                                break
                    
                    if not app_state.stop_market_hour:
                        print(f"[MARKET_HOUR] Analysis completed for {next_hour}:00. Waiting for next execution time.")
            
            elif loop_enabled:
                # Start loop mode
                loop_config = {
                    'analysts_market': analysts_market,
                    'analysts_social': analysts_social,
                    'analysts_news': analysts_news,
                    'analysts_fundamentals': analysts_fundamentals,
                    'analysts_macro': analysts_macro,
                    'research_depth': research_depth,
                    'allow_shorts': allow_shorts,
                    'quick_llm': quick_llm,
                    'deep_llm': deep_llm,
                    'trade_enabled': trade_enabled,
                    'trade_amount': trade_amount
                }
                app_state.start_loop(symbols, loop_config)
                
                loop_iteration = 1
                while not app_state.stop_loop:
                    print(f"[LOOP] Starting iteration {loop_iteration}")
                    
                    # States already initialized above, just add to queue
                    app_state.add_symbols_to_queue(symbols)
                    
                    # Run analysis for all symbols
                    while app_state.analysis_queue and not app_state.stop_loop:
                        symbol = app_state.get_next_symbol()
                        if symbol:
                            print(f"[LOOP] Analyzing {symbol} with current market data...")
                            start_analysis(
                                symbol,
                                analysts_market, analysts_social, analysts_news, analysts_fundamentals, analysts_macro,
                                research_depth, allow_shorts, quick_llm, deep_llm
                            )
                    
                    if app_state.stop_loop:
                        break
                    
                    print(f"[LOOP] Iteration {loop_iteration} completed. Waiting {app_state.loop_interval_minutes} minutes...")
                    
                    # Wait for the specified interval (checking for stop every 30 seconds)
                    wait_time = app_state.loop_interval_minutes * 60  # Convert to seconds
                    elapsed = 0
                    while elapsed < wait_time and not app_state.stop_loop:
                        time.sleep(min(30, wait_time - elapsed))
                        elapsed += 30
                    
                    if not app_state.stop_loop:
                        # Reset analysis results for next iteration but keep states for pagination
                        app_state.reset_for_loop()
                        loop_iteration += 1
                
                print("[LOOP] Loop stopped")
            else:
                # Single run mode (original behavior) - use current date
                # States already initialized above, just add to queue
                app_state.add_symbols_to_queue(symbols)

                while app_state.analysis_queue:
                    symbol = app_state.get_next_symbol()
                    if symbol:
                        print(f"[SINGLE] Analyzing {symbol} with current market data...")
                        start_analysis(
                            symbol,
                            analysts_market, analysts_social, analysts_news, analysts_fundamentals, analysts_macro,
                            research_depth, allow_shorts, quick_llm, deep_llm
                        )
            
            app_state.analysis_running = False

        if not app_state.analysis_running:
            app_state.analysis_running = True
            thread = threading.Thread(target=analysis_thread)
            thread.start()
        
        if market_hour_enabled:
            mode_text = "market hour mode"
            # Format hours for display
            formatted_hours = []
            for hour in market_hours_list:
                if hour < 12:
                    formatted_hours.append(f"{hour}:00 AM")
                else:
                    formatted_hours.append(f"{hour-12}:00 PM" if hour > 12 else "12:00 PM")
            interval_text = f" (at {' and '.join(formatted_hours)} EST/EDT)"
        elif loop_enabled:
            mode_text = "loop mode"
            interval_text = f" (every {app_state.loop_interval_minutes} minutes)"
        else:
            mode_text = "single run mode"
            interval_text = ""
        
        # Store symbols and pagination data in app-store for page refresh recovery
        store_data = {
            "analysis_started": True, 
            "timestamp": time.time(),
            "symbols": symbols,  # Store the symbols list
            "num_symbols": num_symbols,  # Store the count
            "mode": mode_text,
            "interval_text": interval_text
        }
        
        return f"Starting real-time analysis for {', '.join(symbols)} in {mode_text}{interval_text} using current market data...", store_data, num_symbols, 1, num_symbols, 1

    @app.callback(
        [Output("chart-pagination", "max_value", allow_duplicate=True),
         Output("chart-pagination", "active_page", allow_duplicate=True), 
         Output("report-pagination", "max_value", allow_duplicate=True),
         Output("report-pagination", "active_page", allow_duplicate=True)],
        [Input("app-store", "data")],
        prevent_initial_call=True
    )
    def restore_pagination_on_refresh(store_data):
        """Restore pagination and symbol states after page refresh"""
        if not store_data or not store_data.get("symbols"):
            # No stored data, return defaults
            print("[RESTORE] No stored data found, returning defaults")
            return 1, 1, 1, 1
        
        symbols = store_data.get("symbols", [])
        num_symbols = len(symbols)
        
        # Restore symbol states if they don't exist (e.g., after page refresh)
        if not app_state.symbol_states or len(app_state.symbol_states) != num_symbols:
            print(f"[RESTORE] Restoring symbol states for {symbols} after page refresh")
            for symbol in symbols:
                if symbol not in app_state.symbol_states:
                    app_state.init_symbol_state(symbol)
            
            # Set current symbol to first one if none is set
            if not app_state.current_symbol and symbols:
                app_state.current_symbol = symbols[0]
                print(f"[RESTORE] Set current symbol to {symbols[0]}")
        else:
            print(f"[RESTORE] Symbol states already exist for {list(app_state.symbol_states.keys())}")
        
        print(f"[RESTORE] Restoring pagination: max_value={num_symbols}")
        return num_symbols, 1, num_symbols, 1
    
    @app.callback(
        Output("result-text", "children", allow_duplicate=True),
        [Input("app-store", "data")],
        prevent_initial_call=True
    )
    def restore_analysis_status_on_refresh(store_data):
        """Restore analysis status text after page refresh"""
        if not store_data or not store_data.get("analysis_started"):
            return ""
        
        symbols = store_data.get("symbols", [])
        mode = store_data.get("mode", "mode")
        interval_text = store_data.get("interval_text", "")
        
        if symbols:
            return f"📄 Page refreshed - Analysis data for {', '.join(symbols)} has been restored ({mode}{interval_text}). All symbol pages should now be available."
        
        return "" 
