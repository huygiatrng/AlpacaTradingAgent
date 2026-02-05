"""
webui/components/analysis.py
"""

import time
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.dataflows.alpaca_utils import AlpacaUtils
from tradingagents.agents.utils.agent_trading_modes import extract_recommendation
from webui.utils.state import app_state
from webui.utils.charts import create_chart


def execute_trade_after_analysis(ticker, allow_shorts, trade_amount, use_ai_sizing=True):
    """Execute trade based on analysis results"""
    try:
        print(f"[TRADE] Starting trade execution for {ticker}")

        # Get the current state for this symbol
        state = app_state.get_state(ticker)
        if not state:
            print(f"[TRADE] No state found for {ticker}, skipping trade execution")
            return

        if not state.get("analysis_complete"):
            print(f"[TRADE] Analysis not complete for {ticker}, skipping trade execution")
            print(f"[TRADE] Analysis status: {state.get('analysis_complete', 'Unknown')}")
            return

        print(f"[TRADE] Analysis complete for {ticker}, checking for recommended action")

        # Get the recommended action
        recommended_action = state.get("recommended_action")
        print(f"[TRADE] Direct recommended_action: {recommended_action}")

        if not recommended_action:
            # Try to extract from final trade decision
            final_decision = state["current_reports"].get("final_trade_decision")
            print(f"[TRADE] Final decision available: {bool(final_decision)}")
            if final_decision:
                trading_mode = "trading" if allow_shorts else "investment"
                print(f"[TRADE] Extracting recommendation using mode: {trading_mode}")
                recommended_action = extract_recommendation(final_decision, trading_mode)
                print(f"[TRADE] Extracted recommendation: {recommended_action}")

        if not recommended_action:
            print(f"[TRADE] No recommended action found for {ticker}, skipping trade execution")
            print(f"[TRADE] Available reports: {list(state['current_reports'].keys())}")
            return

        # Determine position size (AI-determined or fixed)
        actual_trade_amount = trade_amount  # Default to user-configured amount

        if use_ai_sizing:
            # Try to get AI-recommended position size from analysis results
            analysis_results = state.get("analysis_results", {})
            full_state = analysis_results.get("full_state", {})

            # Priority: Risk Manager > Trader > Fallback
            approved_size = full_state.get("approved_position_size", {})
            trader_size = full_state.get("recommended_position_size", {})

            ai_suggested_dollars = (
                approved_size.get("recommended_size_dollars") or
                trader_size.get("recommended_size_dollars") or
                None
            )

            if ai_suggested_dollars and ai_suggested_dollars > 0:
                # Get account info for validation
                from tradingagents.agents.utils.position_size_extractor import validate_position_size
                account_info = AlpacaUtils.get_account_info()

                # Validation limits (configurable in future)
                limits = {
                    "max_position_pct_of_buying_power": 30,
                    "max_risk_pct_per_trade": 3,
                    "min_position_size": 100
                }

                # Validate AI-suggested size
                validated_size = validate_position_size(
                    ai_suggested_dollars,
                    account_info,
                    limits,
                    ticker
                )

                # Apply user-configured maximum as safety cap
                if validated_size > 0:
                    actual_trade_amount = min(validated_size, trade_amount)
                    print(f"[POSITION SIZE] AI recommended ${ai_suggested_dollars:,.2f}, validated to ${validated_size:,.2f}, capped at ${actual_trade_amount:,.2f}")

                    # Log which agent's recommendation was used
                    if approved_size.get("recommended_size_dollars"):
                        print(f"[POSITION SIZE] Using Risk Manager's approved size")
                    else:
                        print(f"[POSITION SIZE] Using Trader's recommended size")
                else:
                    print(f"[POSITION SIZE] AI-suggested size failed validation, using fallback ${actual_trade_amount:,.2f}")
            else:
                # Extraction failed, use fallback
                print(f"[POSITION SIZE] AI sizing extraction failed, using user-configured amount ${actual_trade_amount:,.2f}")
                if approved_size.get("fallback_used"):
                    print(f"[POSITION SIZE] Risk Manager extraction failed")
                if trader_size.get("fallback_used"):
                    print(f"[POSITION SIZE] Trader extraction failed")
        else:
            print(f"[POSITION SIZE] AI sizing disabled, using fixed amount ${actual_trade_amount:,.2f}")

        print(f"[TRADE] Executing trade for {ticker}: {recommended_action} with ${actual_trade_amount:,.2f}")

        # Get current position
        current_position = AlpacaUtils.get_current_position_state(ticker)
        print(f"[TRADE] Current position for {ticker}: {current_position}")

        # Execute the trading action
        result = AlpacaUtils.execute_trading_action(
            symbol=ticker,
            current_position=current_position,
            signal=recommended_action,
            dollar_amount=actual_trade_amount,
            allow_shorts=allow_shorts
        )
        
        # Check individual action results and provide detailed feedback
        successful_actions = []
        failed_actions = []
        
        for action_result in result.get("actions", []):
            if "result" in action_result:
                action_info = action_result["result"]
                if action_info.get("success"):
                    successful_actions.append(f"{action_result['action']}: {action_info.get('message', 'Success')}")
                else:
                    failed_actions.append(f"{action_result['action']} failed: {action_info.get('error', 'Unknown error')}")
            else:
                successful_actions.append(f"{action_result['action']}: {action_result.get('message', 'Action completed')}")
        
        # Print results based on overall success
        if result.get("success"):
            print(f"[TRADE] Successfully executed trading actions for {ticker}")
            for success in successful_actions:
                print(f"[TRADE] {success}")
            
            # Store trading results in state for UI display
            state["trading_results"] = result
            
            # Signal that a trade occurred to trigger Alpaca data refresh
            app_state.signal_trade_occurred()
        else:
            print(f"[TRADE] Trading execution failed for {ticker}")
            for success in successful_actions:
                print(f"[TRADE] {success}")
            for failure in failed_actions:
                print(f"[TRADE] {failure}")
            
            # Store error information
            state["trading_results"] = {"error": "One or more trading actions failed", "details": failed_actions}
            
    except Exception as e:
        print(f"[TRADE] Error executing trade for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        state = app_state.get_state(ticker)
        if state:
            state["trading_results"] = {"error": f"Trading execution error: {str(e)}"}


def run_analysis(ticker, selected_analysts, research_depth, allow_shorts, quick_llm, deep_llm, parallel_execution=True, progress=None):
    """Run the trading analysis using current/real-time data"""
    try:
        # Always use current date for real-time analysis
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"Starting real-time analysis for {ticker} with current date: {current_date}")
        current_state = app_state.get_state(ticker)
        if not current_state:
            print(f"Error: No state found for {ticker}")
            return
        current_state["analysis_running"] = True
        current_state["analysis_complete"] = False
        
        # Create config with selected options
        config = DEFAULT_CONFIG.copy()
        config["research_depth"] = research_depth  # Use research_depth string (Shallow/Medium/Deep)
        config["allow_shorts"] = allow_shorts
        config["parallel_analysts"] = parallel_execution  # Use user's choice from UI toggle
        config["quick_think_llm"] = quick_llm
        config["deep_think_llm"] = deep_llm
        
        # Initialize TradingAgentsGraph
        print(f"Initializing TradingAgentsGraph with analysts: {selected_analysts}")
        graph = TradingAgentsGraph(selected_analysts, config=config, debug=True)
        
        # Status updates are now handled in the parallel execution coordinator
        
        # Force an initial UI update
        app_state.needs_ui_update = True
        
        # Run analysis with tracing using current date
        print(f"Starting graph stream for {ticker} with current market data")
        trace = []
        for chunk in graph.graph.stream(
            graph.propagator.create_initial_state(ticker, current_date), 
            stream_mode="values",
            config={"recursion_limit": 100}
        ):
            # Track progress
            trace.append(chunk)
            
            # Process intermediate results
            app_state.process_chunk_updates(chunk)
            
            app_state.needs_ui_update = True
            
            # Update progress bar if provided
            if progress is not None:
                # Simulate progress based on steps completed
                completed_agents = sum(1 for status in current_state["agent_statuses"].values() if status == "completed")
                total_agents = len(current_state["agent_statuses"])
                if total_agents > 0:
                    progress(completed_agents / total_agents)
            
            # Small delay to prevent UI lag
            time.sleep(0.1)
        
        # Extract final results
        final_state = trace[-1]
        decision = graph.process_signal(final_state["final_trade_decision"])
        
        # NEW: Persist the extracted decision so the trading engine can act on it directly
        current_state["recommended_action"] = decision

        # Mark all agents as completed
        for agent in current_state["agent_statuses"]:
            app_state.update_agent_status(agent, "completed")
        
        # Set final results
        current_state["analysis_results"] = {
            "ticker": ticker,
            "date": current_date,
            "decision": decision,
            "full_state": final_state,
        }
        
        # Use real chart data with current date (no end_date means most recent data)
        current_state["chart_data"] = create_chart(ticker, period="1y", end_date=None)
        
        current_state["analysis_complete"] = True
        
        # Execute trade if enabled
        trade_enabled = getattr(app_state, 'trade_enabled', False)
        trade_amount = getattr(app_state, 'trade_amount', 1000)
        use_ai_sizing = getattr(app_state, 'use_ai_sizing', True)  # Default to AI sizing enabled
        print(f"[TRADE] Checking trading settings for {ticker}:")
        print(f"[TRADE]   - trade_enabled: {trade_enabled}")
        print(f"[TRADE]   - trade_amount: {trade_amount}")
        print(f"[TRADE]   - use_ai_sizing: {use_ai_sizing}")
        print(f"[TRADE]   - allow_shorts: {allow_shorts}")

        if trade_enabled:
            print(f"[TRADE] Trading enabled for {ticker}, executing trade with max ${trade_amount}")
            execute_trade_after_analysis(ticker, allow_shorts, trade_amount, use_ai_sizing)
        else:
            print(f"[TRADE] Trading disabled for {ticker}, skipping trade execution")
        
        # Final UI update to show completion
        app_state.needs_ui_update = True
        
    except Exception as e:
        print(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()
        if progress is not None:
            progress(1.0)  # Complete the progress bar
    finally:
        # Mark analysis as no longer running
        print(f"Real-time analysis for {ticker} completed")
        current_state["analysis_running"] = False
        
    return "Real-time analysis complete"


def start_analysis(ticker, analysts_market, analysts_social, analysts_news, analysts_fundamentals, analysts_macro,
                 research_depth, allow_shorts, quick_llm, deep_llm, parallel_execution=True, progress=None):
    """Start real-time analysis function for the UI"""
    
    # Parse selected analysts
    selected_analysts = []
    if analysts_market:
        selected_analysts.append("market")
    if analysts_social:
        selected_analysts.append("social")
    if analysts_news:
        selected_analysts.append("news")
    if analysts_fundamentals:
        selected_analysts.append("fundamentals")
    if analysts_macro:
        selected_analysts.append("macro")
    
    if not selected_analysts:
        return "Please select at least one analyst type."

    # Create an initial chart immediately with current data
    try:
        print(f"Creating initial chart for {ticker} with current market data")
        current_state = app_state.get_state(ticker)
        if current_state:
            current_state["chart_data"] = create_chart(ticker, period="1y", end_date=None)
    except Exception as e:
        print(f"Error creating initial chart: {e}")
        import traceback
        traceback.print_exc()
    
    # Run analysis with current data
    run_analysis(ticker, selected_analysts, research_depth, allow_shorts, quick_llm, deep_llm, parallel_execution, progress)
    
    # Update the status message with more details
    trading_mode = "Trading Mode (LONG/NEUTRAL/SHORT)" if allow_shorts else "Investment Mode (BUY/HOLD/SELL)"
    trade_text = f" with ${getattr(app_state, 'trade_amount', 1000)} auto-trading" if getattr(app_state, 'trade_enabled', False) else ""
    return f"Real-time analysis started for {ticker} with {len(selected_analysts)} analysts in {trading_mode}{trade_text} using sequential execution and current market data. Status table will update automatically." 