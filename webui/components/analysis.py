"""
webui/components/analysis.py
"""

import time
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.run_logger import get_run_audit_logger
from tradingagents.dataflows.alpaca_utils import AlpacaUtils
from tradingagents.agents.utils.agent_trading_modes import extract_recommendation
from webui.utils.state import app_state
from webui.utils.charts import create_chart


def execute_trade_after_analysis(ticker, allow_shorts, trade_amount):
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
        
        print(f"[TRADE] Executing trade for {ticker}: {recommended_action} with ${trade_amount}")
        
        # Get current position
        current_position = AlpacaUtils.get_current_position_state(ticker)
        print(f"[TRADE] Current position for {ticker}: {current_position}")
        
        # Execute the trading action
        result = AlpacaUtils.execute_trading_action(
            symbol=ticker,
            current_position=current_position,
            signal=recommended_action,
            dollar_amount=trade_amount,
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


def run_analysis(ticker, selected_analysts, research_depth_config, allow_shorts, quick_llm, deep_llm, progress=None):
    """Run the trading analysis using current/real-time data
    
    Args:
        research_depth_config: Either a dict with "rounds" and "level" keys,
                              or an integer for backward compatibility
    """
    run_logger = get_run_audit_logger()
    run_started = False
    final_state = None
    current_date = None

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
        
        # Handle both new dict format and legacy integer format
        if isinstance(research_depth_config, dict):
            depth_rounds = research_depth_config.get("rounds", 3)
            depth_level = research_depth_config.get("level", "Medium")
        else:
            # Legacy integer format - convert back to string
            depth_rounds = research_depth_config
            depth_map = {1: "Shallow", 3: "Medium", 5: "Deep"}
            depth_level = depth_map.get(research_depth_config, "Medium")
        
        # Create config with selected options
        config = DEFAULT_CONFIG.copy()
        config["max_debate_rounds"] = depth_rounds
        config["max_risk_discuss_rounds"] = depth_rounds
        config["research_depth"] = depth_level  # String for LLM parameter mapping
        config["allow_shorts"] = allow_shorts
        config["parallel_analysts"] = True  # Run analysts in parallel for faster execution
        config["quick_think_llm"] = quick_llm
        config["deep_think_llm"] = deep_llm
        
        # Initialize TradingAgentsGraph
        print(f"Initializing TradingAgentsGraph with analysts: {selected_analysts}")
        graph = TradingAgentsGraph(selected_analysts, config=config, debug=True)
        init_agent_state = graph.propagator.create_initial_state(ticker, current_date)
        run_logger.start_run(
            symbol=ticker,
            trade_date=current_date,
            config=config,
            metadata={"debug": True, "source": "webui_stream"},
        )
        run_started = True
        run_logger.log_state_snapshot(
            stage="initial_state",
            snapshot=init_agent_state,
            symbol=ticker,
        )
        
        # Status updates are now handled in the parallel execution coordinator
        
        # Force an initial UI update
        app_state.needs_ui_update = True
        
        # Run analysis with tracing using current date
        print(f"Starting graph stream for {ticker} with current market data")
        trace = []
        for chunk in graph.graph.stream(
            init_agent_state,
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

        filtered_tool_calls = [
            call for call in app_state.tool_calls_log
            if call.get("symbol") == ticker
        ]
        run_logger.log_state_snapshot(
            stage="webui_runtime_context",
            snapshot={
                "session_id": current_state.get("session_id"),
                "session_start_time": current_state.get("session_start_time"),
                "agent_prompts": current_state.get("agent_prompts", {}),
                "tool_calls": filtered_tool_calls,
                "llm_calls_count": app_state.llm_calls_count,
                "tool_calls_count": app_state.tool_calls_count,
            },
            symbol=ticker,
        )
        run_logger.finish_run(
            symbol=ticker,
            status="completed",
            final_state=final_state,
            final_signal=decision,
        )
        run_started = False
        
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
        print(f"[TRADE] Checking trading settings for {ticker}:")
        print(f"[TRADE]   - trade_enabled: {trade_enabled}")
        print(f"[TRADE]   - trade_amount: {trade_amount}")
        print(f"[TRADE]   - allow_shorts: {allow_shorts}")
        
        if trade_enabled:
            print(f"[TRADE] Trading enabled for {ticker}, executing trade with ${trade_amount}")
            execute_trade_after_analysis(ticker, allow_shorts, trade_amount)
        else:
            print(f"[TRADE] Trading disabled for {ticker}, skipping trade execution")
        
        # Final UI update to show completion
        app_state.needs_ui_update = True
        
    except Exception as e:
        print(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()
        if run_started:
            run_logger.finish_run(
                symbol=ticker,
                status="failed",
                final_state=final_state,
                error_message=str(e),
            )
            run_started = False
        if progress is not None:
            progress(1.0)  # Complete the progress bar
    finally:
        # Mark analysis as no longer running
        print(f"Real-time analysis for {ticker} completed")
        current_state["analysis_running"] = False
        
    return "Real-time analysis complete"


def start_analysis(ticker, analysts_market, analysts_social, analysts_news, analysts_fundamentals, analysts_macro,
                 research_depth, allow_shorts, quick_llm, deep_llm, progress=None):
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
    
    # Convert research depth to integer for debate rounds
    # Also keep the original string for LLM parameter mapping
    if research_depth == "Shallow":
        depth_rounds = 1
    elif research_depth == "Medium":
        depth_rounds = 3
    else:  # Deep
        depth_rounds = 5
    
    # Pass both the string (for LLM params) and rounds (for debates)
    depth_config = {"rounds": depth_rounds, "level": research_depth}
        
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
    run_analysis(ticker, selected_analysts, depth_config, allow_shorts, quick_llm, deep_llm, progress)
    
    # Update the status message with more details
    trading_mode = "Trading Mode (LONG/NEUTRAL/SHORT)" if allow_shorts else "Investment Mode (BUY/HOLD/SELL)"
    trade_text = f" with ${getattr(app_state, 'trade_amount', 1000)} auto-trading" if getattr(app_state, 'trade_enabled', False) else ""
    return f"Real-time analysis started for {ticker} with {len(selected_analysts)} analysts in {trading_mode}{trade_text} using parallel execution and current market data. Status table will update automatically." 
