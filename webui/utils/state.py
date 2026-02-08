"""
Trading Agents Framework - State Management
"""

# Global variables for tracking state
class AppState:
    def __init__(self):
        self.analysis_queue = []
        self.symbol_states = {}
        self.current_symbol = None  # Symbol displayed in UI
        self.analyzing_symbol = None  # Symbol currently being analyzed (backend)
        self.analysis_running = False
        self.analysis_trace = []
        self.tool_calls_count = 0
        self.llm_calls_count = 0
        self.generated_reports_count = 0
        self.needs_ui_update = False
        
        # Session tracking
        self.current_session_id = None
        self.session_start_time = None
        
        # New: Proper tracking lists similar to CLI
        self.tool_calls_log = []  # Store actual tool calls for proper counting
        self.llm_calls_log = []   # Store actual LLM calls for proper counting
        
        # Loop configuration
        self.loop_enabled = False
        self.loop_symbols = []
        self.loop_config = {}
        self.loop_interval_minutes = 60  # Default 1 hour
        self.loop_thread = None
        self.stop_loop = False
        
        # Market hour configuration  
        self.market_hour_enabled = False
        self.market_hour_symbols = []
        self.market_hour_config = {}
        self.market_hours = []
        self.market_hour_thread = None
        self.stop_market_hour = False
        
        # Trading configuration
        self.trade_enabled = False
        self.trade_amount = 1000
        self.trade_occurred = False
        
        self.refresh_interval = 1.0  # seconds
        self.analysis_complete = False
        self.analysis_results = None
        self.ticker_symbol = None
        self.chart_data = None
        self.chart_period = "1y"  # Default chart period
        self.session_id = None
        self.session_start_time = None
        self.report_timestamps = {}  # Track when each report was last updated
        self.agent_statuses = {}
        self.current_reports = {}
        self.investment_debate_state = None
        self.recommended_action = None
        self.last_trade_time = None
        self.alpaca_refresh_needed = False
        self.loop_enabled = False
        self.loop_interval_minutes = 60
        self.loop_symbols = []  # Store original symbols list for looping
        self.loop_config = {}  # Store analysis configuration for looping
        self.loop_thread = None
        self.stop_loop = False  # Flag to stop the loop
        self.market_hour_enabled = False
        self.market_hours = []  # List of hours to trade (e.g., [10, 15] for 10AM and 3PM)
        self.market_hour_symbols = []  # Store original symbols list for market hour trading
        self.market_hour_config = {}  # Store analysis configuration for market hour trading
        self.market_hour_thread = None
        self.stop_market_hour = False  # Flag to stop market hour scheduling
        self.trade_enabled = False
        self.trade_amount = 1000
        self.trade_occurred = False

    def register_llm_call(self, model_name=None, purpose=None):
        """Register an LLM call for accurate UI counting."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        payload = {"model": model_name, "purpose": purpose}
        self.llm_calls_log.append((timestamp, "LLM_CALL", payload))
        self.llm_calls_count = len([call for call in self.llm_calls_log if call[1] == "LLM_CALL"])
        self.needs_ui_update = True
        try:
            from tradingagents.run_logger import get_run_audit_logger
            symbol = self.analyzing_symbol or self.current_symbol
            get_run_audit_logger().log_event(
                event_type="llm_call",
                symbol=symbol,
                payload=payload,
            )
        except Exception:
            pass

    def add_symbols_to_queue(self, symbols):
        """Add a list of symbols to the analysis queue."""
        self.analysis_queue.extend(symbols)

    def get_next_symbol(self):
        """Get the next symbol from the queue for analysis (without changing UI display)."""
        if self.analysis_queue:
            next_symbol = self.analysis_queue.pop(0)
            
            # Set the symbol being analyzed (backend tracking)
            self.analyzing_symbol = next_symbol
            
            # Initialize state if needed
            if next_symbol not in self.symbol_states:
                self.init_symbol_state(next_symbol)
            else:
                # Reset session for existing symbol to start fresh analysis
                self.start_new_session_for_symbol(next_symbol)
            
            # Don't auto-switch UI display - let user control which symbol to view
            # Only set current_symbol if it's not already set (for initial setup)
            if self.current_symbol is None:
                self.current_symbol = next_symbol
            
            return next_symbol
            
        # No more symbols to analyze
        self.analyzing_symbol = None
        return None

    def get_state(self, symbol):
        """Get the state for a specific symbol."""
        return self.symbol_states.get(symbol)

    def get_current_state(self):
        """Get the state for the current symbol."""
        if self.current_symbol:
            return self.symbol_states.get(self.current_symbol)
        return None

    def get_analyzing_state(self):
        """Get the state for the symbol currently being analyzed."""
        if self.analyzing_symbol:
            return self.symbol_states.get(self.analyzing_symbol)
        return None

    def init_symbol_state(self, symbol):
        """Initialize the state for a new symbol."""
        import time
        import uuid
        
        # Generate a new session ID for this symbol analysis
        session_id = str(uuid.uuid4())[:8]
        session_start = time.time()
        
        self.symbol_states[symbol] = {
            "agent_statuses": {
                "Market Analyst": "pending",
                "Social Analyst": "pending",
                "News Analyst": "pending",
                "Fundamentals Analyst": "pending",
                "Macro Analyst": "pending",
                "Bull Researcher": "pending",
                "Bear Researcher": "pending",
                "Research Manager": "pending",
                "Trader": "pending",
                "Risky Analyst": "pending",
                "Safe Analyst": "pending",
                "Neutral Analyst": "pending",
                "Portfolio Manager": "pending"
            },
            "current_reports": {
                "market_report": None,
                "sentiment_report": None,
                "news_report": None,
                "fundamentals_report": None,
                "macro_report": None,
                "bull_report": None,
                "bear_report": None,
                "research_manager_report": None,
                "investment_plan": None, 
                "trader_investment_plan": None,
                "risky_report": None,
                "safe_report": None,
                "neutral_report": None,
                "portfolio_decision": None,
                "final_trade_decision": None
            },
            "agent_prompts": {
                "market_report": None,
                "sentiment_report": None,
                "news_report": None,
                "fundamentals_report": None,
                "macro_report": None,
                "bull_report": None,
                "bear_report": None,
                "research_manager_report": None,
                "investment_plan": None, 
                "trader_investment_plan": None,
                "risky_report": None,
                "safe_report": None,
                "neutral_report": None,
                "portfolio_decision": None,
                "final_trade_decision": None
            },
            "investment_debate_state": None,
            "analysis_complete": False,
            "analysis_results": None,
            "ticker_symbol": symbol,
            "chart_data": None,
            "chart_period": "1y",  # Default chart period
            "session_id": session_id,
            "session_start_time": session_start,
            "report_timestamps": {}  # Track when each report was last updated
        }

    def update_agent_status(self, agent, status, symbol=None):
        """Update the status of an agent for a specific symbol (or current symbol if none specified)."""
        if symbol is None:
            symbol = self.analyzing_symbol or self.current_symbol
            
        state = self.get_state(symbol)
        if state:
            if agent in state["agent_statuses"]:
                if status not in ["pending", "in_progress", "completed"]:
                    print(f"Warning: Invalid status '{status}' for agent '{agent}', defaulting to 'pending'")
                    status = "pending"
                
                if state["agent_statuses"][agent] != status:
                    state["agent_statuses"][agent] = status
                    print(f"[STATE - {symbol}] Updated {agent} status to {status}")
                    self.needs_ui_update = True

    def store_agent_prompt(self, report_type, prompt_text, symbol=None):
        """Store the prompt used by an agent for a specific report type."""
        if symbol is None:
            symbol = self.analyzing_symbol or self.current_symbol
            
        state = self.get_state(symbol)
        if state and "agent_prompts" in state:
            state["agent_prompts"][report_type] = prompt_text
            print(f"[STATE - {symbol}] Stored prompt for {report_type} ({len(prompt_text)} chars)")

    def get_agent_prompt(self, report_type, symbol=None):
        """Get the prompt used by an agent for a specific report type."""
        if symbol is None:
            symbol = self.current_symbol
            
        state = self.get_state(symbol)
        if state and "agent_prompts" in state:
            return state["agent_prompts"].get(report_type)
        return None

    def reset(self):
        """Reset the application state for all symbols."""
        print("[STATE] Resetting application state")
        self.analysis_queue = []
        self.symbol_states = {}
        self.current_symbol = None
        self.analysis_running = False
        self.analysis_trace = []
        self.tool_calls_count = 0
        self.llm_calls_count = 0
        # Reset the new tracking lists
        self.tool_calls_log = []
        self.llm_calls_log = []
        self.generated_reports_count = 0
        # Reset session tracking
        self.current_session_id = None
        self.session_start_time = None

    def get_tool_calls_for_display(self, agent_filter=None, symbol_filter=None):
        """Get tool calls in a consistent format for UI display, optionally filtered by agent type and symbol"""
        formatted_calls = []
        
        for call in self.tool_calls_log:
            if isinstance(call, dict):
                # New format - already has all the data we need
                formatted_calls.append(call)
            elif isinstance(call, tuple) and len(call) >= 3:
                # Old format - convert to new format
                timestamp, tool_name, inputs = call[:3]
                formatted_call = {
                    "timestamp": timestamp,
                    "tool_name": tool_name,
                    "inputs": inputs,
                    "output": "Output not available (old format)",
                    "execution_time": "Unknown",
                    "status": "completed",
                    "agent_type": "Unknown Agent"  # Old format doesn't have agent info
                }
                formatted_calls.append(formatted_call)
            else:
                # Invalid format - create error entry
                formatted_call = {
                    "timestamp": "Unknown",
                    "tool_name": "Invalid Entry",
                    "inputs": {},
                    "output": f"Invalid tool call format: {str(call)}",
                    "execution_time": "Unknown",
                    "status": "error",
                    "agent_type": "Unknown Agent"
                }
                formatted_calls.append(formatted_call)
        
        # Filter by agent type if specified
        if agent_filter:
            filtered_calls = []
            for call in formatted_calls:
                agent_type = call.get("agent_type", "").lower()
                # Match agent types using flexible matching
                if agent_filter.lower() in agent_type or self._matches_agent_type(agent_filter, agent_type):
                    filtered_calls.append(call)
            formatted_calls = filtered_calls
            
        # Filter by symbol if specified
        if symbol_filter:
            filtered_calls = []
            for call in formatted_calls:
                call_symbol = call.get("symbol")
                # Match symbol (case-insensitive)
                if call_symbol and call_symbol.upper() == symbol_filter.upper():
                    filtered_calls.append(call)
            formatted_calls = filtered_calls
        
        return formatted_calls
    
    def _matches_agent_type(self, filter_type, agent_type):
        """Helper method to match agent types with flexible naming"""
        # Mapping of report types to agent names
        agent_mappings = {
            "market_report": ["market analyst", "market", "technical analyst"],
            "sentiment_report": ["social analyst", "social", "sentiment analyst"],
            "news_report": ["news analyst", "news"],
            "fundamentals_report": ["fundamentals analyst", "fundamental analyst", "fundamentals"],
            "macro_report": ["macro analyst", "macro", "macroeconomic analyst"],
            "bull_report": ["bull researcher", "bull", "optimistic researcher"],
            "bear_report": ["bear researcher", "bear", "pessimistic researcher"],
            "research_manager_report": ["research manager", "manager"],
            "trader_investment_plan": ["trader", "trading", "portfolio manager"],
            "final_trade_decision": ["portfolio manager", "final", "decision"]
        }
        
        filter_lower = filter_type.lower()
        agent_lower = agent_type.lower()
        
        # Direct match
        if filter_lower in agent_lower:
            return True
            
        # Check mappings
        if filter_type in agent_mappings:
            for mapping in agent_mappings[filter_type]:
                if mapping in agent_lower:
                    return True
                    
        return False
        
    def reset_for_loop(self):
        """Reset state for the next loop iteration without stopping the loop."""
        import time
        import uuid
        print("[STATE] Resetting state for next loop iteration")
        self.analysis_queue = []
        
        # Reset analysis data for each symbol but KEEP the symbol states for pagination
        for symbol in self.symbol_states:
            state = self.symbol_states[symbol]
            # Generate new session tracking for this reset
            new_session_id = str(uuid.uuid4())[:8]
            new_session_start = time.time()
            
            # Reset analysis-specific data but keep ticker_symbol for pagination
            state.update({
                "analysis_running": False,
                "analysis_complete": False,
                "current_reports": {
                    "market_report": None,
                    "sentiment_report": None,
                    "news_report": None,
                    "fundamentals_report": None,
                    "macro_report": None,
                    "bull_report": None,
                    "bear_report": None,
                    "research_manager_report": None,
                    "trader_investment_plan": None,
                    "risky_report": None,
                    "safe_report": None,
                    "neutral_report": None,
                    "portfolio_decision": None,
                    "final_trade_decision": None
                },
                "agent_prompts": {
                    "market_report": None,
                    "sentiment_report": None,
                    "news_report": None,
                    "fundamentals_report": None,
                    "macro_report": None,
                    "bull_report": None,
                    "bear_report": None,
                    "research_manager_report": None,
                    "trader_investment_plan": None,
                    "risky_report": None,
                    "safe_report": None,
                    "neutral_report": None,
                    "portfolio_decision": None,
                    "final_trade_decision": None
                },
                "agent_statuses": {
                    "Market Analyst": "pending",
                    "Social Analyst": "pending", 
                    "News Analyst": "pending",
                    "Fundamentals Analyst": "pending",
                    "Macro Analyst": "pending",
                    "Bull Researcher": "pending",
                    "Bear Researcher": "pending",
                    "Research Manager": "pending",
                    "Trader": "pending",
                    "Risky Analyst": "pending",
                    "Safe Analyst": "pending",
                    "Neutral Analyst": "pending",
                    "Portfolio Manager": "pending"
                },
                "analysis_results": None,
                "recommended_action": None,
                "chart_data": state.get("chart_data"),  # Preserve chart data
                "chart_period": state.get("chart_period", "1y"),  # Preserve chart period
                # Keep ticker_symbol for pagination
                "ticker_symbol": state.get("ticker_symbol"),
                # Reset session tracking
                "session_id": new_session_id,
                "session_start_time": new_session_start,
                "report_timestamps": {}
            })
        
        self.current_symbol = None
        self.analysis_trace = []
        self.tool_calls_count = 0
        self.llm_calls_count = 0
        # Reset the new tracking lists
        self.tool_calls_log = []
        self.llm_calls_log = []
        self.generated_reports_count = 0
        self.needs_ui_update = True

    def start_loop(self, symbols, config):
        """Start the looping mode with given symbols and configuration."""
        self.loop_enabled = True
        self.loop_symbols = symbols
        self.loop_config = config
        self.stop_loop = False
        print(f"[STATE] Starting loop mode with {len(symbols)} symbols, interval: {self.loop_interval_minutes} minutes")

    def stop_loop_mode(self):
        """Stop the looping mode."""
        self.stop_loop = True
        self.loop_enabled = False
        self.analysis_running = False
        print("[STATE] Stopping loop mode")

    def start_market_hour_mode(self, symbols, config, hours):
        """Start the market hour trading mode with given symbols, configuration, and hours."""
        self.market_hour_enabled = True
        self.market_hour_symbols = symbols
        self.market_hour_config = config
        self.market_hours = hours
        self.stop_market_hour = False
        print(f"[STATE] Starting market hour mode with {len(symbols)} symbols, hours: {hours}")

    def stop_market_hour_mode(self):
        """Stop the market hour trading mode."""
        self.stop_market_hour = True
        self.market_hour_enabled = False
        self.analysis_running = False
        print("[STATE] Stopping market hour mode")
    
    def start_new_session_for_symbol(self, symbol):
        """Start a new analysis session for an existing symbol."""
        import time
        import uuid
        
        if symbol in self.symbol_states:
            state = self.symbol_states[symbol]
            # Generate new session tracking
            state["session_id"] = str(uuid.uuid4())[:8]
            state["session_start_time"] = time.time()
            state["report_timestamps"] = {}
            print(f"[STATE] Started new analysis session {state['session_id']} for {symbol}")

    def signal_trade_occurred(self):
        """Signal that a trade has occurred and Alpaca data should be refreshed."""
        import time
        self.last_trade_time = time.time()
        self.alpaca_refresh_needed = True
        print("[STATE] Trading event signaled - Alpaca refresh needed")
    
    def update_reports_count(self):
        """Update the generated reports count across all symbols."""
        total_reports = 0
        for symbol_state in self.symbol_states.values():
            reports = symbol_state.get("current_reports", {})
            total_reports += sum(1 for content in reports.values() if content is not None and str(content).strip())
        self.generated_reports_count = total_reports

    def is_all_symbols_complete(self):
        """Check if all symbols in the current analysis are complete."""
        if not self.loop_symbols:
            return False
        
        for symbol in self.loop_symbols:
            state = self.get_state(symbol)
            if not state or not state.get("analysis_complete", False):
                return False
        return True

    def process_chunk_updates(self, chunk):
        """Process chunk updates from the graph stream for the symbol currently being analyzed."""
        state = self.get_analyzing_state()
        if not state:
            # Fallback to current symbol if no analyzing symbol is set
            state = self.get_current_state()
            if not state:
                return

        analyzing_symbol = self.analyzing_symbol or self.current_symbol
        ui_update_needed = False

        # Map report types to agent names
        report_to_agent = {
            "market_report": "Market Analyst",
            "sentiment_report": "Social Analyst", 
            "news_report": "News Analyst",
            "fundamentals_report": "Fundamentals Analyst",
            "macro_report": "Macro Analyst",
            "bull_report": "Bull Researcher",
            "bear_report": "Bear Researcher",
            "research_manager_report": "Research Manager",
            "investment_plan": "Trader",
            "trader_investment_plan": "Trader",
            "risky_report": "Risky Analyst",
            "safe_report": "Safe Analyst",
            "neutral_report": "Neutral Analyst",
        }
        
        # Determine the analyst execution sequence based on user selection (if available)
        default_sequence = [
            "Market Analyst",
            "Social Analyst",
            "News Analyst",
            "Fundamentals Analyst",
            "Macro Analyst",
        ]
        # If the UI has stored the list of active analysts, respect that (and preserve order)
        if hasattr(self, "active_analysts") and self.active_analysts:
            # Keep only those analysts that are in the default ordering to avoid typos
            analyst_sequence = [a for a in default_sequence if a in self.active_analysts]
            # Fallback: if somehow none matched (e.g., custom ordering), just use the provided list
            if not analyst_sequence:
                analyst_sequence = list(self.active_analysts)
        else:
            analyst_sequence = default_sequence
        
        # Update analyst reports and manage status transitions
        for report_type in ["market_report", "sentiment_report", "news_report", "fundamentals_report", "macro_report"]:
            if report_type in chunk:
                new_report = chunk[report_type]
                # Skip if report content is None or empty/whitespace only
                if new_report is None or (isinstance(new_report, str) and new_report.strip() == ""):
                    continue
                
                # Check for duplicate content using session-aware logic
                import time
                current_report = state["current_reports"].get(report_type)
                agent = report_to_agent[report_type]
                current_status = state["agent_statuses"].get(agent)
                current_time = time.time()
                
                # Get the last update timestamp for this report type
                last_update_time = state.get("report_timestamps", {}).get(report_type, 0)
                
                # Skip duplicates only if:
                # 1. Content is exactly the same AND
                # 2. Agent is already completed AND 
                # 3. It was updated recently (within 5 seconds to avoid stream spam)
                is_duplicate = (
                    new_report == current_report and 
                    current_status == "completed" and 
                    (current_time - last_update_time) < 5
                )
                
                if is_duplicate:
                    continue
                
                # Store the report if it's genuinely new or different
                if new_report != current_report or current_status != "completed":
                    # 🛡️ PROTECTION: Ensure UI gets final reports, not intermediate ones
                    # Once an analyst is completed, only accept significantly longer reports
                    # This prevents UI from showing incomplete streaming chunks
                    if current_status == "completed":
                        current_length = len(current_report or "")
                        new_length = len(new_report or "")
                        
                        # Only accept new reports if they're at least 20% longer than current
                        # This allows final complete reports while blocking minor streaming updates
                        min_required_length = current_length * 1.2
                        
                        if new_length < min_required_length:
                            # print(f"[STATE - {self.current_symbol}] 🛡️ Blocking {report_type} update: {new_length} chars < required {min_required_length:.0f} chars (analyst completed)")
                            continue
                        else:
                            print(f"[STATE - {analyzing_symbol}] 📊 Accepting larger final {report_type}: {new_length} chars (was {current_length})")
                    
                    # Add safety check for excessive updates from the same analyst
                    update_count_key = f"{report_type}_update_count"
                    current_count = state.get(update_count_key, 0)
                    
                    # If an analyst is producing too many different reports, there might be an issue
                    if current_count > 10:  # Allow max 10 updates per report type
                        print(f"[STATE - {analyzing_symbol}] ⚠️ WARNING: {report_type} has been updated {current_count} times. Possible infinite loop detected.")
                        if current_count > 15:  # Hard limit
                            print(f"[STATE - {analyzing_symbol}] 🛑 BLOCKING further {report_type} updates to prevent hang")
                            continue
                    
                    state["current_reports"][report_type] = new_report
                    state.setdefault("report_timestamps", {})[report_type] = current_time
                    state[update_count_key] = current_count + 1
                    
                    # Count unique non-empty reports across all symbols
                    if new_report:
                        self.update_reports_count()
                        
                        # Add debug logging for all analyst reports
                        print(f"[STATE - {analyzing_symbol}] ✅ Updated {report_type} with content length: {len(new_report)} (update #{current_count + 1})")
                        print(f"[STATE - {analyzing_symbol}] 📊 Total Generated Reports: {self.generated_reports_count}")
                        
                        # Special debugging for macro report (simplified)
                        if report_type == "macro_report":
                            print(f"[STATE - {analyzing_symbol}] 📊 MACRO REPORT RECEIVED: {len(new_report)} chars")
                            
                    ui_update_needed = True

                # Special debugging for macro analyst (only when transitioning to in_progress)
                if agent == "Macro Analyst" and current_status == "pending":
                    print(f"[STATE - {analyzing_symbol}] 📊 MACRO ANALYST STATUS TRANSITION:")
                    print(f"[STATE - {analyzing_symbol}] 📊   - Current status: {current_status}")
                    print(f"[STATE - {analyzing_symbol}] 📊   - Report type: {report_type}")

                # Transition logic:
                #   - If the agent is already "in_progress", receiving a report marks it "completed".
                #   - Do NOT automatically move a "pending" agent to "in_progress" when a report
                #     appears; the progression to "in_progress" is controlled explicitly when the
                #     previous analyst completes.
                if current_status == "in_progress":
                    # Mark this analyst as completed and advance workflow
                    self.update_agent_status(agent, "completed", analyzing_symbol)
                    ui_update_needed = True

                    # Special debugging for macro analyst completion
                    if agent == "Macro Analyst":
                        print(f"[STATE - {analyzing_symbol}] 📊 MACRO ANALYST COMPLETED!")
                        print(f"[STATE - {analyzing_symbol}] 📊   - Transitioning from 'in_progress' to 'completed'")

                    # Advance to the next analyst in the predefined sequence
                    if agent in analyst_sequence and agent != analyst_sequence[-1]:
                        next_analyst = analyst_sequence[analyst_sequence.index(agent) + 1]
                        if state["agent_statuses"].get(next_analyst) == "pending":
                            self.update_agent_status(next_analyst, "in_progress", analyzing_symbol)
                            ui_update_needed = True
                            print(f"[STATE - {analyzing_symbol}] ➡️ Advanced to next analyst: {next_analyst}")
                    elif agent == analyst_sequence[-1]:
                        print(f"[STATE - {analyzing_symbol}] ✅ All {len(analyst_sequence)} analysts completed. Ready for research phase.")
                        # Special debugging for macro analyst being the last
                        if agent == "Macro Analyst":
                            print(f"[STATE - {analyzing_symbol}] 📊 MACRO ANALYST was the final analyst in sequence!")
                elif current_status == "pending" and new_report:
                    # This might be a timing issue where report arrives before status is set to in_progress
                    # Just log it as info, not a warning
                    print(f"[STATE - {analyzing_symbol}] 📝 Received {report_type} for {agent} (status: {current_status})")

        # Research team debate state
        if "investment_debate_state" in chunk:
            debate_state = chunk["investment_debate_state"]
            
            # Store the full debate state for chat UI access
            state["investment_debate_state"] = debate_state
            
            # Bull researcher
            if "bull_history" in debate_state and debate_state["bull_history"]:
                # Only set to in_progress if currently pending (don't override completed status)
                current_status = state["agent_statuses"].get("Bull Researcher")
                if current_status == "pending":
                    self.update_agent_status("Bull Researcher", "in_progress", analyzing_symbol)
                # Use the latest message from bull_messages array if available, otherwise use full history
                if "bull_messages" in debate_state and debate_state["bull_messages"]:
                    latest_bull_message = debate_state["bull_messages"][-1]
                    state["current_reports"]["bull_report"] = latest_bull_message
                else:
                    state["current_reports"]["bull_report"] = debate_state["bull_history"]
                self.update_reports_count()
                ui_update_needed = True
            
            # Bear researcher
            if "bear_history" in debate_state and debate_state["bear_history"]:
                # Only set to in_progress if currently pending (don't override completed status)
                current_status = state["agent_statuses"].get("Bear Researcher")
                if current_status == "pending":
                    self.update_agent_status("Bear Researcher", "in_progress", analyzing_symbol)
                # Use the latest message from bear_messages array if available, otherwise use full history
                if "bear_messages" in debate_state and debate_state["bear_messages"]:
                    latest_bear_message = debate_state["bear_messages"][-1]
                    state["current_reports"]["bear_report"] = latest_bear_message
                else:
                    state["current_reports"]["bear_report"] = debate_state["bear_history"]
                self.update_reports_count()
                ui_update_needed = True
            
            # Research manager
            if "judge_decision" in debate_state and debate_state["judge_decision"]:
                self.update_agent_status("Bull Researcher", "completed", analyzing_symbol)
                self.update_agent_status("Bear Researcher", "completed", analyzing_symbol)
                self.update_agent_status("Research Manager", "completed", analyzing_symbol)
                state["current_reports"]["research_manager_report"] = debate_state["judge_decision"]
                state["current_reports"]["investment_plan"] = debate_state["judge_decision"]
                self.update_agent_status("Trader", "in_progress", analyzing_symbol)
                ui_update_needed = True
        
        # Trader plan
        if "trader_investment_plan" in chunk and chunk["trader_investment_plan"]:
            state["current_reports"]["trader_investment_plan"] = chunk["trader_investment_plan"]
            self.update_reports_count()
            self.update_agent_status("Trader", "completed", analyzing_symbol)
            self.update_agent_status("Risky Analyst", "in_progress", analyzing_symbol)
            ui_update_needed = True
        
        # Risk debate state
        if "risk_debate_state" in chunk:
            risk_state = chunk["risk_debate_state"]
            
            # Store the full risk debate state for debugging and chat UI access
            state["risk_debate_state"] = risk_state
            
            # Risky analyst
            if "current_risky_response" in risk_state and risk_state["current_risky_response"]:
                # Only set to in_progress if currently pending (don't override completed status)
                current_status = state["agent_statuses"].get("Risky Analyst")
                if current_status == "pending":
                    self.update_agent_status("Risky Analyst", "in_progress", analyzing_symbol)
                # Extract just the content without the "Risky Analyst:" prefix if present
                risky_content = risk_state["current_risky_response"]
                if risky_content.startswith("Risky Analyst: "):
                    risky_content = risky_content[15:]  # Remove "Risky Analyst: " prefix
                state["current_reports"]["risky_report"] = risky_content
                self.update_reports_count()
                # print(f"[STATE - {self.current_symbol}] Updated risky_report with content length: {len(risky_content)}")
                ui_update_needed = True
            
            # Safe analyst
            if "current_safe_response" in risk_state and risk_state["current_safe_response"]:
                # Only set to in_progress if currently pending (don't override completed status)
                current_status = state["agent_statuses"].get("Safe Analyst")
                if current_status == "pending":
                    self.update_agent_status("Safe Analyst", "in_progress", analyzing_symbol)
                # Extract just the content without the "Safe Analyst:" prefix if present
                safe_content = risk_state["current_safe_response"]
                if safe_content.startswith("Safe Analyst: "):
                    safe_content = safe_content[14:]  # Remove "Safe Analyst: " prefix
                state["current_reports"]["safe_report"] = safe_content
                self.update_reports_count()
                # print(f"[STATE - {self.current_symbol}] Updated safe_report with content length: {len(safe_content)}")
                ui_update_needed = True
            
            # Neutral analyst
            if "current_neutral_response" in risk_state and risk_state["current_neutral_response"]:
                # Only set to in_progress if currently pending (don't override completed status)
                current_status = state["agent_statuses"].get("Neutral Analyst")
                if current_status == "pending":
                    self.update_agent_status("Neutral Analyst", "in_progress", analyzing_symbol)
                # Extract just the content without the "Neutral Analyst:" prefix if present
                neutral_content = risk_state["current_neutral_response"]
                if neutral_content.startswith("Neutral Analyst: "):
                    neutral_content = neutral_content[17:]  # Remove "Neutral Analyst: " prefix
                state["current_reports"]["neutral_report"] = neutral_content
                self.update_reports_count()
                # print(f"[STATE - {self.current_symbol}] Updated neutral_report with content length: {len(neutral_content)}")
                ui_update_needed = True
            
            # Portfolio manager - preserve individual reports when final decision is made
            if "judge_decision" in risk_state and risk_state["judge_decision"]:
                # Ensure individual reports are preserved from the debate history
                if not state["current_reports"]["risky_report"] and "risky_history" in risk_state:
                    risky_history = risk_state["risky_history"]
                    if risky_history:
                        state["current_reports"]["risky_report"] = risky_history.replace("Risky Analyst: ", "").strip()
                
                if not state["current_reports"]["safe_report"] and "safe_history" in risk_state:
                    safe_history = risk_state["safe_history"]
                    if safe_history:
                        state["current_reports"]["safe_report"] = safe_history.replace("Safe Analyst: ", "").strip()
                
                if not state["current_reports"]["neutral_report"] and "neutral_history" in risk_state:
                    neutral_history = risk_state["neutral_history"]
                    if neutral_history:
                        state["current_reports"]["neutral_report"] = neutral_history.replace("Neutral Analyst: ", "").strip()
                
                # Mark all as completed
                self.update_agent_status("Risky Analyst", "completed", analyzing_symbol)
                self.update_agent_status("Safe Analyst", "completed", analyzing_symbol)
                self.update_agent_status("Neutral Analyst", "completed", analyzing_symbol)
                self.update_agent_status("Portfolio Manager", "completed", analyzing_symbol)
                
                # Set final decisions
                state["current_reports"]["portfolio_decision"] = risk_state["judge_decision"]
                state["current_reports"]["final_trade_decision"] = risk_state["judge_decision"]
                
                # Store extracted recommendation if available
                if "recommended_action" in chunk:
                    state["recommended_action"] = chunk["recommended_action"]
                
                # Mark the overall analysis as complete once the Portfolio Manager has delivered the final decision
                state["analysis_complete"] = True
                
                print(f"[STATE - {analyzing_symbol}] Final decision set. Reports status:")
                print(f"  risky_report: {len(state['current_reports']['risky_report'] or '') > 0}")
                print(f"  safe_report: {len(state['current_reports']['safe_report'] or '') > 0}")
                print(f"  neutral_report: {len(state['current_reports']['neutral_report'] or '') > 0}")
                print(f"  final_trade_decision: {len(state['current_reports']['final_trade_decision'] or '') > 0}")
                
                ui_update_needed = True
        
        # Proper tracking of LLM calls and tool calls (similar to CLI implementation)
        if "messages" in chunk and len(chunk.get("messages", [])) > 0:
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            
            # Process each message in the chunk
            for message in chunk["messages"]:
                # Extract message content and type
                if hasattr(message, "content"):
                    content = message.content
                    msg_type = "Reasoning"  # LLM reasoning calls
                else:
                    content = str(message)
                    msg_type = "System"
                
                # Add to LLM calls log
                self.llm_calls_log.append((timestamp, msg_type, content))
                
                # Note: Tool calls are now tracked directly in agent_utils.py timing_wrapper
                # No need to parse them from message chunks
            
            # Update LLM calls count
            has_llm_calls = any(call[1] == "LLM_CALL" for call in self.llm_calls_log)
            if has_llm_calls:
                self.llm_calls_count = len([call for call in self.llm_calls_log if call[1] == "LLM_CALL"])
            else:
                self.llm_calls_count = len([call for call in self.llm_calls_log if call[1] == "Reasoning"])
            
            # Tool calls count is updated directly in timing_wrapper, no need to recalculate here
            
            # Debug output for message processing
            if len(chunk.get("messages", [])) > 0:
                print(f"[STATE] Processed {len(chunk['messages'])} messages")
                print(f"[STATE] Updated counts - Tool Calls: {self.tool_calls_count}, LLM Calls: {self.llm_calls_count}")
            
            ui_update_needed = True
                
        # Set the first analyst to in_progress when analysis starts (detect initial human message)
        if "messages" in chunk and chunk["messages"]:
            for message in chunk["messages"]:
                # Check if this is the initial human message that starts the analysis
                if hasattr(message, 'type') and message.type == "human":
                    # Only set an analyst to in_progress if NONE are currently in progress
                    if not any(status == "in_progress" for status in state["agent_statuses"].values()):
                        # Use the same dynamic analyst_sequence defined above
                        for analyst in analyst_sequence:
                            if state["agent_statuses"].get(analyst) == "pending":
                                self.update_agent_status(analyst, "in_progress", analyzing_symbol)
                                ui_update_needed = True
                                break  # Only set the first pending analyst
                    break  # Only process the first human message

        # Set the UI update flag if any changes were made
        if ui_update_needed:
            self.needs_ui_update = True
            # print(f"[STATE] Setting needs_ui_update flag due to chunk updates")

# Create a global instance
app_state = AppState() 
