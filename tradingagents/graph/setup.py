# TradingAgents/graph/setup.py

import concurrent.futures
import threading
import copy
import time
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.analysts.macro_analyst import create_macro_analyst
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.utils.agent_utils import Toolkit
from tradingagents.agents.utils.report_context import create_report_context_node
from tradingagents.run_logger import get_run_audit_logger

from .conditional_logic import ConditionalLogic


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: ChatOpenAI,
        deep_thinking_llm: ChatOpenAI,
        toolkit: Toolkit,
        tool_nodes: Dict[str, ToolNode],
        bull_memory,
        bear_memory,
        trader_memory,
        invest_judge_memory,
        risk_manager_memory,
        conditional_logic: ConditionalLogic,
        config: Dict[str, Any] = None,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.toolkit = toolkit
        self.tool_nodes = tool_nodes
        self.bull_memory = bull_memory
        self.bear_memory = bear_memory
        self.trader_memory = trader_memory
        self.invest_judge_memory = invest_judge_memory
        self.risk_manager_memory = risk_manager_memory
        self.conditional_logic = conditional_logic
        self.config = config

    def _wrap_node_with_run_logging(self, node_name: str, node_fn):
        """Wrap a graph node so its outputs are persisted to the per-run audit log."""

        def wrapped_node(state):
            logger = get_run_audit_logger()
            symbol = state.get("company_of_interest")
            start_time = time.time()

            try:
                result = node_fn(state)
                elapsed = time.time() - start_time
                logger.log_event(
                    event_type="node_execution",
                    symbol=symbol,
                    payload={
                        "node_name": node_name,
                        "status": "success",
                        "elapsed_seconds": round(elapsed, 4),
                    },
                )
            except Exception as e:
                elapsed = time.time() - start_time
                logger.log_event(
                    event_type="node_error",
                    symbol=symbol,
                    payload={
                        "node_name": node_name,
                        "status": "error",
                        "elapsed_seconds": round(elapsed, 4),
                        "error_message": str(e),
                    },
                )
                raise

            if not isinstance(result, dict):
                return result

            output_keys = [
                "market_report",
                "sentiment_report",
                "news_report",
                "fundamentals_report",
                "macro_report",
                "investment_plan",
                "trader_investment_plan",
                "final_trade_decision",
            ]
            for output_key in output_keys:
                output_value = result.get(output_key)
                if output_value:
                    logger.log_agent_output(
                        output_type=output_key,
                        content=output_value,
                        symbol=symbol,
                        metadata={"node_name": node_name},
                    )

            if "report_context" in result and isinstance(result["report_context"], dict):
                logger.log_agent_output(
                    output_type="report_context_stats",
                    content=result["report_context"].get("stats", {}),
                    symbol=symbol,
                    metadata={"node_name": node_name},
                )

            investment_debate_state = result.get("investment_debate_state")
            if isinstance(investment_debate_state, dict):
                current_response = investment_debate_state.get("current_response")
                if current_response:
                    logger.log_agent_output(
                        output_type="investment_debate_response",
                        content=current_response,
                        symbol=symbol,
                        metadata={"node_name": node_name},
                    )

            risk_debate_state = result.get("risk_debate_state")
            if isinstance(risk_debate_state, dict):
                latest_speaker = risk_debate_state.get("latest_speaker")
                speaker_key_map = {
                    "Risky": "current_risky_response",
                    "Safe": "current_safe_response",
                    "Neutral": "current_neutral_response",
                }
                speaker_response = risk_debate_state.get(speaker_key_map.get(latest_speaker, ""))
                if speaker_response:
                    logger.log_agent_output(
                        output_type="risk_debate_response",
                        content=speaker_response,
                        symbol=symbol,
                        metadata={
                            "node_name": node_name,
                            "latest_speaker": latest_speaker,
                        },
                    )

            return result

        return wrapped_node

    def _create_parallel_analysts_coordinator(self, selected_analysts, analyst_nodes, tool_nodes, delete_nodes):
        """Create a coordinator that runs selected analysts in parallel"""
        
        def parallel_analysts_execution(state: AgentState):
            """Execute selected analysts in parallel"""
            print(f"[PARALLEL] Starting parallel execution of analysts: {selected_analysts}")
            print(f"[PARALLEL] State keys available: {list(state.keys())}")
            
            # Check if UI state management is available
            ui_available = False
            try:
                from webui.utils.state import app_state
                ui_available = True
            except ImportError:
                pass
            
            # Update UI status for all analysts as in_progress
            if ui_available:
                for analyst_type in selected_analysts:
                    analyst_name = f"{analyst_type.capitalize()} Analyst"
                    app_state.update_agent_status(analyst_name, "in_progress")
            
            def execute_single_analyst(analyst_info):
                """Execute a single analyst in a separate thread"""
                analyst_type, analyst_node = analyst_info
                
                # Create a deep copy of the state for this analyst
                analyst_state = copy.deepcopy(state)
                
                print(f"[PARALLEL] Starting {analyst_type} analyst")
                
                # Execute the analyst
                try:
                    # Add a small delay before starting analyst execution
                    analyst_call_delay = self.config.get("analyst_call_delay", 0.1)
                    time.sleep(analyst_call_delay)  # Configurable delay before starting
                    
                    result_state = analyst_node(analyst_state)
                    
                    # Check if the analyst made tool calls
                    has_tool_calls = False
                    if result_state.get("messages") and len(result_state["messages"]) > 0:
                        last_message = result_state["messages"][-1]
                        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                            has_tool_calls = True
                    
                    if has_tool_calls:
                        print(f"[PARALLEL] {analyst_type} analyst making tool calls")
                        tool_result = tool_nodes[analyst_type].invoke(result_state)
                        
                        if tool_result and tool_result.get("messages"):
                            # Preserve original state fields when merging tool results
                            merged_state = analyst_state.copy()  # Start with original state
                            
                            # Update with tool result messages
                            merged_state["messages"] = tool_result["messages"]
                            
                            # Add a small delay before making the next LLM call
                            tool_result_delay = self.config.get("tool_result_delay", 0.2)
                            time.sleep(tool_result_delay)  # Configurable delay between tool result and next analyst call
                            
                            # Run analyst again with tool results
                            result_state = analyst_node(merged_state)
                    else:
                        print(f"[PARALLEL] {analyst_type} analyst completed without tool calls")
                    
                    # Clean up messages safely
                    if result_state.get("messages"):
                        # Check if all messages have valid IDs before cleaning
                        valid_messages = [m for m in result_state["messages"] if m is not None and hasattr(m, 'id') and m.id is not None]
                        if valid_messages:
                            # Create a temporary state with only valid messages for cleanup
                            temp_state = {"messages": valid_messages}
                            final_state = delete_nodes[analyst_type](temp_state)
                            # Preserve other fields from result_state
                            for key, value in result_state.items():
                                if key != "messages":
                                    final_state[key] = value
                        else:
                            # No valid messages to clean, use result_state as is
                            final_state = result_state
                    else:
                        final_state = result_state
                    
                    print(f"[PARALLEL] {analyst_type} analyst completed")
                    
                    # Determine report field name
                    report_field = f"{analyst_type}_report"
                    if analyst_type == "social":
                        report_field = "sentiment_report"
                    
                    # Extract report content immediately
                    report_content = None
                    if final_state.get("messages"):
                        last_msg = final_state["messages"][-1]
                        if hasattr(last_msg, 'content') and last_msg.content:
                            report_content = last_msg.content
                    if not report_content and report_field in final_state:
                        report_content = final_state.get(report_field)
                    
                    # Update UI state immediately (real-time update)
                    if ui_available:
                        analyst_name = f"{analyst_type.capitalize()} Analyst"
                        app_state.update_agent_status(analyst_name, "completed")
                        
                        # Store report in UI state immediately for real-time display
                        if report_content:
                            ticker = state.get("company_of_interest", "")
                            if ticker:
                                ui_state = app_state.get_state(ticker)
                                if ui_state:
                                    ui_state["current_reports"][report_field] = report_content
                                    print(f"[PARALLEL] Real-time update: {analyst_type} report ({len(report_content)} chars) stored for {ticker}")
                    
                    return analyst_type, final_state
                    
                except Exception as e:
                    print(f"[PARALLEL] Error in {analyst_type} analyst: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Update UI status to error (completed with issues)
                    if ui_available:
                        analyst_name = f"{analyst_type.capitalize()} Analyst"
                        app_state.update_agent_status(analyst_name, "completed")
                    
                    return analyst_type, analyst_state
            
            # Execute all analysts in parallel with staggered starts
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected_analysts)) as executor:
                # Submit analyst tasks with delays to avoid API rate limits
                future_to_analyst = {}
                analyst_start_delay = self.config.get("analyst_start_delay", 0.5)
                for i, analyst_type in enumerate(selected_analysts):
                    # Add a small delay between each analyst start (configurable to avoid API overload)
                    if i > 0:
                        time.sleep(analyst_start_delay)
                    
                    analyst_node = analyst_nodes[analyst_type]
                    future = executor.submit(execute_single_analyst, (analyst_type, analyst_node))
                    future_to_analyst[future] = analyst_type
                    print(f"[PARALLEL] Submitted {analyst_type} analyst (delay: {i * analyst_start_delay}s)")
                
                # Collect results as they complete
                completed_results = {}
                for future in concurrent.futures.as_completed(future_to_analyst):
                    analyst_type = future_to_analyst[future]
                    try:
                        result_analyst_type, result_state = future.result()
                        completed_results[result_analyst_type] = result_state
                        print(f"[PARALLEL] {result_analyst_type} analyst completed successfully")
                    except Exception as e:
                        print(f"[PARALLEL] {analyst_type} analyst failed: {e}")
                        completed_results[analyst_type] = state  # Use original state as fallback
            
            print(f"[PARALLEL] All analysts completed. Merging results...")
            
            # Merge all results into the final state
            final_state = copy.deepcopy(state)
            
            # Collect all analyst reports
            for analyst_type, result_state in completed_results.items():
                # Determine the report field name
                report_field = f"{analyst_type}_report"
                if analyst_type == "social":
                    report_field = "sentiment_report"
                
                # Try to extract content from the result state
                content = None
                
                # First, try to get from messages
                if result_state.get("messages"):
                    final_message = result_state["messages"][-1]
                    if hasattr(final_message, 'content') and final_message.content:
                        content = final_message.content
                
                # If no content from messages, check if the report field was set directly
                if not content and report_field in result_state:
                    content = result_state.get(report_field)
                
                # Store the content if we have any
                if content:
                    final_state[report_field] = content
                    print(f"[PARALLEL] Stored {analyst_type} report ({len(content)} chars)")
                    print(f"[PARALLEL]   Preview: {content[:150]}..." if len(content) > 150 else f"[PARALLEL]   Content: {content}")
                    
                    # Update report in UI state as well
                    if ui_available:
                        ticker = state.get("ticker", "")
                        if ticker:
                            ui_state = app_state.get_state(ticker)
                            if ui_state:
                                ui_state["current_reports"][report_field] = content
                else:
                    # Ensure the field exists even if empty
                    if report_field not in final_state:
                        final_state[report_field] = ""
                    print(f"[PARALLEL] Warning: No content for {analyst_type} report")
                    # Debug: show what we have in the result_state
                    print(f"[PARALLEL]   result_state keys: {list(result_state.keys())}")
                    if result_state.get("messages"):
                        last_msg = result_state["messages"][-1]
                        print(f"[PARALLEL]   Last message type: {type(last_msg).__name__}")
                        if hasattr(last_msg, 'content'):
                            print(f"[PARALLEL]   Last message content: {last_msg.content[:200] if last_msg.content else 'None'}...")
            
            print(f"[PARALLEL] Parallel analyst execution completed")
            return final_state
        
        return parallel_analysts_execution

    def _create_parallel_risk_round_one_coordinator(self, risk_nodes):
        """Run Risky/Safe/Neutral analysts in parallel for round 1 only."""

        def _append_history(existing: str, new_text: str) -> str:
            existing = (existing or "").strip()
            new_text = (new_text or "").strip()
            if not new_text:
                return existing
            if not existing:
                return new_text
            return f"{existing}\n{new_text}"

        def parallel_risk_round_one(state: AgentState):
            print("[RISK_PARALLEL] Starting first-round parallel execution")

            ui_available = False
            try:
                from webui.utils.state import app_state
                ui_available = True
            except ImportError:
                app_state = None

            if ui_available:
                for analyst_name in ("Risky Analyst", "Safe Analyst", "Neutral Analyst"):
                    app_state.update_agent_status(analyst_name, "in_progress")

            def execute_single(analyst_name: str, analyst_node):
                local_state = copy.deepcopy(state)
                try:
                    result_state = analyst_node(local_state)
                    if ui_available:
                        app_state.update_agent_status(analyst_name, "completed")
                    return analyst_name, result_state
                except Exception as e:
                    print(f"[RISK_PARALLEL] Error in {analyst_name}: {e}")
                    if ui_available:
                        app_state.update_agent_status(analyst_name, "completed")
                    return analyst_name, local_state

            completed_results = {}
            analyst_start_delay = self.config.get(
                "risk_analyst_start_delay",
                self.config.get("analyst_start_delay", 0.5),
            )
            analyst_order = [
                ("Risky Analyst", risk_nodes["Risky"]),
                ("Safe Analyst", risk_nodes["Safe"]),
                ("Neutral Analyst", risk_nodes["Neutral"]),
            ]

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = {}
                for i, (analyst_name, analyst_node) in enumerate(analyst_order):
                    if i > 0:
                        time.sleep(analyst_start_delay)
                    fut = executor.submit(execute_single, analyst_name, analyst_node)
                    futures[fut] = analyst_name

                for future in concurrent.futures.as_completed(futures):
                    analyst_name = futures[future]
                    try:
                        result_name, result_state = future.result()
                        completed_results[result_name] = result_state
                        print(f"[RISK_PARALLEL] {result_name} completed")
                    except Exception as e:
                        print(f"[RISK_PARALLEL] {analyst_name} failed: {e}")
                        completed_results[analyst_name] = copy.deepcopy(state)

            base = copy.deepcopy(state.get("risk_debate_state", {}))
            merged = {
                "history": base.get("history", ""),
                "risky_history": base.get("risky_history", ""),
                "safe_history": base.get("safe_history", ""),
                "neutral_history": base.get("neutral_history", ""),
                "risky_messages": list(base.get("risky_messages", [])),
                "safe_messages": list(base.get("safe_messages", [])),
                "neutral_messages": list(base.get("neutral_messages", [])),
                "latest_speaker": base.get("latest_speaker", "Risky"),
                "current_risky_response": base.get("current_risky_response", ""),
                "current_safe_response": base.get("current_safe_response", ""),
                "current_neutral_response": base.get("current_neutral_response", ""),
                "judge_decision": base.get("judge_decision", ""),
                "count": int(base.get("count", 0)),
            }

            merge_spec = [
                ("Risky Analyst", "current_risky_response", "risky_history", "risky_messages", "Risky"),
                ("Safe Analyst", "current_safe_response", "safe_history", "safe_messages", "Safe"),
                ("Neutral Analyst", "current_neutral_response", "neutral_history", "neutral_messages", "Neutral"),
            ]

            appended_count = 0
            for analyst_name, current_key, history_key, messages_key, speaker_label in merge_spec:
                result_state = completed_results.get(analyst_name, {})
                result_debate = result_state.get("risk_debate_state", {})
                response_text = (result_debate.get(current_key, "") or "").strip()
                if not response_text:
                    continue

                merged[current_key] = response_text
                merged[history_key] = _append_history(merged.get(history_key, ""), response_text)
                merged[messages_key].append(response_text)
                merged["history"] = _append_history(merged.get("history", ""), response_text)
                merged["latest_speaker"] = speaker_label
                appended_count += 1

            merged["count"] = int(base.get("count", 0)) + appended_count
            print(
                f"[RISK_PARALLEL] Merge complete: "
                f"count={merged['count']}, latest={merged['latest_speaker']}"
            )
            return {"risk_debate_state": merged}

        return parallel_risk_round_one

    def setup_graph(
        self, selected_analysts=["market", "social", "news", "fundamentals", "macro"]
    ):
        """Set up and compile the agent workflow graph with configurable parallel/sequential analyst execution.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst  
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")
        
        # Check if parallel execution is enabled
        parallel_mode = self.config.get("parallel_analysts", True)
        print(f"[SETUP] Using {'parallel' if parallel_mode else 'sequential'} analyst execution mode")

        # Create analyst nodes
        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}

        if "market" in selected_analysts:
            analyst_nodes["market"] = self._wrap_node_with_run_logging(
                "Market Analyst",
                create_market_analyst(
                    self.quick_thinking_llm, self.toolkit
                ),
            )
            delete_nodes["market"] = create_msg_delete()
            tool_nodes["market"] = self.tool_nodes["market"]

        if "social" in selected_analysts:
            analyst_nodes["social"] = self._wrap_node_with_run_logging(
                "Social Analyst",
                create_social_media_analyst(
                    self.quick_thinking_llm, self.toolkit
                ),
            )
            delete_nodes["social"] = create_msg_delete()
            tool_nodes["social"] = self.tool_nodes["social"]

        if "news" in selected_analysts:
            analyst_nodes["news"] = self._wrap_node_with_run_logging(
                "News Analyst",
                create_news_analyst(
                    self.quick_thinking_llm, self.toolkit
                ),
            )
            delete_nodes["news"] = create_msg_delete()
            tool_nodes["news"] = self.tool_nodes["news"]

        if "fundamentals" in selected_analysts:
            analyst_nodes["fundamentals"] = self._wrap_node_with_run_logging(
                "Fundamentals Analyst",
                create_fundamentals_analyst(
                    self.quick_thinking_llm, self.toolkit
                ),
            )
            delete_nodes["fundamentals"] = create_msg_delete()
            tool_nodes["fundamentals"] = self.tool_nodes["fundamentals"]

        if "macro" in selected_analysts:
            analyst_nodes["macro"] = self._wrap_node_with_run_logging(
                "Macro Analyst",
                create_macro_analyst(
                    self.quick_thinking_llm, self.toolkit
                ),
            )
            delete_nodes["macro"] = create_msg_delete()
            tool_nodes["macro"] = self.tool_nodes["macro"]

        # Create researcher and manager nodes
        bull_researcher_node = self._wrap_node_with_run_logging(
            "Bull Researcher",
            create_bull_researcher(
                self.quick_thinking_llm, self.bull_memory
            ),
        )
        bear_researcher_node = self._wrap_node_with_run_logging(
            "Bear Researcher",
            create_bear_researcher(
                self.quick_thinking_llm, self.bear_memory
            ),
        )
        research_manager_node = self._wrap_node_with_run_logging(
            "Research Manager",
            create_research_manager(
                self.deep_thinking_llm, self.invest_judge_memory
            ),
        )
        trader_node = self._wrap_node_with_run_logging(
            "Trader",
            create_trader(self.deep_thinking_llm, self.trader_memory, self.config),
        )

        # Create risk analysis nodes
        risky_analyst = self._wrap_node_with_run_logging(
            "Risky Analyst",
            create_risky_debator(self.quick_thinking_llm, self.config),
        )
        neutral_analyst = self._wrap_node_with_run_logging(
            "Neutral Analyst",
            create_neutral_debator(self.quick_thinking_llm, self.config),
        )
        safe_analyst = self._wrap_node_with_run_logging(
            "Safe Analyst",
            create_safe_debator(self.quick_thinking_llm, self.config),
        )
        risk_manager_node = self._wrap_node_with_run_logging(
            "Risk Judge",
            create_risk_manager(
                self.deep_thinking_llm, self.risk_manager_memory, self.config
            ),
        )
        parallel_risk_round_one_mode = self.config.get("parallel_risk_first_round", True)

        # Create workflow
        workflow = StateGraph(AgentState)
        report_context_node = self._wrap_node_with_run_logging(
            "Build Report Context",
            create_report_context_node(self.config),
        )
        workflow.add_node("Build Report Context", report_context_node)

        if parallel_mode:
            # Create parallel analysts coordinator
            parallel_analysts_node = self._create_parallel_analysts_coordinator(
                selected_analysts, analyst_nodes, tool_nodes, delete_nodes
            )
            
            # Add the parallel analysts node
            workflow.add_node("Parallel Analysts", parallel_analysts_node)
            
            # Define edges for parallel execution
            # Start with parallel analysts execution
            workflow.add_edge(START, "Parallel Analysts")
            
            # Build shared context packet before downstream agents consume reports.
            workflow.add_edge("Parallel Analysts", "Build Report Context")
            workflow.add_edge("Build Report Context", "Bull Researcher")
        else:
            # Add individual analyst nodes for sequential execution
            for analyst_type, node in analyst_nodes.items():
                workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
                workflow.add_node(
                    f"Msg Clear {analyst_type.capitalize()}", delete_nodes[analyst_type]
                )
                workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])
            
            # Define edges for sequential execution
            # Start with the first analyst
            first_analyst = selected_analysts[0]
            workflow.add_edge(START, f"{first_analyst.capitalize()} Analyst")

            # Connect analysts in sequence
            for i, analyst_type in enumerate(selected_analysts):
                current_analyst = f"{analyst_type.capitalize()} Analyst"
                current_tools = f"tools_{analyst_type}"
                current_clear = f"Msg Clear {analyst_type.capitalize()}"

                # Add conditional edges for current analyst
                workflow.add_conditional_edges(
                    current_analyst,
                    getattr(self.conditional_logic, f"should_continue_{analyst_type}"),
                    [current_tools, current_clear],
                )
                workflow.add_edge(current_tools, current_analyst)

                # Connect to next analyst or to Bull Researcher if this is the last analyst
                if i < len(selected_analysts) - 1:
                    next_analyst = f"{selected_analysts[i+1].capitalize()} Analyst"
                    workflow.add_edge(current_clear, next_analyst)
                else:
                    workflow.add_edge(current_clear, "Build Report Context")

            workflow.add_edge("Build Report Context", "Bull Researcher")

        # Add other nodes (common to both modes)
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Risky Analyst", risky_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Safe Analyst", safe_analyst)
        workflow.add_node("Risk Judge", risk_manager_node)
        if parallel_risk_round_one_mode:
            parallel_risk_round_one_node = self._wrap_node_with_run_logging(
                "Parallel Risk Round 1",
                self._create_parallel_risk_round_one_coordinator(
                    {
                        "Risky": risky_analyst,
                        "Safe": safe_analyst,
                        "Neutral": neutral_analyst,
                    }
                ),
            )
            workflow.add_node("Parallel Risk Round 1", parallel_risk_round_one_node)

        # Add remaining edges (unchanged from original)
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")
        if parallel_risk_round_one_mode:
            workflow.add_edge("Trader", "Parallel Risk Round 1")
            workflow.add_conditional_edges(
                "Parallel Risk Round 1",
                self.conditional_logic.should_continue_risk_analysis,
                {
                    "Risky Analyst": "Risky Analyst",
                    "Safe Analyst": "Safe Analyst",
                    "Neutral Analyst": "Neutral Analyst",
                    "Risk Judge": "Risk Judge",
                },
            )
        else:
            workflow.add_edge("Trader", "Risky Analyst")
        workflow.add_conditional_edges(
            "Risky Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Safe Analyst": "Safe Analyst",
                "Risk Judge": "Risk Judge",
            },
        )
        workflow.add_conditional_edges(
            "Safe Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Risk Judge": "Risk Judge",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Risky Analyst": "Risky Analyst",
                "Risk Judge": "Risk Judge",
            },
        )
        workflow.add_edge("Risk Judge", END)

        return workflow.compile()
