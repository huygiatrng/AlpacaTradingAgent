# TradingAgents/graph/trading_graph.py

import os
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any, Tuple, List, Optional

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.agents.utils.gpt5_llm import get_chat_model, is_gpt5_model, get_model_params_for_depth, describe_model_params
from tradingagents.run_logger import get_run_audit_logger
from tradingagents.dataflows.interface import set_config
from tradingagents.dataflows.config import get_api_key

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals", "macro"],
        debug=False,
        config: Dict[str, Any] = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Get API key from environment variables or config
        api_key = get_api_key("openai_api_key", "OPENAI_API_KEY")

        # Initialize LLMs with appropriate parameters based on model type and research depth
        deep_think_model = self.config["deep_think_llm"]
        quick_think_model = self.config["quick_think_llm"]
        
        # Get research depth from config (UI setting: "Shallow", "Medium", "Deep")
        # This controls reasoning_effort, verbosity, and temperature based on model role
        research_depth = self.config.get("research_depth", "Medium")
        
        # Convert integer research_depth (from debate rounds) back to string if needed
        if isinstance(research_depth, int):
            depth_map = {1: "Shallow", 2: "Medium", 3: "Deep"}
            research_depth = depth_map.get(research_depth, "Medium")
        
        # Get model parameters based on research depth and model role
        # Quick thinker: prioritizes speed, lower reasoning effort
        # Deep thinker: prioritizes quality, higher reasoning effort
        quick_think_kwargs = get_model_params_for_depth(quick_think_model, research_depth, "quick")
        deep_think_kwargs = get_model_params_for_depth(deep_think_model, research_depth, "deep")
        
        # Log the configuration being used
        quick_params_desc = describe_model_params(quick_think_model, research_depth, "quick")
        deep_params_desc = describe_model_params(deep_think_model, research_depth, "deep")
        print(f"[LLM CONFIG] Research Depth: {research_depth}")
        print(f"[LLM CONFIG] Quick Thinker ({quick_think_model}): {quick_params_desc}")
        print(f"[LLM CONFIG] Deep Thinker ({deep_think_model}): {deep_params_desc}")
        
        self.deep_thinking_llm = get_chat_model(
            deep_think_model, 
            api_key=api_key,
            **deep_think_kwargs
        )
        
        self.quick_thinking_llm = get_chat_model(
            quick_think_model, 
            api_key=api_key,
            **quick_think_kwargs
        )
        
        self.toolkit = Toolkit(config=self.config)

        # Initialize memories
        self.bull_memory = FinancialSituationMemory("bull_memory")
        self.bear_memory = FinancialSituationMemory("bear_memory")
        self.trader_memory = FinancialSituationMemory("trader_memory")
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory")
        self.risk_manager_memory = FinancialSituationMemory("risk_manager_memory")

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config.get("max_debate_rounds", 2), 
            max_risk_discuss_rounds=self.config.get("max_risk_discuss_rounds", 2)
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.toolkit,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.risk_manager_memory,
            self.conditional_logic,
            self.config,
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph
        self.graph = self.graph_setup.setup_graph(selected_analysts)

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources."""
        return {
            "market": ToolNode(
                [
                    # online tools
                    self.toolkit.get_alpaca_data,
                    self.toolkit.get_stockstats_indicators_report_online,
                    # offline tools
                    self.toolkit.get_stockstats_indicators_report,
                    self.toolkit.get_alpaca_data_report,
                    # crypto
                    self.toolkit.get_coindesk_news,
                ]
            ),
            "social": ToolNode(
                [
                    # online tools
                    self.toolkit.get_stock_news_openai,
                    # offline tools
                    self.toolkit.get_reddit_stock_info,
                    # crypto
                    self.toolkit.get_coindesk_news,
                ]
            ),
            "news": ToolNode(
                [
                    # online tools
                    self.toolkit.get_global_news_openai,
                    self.toolkit.get_google_news,
                    # offline tools
                    self.toolkit.get_finnhub_news,
                    self.toolkit.get_reddit_news,
                    # crypto
                    self.toolkit.get_coindesk_news,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # online tools
                    self.toolkit.get_fundamentals_openai,
                    self.toolkit.get_defillama_fundamentals,
                    # offline tools
                    self.toolkit.get_finnhub_company_insider_sentiment,
                    self.toolkit.get_finnhub_company_insider_transactions,
                    self.toolkit.get_simfin_balance_sheet,
                    self.toolkit.get_simfin_cashflow,
                    self.toolkit.get_simfin_income_stmt,
                    # earnings tools
                    self.toolkit.get_earnings_calendar,
                    self.toolkit.get_earnings_surprise_analysis,
                ]
            ),
            "macro": ToolNode(
                [
                    # macro economic tools
                    self.toolkit.get_macro_analysis,
                    self.toolkit.get_economic_indicators,
                    self.toolkit.get_yield_curve_analysis,
                ]
            ),
        }

    def propagate(self, company_name, trade_date):
        """Run the trading agents graph for a company on a specific date."""

        self.ticker = company_name
        run_logger = get_run_audit_logger()

        # Initialize state
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date
        )
        args = self.propagator.get_graph_args()
        run_logger.start_run(
            symbol=company_name,
            trade_date=str(trade_date),
            config=self.config,
            metadata={"debug": self.debug},
        )
        run_logger.log_state_snapshot(
            stage="initial_state",
            snapshot=init_agent_state,
            symbol=company_name,
        )

        try:
            if self.debug:
                # Debug mode with tracing
                trace = []
                for chunk in self.graph.stream(init_agent_state, **args):
                    if len(chunk["messages"]) == 0:
                        pass
                    else:
                        chunk["messages"][-1].pretty_print()
                        trace.append(chunk)

                final_state = trace[-1]
            else:
                # Standard mode without tracing
                final_state = self.graph.invoke(init_agent_state, **args)
        except Exception as e:
            run_logger.finish_run(
                symbol=company_name,
                status="failed",
                error_message=str(e),
            )
            raise

        # Store current state for reflection
        self.curr_state = final_state

        # Log state
        self._log_state(trade_date, final_state)

        # Return decision and processed signal
        try:
            final_signal = self.process_signal(final_state["final_trade_decision"])
            try:
                from webui.utils.state import app_state

                symbol_state = app_state.get_state(company_name) or {}
                filtered_tool_calls = [
                    call for call in app_state.tool_calls_log
                    if call.get("symbol") == company_name
                ]
                run_logger.log_state_snapshot(
                    stage="webui_runtime_context",
                    snapshot={
                        "session_id": symbol_state.get("session_id"),
                        "session_start_time": symbol_state.get("session_start_time"),
                        "agent_prompts": symbol_state.get("agent_prompts", {}),
                        "tool_calls": filtered_tool_calls,
                        "llm_calls_count": app_state.llm_calls_count,
                        "tool_calls_count": app_state.tool_calls_count,
                    },
                    symbol=company_name,
                )
            except Exception:
                pass

            run_logger.finish_run(
                symbol=company_name,
                status="completed",
                final_state=final_state,
                final_signal=final_signal,
            )
            return final_state, final_signal
        except Exception as e:
            run_logger.finish_run(
                symbol=company_name,
                status="failed",
                final_state=final_state,
                error_message=str(e),
            )
            raise

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "report_context_stats": final_state.get("report_context", {}).get("stats", {}),
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "risky_history": final_state["risk_debate_state"]["risky_history"],
                "safe_history": final_state["risk_debate_state"]["safe_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        # Save to file
        directory = Path(f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/")
        directory.mkdir(parents=True, exist_ok=True)

        with open(
            f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/full_states_log.json",
            "w",
        ) as f:
            json.dump(self.log_states_dict, f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """Reflect on decisions and update memory based on returns."""
        self.reflector.reflect_bull_researcher(
            self.curr_state, returns_losses, self.bull_memory
        )
        self.reflector.reflect_bear_researcher(
            self.curr_state, returns_losses, self.bear_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.trader_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        self.reflector.reflect_risk_manager(
            self.curr_state, returns_losses, self.risk_manager_memory
        )

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
