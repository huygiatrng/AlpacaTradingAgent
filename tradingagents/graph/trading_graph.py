# TradingAgents/graph/trading_graph.py

import os
from pathlib import Path
import json
from datetime import date, datetime, timedelta
from typing import Dict, Any, Tuple, List, Optional

import yfinance as yf
from langgraph.prebuilt import ToolNode

from tradingagents.llm_clients import create_llm_client
from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory, TradingMemoryLog
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.openai_model_registry import normalize_model_params, describe_model_params
from tradingagents.run_logger import get_run_audit_logger
from tradingagents.dataflows.config import (
    get_llm_api_key,
    get_openai_base_url,
    is_local_openai_enabled,
    set_config,
)
from tradingagents.dataflows.ticker_utils import TickerUtils, is_crypto_ticker
from tradingagents.dataflows.utils import safe_ticker_component

from .checkpointer import clear_checkpoint, get_checkpointer, thread_id
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
        callbacks: Optional[List] = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        os.makedirs(self.config.get("results_dir", "eval_results"), exist_ok=True)

        # Initialize LLMs with appropriate parameters based on model type and research depth
        deep_think_model = self.config["deep_think_llm"]
        quick_think_model = self.config["quick_think_llm"]
        
        # Research depth now controls debate rounds. Model parameters are explicit
        # per selected model and can be adjusted in the UI.
        research_depth = self.config.get("research_depth", "Medium")
        
        # Convert integer research_depth (from debate rounds) back to string if needed
        if isinstance(research_depth, int):
            depth_map = {1: "Shallow", 2: "Medium", 3: "Deep"}
            research_depth = depth_map.get(research_depth, "Medium")
        
        quick_think_kwargs = normalize_model_params(
            quick_think_model,
            self.config.get("quick_llm_params"),
            role="quick",
        )
        deep_think_kwargs = normalize_model_params(
            deep_think_model,
            self.config.get("deep_llm_params"),
            role="deep",
        )
        
        # Log the configuration being used
        quick_params_desc = describe_model_params(quick_think_model, quick_think_kwargs, "quick")
        deep_params_desc = describe_model_params(deep_think_model, deep_think_kwargs, "deep")
        print(f"[LLM CONFIG] Research Depth: {research_depth} (debate rounds only)")
        print(f"[LLM CONFIG] Quick Thinker ({quick_think_model}): {quick_params_desc}")
        print(f"[LLM CONFIG] Deep Thinker ({deep_think_model}): {deep_params_desc}")

        provider = self.config.get("llm_provider", "openai").lower()
        backend_url = self.config.get("backend_url")
        if provider == "openai" and is_local_openai_enabled():
            provider = "local_openai"
            backend_url = backend_url or get_openai_base_url()
        elif provider == "local_openai":
            backend_url = backend_url or get_openai_base_url()
        if backend_url:
            print(f"[LLM CONFIG] Using OpenAI-compatible endpoint: {backend_url}")

        base_llm_kwargs = self._get_provider_kwargs(provider)
        if self.callbacks:
            base_llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=provider,
            model=deep_think_model,
            base_url=backend_url,
            api_key=get_llm_api_key(provider),
            model_role="deep",
            **base_llm_kwargs,
            **deep_think_kwargs,
        )
        quick_client = create_llm_client(
            provider=provider,
            model=quick_think_model,
            base_url=backend_url,
            api_key=get_llm_api_key(provider),
            model_role="quick",
            **base_llm_kwargs,
            **quick_think_kwargs,
        )

        self.deep_thinking_llm = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()
        
        self.toolkit = Toolkit(config=self.config)

        # Initialize memories
        self.bull_memory = FinancialSituationMemory("bull_memory")
        self.bear_memory = FinancialSituationMemory("bear_memory")
        self.trader_memory = FinancialSituationMemory("trader_memory")
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory")
        self.risk_manager_memory = FinancialSituationMemory("risk_manager_memory")
        self.memory_log = TradingMemoryLog(self.config)

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

        self.propagator = Propagator(max_recur_limit=self.config.get("max_recur_limit", 200))
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph
        self.workflow = self.graph_setup.setup_graph(selected_analysts)
        self.graph = self.workflow.compile()
        self._checkpointer_ctx = None

    def _get_provider_kwargs(self, provider: str) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        provider = (provider or "openai").lower()
        if provider == "google" and self.config.get("google_thinking_level"):
            kwargs["thinking_level"] = self.config["google_thinking_level"]
        elif provider in ("openai", "local_openai") and self.config.get("openai_reasoning_effort"):
            kwargs["reasoning_effort"] = self.config["openai_reasoning_effort"]
        elif provider == "anthropic" and self.config.get("anthropic_effort"):
            kwargs["effort"] = self.config["anthropic_effort"]
        return kwargs

    def _graph_for_run(self, ticker: str, trade_date: str):
        """Return a compiled graph and optional checkpointer context for one run."""
        if not self.config.get("checkpoint_enabled", False):
            return self.graph, None

        checkpointer_ctx = get_checkpointer(self.config["data_cache_dir"], ticker)
        checkpointer = checkpointer_ctx.__enter__()
        return self.workflow.compile(checkpointer=checkpointer), checkpointer_ctx

    def _graph_args_for_run(self, ticker: str, trade_date: str) -> Dict[str, Any]:
        args = self.propagator.get_graph_args()
        if self.config.get("checkpoint_enabled", False):
            args["config"].setdefault("configurable", {})["thread_id"] = thread_id(
                ticker, str(trade_date)
            )
        return args

    def _ticker_for_yfinance(self, ticker: str) -> str:
        try:
            return TickerUtils.convert_for_api(ticker, "yahoo")
        except Exception:
            return ticker.replace("/", "-")

    def _benchmark_for(self, ticker: str) -> Optional[str]:
        if is_crypto_ticker(ticker):
            base = self._ticker_for_yfinance(ticker).split("-")[0].upper()
            return None if base == "BTC" else "BTC-USD"
        return None if ticker.upper() == "SPY" else "SPY"

    def _fetch_return(self, ticker: str, start_date: date, holding_days: int) -> Optional[float]:
        if datetime.now().date() < start_date + timedelta(days=holding_days):
            return None

        symbol = self._ticker_for_yfinance(ticker)
        start = start_date.isoformat()
        end = (start_date + timedelta(days=holding_days + 7)).isoformat()
        try:
            data = yf.download(
                symbol,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                actions=False,
                threads=False,
            )
        except Exception:
            return None

        if data is None or data.empty or "Close" not in data:
            return None

        close = data["Close"]
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        close = close.dropna()
        if len(close) < 2:
            return None
        start_price = float(close.iloc[0])
        end_price = float(close.iloc[-1])
        if start_price == 0:
            return None
        return (end_price / start_price) - 1.0

    def _resolve_memory_log_outcomes(self, ticker: str, trade_date: str) -> None:
        holding_days = int(self.config.get("memory_outcome_holding_days", 5))
        try:
            current_date = datetime.strptime(str(trade_date), "%Y-%m-%d").date()
        except ValueError:
            current_date = date.today()

        for entry in self.memory_log.get_pending_entries(ticker):
            entry_date_text = entry.get("date")
            if not entry_date_text:
                continue
            try:
                entry_date = datetime.strptime(entry_date_text, "%Y-%m-%d").date()
            except ValueError:
                continue
            if entry_date >= current_date:
                continue

            raw_return = self._fetch_return(ticker, entry_date, holding_days)
            if raw_return is None:
                continue

            benchmark = self._benchmark_for(ticker)
            benchmark_return = (
                self._fetch_return(benchmark, entry_date, holding_days)
                if benchmark
                else None
            )
            alpha_return = (
                raw_return - benchmark_return
                if benchmark_return is not None
                else None
            )
            try:
                reflection = self.reflector.reflect_on_final_decision(
                    entry.get("decision", ""), raw_return, alpha_return
                )
            except Exception:
                reflection = "Outcome resolved, but reflection generation failed."
            self.memory_log.update_with_outcome(
                ticker=ticker,
                trade_date=entry_date_text,
                raw_return=raw_return,
                alpha_return=alpha_return,
                holding_days=holding_days,
                reflection=reflection,
            )

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources."""
        news_tools = [
            self.toolkit.get_google_news,
            self.toolkit.get_finnhub_news_recent,
            self.toolkit.get_coindesk_news,
        ]
        if self.config.get("news_global_openai_enabled", False):
            news_tools.insert(0, self.toolkit.get_global_news_openai)

        return {
            "market": ToolNode(
                [
                    # online tools
                    self.toolkit.get_alpaca_data,
                    self.toolkit.get_stockstats_indicators_report_online,
                    # offline tools
                    self.toolkit.get_stockstats_indicators_report,
                    self.toolkit.get_alpaca_data_report,
                ]
            ),
            "social": ToolNode(
                [
                    # online tools
                    self.toolkit.get_stock_news_openai,
                    # direct social tools
                    self.toolkit.get_reddit_stock_info,
                    self.toolkit.get_reddit_news,
                ]
            ),
            "news": ToolNode(news_tools),
            "fundamentals": ToolNode(
                [
                    # online tools
                    self.toolkit.get_fundamentals_openai,
                    self.toolkit.get_defillama_fundamentals,
                    # direct data tools
                    self.toolkit.get_finnhub_company_insider_sentiment,
                    self.toolkit.get_finnhub_company_insider_transactions,
                    self.toolkit.get_simfin_balance_sheet,
                    self.toolkit.get_simfin_cashflow,
                    self.toolkit.get_simfin_income_stmt,
                ]
            ),
            "macro": ToolNode(
                [
                    # macro economic tools
                    self.toolkit.get_macro_analysis,
                    self.toolkit.get_economic_indicators,
                    self.toolkit.get_yield_curve_analysis,
                    self.toolkit.get_macro_news_openai,
                ]
            ),
        }

    def propagate(self, company_name, trade_date):
        """Run the trading agents graph for a company on a specific date."""

        self.ticker = company_name
        run_logger = get_run_audit_logger()
        self._resolve_memory_log_outcomes(company_name, str(trade_date))

        # Initialize state
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date
        )
        args = self._graph_args_for_run(company_name, str(trade_date))
        graph, checkpointer_ctx = self._graph_for_run(company_name, str(trade_date))
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
                for chunk in graph.stream(init_agent_state, **args):
                    if len(chunk["messages"]) == 0:
                        pass
                    else:
                        chunk["messages"][-1].pretty_print()
                        trace.append(chunk)

                final_state = trace[-1]
            else:
                # Standard mode without tracing
                final_state = graph.invoke(init_agent_state, **args)
        except Exception as e:
            run_logger.finish_run(
                symbol=company_name,
                status="failed",
                error_message=str(e),
            )
            raise
        finally:
            if checkpointer_ctx is not None:
                checkpointer_ctx.__exit__(None, None, None)

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
            self.memory_log.store_decision(
                ticker=company_name,
                trade_date=str(trade_date),
                final_trade_decision=final_state["final_trade_decision"],
                trading_mode=final_state.get(
                    "trading_mode",
                    self.config.get("trading_mode", "investment"),
                ),
            )
            if self.config.get("checkpoint_enabled", False):
                clear_checkpoint(self.config["data_cache_dir"], company_name, str(trade_date))
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
        safe_ticker = safe_ticker_component(self.ticker)
        directory = (
            Path(self.config.get("results_dir", "eval_results"))
            / safe_ticker
            / "TradingAgentsStrategy_logs"
        )
        directory.mkdir(parents=True, exist_ok=True)

        with open(directory / "full_states_log.json", "w", encoding="utf-8") as f:
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
