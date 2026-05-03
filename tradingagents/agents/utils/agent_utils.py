from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from typing import List
from typing import Annotated
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import RemoveMessage
from langchain_core.tools import tool
from datetime import date, timedelta, datetime
import functools
import pandas as pd
import os
from dateutil.relativedelta import relativedelta
from langchain_openai import ChatOpenAI
import tradingagents.dataflows.interface as interface
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.run_logger import get_run_audit_logger
from tradingagents.dataflows.config import get_api_key
import json
import time
from functools import wraps


TOOL_MIN_OUTPUT_CHARS = {
    "get_stock_news_openai": 500,
    "get_global_news_openai": 500,
    "get_fundamentals_openai": 350,
    "get_macro_analysis": 280,
}

DEFAULT_SEMANTIC_RETRY_DISABLED_TOOLS = {
    "get_global_news_openai",
    "get_macro_news_openai",
}

INTERACTIVE_FOLLOWUP_PATTERNS = (
    "would you like",
    "if you'd like",
    "if you would like",
    "if you want, i can",
    "if you want i can",
    "do you want me to",
    "i can do this, but i need",
    "should i",
    "want me to",
    "i can fetch",
    "which follow-up",
    "which follow up",
)

VALID_SHORT_OUTPUT_PATTERNS = (
    "no earnings data found",
    "not found for",
    "no data available",
    "fallback used because openai",
)


def _is_trailing_interactive_followup(text: str) -> bool:
    if not text:
        return False
    tail = (
        text.strip()
        .lower()
        .replace("’", "'")
        .replace("‘", "'")
        .replace("‑", "-")
        .replace("–", "-")
        .replace("—", "-")
    )[-700:]
    return any(pattern in tail for pattern in INTERACTIVE_FOLLOWUP_PATTERNS)


def _strip_trailing_interactive_followup(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned

    lines = cleaned.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()

    removed_any = False
    while lines:
        tail = (
            lines[-1]
            .strip()
            .lower()
            .replace("’", "'")
            .replace("‘", "'")
            .replace("‑", "-")
            .replace("–", "-")
            .replace("—", "-")
        )
        if any(pattern in tail for pattern in INTERACTIVE_FOLLOWUP_PATTERNS):
            removed_any = True
            lines.pop()
            while lines and lines[-1].strip().startswith(("- ", "* ")):
                lines.pop()
            while lines and not lines[-1].strip():
                lines.pop()
            continue
        break

    candidate = "\n".join(lines).strip()
    if removed_any and candidate:
        return candidate
    return cleaned


def _score_output_quality(tool_name: str, output: object) -> dict:
    text = str(output or "").strip()
    lower = (
        text.lower()
        .replace("’", "'")
        .replace("‘", "'")
        .replace("‑", "-")
        .replace("–", "-")
        .replace("—", "-")
    )
    flags = []
    min_chars = TOOL_MIN_OUTPUT_CHARS.get(tool_name, 0)
    trailing_interactive = _is_trailing_interactive_followup(text)

    if not text:
        flags.append("empty_output")

    if any(pattern in lower for pattern in INTERACTIVE_FOLLOWUP_PATTERNS):
        substantial_output = len(text) >= max(1200, min_chars * 2)
        if not (trailing_interactive and substantial_output):
            flags.append("interactive_followup")

    if lower.startswith("error:") or lower.startswith("exception:"):
        flags.append("error_prefixed_output")

    if min_chars and len(text) < min_chars:
        if not any(pattern in lower for pattern in VALID_SHORT_OUTPUT_PATTERNS):
            flags.append("undersized_output")

    suspect = any(
        flag in ("empty_output", "interactive_followup", "undersized_output")
        for flag in flags
    )
    retry_recommended = suspect and "error_prefixed_output" not in flags

    score = 1.0
    for flag in flags:
        if flag in ("empty_output", "interactive_followup"):
            score -= 0.4
        elif flag == "undersized_output":
            score -= 0.2
        elif flag == "error_prefixed_output":
            score -= 0.1
    score = max(0.0, round(score, 3))

    return {
        "score": score,
        "flags": flags,
        "is_suspect": suspect,
        "retry_recommended": retry_recommended,
        "output_chars": len(text),
        "output_preview": text[:220],
        "trailing_interactive_followup": trailing_interactive,
    }


def _semantic_retry_disabled_tools(config: dict | None = None) -> set[str]:
    config = config or {}
    configured = config.get(
        "tool_semantic_retry_disabled_tools",
        DEFAULT_SEMANTIC_RETRY_DISABLED_TOOLS,
    )
    if configured is None:
        return set()
    if isinstance(configured, str):
        return {name.strip() for name in configured.split(",") if name.strip()}
    try:
        return {str(name).strip() for name in configured if str(name).strip()}
    except TypeError:
        return set(DEFAULT_SEMANTIC_RETRY_DISABLED_TOOLS)


def _should_retry_tool_output(
    tool_name: str,
    *,
    uses_web_search: bool,
    semantic_retry_enabled: bool,
    retry_count: int,
    max_semantic_retries: int,
    quality: dict,
    config: dict | None = None,
) -> bool:
    if not (
        uses_web_search
        and semantic_retry_enabled
        and retry_count < max_semantic_retries
        and quality.get("retry_recommended", False)
    ):
        return False

    if tool_name in _semantic_retry_disabled_tools(config):
        return False

    # Avoid expensive second global-news web search when the first result
    # is already substantive, even if it ends with an interactive tail.
    if (
        tool_name == "get_global_news_openai"
        and quality.get("output_chars", 0)
        >= TOOL_MIN_OUTPUT_CHARS.get(tool_name, 0)
    ):
        return False

    return True


def timing_wrapper(analyst_type, timeout_seconds=120, uses_web_search=False):
    """
    Decorator to time function calls and track them for UI display with timeout protection
    
    Args:
        analyst_type: Type of analyst (MARKET, SOCIAL, etc.)
        timeout_seconds: Maximum execution time allowed (default 120s)
        uses_web_search: If True, applies a small configurable timeout extension for web tools
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Start timing
            start_time = time.time()
            
            # Get the function (tool) name
            tool_name = func.__name__
            
            # Timeout handling using ThreadPoolExecutor (cross-platform)
            import concurrent.futures
            
            def run_function():
                return func(*args, **kwargs)
            
            # Format tool inputs for display
            input_summary = {}
            input_summary_full = {}
            
            # Get function signature to map args to parameter names
            import inspect
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            
            # Map positional args to parameter names
            for i, arg in enumerate(args):
                if i < len(param_names):
                    param_name = param_names[i]
                    # Truncate long string arguments for display
                    input_summary_full[param_name] = arg
                    if isinstance(arg, str) and len(arg) > 100:
                        input_summary[param_name] = arg[:97] + "..."
                    else:
                        input_summary[param_name] = arg
            
            # Add keyword arguments
            for key, value in kwargs.items():
                input_summary_full[key] = value
                if isinstance(value, str) and len(value) > 100:
                    input_summary[key] = value[:97] + "..."
                else:
                    input_summary[key] = value

            print(f"[{analyst_type}] 🔧 Starting tool '{tool_name}' with inputs: {input_summary}")
            
            # Notify the state management system of tool call execution
            try:
                from webui.utils.state import app_state
                import datetime
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                
                # Execute the function with timeout protection + semantic retry
                semantic_retry_enabled = Toolkit._config.get("tool_semantic_retry_enabled", True)
                max_semantic_retries = int(Toolkit._config.get("tool_semantic_retry_max_retries", 1))
                retry_backoff_seconds = float(Toolkit._config.get("tool_semantic_retry_backoff_seconds", 0.8))
                base_timeout_seconds = float(timeout_seconds)
                web_search_timeout_extension = 0.0
                if uses_web_search:
                    web_search_timeout_extension = float(
                        Toolkit._config.get("web_search_timeout_extension_seconds", 45)
                    )
                effective_timeout_seconds = max(
                    10.0, base_timeout_seconds + web_search_timeout_extension
                )

                def _execute_once():
                    # Do not use context manager here: on timeout, __exit__ waits for worker completion.
                    # That defeats timeout enforcement and can block for several extra minutes.
                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    future = executor.submit(run_function)
                    try:
                        return future.result(timeout=effective_timeout_seconds)
                    except concurrent.futures.TimeoutError:
                        future.cancel()
                        raise
                    finally:
                        executor.shutdown(wait=False, cancel_futures=True)

                retry_count = 0
                quality_details = {}
                first_quality = {}
                result = None
                best_result = None
                best_quality = None

                while True:
                    try:
                        result = _execute_once()
                    except concurrent.futures.TimeoutError:
                        elapsed = time.time() - start_time
                        timeout_msg = (
                            f"TIMEOUT: Tool '{tool_name}' exceeded {effective_timeout_seconds:.1f}s "
                            f"limit (stopped at {elapsed:.1f}s)"
                        )
                        print(f"[{analyst_type}] ⏰ {timeout_msg}")

                        tool_call_info = {
                            "timestamp": timestamp,
                            "tool_name": tool_name,
                            "inputs": input_summary,
                            "output": f"TIMEOUT ERROR: {timeout_msg}",
                            "execution_time": f"{elapsed:.2f}s",
                            "status": "timeout",
                            "agent_type": analyst_type,
                            "symbol": getattr(app_state, 'analyzing_symbol', None) or getattr(app_state, 'current_symbol', None),
                            "error_details": {
                                "error_type": "TimeoutError",
                                "timeout_seconds": effective_timeout_seconds,
                                "actual_time": elapsed
                            },
                            "retry_count": retry_count,
                        }

                        app_state.tool_calls_log.append(tool_call_info)
                        app_state.tool_calls_count = len(app_state.tool_calls_log)
                        app_state.needs_ui_update = True
                        get_run_audit_logger().log_tool_call(
                            tool_name=tool_name,
                            inputs=input_summary_full,
                            output=tool_call_info["output"],
                            status="timeout",
                            execution_time_seconds=elapsed,
                            agent_type=analyst_type,
                            symbol=tool_call_info["symbol"],
                            error_details=tool_call_info.get("error_details", {}),
                            quality_details={"flags": ["timeout"], "is_suspect": True},
                            retry_count=retry_count,
                        )

                        return (
                            f"Error: Tool '{tool_name}' timed out after {effective_timeout_seconds:.1f}s. "
                            "This may indicate network issues, API problems, or insufficient data."
                        )

                    # Check for very slow execution
                    partial_elapsed = time.time() - start_time
                    if partial_elapsed > 120:
                        print(f"[{analyst_type}] ⚠️ Slow execution warning: {tool_name} took {partial_elapsed:.1f}s")

                    if isinstance(result, str):
                        sanitized_result = _strip_trailing_interactive_followup(result)
                        if sanitized_result != result:
                            result = sanitized_result

                    quality = _score_output_quality(tool_name, result)
                    if not first_quality:
                        first_quality = quality

                    if (
                        best_quality is None
                        or quality["score"] > best_quality["score"]
                        or (
                            quality["score"] == best_quality["score"]
                            and quality["output_chars"] > best_quality["output_chars"]
                        )
                    ):
                        best_quality = quality
                        best_result = result

                    should_retry = _should_retry_tool_output(
                        tool_name,
                        uses_web_search=uses_web_search,
                        semantic_retry_enabled=semantic_retry_enabled,
                        retry_count=retry_count,
                        max_semantic_retries=max_semantic_retries,
                        quality=quality,
                        config=Toolkit._config,
                    )
                    if not should_retry:
                        quality_details = quality
                        break

                    retry_count += 1
                    get_run_audit_logger().log_event(
                        event_type="tool_retry",
                        symbol=getattr(app_state, "analyzing_symbol", None) or getattr(app_state, "current_symbol", None),
                        payload={
                            "tool_name": tool_name,
                            "agent_type": analyst_type,
                            "retry_count": retry_count,
                            "reason": quality.get("flags", []),
                        },
                    )
                    print(
                        f"[{analyst_type}] 🔄 Retrying tool '{tool_name}' (attempt {retry_count + 1}) "
                        f"due to quality flags: {quality.get('flags', [])}"
                    )
                    time.sleep(retry_backoff_seconds)

                if best_result is not None and best_quality is not None:
                    result = best_result
                    quality_details = best_quality
                
                # Calculate execution time
                elapsed = time.time() - start_time
                print(f"[{analyst_type}] ✅ Tool '{tool_name}' completed in {elapsed:.2f}s")
                
                # Format the result for display (truncate if too long)
                result_summary = result
                
                # Store the complete tool call information including the output
                # Get current symbol from app_state for filtering
                current_symbol = getattr(app_state, 'analyzing_symbol', None) or getattr(app_state, 'current_symbol', None)
                
                tool_call_info = {
                    "timestamp": timestamp,
                    "tool_name": tool_name,
                    "inputs": input_summary,
                    "output": result_summary,
                    "execution_time": f"{elapsed:.2f}s",
                    "status": "success",
                    "agent_type": analyst_type,  # Add agent type for filtering
                    "symbol": current_symbol,  # Add symbol for filtering
                    "retry_count": retry_count,
                    "quality_details": quality_details,
                    "initial_quality": first_quality or quality_details,
                }
                
                app_state.tool_calls_log.append(tool_call_info)
                app_state.tool_calls_count = len(app_state.tool_calls_log)
                app_state.needs_ui_update = True
                print(f"[TOOL TRACKER] Registered tool call: {tool_name} for {analyst_type} (Total: {app_state.tool_calls_count})")
                get_run_audit_logger().log_tool_call(
                    tool_name=tool_name,
                    inputs=input_summary_full,
                    output=result_summary,
                    status="success",
                    execution_time_seconds=elapsed,
                    agent_type=analyst_type,
                    symbol=current_symbol,
                    quality_details=quality_details,
                    retry_count=retry_count,
                )
                
                return result
                
            except Exception as e:
                elapsed = time.time() - start_time
                
                # Enhanced error logging with detailed debugging info
                error_details = {
                    "tool_name": tool_name,
                    "inputs": input_summary,
                    "execution_time": f"{elapsed:.2f}s",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
                
                # Add specific error handling for common issues
                detailed_error = str(e)
                if "api key" in str(e).lower():
                    detailed_error = f"API KEY ERROR: {str(e)}\n💡 SOLUTION: Check your API key configuration in the .env file"
                elif "organization" in str(e).lower() and "verification" in str(e).lower():
                    detailed_error = f"OPENAI ORG ERROR: {str(e)}\n💡 SOLUTION: Your OpenAI organization may need verification or you may have billing issues"
                elif "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    detailed_error = f"TIMEOUT ERROR: {str(e)}\n💡 SOLUTION: Network or API service may be slow. Try again in a few minutes"
                elif "rate limit" in str(e).lower():
                    detailed_error = f"RATE LIMIT ERROR: {str(e)}\n💡 SOLUTION: You've hit API rate limits. Wait before retrying"
                elif "connection" in str(e).lower():
                    detailed_error = f"CONNECTION ERROR: {str(e)}\n💡 SOLUTION: Check your internet connection and API service status"
                elif "insufficient data" in str(e).lower():
                    detailed_error = f"DATA ERROR: {str(e)}\n💡 SOLUTION: Try a different date range or check if the symbol is correct"
                
                print(f"[{analyst_type}] ❌ Tool '{tool_name}' failed after {elapsed:.2f}s")
                print(f"[{analyst_type}] 🔍 ERROR DETAILS:")
                print(f"   Error Type: {error_details['error_type']}")
                print(f"   Error Message: {detailed_error}")
                print(f"   Tool Inputs: {input_summary}")
                
                # Store the failed tool call information with enhanced details
                try:
                    from webui.utils.state import app_state
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    
                    # Get current symbol from app_state for filtering
                    current_symbol = getattr(app_state, 'analyzing_symbol', None) or getattr(app_state, 'current_symbol', None)
                    
                    tool_call_info = {
                        "timestamp": timestamp,
                        "tool_name": tool_name,
                        "inputs": input_summary,
                        "output": f"ERROR ({error_details['error_type']}): {detailed_error}",
                        "execution_time": f"{elapsed:.2f}s",
                        "status": "error",
                        "agent_type": analyst_type,  # Add agent type for filtering
                        "symbol": current_symbol,  # Add symbol for filtering
                        "error_details": error_details  # Add structured error details
                    }
                    
                    app_state.tool_calls_log.append(tool_call_info)
                    app_state.tool_calls_count = len(app_state.tool_calls_log)
                    app_state.needs_ui_update = True
                    print(f"[TOOL TRACKER] Registered failed tool call: {tool_name} for {analyst_type} (Total: {app_state.tool_calls_count})")
                    get_run_audit_logger().log_tool_call(
                        tool_name=tool_name,
                        inputs=input_summary_full,
                        output=tool_call_info["output"],
                        status="error",
                        execution_time_seconds=elapsed,
                        agent_type=analyst_type,
                        symbol=current_symbol,
                        error_details=error_details,
                        quality_details={"flags": ["runtime_error"], "is_suspect": True},
                        retry_count=0,
                    )
                except Exception as track_error:
                    print(f"[TOOL TRACKER] Failed to track failed tool call: {track_error}")
                
                raise  # Re-raise the exception
                
        return wrapper
    return decorator


def create_msg_delete():
    def delete_messages(state):
        """To prevent message history from overflowing, regularly clear message history after a stage of the pipeline is done"""
        messages = state["messages"]
        return {"messages": [RemoveMessage(id=m.id) for m in messages]}

    return delete_messages


class Toolkit:
    _config = DEFAULT_CONFIG.copy()

    @classmethod
    def update_config(cls, config):
        """Update the class-level configuration."""
        cls._config.update(config)

    @property
    def config(self):
        """Access the configuration."""
        return self._config

    def __init__(self, config=None):
        if config:
            self.update_config(config)

    @staticmethod
    def _has_key(config_key: str, env_key: str) -> bool:
        try:
            return bool(get_api_key(config_key, env_key))
        except Exception:
            return False

    def has_openai_web_search(self) -> bool:
        return self._has_key("openai_api_key", "OPENAI_API_KEY")

    def has_finnhub(self) -> bool:
        return self._has_key("finnhub_api_key", "FINNHUB_API_KEY")

    def has_alpaca_credentials(self) -> bool:
        return self._has_key("alpaca_api_key", "ALPACA_API_KEY") and self._has_key(
            "alpaca_secret_key", "ALPACA_SECRET_KEY"
        )

    def has_fred(self) -> bool:
        return self._has_key("fred_api_key", "FRED_API_KEY")

    def has_coindesk(self) -> bool:
        return self._has_key("coindesk_api_key", "COINDESK_API_KEY")

    def has_simfin_data(self) -> bool:
        data_dir = self.config.get("data_dir", "")
        if not data_dir:
            return False

        statement_paths = {
            "balance": ("balance_sheets", "us-balance-{freq}.csv"),
            "cashflow": ("cashflow", "us-cashflow-{freq}.csv"),
            "income": ("income_statements", "us-income-{freq}.csv"),
        }

        # Consider SimFin available if one full frequency set exists.
        for freq in ("annual", "quarterly"):
            all_present = True
            for folder, pattern in statement_paths.values():
                full_path = os.path.join(
                    data_dir,
                    "simfin_data_all",
                    folder,
                    "companies",
                    "us",
                    pattern.format(freq=freq),
                )
                if not os.path.exists(full_path):
                    all_present = False
                    break
            if all_present:
                return True

        return False

    @staticmethod
    @tool
    @timing_wrapper("NEWS")
    def get_reddit_news(
        curr_date: Annotated[str, "Date you want to get news for in yyyy-mm-dd format"],
    ) -> str:
        """
        Retrieve global news from Reddit within a specified time frame.
        Args:
            curr_date (str): Date you want to get news for in yyyy-mm-dd format
        Returns:
            str: A formatted dataframe containing the latest global news from Reddit in the specified time frame.
        """
        
        global_news_result = interface.get_reddit_global_news(curr_date, 7, 5)

        return global_news_result

    @staticmethod
    @tool
    @timing_wrapper("NEWS")
    def get_finnhub_news_recent(
        ticker: Annotated[
            str,
            "Search query of a company, e.g. 'AAPL, TSM, NVDA'",
        ],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
        look_back_days: Annotated[int, "How many days to look back"] = 7,
    ):
        """
        Retrieve recent company news from Finnhub over a lookback window.
        Args:
            ticker (str): Company ticker symbol
            curr_date (str): Current date in yyyy-mm-dd format
            look_back_days (int): Days to look back from curr_date
        Returns:
            str: Formatted Finnhub news report
        """

        finnhub_news_result = interface.get_finnhub_news(
            ticker, curr_date, look_back_days
        )

        return finnhub_news_result

    @staticmethod
    @tool
    @timing_wrapper("NEWS")
    def get_finnhub_news(
        ticker: Annotated[
            str,
            "Search query of a company, e.g. 'AAPL, TSM, etc.",
        ],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    ):
        """
        Retrieve the latest news about a given stock from Finnhub within a date range
        Args:
            ticker (str): Ticker of a company. e.g. AAPL, TSM
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
        Returns:
            str: A formatted dataframe containing news about the company within the date range from start_date to end_date
        """

        end_date_str = end_date

        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        look_back_days = (end_date - start_date).days

        finnhub_news_result = interface.get_finnhub_news(
            ticker, end_date_str, look_back_days
        )

        return finnhub_news_result

    @staticmethod
    @tool
    @timing_wrapper("SOCIAL")
    def get_reddit_stock_info(
        ticker: Annotated[
            str,
            "Ticker of a company. e.g. AAPL, TSM",
        ],
        curr_date: Annotated[str, "Current date you want to get news for"],
    ) -> str:
        """
        Retrieve the latest news about a given stock from Reddit, given the current date.
        Args:
            ticker (str): Ticker of a company. e.g. AAPL, TSM
            curr_date (str): current date in yyyy-mm-dd format to get news for
        Returns:
            str: A formatted dataframe containing the latest news about the company on the given date
        """

        stock_news_results = interface.get_reddit_company_news(ticker, curr_date, 7, 5)

        return stock_news_results

    @staticmethod
    @tool
    @timing_wrapper("MARKET")
    def get_alpaca_data(
        symbol: Annotated[str, "ticker symbol (stocks: AAPL, TSM; crypto: ETH/USD, BTC/USD)"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
        timeframe: Annotated[str, "Timeframe for data: 1Min, 5Min, 15Min, 1Hour, 1Day"] = "1Day",
    ) -> str:
        """
        Retrieve stock and cryptocurrency price data from Alpaca.
        For crypto symbols, use format with slash: ETH/USD, BTC/USD, SOL/USD
        For stock symbols, use standard format: AAPL, TSM, NVDA
        Args:
            symbol (str): Ticker symbol - stocks: AAPL, TSM; crypto: ETH/USD, BTC/USD
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
            timeframe (str): Timeframe for data (1Min, 5Min, 15Min, 1Hour, 1Day)
        Returns:
            str: A formatted dataframe containing the price data for the specified ticker symbol in the specified date range.
        """

        result_data = interface.get_alpaca_data(symbol, start_date, end_date, timeframe)

        return result_data

    @staticmethod
    @tool
    @timing_wrapper("MARKET")
    def get_stockstats_indicators_report(
        symbol: Annotated[str, "ticker symbol of the company"],
        indicator: Annotated[
            str, "technical indicator to get the analysis and report of"
        ],
        curr_date: Annotated[
            str, "The current trading date you are trading on, YYYY-mm-dd"
        ],
        look_back_days: Annotated[int, "how many days to look back"] = 30,
    ) -> str:
        """
        Retrieve stock stats indicators for a given ticker symbol and indicator.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            indicator (str): Technical indicator to get the analysis and report of
            curr_date (str): The current trading date you are trading on, YYYY-mm-dd
            look_back_days (int): How many days to look back, default is 30
        Returns:
            str: A formatted dataframe containing the stock stats indicators for the specified ticker symbol and indicator.
        """

        result_stockstats = interface.get_stock_stats_indicators_window(
            symbol, indicator, curr_date, look_back_days, False
        )

        return result_stockstats

    @staticmethod
    @tool
    @timing_wrapper("MARKET")
    def get_stockstats_indicators_report_online(
        symbol: Annotated[str, "ticker symbol (stocks: AAPL, TSM; crypto: ETH/USD, BTC/USD)"],
        indicator: Annotated[
            str, "technical indicator to get the analysis and report of"
        ],
        curr_date: Annotated[
            str, "The current trading date you are trading on, YYYY-mm-dd"
        ],
        look_back_days: Annotated[int, "how many days to look back"] = 30,
        timeframe: Annotated[str, "Chart timeframe: 1Hour, 4Hour, 1Day"] = "1Day",
        max_points: Annotated[int, "Maximum historical data points to return"] = 30,
    ) -> str:
        """
        Retrieve indicator history with configurable indicator + timeframe.
        This tool is designed for iterative technical analysis:
        call it multiple times with different indicator/timeframe combinations.

        Args:
            symbol (str): Ticker symbol - stocks: AAPL, TSM; crypto: ETH/USD, BTC/USD
            indicator (str): Indicator name (e.g. rsi_14, macd, close_8_ema, atr_14, all)
            curr_date (str): The current trading date you are trading on, YYYY-mm-dd
            look_back_days (int): Calendar days to include in history
            timeframe (str): 1Hour, 4Hour, or 1Day
            max_points (int): Max rows returned in history table
        Returns:
            str: Indicator history report with latest values and a history table.
        """
        return interface.get_stockstats_indicator_history(
            symbol=symbol,
            indicator=indicator,
            curr_date=curr_date,
            timeframe=timeframe,
            look_back_days=look_back_days,
            max_points=max_points,
        )

    @staticmethod
    @tool
    @timing_wrapper("FUNDAMENTALS")
    def get_finnhub_company_insider_sentiment(
        ticker: Annotated[str, "ticker symbol for the company"],
        curr_date: Annotated[
            str,
            "current date of you are trading at, yyyy-mm-dd",
        ],
    ):
        """
        Retrieve insider sentiment information about a company (retrieved from public SEC information) for the past 30 days
        Args:
            ticker (str): ticker symbol of the company
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
            str: a report of the sentiment in the past 30 days starting at curr_date
        """

        data_sentiment = interface.get_finnhub_company_insider_sentiment(
            ticker, curr_date, 30
        )

        return data_sentiment

    @staticmethod
    @tool
    @timing_wrapper("FUNDAMENTALS")
    def get_finnhub_company_insider_transactions(
        ticker: Annotated[str, "ticker symbol"],
        curr_date: Annotated[
            str,
            "current date you are trading at, yyyy-mm-dd",
        ],
    ):
        """
        Retrieve insider transaction information about a company (retrieved from public SEC information) for the past 30 days
        Args:
            ticker (str): ticker symbol of the company
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
            str: a report of the company's insider transactions/trading information in the past 30 days
        """

        data_trans = interface.get_finnhub_company_insider_transactions(
            ticker, curr_date, 30
        )

        return data_trans

    @staticmethod
    @tool
    @timing_wrapper("FUNDAMENTALS")
    def get_simfin_balance_sheet(
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[
            str,
            "reporting frequency of the company's financial history: annual/quarterly",
        ],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    ):
        """
        Retrieve the most recent balance sheet of a company
        Args:
            ticker (str): ticker symbol of the company
            freq (str): reporting frequency of the company's financial history: annual / quarterly
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
            str: a report of the company's most recent balance sheet
        """

        data_balance_sheet = interface.get_simfin_balance_sheet(ticker, freq, curr_date)

        return data_balance_sheet

    @staticmethod
    @tool
    @timing_wrapper("FUNDAMENTALS")
    def get_simfin_cashflow(
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[
            str,
            "reporting frequency of the company's financial history: annual/quarterly",
        ],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    ):
        """
        Retrieve the most recent cash flow statement of a company
        Args:
            ticker (str): ticker symbol of the company
            freq (str): reporting frequency of the company's financial history: annual / quarterly
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
                str: a report of the company's most recent cash flow statement
        """

        data_cashflow = interface.get_simfin_cashflow(ticker, freq, curr_date)

        return data_cashflow

    @staticmethod
    @tool
    @timing_wrapper("NEWS")
    def get_coindesk_news(
        ticker: Annotated[str, "Ticker symbol, e.g. 'BTC/USD', 'ETH/USD', 'ETH', etc."],
        num_sentences: Annotated[int, "Number of sentences to include from news body."] = 5,
    ):
        """
        Retrieve news for a cryptocurrency.
        This function checks if the ticker is a crypto pair (like BTC/USD) and extracts the base currency.
        Then it fetches news for that cryptocurrency from CryptoCompare.

        Args:
            ticker (str): Ticker symbol for the cryptocurrency.
            num_sentences (int): Number of sentences to extract from the body of each news article.

        Returns:
            str: Formatted string containing news.
        """
        return interface.get_coindesk_news(ticker, num_sentences)

    @staticmethod
    @tool
    @timing_wrapper("FUNDAMENTALS")
    def get_simfin_income_stmt(
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[
            str,
            "reporting frequency of the company's financial history: annual/quarterly",
        ],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    ):
        """
        Retrieve the most recent income statement of a company
        Args:
            ticker (str): ticker symbol of the company
            freq (str): reporting frequency of the company's financial history: annual / quarterly
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
                str: a report of the company's most recent income statement
        """

        data_income_stmt = interface.get_simfin_income_statements(
            ticker, freq, curr_date
        )

        return data_income_stmt

    @staticmethod
    @tool
    @timing_wrapper("NEWS")
    def get_google_news(
        query: Annotated[str, "Query to search with"],
        curr_date: Annotated[str, "Curr date in yyyy-mm-dd format"],
    ):
        """
        Retrieve the latest news from Google News based on a query and date range.
        Args:
            query (str): Query to search with
            curr_date (str): Current date in yyyy-mm-dd format
            look_back_days (int): How many days to look back
        Returns:
            str: A formatted string containing the latest news from Google News based on the query and date range.
        """

        google_news_results = interface.get_google_news(query, curr_date, 7)

        return google_news_results

    @staticmethod
    @tool
    @timing_wrapper("SOCIAL", uses_web_search=True)
    def get_stock_news_openai(
        ticker: Annotated[str, "the company's ticker"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    ):
        """
        Retrieve the latest news about a given stock by using OpenAI's news API.
        Args:
            ticker (str): Ticker of a company. e.g. AAPL, TSM
            curr_date (str): Current date in yyyy-mm-dd format
        Returns:
            str: A formatted string containing the latest news about the company on the given date.
        """

        openai_news_results = interface.get_stock_news_openai(ticker, curr_date)

        return openai_news_results

    @staticmethod
    @tool
    @timing_wrapper("NEWS", uses_web_search=True)
    def get_global_news_openai(
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
        ticker_context: Annotated[str, "Ticker symbol for context-aware news (e.g., ETH/USD, AAPL)"] = None,
    ):
        """
        Retrieve the latest global news relevant to the asset being analyzed using OpenAI with web search.
        For crypto assets (BTC, ETH, etc.), focuses on crypto-relevant global news like regulation, institutional adoption, DeFi developments.
        For stocks, focuses on macro-economic and sector-specific global news.
        
        Args:
            curr_date (str): Current date in yyyy-mm-dd format
            ticker_context (str): Ticker symbol to provide context for relevant news (e.g., ETH/USD for crypto, AAPL for stocks)
            
        Returns:
            str: A formatted string containing the latest relevant global news for the asset being analyzed.
        """

        openai_news_results = interface.get_global_news_openai(curr_date, ticker_context)

        return openai_news_results

    @staticmethod
    @tool
    @timing_wrapper("FUNDAMENTALS", uses_web_search=True)
    def get_fundamentals_openai(
        ticker: Annotated[str, "the company's ticker"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    ):
        """
        Retrieve the latest fundamental information about a given stock on a given date by using OpenAI's news API.
        Args:
            ticker (str): Ticker of a company. e.g. AAPL, TSM
            curr_date (str): Current date in yyyy-mm-dd format
        Returns:
            str: A formatted string containing the latest fundamental information about the company on the given date.
        """

        openai_fundamentals_results = interface.get_fundamentals_openai(
            ticker, curr_date
        )

        return openai_fundamentals_results

    @staticmethod
    @tool
    @timing_wrapper("FUNDAMENTALS")
    def get_earnings_calendar(
        ticker: Annotated[str, "Stock or crypto ticker symbol"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    ) -> str:
        """
        Retrieve earnings calendar data for stocks or major events for crypto.
        For stocks: Shows earnings dates, EPS estimates vs actuals, revenue estimates vs actuals, and surprise analysis.
        For crypto: Shows major protocol events, upgrades, and announcements that could impact price.
        
        Args:
            ticker (str): Stock ticker (e.g. AAPL, TSLA) or crypto ticker (e.g. BTC/USD, ETH/USD, SOL/USD)
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
            
        Returns:
            str: Formatted earnings calendar data with estimates, actuals, and surprise analysis
        """
        
        earnings_calendar_results = interface.get_earnings_calendar(
            ticker, start_date, end_date
        )
        
        return earnings_calendar_results

    @staticmethod
    @tool
    @timing_wrapper("FUNDAMENTALS")
    def get_earnings_surprise_analysis(
        ticker: Annotated[str, "Stock ticker symbol"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
        lookback_quarters: Annotated[int, "Number of quarters to analyze"] = 8,
    ) -> str:
        """
        Analyze historical earnings surprises to identify patterns and trading implications.
        Shows consistency of beats/misses, magnitude of surprises, and seasonal patterns.
        
        Args:
            ticker (str): Stock ticker symbol, e.g. AAPL, TSLA
            curr_date (str): Current date in yyyy-mm-dd format
            lookback_quarters (int): Number of quarters to analyze (default 8 = ~2 years)
            
        Returns:
            str: Analysis of earnings surprise patterns with trading implications
        """
        
        earnings_surprise_results = interface.get_earnings_surprise_analysis(
            ticker, curr_date, lookback_quarters
        )
        
        return earnings_surprise_results

    @staticmethod
    @tool
    @timing_wrapper("MACRO")
    def get_macro_analysis(
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
        lookback_days: Annotated[int, "Number of days to look back for data"] = 90,
    ) -> str:
        """
        Retrieve comprehensive macro economic analysis including Fed funds, CPI, PPI, NFP, GDP, PMI, Treasury curve, VIX.
        Provides economic indicators, yield curve analysis, and Fed policy updates with trading implications.
        
        Args:
            curr_date (str): Current date in yyyy-mm-dd format
            lookback_days (int): Number of days to look back for data (default 90)
            
        Returns:
            str: Comprehensive macro economic analysis with trading implications
        """
        
        macro_analysis_results = interface.get_macro_analysis(
            curr_date, lookback_days
        )
        
        return macro_analysis_results

    @staticmethod
    @tool
    @timing_wrapper("MACRO", uses_web_search=True)
    def get_macro_news_openai(
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
        ticker_context: Annotated[
            str,
            "Optional context for macro web search (e.g. NVDA, BTC/USD, or 'financial markets')",
        ] = "financial markets",
    ) -> str:
        """
        Retrieve macro-relevant global news using OpenAI web search.

        Args:
            curr_date (str): Current date in yyyy-mm-dd format
            ticker_context (str): Context string for relevance filtering

        Returns:
            str: Macro/global-news analysis grounded in web search.
        """

        context = str(ticker_context or "financial markets").strip()
        if "financial markets" not in context.lower():
            context = f"financial markets and {context}"
        return interface.get_global_news_openai(curr_date, context)

    @staticmethod
    @tool
    @timing_wrapper("MACRO")
    def get_economic_indicators(
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
        lookback_days: Annotated[int, "Number of days to look back for data"] = 90,
    ) -> str:
        """
        Retrieve key economic indicators report including Fed funds, CPI, PPI, unemployment, NFP, GDP, PMI, VIX.
        
        Args:
            curr_date (str): Current date in yyyy-mm-dd format
            lookback_days (int): Number of days to look back for data (default 90)
            
        Returns:
            str: Economic indicators report with analysis and interpretations
        """
        
        economic_indicators_results = interface.get_economic_indicators(
            curr_date, lookback_days
        )
        
        return economic_indicators_results

    @staticmethod
    @tool
    @timing_wrapper("MACRO")
    def get_yield_curve_analysis(
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    ) -> str:
        """
        Retrieve Treasury yield curve analysis including inversion signals and recession indicators.
        
        Args:
            curr_date (str): Current date in yyyy-mm-dd format
            
        Returns:
            str: Treasury yield curve data with inversion analysis
        """
        
        yield_curve_results = interface.get_yield_curve_analysis(curr_date)
        
        return yield_curve_results

    @staticmethod
    @tool
    @timing_wrapper("FUNDAMENTALS")
    def get_defillama_fundamentals(
        ticker: Annotated[str, "Crypto ticker symbol (without USD/USDT suffix)"],
        lookback_days: Annotated[int, "Number of days to look back for data"] = 30,
    ):
        """
        Retrieve fundamental data for a cryptocurrency from DeFi Llama.
        This includes TVL (Total Value Locked), TVL change over lookback period,
        fees collected, and revenue data.
        
        Args:
            ticker (str): Crypto ticker symbol (e.g., BTC, ETH, UNI)
            lookback_days (int): Number of days to look back for data
            
        Returns:
            str: A markdown-formatted report of crypto fundamentals from DeFi Llama
        """
        
        defillama_results = interface.get_defillama_fundamentals(
            ticker, lookback_days
        )
        
        return defillama_results

    @staticmethod
    @tool
    @timing_wrapper("MARKET")
    def get_alpaca_data_report(
        symbol: Annotated[str, "ticker symbol of the company"],
        curr_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        look_back_days: Annotated[int, "how many days to look back"],
        timeframe: Annotated[str, "Timeframe for data: 1Min, 5Min, 15Min, 1Hour, 1Day"] = "1Day",
    ) -> str:
        """
        Retrieve Alpaca data for a given ticker symbol.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            curr_date (str): The current trading date in YYYY-mm-dd format
            look_back_days (int): How many days to look back
            timeframe (str): Timeframe for data (1Min, 5Min, 15Min, 1Hour, 1Day)
        Returns:
            str: A formatted dataframe containing the Alpaca data for the specified ticker symbol.
        """

        result_alpaca = interface.get_alpaca_data_window(
            symbol, curr_date, look_back_days, timeframe
        )

        return result_alpaca

    @staticmethod
    @tool
    @timing_wrapper("MARKET")
    def get_stock_data_table(
        symbol: Annotated[str, "ticker symbol of the company"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
        look_back_days: Annotated[int, "how many days to look back"] = 90,
        timeframe: Annotated[str, "Timeframe for data: 1Min, 5Min, 15Min, 1Hour, 1Day"] = "1Day",
    ) -> str:
        """
        Retrieve comprehensive stock data table for a given ticker symbol over a lookback period.
        Returns a clean table with Date, Open, High, Low, Close, Volume, VWAP columns for swing trading analysis.
        
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, NVDA
            curr_date (str): The current trading date in YYYY-mm-dd format
            look_back_days (int): How many days to look back (default 60)
            timeframe (str): Timeframe for data (1Min, 5Min, 15Min, 1Hour, 1Day)
            
        Returns:
            str: A comprehensive table containing Date, OHLCV, VWAP data for the lookback period
        """

        # Get the raw data from the interface
        raw_result = interface.get_alpaca_data_window(
            symbol, curr_date, look_back_days, timeframe
        )
        
        # Parse and reformat the timestamp column to be more readable
        import re
        
        try:
            # Use regex to replace complex timestamps with simple dates
            # Pattern: 2025-07-08 04:00:00+00:00 -> 2025-07-08
            timestamp_pattern = r'(\d{4}-\d{2}-\d{2})\s+\d{2}:\d{2}:\d{2}[+\-]\d{2}:\d{2}'
            
            # Replace the header line
            result = raw_result.replace('timestamp', 'Date')
            
            # Replace all timestamp values with just the date
            result = re.sub(timestamp_pattern, r'\1', result)
            
            # Also clean up any remaining timezone info
            result = re.sub(r'\s+\d{2}:\d{2}:\d{2}[+\-]\d{2}:\d{2}', '', result)
            
            # Update the title
            result = result.replace('Stock data for', 'Stock Data Table for')
            result = result.replace('from 2025-', f'({look_back_days}-day lookback)\nFrom 2025-')
            
            return result
                
        except Exception as e:
            # Fallback to original if any processing fails
            return raw_result

    @staticmethod
    @tool
    @timing_wrapper("MARKET")
    def get_indicators_table(
        symbol: Annotated[str, "ticker symbol (stocks: AAPL, NVDA; crypto: ETH/USD, BTC/USD)"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
        look_back_days: Annotated[int, "how many days to look back"] = 90,
    ) -> str:
        """
        Retrieve comprehensive technical indicators table for stocks and crypto over a lookback period.
        Returns a full table with Date and all key technical indicators calculated over the specified time window.
        Includes: EMAs, SMAs, RSI, MACD, Bollinger Bands, Stochastic, Williams %R, OBV, MFI, ATR.
        
        For crypto symbols, use format with slash: ETH/USD, BTC/USD, SOL/USD
        For stock symbols, use standard format: AAPL, NVDA, TSLA
        
        Args:
            symbol (str): Ticker symbol - stocks: AAPL, NVDA; crypto: ETH/USD, BTC/USD
            curr_date (str): The current trading date in YYYY-mm-dd format
            look_back_days (int): How many days to look back (default 90)
            
        Returns:
            str: A comprehensive table containing Date and all technical indicators for the lookback period
        """
        
        # Define the key indicators optimized for swing trading
        key_indicators = [
            'close_8_ema',      # 8-day EMA (faster trend detection for swing trading)
            'close_21_ema',     # 21-day EMA (key swing level)
            'close_50_sma',     # 50-day SMA (major trend)
            'rsi_14',           # 14-day RSI (optimal for daily signals)
            'macd',             # MACD Line (12,26,9 default)
            'macds',            # MACD Signal Line
            'macdh',            # MACD Histogram
            'boll_ub',          # Bollinger Upper (20,2 default)
            'boll_lb',          # Bollinger Lower (20,2 default)
            'kdjk_9',           # Stochastic %K (9-period)
            'kdjd_9',           # Stochastic %D (9-period)
            'wr_14',            # Williams %R (14-period)
            'atr_14',           # ATR (14-period for position sizing)
            'obv'               # On-Balance Volume (volume confirmation)
        ]
        
        # Get indicator data for each indicator across the time window
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Calculate date range
        curr_dt = pd.to_datetime(curr_date)
        start_dt = curr_dt - pd.Timedelta(days=look_back_days)
        
        results = []
        results.append(f"# Technical Indicators Table for {symbol}")
        results.append(f"**Period:** {start_dt.strftime('%Y-%m-%d')} to {curr_date} ({look_back_days} days lookback)")
        results.append(f"**Showing:** Last 25 trading days for swing trading analysis")
        results.append("")
        
        # Create table header
        header_row = "| Date | " + " | ".join([ind.replace('_', ' ').title() for ind in key_indicators]) + " |"
        separator_row = "|------|" + "|".join(["------" for _ in key_indicators]) + "|"
        
        results.append(header_row)
        results.append(separator_row)
        
        # Generate dates for the lookback period - only trading days
        dates = []
        trading_days_found = 0
        days_back = 0
        
        # Get the last 45 trading days (roughly 9 weeks of trading data)
        while trading_days_found < 45 and days_back <= look_back_days:
            date = curr_dt - pd.Timedelta(days=days_back)
            # Skip weekends (Saturday=5, Sunday=6)
            if date.weekday() < 5:  # Monday=0, Friday=4
                dates.append(date.strftime("%Y-%m-%d"))
                trading_days_found += 1
            days_back += 1
        
        # Reverse to get chronological order, then take the most recent portion
        dates = dates[::-1]
        recent_dates = dates[-25:] if len(dates) > 25 else dates  # Show last 25 trading days
        
        # OPTIMIZED: Use batch processing instead of 350+ individual calls
        print(f"[INDICATORS] Getting batch indicator data for {symbol} over {len(recent_dates)} dates...")
        
        # Get raw stock data first to calculate all indicators at once
        try:
            from tradingagents.dataflows.alpaca_utils import AlpacaUtils
            import pandas as pd
            
            # Get extended data for proper indicator calculation (need more history)
            start_date_extended = curr_dt - pd.Timedelta(days=200)  # More history for proper indicators
            
            # Get stock data
            stock_data = AlpacaUtils.get_stock_data(
                symbol=symbol,
                start_date=start_date_extended.strftime('%Y-%m-%d'),
                end_date=curr_date,
                timeframe="1Day"
            )
            
            if stock_data.empty:
                results.append("| ERROR | No stock data available for indicator calculations |")
                return "\n".join(results)
            
            # Clean data and ensure proper indexing
            stock_data = stock_data.dropna()
            stock_data = stock_data.reset_index(drop=True)
            
            # Ensure we have enough data for indicators
            if len(stock_data) < 50:
                results.append(f"| WARNING | Only {len(stock_data)} days of data available, indicators may be incomplete |")
            
            print(f"[INDICATORS] Processing {len(stock_data)} days of data for {symbol}")
            
            # Calculate all indicators using stockstats
            import stockstats
            stock_stats = stockstats.StockDataFrame.retype(stock_data.copy())
            
            # Calculate all indicators efficiently
            indicator_data = {}
            for indicator in key_indicators:
                try:
                    if indicator == 'close_8_ema':
                        indicator_data[indicator] = stock_stats['close_8_ema']
                    elif indicator == 'close_21_ema':
                        indicator_data[indicator] = stock_stats['close_21_ema']  
                    elif indicator == 'close_50_sma':
                        indicator_data[indicator] = stock_stats['close_50_sma']
                    elif indicator == 'rsi_14':
                        indicator_data[indicator] = stock_stats['rsi_14']
                    elif indicator == 'macd':
                        indicator_data[indicator] = stock_stats['macd']
                    elif indicator == 'macds':
                        indicator_data[indicator] = stock_stats['macds']
                    elif indicator == 'macdh':
                        indicator_data[indicator] = stock_stats['macdh']
                    elif indicator == 'boll_ub':
                        indicator_data[indicator] = stock_stats['boll_ub']
                    elif indicator == 'boll_lb':
                        indicator_data[indicator] = stock_stats['boll_lb']
                    elif indicator == 'kdjk_9':
                        indicator_data[indicator] = stock_stats['kdjk_9']
                    elif indicator == 'kdjd_9':
                        indicator_data[indicator] = stock_stats['kdjd_9']
                    elif indicator == 'wr_14':
                        indicator_data[indicator] = stock_stats['wr_14']
                    elif indicator == 'atr_14':
                        indicator_data[indicator] = stock_stats['atr_14']
                    elif indicator == 'obv':
                        # OBV calculation - handle the parsing issue
                        try:
                            indicator_data[indicator] = stock_stats['obv']
                        except Exception as obv_error:
                            print(f"[INDICATORS] OBV calculation failed, using manual method: {obv_error}")
                            # Manual OBV calculation
                            obv_values = []
                            obv = 0
                            for i in range(len(stock_data)):
                                if i == 0:
                                    obv_values.append(stock_data['volume'].iloc[i])
                                else:
                                    if stock_data['close'].iloc[i] > stock_data['close'].iloc[i-1]:
                                        obv += stock_data['volume'].iloc[i]
                                    elif stock_data['close'].iloc[i] < stock_data['close'].iloc[i-1]:
                                        obv -= stock_data['volume'].iloc[i]
                                    obv_values.append(obv)
                            indicator_data[indicator] = pd.Series(obv_values, index=stock_data.index)
                    else:
                        indicator_data[indicator] = None
                except Exception as e:
                    print(f"[INDICATORS] Warning: Failed to calculate {indicator}: {e}")
                    indicator_data[indicator] = None
            
            # Convert date strings to datetime for matching
            recent_dates_dt = [pd.to_datetime(d) for d in recent_dates]
            
            # Build table rows efficiently
            for date_str in recent_dates:
                row_values = [date_str]
                date_dt = pd.to_datetime(date_str)
                
                for indicator in key_indicators:
                    try:
                        # Find matching date in indicator data
                        indicator_series = indicator_data.get(indicator)
                        if indicator_series is not None and len(indicator_series) > 0:
                            try:
                                # Convert recent_dates to match stock_data index
                                # Find the closest date index in our data
                                target_date = pd.to_datetime(date_str)
                                
                                # If stock_data has a date column, use it for matching
                                if 'date' in stock_data.columns:
                                    date_matches = stock_data[stock_data['date'] == target_date.strftime('%Y-%m-%d')]
                                    if not date_matches.empty:
                                        idx = date_matches.index[0]
                                        if idx < len(indicator_series):
                                            value = indicator_series.iloc[idx]
                                        else:
                                            value = indicator_series.iloc[-1]  # Use last available
                                    else:
                                        # Use the most recent available data
                                        value = indicator_series.iloc[-1] if len(indicator_series) > 0 else None
                                else:
                                    # Use index-based matching (most recent data)
                                    days_from_end = (pd.to_datetime(recent_dates[-1]) - target_date).days
                                    idx = max(0, len(indicator_series) - 1 - days_from_end)
                                    idx = min(idx, len(indicator_series) - 1)
                                    value = indicator_series.iloc[idx]
                                
                                if pd.isna(value) or value is None:
                                    row_values.append("N/A")
                                else:
                                    # Format value appropriately
                                    if indicator in ['rsi_14', 'kdjk_9', 'kdjd_9', 'wr_14']:
                                        row_values.append(f"{float(value):.1f}")
                                    elif 'macd' in indicator:
                                        row_values.append(f"{float(value):.3f}")
                                    else:
                                        row_values.append(f"{float(value):.2f}")
                            except Exception as match_error:
                                print(f"[INDICATORS] Date matching error for {indicator}: {match_error}")
                                row_values.append("N/A")
                        else:
                            row_values.append("N/A")
                    except Exception as e:
                        row_values.append("N/A")
                
                # Format the table row
                table_row = "| " + " | ".join(row_values) + " |"
                results.append(table_row)
                
        except Exception as e:
            print(f"[INDICATORS] ERROR: Batch indicator calculation failed: {e}")
            # Fallback to individual calls (original slow method) with timeout
            import time
            timeout_per_call = 2.0  # 2 second timeout per call
            
            for date in recent_dates:
                row_values = [date]
                
                for indicator in key_indicators:
                    start_time = time.time()
                    try:
                        # Get indicator value with timeout protection
                        value = interface.get_stock_stats_indicators_window(
                            symbol, indicator, date, 1, True
                        )
                        
                        # Check if call took too long
                        elapsed = time.time() - start_time
                        if elapsed > timeout_per_call:
                            print(f"[INDICATORS] Warning: {indicator} took {elapsed:.1f}s (slow)")
                        
                        # Extract numeric value
                        if ":" in value:
                            numeric_part = value.split(":")[-1].strip().split("(")[0].strip()
                            try:
                                float_val = float(numeric_part)
                                if indicator in ['rsi_14', 'kdjk_9', 'kdjd_9', 'wr_14']:
                                    row_values.append(f"{float_val:.1f}")
                                elif 'macd' in indicator:
                                    row_values.append(f"{float_val:.3f}")
                                else:
                                    row_values.append(f"{float_val:.2f}")
                            except:
                                row_values.append("N/A")
                        else:
                            row_values.append("N/A")
                    except Exception as ind_e:
                        print(f"[INDICATORS] Error getting {indicator} for {date}: {ind_e}")
                        row_values.append("N/A")
                
                # Format the table row
                table_row = "| " + " | ".join(row_values) + " |"
                results.append(table_row)
        
        results.append("")
        results.append("## Key Swing Trading Signals Analysis:")
        results.append("- **Trend Structure:** 8-EMA > 21-EMA > 50-SMA = Strong uptrend | Price above all EMAs = Bullish")
        results.append("- **Momentum:** RSI 30-50 = Accumulation zone | RSI 50-70 = Trending | RSI >70 = Overbought")
        results.append("- **MACD Signals:** MACD > Signal = Bullish momentum | Histogram growing = Acceleration")
        results.append("- **Bollinger Bands:** Price at Upper Band = Breakout potential | Price at Lower Band = Support test")
        results.append("- **Stochastic:** %K crossing above %D in oversold (<20) = Buy signal | In overbought (>80) = Sell signal")
        results.append("- **Williams %R:** Values -20 to -80 = Normal range | Below -80 = Oversold (buy) | Above -20 = Overbought (sell)")
        results.append("- **ATR:** Use for position sizing (1-2x ATR for stop loss) | Higher ATR = More volatile")
        results.append("")
        results.append("**Swing Strategy:** Look for multi-timeframe trend + momentum + volume confirmation for 2-10 day positions")
        
        return "\n".join(results)

    @staticmethod
    @tool
    @timing_wrapper("MARKET")
    def get_technical_brief(
        symbol: Annotated[str, "ticker symbol (stocks: AAPL, NVDA; crypto: ETH/USD, BTC/USD)"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    ) -> str:
        """
        Get a standardized Technical Brief with pre-analyzed TA across 1h / 4h / 1d timeframes.

        Returns compact structured JSON containing:
        - trend: direction + strength per timeframe
        - momentum: RSI zone, divergence flags, MACD cross
        - vwap_state: above/below + zscore distance
        - volatility: ATR percentile, Bollinger squeeze/breakout
        - market_structure: BOS/CHOCH + last swing points
        - key_levels: 3-5 important cross-timeframe price levels
        - signal_summary: classified setup (breakout/pullback/mean_reversion/trend_continuation) + confidence

        This is optimized for LLM consumption -- all indicator interpretation is
        done deterministically so the LLM receives pre-digested conclusions.
        """
        return interface.get_technical_brief(symbol, curr_date)
