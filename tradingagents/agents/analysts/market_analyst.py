from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, ToolMessage
import time
import json
from tradingagents.agents.utils.agent_utils import log_llm_start, log_llm_end

# Import prompt capture utility
try:
    from webui.utils.prompt_capture import capture_agent_prompt
except ImportError:
    # Fallback for when webui is not available
    def capture_agent_prompt(report_type, prompt_content, symbol=None):
        pass


def create_market_analyst(llm, toolkit):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        is_crypto = "/" in ticker or "USD" in ticker.upper() or "USDT" in ticker.upper()

        # All tools now use smart caching - no need for online_tools flag
        if is_crypto:
            # Crypto gets the same data tools as stocks since Alpaca supports crypto
            tools = [
                toolkit.get_alpaca_data,  # Alpaca supports crypto price data (with caching)
                toolkit.get_indicators_table,
                toolkit.get_stockstats_indicators_report_online,
                toolkit.get_coindesk_news  # Plus crypto-specific news (with caching)
            ]
        else:
            # Stock tools with smart caching
            tools = [
                toolkit.get_stock_data_table,
                toolkit.get_indicators_table,
                toolkit.get_stockstats_indicators_report_online,  # Keep for custom indicators
            ]

        system_message = (
            """You are an EOD TRADING technical analyst specializing in identifying optimal entry/exit points for overnight positions based on daily market close data. Your role is to select the **most relevant indicators** (up to **8**) for EOD trading setups from the following list, focusing on daily chart patterns and end-of-day signals.

**EOD TRADING FOCUS:**
- Target holding periods: Overnight with daily reassessment
- Entry signals: End-of-day breakouts, daily closing patterns, overnight setups
- Exit signals: Daily market close evaluation, next-day gap management
- Volume confirmation: Essential for validating EOD moves and overnight positioning

Categories and Indicators for EOD TRADING:

**Daily Momentum & Trend (Priority for EOD):**
- close_10_ema: 10-period EMA – Critical for daily momentum. EOD price above = bullish overnight bias
- close_20_sma: 20-period SMA – Key daily level. EOD breaks often signal overnight moves
- close_50_sma: 50-period SMA – Major daily support/resistance. EOD tests create overnight setups
- rsi: RSI (14) – Daily overbought >70, oversold <30. EOD divergences signal overnight reversals

**MACD for EOD Timing:**
- macd: MACD line – Daily momentum detector. EOD zero-line crosses excellent for overnight entries
- macds: MACD Signal – Daily bullish/bearish crossovers for overnight position timing
- macdh: MACD Histogram – Growing histogram confirms EOD momentum for overnight holds

**Oscillators for EOD Entry/Exit:**
- kdjk: %K Stochastic – Daily oversold <20, overbought >80. Use for EOD entry timing
- kdjd: %D – Smoother daily signal. %K crossing above %D at EOD = overnight buy signal
- wr: Williams %R – Fast daily oscillator. EOD -20 to -80 range ideal for overnight entries

**Daily Volatility & Support/Resistance:**
- atr: Average True Range – Size stop losses at 1-2x daily ATR for overnight positions
- boll_ub: Bollinger Upper Band – Daily resistance. EOD breakouts above = overnight momentum
- boll_lb: Bollinger Lower Band – Daily support. EOD bounces create overnight opportunities

**Volume Confirmation:**
- obv: On-Balance Volume – Must confirm daily price moves. EOD divergences signal overnight reversals
- mfi: Money Flow Index – Daily volume-weighted momentum. >80 overbought, <20 oversold for EOD

**EOD TRADING ANALYSIS REQUIREMENTS:**
1. **Daily Setup Analysis:** Identify specific price levels for EOD entries based on daily close patterns
2. **Overnight Target Identification:** Set realistic profit targets for next day based on daily ranges
3. **Risk Management:** Define stop-loss levels below daily support (usually 1-3% risk)
4. **Volume Confirmation:** Ensure daily volume supports the anticipated overnight move
5. **Daily Timeframe Context:** Focus on daily chart patterns, ignore intraday noise
6. **Catalyst Awareness:** Consider overnight events that could drive next-day price moves

Select indicators that provide **EOD-specific** insights. Prioritize daily momentum, trend strength, and daily support/resistance levels over intraday scalping or long-term investment metrics. Always analyze from an **EOD trader's perspective** focusing on overnight positioning.

When you call tools, use **exact** indicator names (case-sensitive). Follow this workflow:
1. **Call `get_stock_data_table` first** (lookback **90 days** by default) to get comprehensive OHLCV + VWAP data table
2. **Call `get_indicators_table`** to get comprehensive technical indicators table optimized for EOD trading
3. **Optionally call `get_stockstats_indicators_report_online`** for specific custom indicators with non-default parameters

The indicators table includes EOD-optimized signals: 8-EMA/21-EMA/50-SMA (trend), RSI-14 (momentum), MACD (12,26,9), Bollinger Bands (20,2), Stochastic (9-period), Williams %R-14, ATR-14 (position sizing), and OBV (volume confirmation). This provides you with complete tabular data showing price action and all key indicators over the 90-day window for comprehensive EOD analysis. Provide specific EOD trading recommendations with entry points, targets, and stop levels based on these historical data tables.
            """
            + """ 

**EOD TRADING SUMMARY TABLE REQUIRED:**
Make sure to append a Markdown table at the end with:
| Metric | Value | EOD Signal | Entry Level | Overnight Target | Stop Loss |
|--------|-------|------------|-------------|------------------|-----------|
| [Indicator] | [Current Value] | [Bullish/Bearish/Neutral] | [Price Level] | [Target Price] | [Stop Price] |

Focus on actionable EOD trading insights, not generic market commentary."""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    " You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. The company we want to look at is {ticker}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        # Capture the COMPLETE resolved prompt that gets sent to the LLM
        try:
            # Get the formatted messages with all variables resolved
            messages_history = list(state["messages"])
            formatted_messages = prompt.format_messages(messages=messages_history)
            
            # Extract the complete system message (first message)
            if formatted_messages and hasattr(formatted_messages[0], 'content'):
                complete_prompt = formatted_messages[0].content
            else:
                # Fallback: manually construct the complete prompt
                tool_names_str = ", ".join([tool.name for tool in tools])
                complete_prompt = f""" You are a helpful AI assistant, collaborating with other assistants. Use the provided tools to progress towards answering the question. If you are unable to fully answer, that's OK; another assistant with different tools will help where you left off. Execute what you can to make progress. If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable, prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop. You have access to the following tools: {tool_names_str}.

{system_message}

For your reference, the current date is {current_date}. The company we want to look at is {ticker}"""
            
            capture_agent_prompt("market_report", complete_prompt, ticker)
        except Exception as e:
            print(f"[MARKET] Warning: Could not capture complete prompt: {e}")
            # Fallback to system message only
            capture_agent_prompt("market_report", system_message, ticker)

        chain = prompt | llm.bind_tools(tools)

        # Copy the incoming conversation history so we can append to it when the model makes tool calls
        messages_history = list(state["messages"])

        # Track tools that have failed to prevent LLM retries
        failed_tools = set()
        max_iterations = 5  # Prevent infinite loops
        iteration_count = 0

        # First LLM response
        model_name = getattr(llm, 'model_name', 'unknown')
        start_time = log_llm_start("MARKET_ANALYST", model_name)
        try:
            result = chain.invoke(messages_history)
            log_llm_end("MARKET_ANALYST", model_name, start_time, result)
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[LLM - MARKET_ANALYST] ❌ {model_name} failed after {elapsed:.2f}s: {str(e)}")
            raise

        # Handle iterative tool calls until the model stops requesting them
        # Check both result.tool_calls (modern LangChain) and additional_kwargs["tool_calls"] (legacy)
        tool_calls_list = getattr(result, "tool_calls", None) or getattr(result, "additional_kwargs", {}).get("tool_calls", [])
        while tool_calls_list and iteration_count < max_iterations:
            iteration_count += 1
            # Append the AI message with tool calls (it's already properly formatted)
            messages_history.append(result)

            # Execute each tool and collect results
            for tool_call in tool_calls_list:
                # Handle different tool call structures
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name") or tool_call.get("function", {}).get("name")
                    tool_args = tool_call.get("args", {}) or tool_call.get("function", {}).get("arguments", {})
                    tool_call_id = tool_call.get("id") or tool_call.get("tool_call_id")
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {}
                else:
                    # Handle LangChain ToolCall objects
                    tool_name = getattr(tool_call, 'name', None)
                    tool_args = getattr(tool_call, 'args', {})
                    tool_call_id = getattr(tool_call, 'id', None)

                # Check if this tool has already failed - SKIP RETRY
                if tool_name in failed_tools:
                    tool_result = f"PERMANENT FAILURE: Tool '{tool_name}' already failed in this analysis. Will not retry. Proceed with analysis using other available data sources."
                    print(f"[MARKET] ⚠️ Skipping retry of failed tool: {tool_name}")
                    tool_msg = ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call_id,
                    )
                    messages_history.append(tool_msg)
                    continue

                # Find the matching tool by name
                tool_fn = next((t for t in tools if t.name == tool_name), None)

                if tool_fn is None:
                    tool_result = f"Tool '{tool_name}' not found."
                    print(f"[MARKET] ⚠️ {tool_result}")
                else:
                    try:
                        print(f"[MARKET] 🔧 Starting tool '{tool_name}'...")
                        start_time = time.time()

                        # LangChain Tool objects expose `.run` (string IO) as well as `.invoke` (dict/kwarg IO)
                        if hasattr(tool_fn, "invoke"):
                            tool_result = tool_fn.invoke(tool_args)
                        else:
                            tool_result = tool_fn.run(**tool_args)

                        elapsed = time.time() - start_time
                        print(f"[MARKET] ✅ Tool '{tool_name}' completed (took {elapsed:.2f}s)")

                    except TimeoutError as timeout_err:
                        elapsed = time.time() - start_time
                        # Mark this tool as failed to prevent retries
                        failed_tools.add(tool_name)
                        tool_result = f"PERMANENT FAILURE: Tool '{tool_name}' timed out after {elapsed:.0f}s. This tool will not be retried. Proceed with analysis using other available data sources."
                        print(f"[MARKET] ⏰ TIMEOUT: {tool_result}")
                    except Exception as tool_err:
                        elapsed = time.time() - start_time
                        tool_result = f"Error running tool '{tool_name}': {str(tool_err)}"
                        print(f"[MARKET] ❌ Tool '{tool_name}' failed after {elapsed:.2f}s: {str(tool_err)}")

                # Append the tool result message
                tool_msg = ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call_id,
                )
                messages_history.append(tool_msg)

            # Ask the LLM to continue with the new context
            model_name = getattr(llm, 'model_name', 'unknown')
            start_time = log_llm_start("MARKET_ANALYST", model_name)
            try:
                result = chain.invoke(messages_history)
                log_llm_end("MARKET_ANALYST", model_name, start_time, result)
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"[LLM - MARKET_ANALYST] ❌ {model_name} failed after {elapsed:.2f}s: {str(e)}")
                raise

            # Update tool_calls_list for next iteration
            tool_calls_list = getattr(result, "tool_calls", None) or getattr(result, "additional_kwargs", {}).get("tool_calls", [])

        # Check if we hit max iterations
        if iteration_count >= max_iterations:
            print(f"[MARKET] ⚠️ Reached max tool iterations ({max_iterations}). Proceeding with available data.")
        
        # Check if the result already contains FINAL TRANSACTION PROPOSAL
        if "FINAL TRANSACTION PROPOSAL:" not in result.content:
            # Create a simple prompt that includes the analysis content directly
            final_prompt = f"""Based on the following market and technical analysis for {ticker}, please provide your final trading recommendation.

Analysis:
{result.content}

You must conclude with: FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** followed by a brief justification."""

            # Use a simple chain without tools for the final recommendation
            final_chain = llm
            model_name = getattr(llm, 'model_name', 'unknown')
            start_time = log_llm_start("MARKET_ANALYST", model_name)
            try:
                final_result = final_chain.invoke(final_prompt)
                log_llm_end("MARKET_ANALYST", model_name, start_time, final_result)
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"[LLM - MARKET_ANALYST] ❌ {model_name} failed after {elapsed:.2f}s: {str(e)}")
                raise

            # Combine the analysis with the final proposal
            combined_content = result.content + "\n\n" + final_result.content
            result = AIMessage(content=combined_content)

        return {
            "messages": [result],
            "market_report": result.content,
        }

    return market_analyst_node
