from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, ToolMessage
import time
import json

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

        # ── Primary tool: structured Technical Brief (Tier-1 quant output) ──
        # The brief pre-computes indicators across 1h/4h/1d and returns a
        # compact JSON -- the LLM no longer needs to interpret raw tables.
        if toolkit.config["online_tools"]:
            tools = [toolkit.get_technical_brief]
            # Keep legacy tools as optional fallbacks for edge cases
            if is_crypto:
                tools.append(toolkit.get_coindesk_news)
        else:
            # Offline mode: fall back to legacy tools (no Alpaca live data)
            tools = [
                toolkit.get_alpaca_data_report,
                toolkit.get_stockstats_indicators_report,
            ]
            if is_crypto:
                tools.append(toolkit.get_coindesk_news)

        system_message = (
            """You are a **multi-timeframe technical analyst**. Your input is a structured Technical Brief (JSON) that has already been computed deterministically across three timeframes: **1 h, 4 h, and 1 d**.

## Your workflow

1. **Call `get_technical_brief`** with the ticker and current date.
   You will receive a JSON object with the following pre-analyzed sections for each timeframe:
   - `trend` – direction (bullish/bearish/neutral), strength, EMA slope, HH/HL flags
   - `momentum` – RSI value & zone, divergence flag, MACD cross & histogram trend
   - `vwap_state` – above/below VWAP with z-score distance
   - `volatility` – ATR value & percentile, Bollinger squeeze/breakout flags
   - `market_structure` – BOS (Break of Structure) / CHOCH (Change of Character), last swing points
   Plus cross-timeframe fields:
   - `key_levels` – 3-5 most important support/resistance/pivot levels
   - `signal_summary` – classified setup type and confidence

2. **Analyze the brief** — look for:
   - **Timeframe confluence**: Do 1 h, 4 h, and 1 d agree on trend direction?
   - **Key level proximity**: Is price near a major support/resistance?
   - **Momentum confirmation**: Does RSI / MACD support the trend?
   - **Volatility context**: Is a squeeze building or a breakout underway?
   - **Market structure**: Has structure broken (BOS) or character changed (CHOCH)?

3. **Produce your analysis** with these required sections:

### a) Conclusion
State **BULLISH**, **BEARISH**, or **NEUTRAL** with a 1-sentence rationale.

### b) Entry Conditions
Specify the price level and conditions for entering a position (e.g., "Enter long on a pullback to $185 if 1 h RSI bounces from neutral zone").

### c) Invalidation
The price level or condition that would invalidate the thesis (e.g., "Below $180 — daily swing low broken").

### d) Risk Sizing Hint
A brief note on position sizing based on ATR (e.g., "ATR $3.20 → stop 1.5x ATR = $4.80 risk per share").

### e) Narrative
2-3 sentences explaining *why* the setup works, connecting multi-timeframe evidence.

### f) Summary Table
| Field | Value |
|-------|-------|
| Bias | Bullish / Bearish / Neutral |
| Setup | breakout / pullback / mean_reversion / trend_continuation |
| Confidence | high / medium / low |
| Entry | $xxx |
| Target | $xxx |
| Stop | $xxx |
| R:R | x.x : 1 |

Conclude with: **FINAL TRANSACTION PROPOSAL: BUY/HOLD/SELL** and a brief justification.

**Important**: Do NOT request raw indicator tables — the Technical Brief already contains all the analysis you need in pre-digested form."""
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

        # First LLM response
        result = chain.invoke(messages_history)

        # Handle iterative tool calls until the model stops requesting them
        while getattr(result, "additional_kwargs", {}).get("tool_calls"):
            for tool_call in result.additional_kwargs["tool_calls"]:
                # Handle different tool call structures
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name") or tool_call.get("function", {}).get("name")
                    tool_args = tool_call.get("args", {}) or tool_call.get("function", {}).get("arguments", {})
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {}
                else:
                    # Handle LangChain ToolCall objects
                    tool_name = getattr(tool_call, 'name', None)
                    tool_args = getattr(tool_call, 'args', {})

                # Find the matching tool by name
                tool_fn = next((t for t in tools if t.name == tool_name), None)

                if tool_fn is None:
                    tool_result = f"Tool '{tool_name}' not found."
                    print(f"[MARKET] ⚠️ {tool_result}")
                else:
                    try:
                        # LangChain Tool objects expose `.run` (string IO) as well as `.invoke` (dict/kwarg IO)
                        if hasattr(tool_fn, "invoke"):
                            tool_result = tool_fn.invoke(tool_args)
                        else:
                            tool_result = tool_fn.run(**tool_args)
                        
                    except Exception as tool_err:
                        tool_result = f"Error running tool '{tool_name}': {str(tool_err)}"

                # Append the assistant tool call and tool result messages so the LLM can continue the conversation
                tool_call_id = tool_call.get("id") or tool_call.get("tool_call_id")
                ai_tool_call_msg = AIMessage(
                    content="",
                    additional_kwargs={"tool_calls": [tool_call]},
                )
                tool_msg = ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call_id,
                )

                messages_history.append(ai_tool_call_msg)
                messages_history.append(tool_msg)

            # Ask the LLM to continue with the new context
            result = chain.invoke(messages_history)
        
        # Check if the result already contains FINAL TRANSACTION PROPOSAL
        if "FINAL TRANSACTION PROPOSAL:" not in result.content:
            # Create a simple prompt that includes the analysis content directly
            final_prompt = f"""Based on the following market and technical analysis for {ticker}, please provide your final trading recommendation.

Analysis:
{result.content}

You must conclude with: FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** followed by a brief justification."""
            
            # Use a simple chain without tools for the final recommendation
            final_chain = llm
            final_result = final_chain.invoke(final_prompt)
            
            # Combine the analysis with the final proposal
            combined_content = result.content + "\n\n" + final_result.content
            result = AIMessage(content=combined_content)

        return {
            "messages": [result],
            "market_report": result.content,
        }

    return market_analyst_node
