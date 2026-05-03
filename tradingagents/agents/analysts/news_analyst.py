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


def create_news_analyst(llm, toolkit):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        
        is_crypto = "/" in ticker or "USD" in ticker.upper() or "USDT" in ticker.upper()
        openai_available = toolkit.has_openai_web_search()
        finnhub_available = toolkit.has_finnhub()
        coindesk_available = toolkit.has_coindesk()

        global_news_available = (
            toolkit.config["online_tools"]
            and openai_available
            and bool(toolkit.config.get("news_global_openai_enabled", False))
        )
        tools = [toolkit.get_google_news]
        if global_news_available:
            tools.append(toolkit.get_global_news_openai)
        if is_crypto:
            if coindesk_available:
                tools.append(toolkit.get_coindesk_news)
        else:
            if finnhub_available:
                tools.append(toolkit.get_finnhub_news_recent)

        source_labels = ["Google News"]
        if global_news_available:
            source_labels.append("OpenAI global web search")
        if is_crypto and coindesk_available:
            source_labels.append("CoinDesk/CryptoCompare")
        if (not is_crypto) and finnhub_available:
            source_labels.append("Finnhub")

        global_news_guidance = (
            f"**IMPORTANT:** When using get_global_news_openai, ALWAYS pass ticker_context='{ticker}' to get "
            + (
                "crypto-relevant global news (regulation, institutional adoption, DeFi developments)"
                if is_crypto
                else "sector-relevant global news"
            )
            + " instead of generic macro news.\n"
            if global_news_available
            else ""
        )

        source_guidance = (
            " Use all currently available news tools before concluding."
            f" Active sources: {', '.join(source_labels)}."
            " For `get_finnhub_news_recent`, pass ticker and curr_date from context."
            " Do not request broad macro/global web searches unless OpenAI global web search is listed as an active source;"
            " the Macro analyst handles that context when enabled."
        )
        system_message = (
            f"You are a SWING TRADING news analyst specializing in identifying news events and market developments that could drive multi-day price movements for {ticker}. Focus on catalysts and sentiment shifts that affect swing trading positions (2-10 day holds)."
            + " **SWING TRADING NEWS ANALYSIS:** \n"
            + "1. **Multi-Day Catalyst Identification:** Events, announcements, and data releases that could sustain price trends over 2-10 days \n"
            + "2. **Sentiment Trends:** Changes in market narrative, analyst sentiment, or sector rotation that persist across multiple days \n"
            + "3. **Event Calendar:** Specific dates for earnings, FDA approvals, product launches, economic data during the swing holding period \n"
            + "4. **Momentum Drivers:** News creating sustained multi-day price momentum suitable for swing positioning \n"
            + "5. **Swing Risk Events:** Upcoming geopolitical developments, Fed decisions, sector-specific risks during the holding period \n"
            + "6. **Sector & Relative Strength:** How similar companies and the broader sector are trending — multi-day momentum patterns \n"
            + "**ANALYSIS PRIORITIES:** \n"
            + "- Focus on news that could sustain or reverse a multi-day price swing \n"
            + "- Identify both bullish and bearish catalysts over the coming 2-10 trading days \n"
            + "- Assess news impact magnitude (minor <2%, moderate 2-5%, major >5% multi-day moves) \n"
            + "- Consider news durability (will impact persist through the swing period?) \n"
            + "- Analyze market reaction patterns to similar news for multi-day follow-through \n"
            + global_news_guidance
            + f"{source_guidance}\n"
            + "**AVOID:** Generic market commentary, intraday noise. Focus on news with multi-day impact potential relevant to swing trades.\n"
            + """ Make sure to append a Markdown table at the end organizing:
| News Event | Date/Time | Impact Level | Price Direction | Swing Trading Implication |
|------------|-----------|--------------|----------------|------------------------|\n
| [Specific Event] | [Date/Time] | [High/Med/Low] | [Bullish/Bearish/Neutral] | [Entry/Exit/Hold Strategy] |

Provide specific, actionable news analysis for swing trading decisions with clear timing and multi-day impact assessment."""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. We are looking at the ticker: {ticker}",
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
                complete_prompt = f"""You are a helpful AI assistant, collaborating with other assistants. Use the provided tools to progress towards answering the question. If you are unable to fully answer, that's OK; another assistant with different tools will help where you left off. Execute what you can to make progress. If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable, prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop. You have access to the following tools: {tool_names_str}.

{system_message}

For your reference, the current date is {current_date}. We are looking at the ticekr: {ticker}"""
            
            capture_agent_prompt("news_report", complete_prompt, ticker)
        except Exception as e:
            print(f"[NEWS] Warning: Could not capture complete prompt: {e}")
            # Fallback to system message only
            capture_agent_prompt("news_report", system_message, ticker)

        chain = prompt | (llm.bind_tools(tools) if tools else llm)
        
        # Copy the incoming conversation history so we can append to it when the model makes tool calls
        messages_history = list(state["messages"])

        # First LLM response
        result = chain.invoke(messages_history)
        max_tool_iterations = int(toolkit.config.get("max_tool_iterations_per_agent", 8))
        max_same_call_repeats = int(toolkit.config.get("max_same_tool_call_repeats", 1))
        tool_call_counts = {}
        tool_result_cache = {}
        iteration_count = 0

        # Handle iterative tool calls until the model stops requesting them
        while tools and getattr(result, "additional_kwargs", {}).get("tool_calls") and iteration_count < max_tool_iterations:
            iteration_count += 1
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
                    print(f"[NEWS] ⚠️ {tool_result}")
                else:
                    call_signature = f"{tool_name}:{json.dumps(tool_args, sort_keys=True, default=str)}"
                    call_count = tool_call_counts.get(call_signature, 0) + 1
                    tool_call_counts[call_signature] = call_count

                    if call_signature in tool_result_cache:
                        tool_result = tool_result_cache[call_signature]
                    elif call_count > max_same_call_repeats:
                        tool_result = (
                            f"Skipped repeated tool call '{tool_name}' with identical arguments "
                            f"after {max_same_call_repeats} execution(s). Reuse previous evidence."
                        )
                    else:
                        try:
                            # LangChain Tool objects expose `.run` (string IO) as well as `.invoke` (dict/kwarg IO)
                            if hasattr(tool_fn, "invoke"):
                                tool_result = tool_fn.invoke(tool_args)
                            else:
                                tool_result = tool_fn.run(**tool_args)
                            tool_result_cache[call_signature] = str(tool_result)
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

        if tools and getattr(result, "additional_kwargs", {}).get("tool_calls"):
            result = AIMessage(
                content=(
                    (result.content or "").strip()
                    + f"\n\nTool-loop halted after {max_tool_iterations} iterations to prevent endless retries."
                ).strip()
            )
        
        # Check if the result already contains FINAL TRANSACTION PROPOSAL
        if "FINAL TRANSACTION PROPOSAL:" not in result.content:
            # Create a simple prompt that includes the analysis content directly
            final_prompt = f"""Based on the following news analysis for {ticker}, please provide your final trading recommendation considering the overall news sentiment and implications.

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
            "news_report": result.content,
        }

    return news_analyst_node
