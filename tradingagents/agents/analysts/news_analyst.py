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


def create_news_analyst(llm, toolkit):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        
        is_crypto = "/" in ticker or "USD" in ticker.upper() or "USDT" in ticker.upper()

        if is_crypto:
            tools = [
                toolkit.get_coindesk_news,
                toolkit.get_reddit_news,
                toolkit.get_google_news,
            ]
        else:
            tools = [
                toolkit.get_finnhub_news,
                toolkit.get_reddit_news,
                toolkit.get_google_news,
            ]

        system_message = (
            f"You are an EOD TRADING news analyst specializing in identifying news events and market developments that could drive overnight and next-day price movements for {ticker}. Focus on after-hours catalysts and sentiment shifts that create EOD trading opportunities."
            + " **EOD TRADING NEWS ANALYSIS:** \n"
            + "1. **Overnight Catalyst Identification:** After-hours events, announcements, data releases that could create next-day gaps or moves \n"
            + "2. **End-of-Day Sentiment Shifts:** Changes in market narrative, analyst sentiment, or sector rotation trends affecting overnight positions \n"
            + "3. **Event Timing:** Specific dates/times for earnings, FDA approvals, product launches, economic data that EOD traders should know \n"
            + "4. **After-Hours Momentum Drivers:** Breaking news creating overnight price momentum suitable for next-day positioning \n"
            + "5. **Overnight Risk Events:** Geopolitical developments, Fed decisions, sector-specific risks that could impact overnight positions \n"
            + "6. **Pre-Market Analysis:** How similar companies are reacting to news - sector momentum and relative strength patterns for next day \n"
            + "**ANALYSIS PRIORITIES:** \n"
            + "- Focus on actionable news with clear timing implications for overnight trades \n"
            + "- Identify both bullish and bearish catalysts affecting next trading day \n"
            + "- Assess news impact magnitude (minor <2%, moderate 2-5%, major >5% overnight/next-day moves) \n"
            + "- Consider news durability (will impact persist through next day or just overnight?) \n"
            + "- Analyze market reaction patterns to similar news in overnight/pre-market sessions \n"
            + "**AVOID:** Generic market commentary, long-term trends, intraday noise. Focus on EOD-relevant news with overnight impact potential.\n"
            + """ Make sure to append a Markdown table at the end organizing:
| News Event | Date/Time | Impact Level | Price Direction | EOD Trading Implication |
|------------|-----------|--------------|----------------|------------------------|\n
| [Specific Event] | [Date/Time] | [High/Med/Low] | [Bullish/Bearish/Neutral] | [Entry/Exit/Hold Strategy] |

Provide specific, actionable news analysis for EOD trading decisions with clear timing and impact assessment."""
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
                    "For your reference, the current date is {current_date}. We are looking at the ticekr: {ticker}",
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

        chain = prompt | llm.bind_tools(tools)

        # Copy the incoming conversation history so we can append to it when the model makes tool calls
        messages_history = list(state["messages"])

        # Track tools that have failed to prevent LLM retries
        failed_tools = set()
        max_iterations = 5  # Prevent infinite loops
        iteration_count = 0

        # First LLM response
        model_name = getattr(llm, 'model_name', 'unknown')
        start_time = log_llm_start("NEWS_ANALYST", model_name)
        try:
            result = chain.invoke(messages_history)
            log_llm_end("NEWS_ANALYST", model_name, start_time, result)
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[LLM - NEWS_ANALYST] ❌ {model_name} failed after {elapsed:.2f}s: {str(e)}")
            raise

        # Handle iterative tool calls until the model stops requesting them
        tool_calls_list = getattr(result, "tool_calls", None) or getattr(result, "additional_kwargs", {}).get("tool_calls", [])
        while tool_calls_list and iteration_count < max_iterations:
            iteration_count += 1
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
                    tool_result = f"Tool '{tool_name}' already failed. Skipping retry."
                    print(f"[NEWS] ⚠️ Skipping retry of failed tool: {tool_name}")
                    messages_history.append(ToolMessage(content=tool_result, tool_call_id=tool_call_id))
                    continue

                # Find the matching tool by name
                tool_fn = next((t for t in tools if t.name == tool_name), None)

                if tool_fn is None:
                    tool_result = f"Tool '{tool_name}' not found."
                    print(f"[NEWS] ⚠️ {tool_result}")
                else:
                    try:
                        print(f"[NEWS] 🔧 Calling tool '{tool_name}'...")
                        start_time = time.time()

                        # LangChain Tool objects expose `.run` (string IO) as well as `.invoke` (dict/kwarg IO)
                        if hasattr(tool_fn, "invoke"):
                            tool_result = tool_fn.invoke(tool_args)
                        else:
                            tool_result = tool_fn.run(**tool_args)

                        elapsed = time.time() - start_time
                        print(f"[NEWS] ✅ Tool '{tool_name}' completed ({elapsed:.2f}s)")

                    except TimeoutError as timeout_err:
                        elapsed = time.time() - start_time
                        failed_tools.add(tool_name)
                        tool_result = f"Tool '{tool_name}' timed out after {elapsed:.0f}s. Skipping."
                        print(f"[NEWS] ⏰ TIMEOUT: {tool_result}")
                    except Exception as tool_err:
                        elapsed = time.time() - start_time
                        tool_result = f"Error running tool '{tool_name}': {str(tool_err)}"
                        print(f"[NEWS] ❌ Tool '{tool_name}' failed after {elapsed:.2f}s: {str(tool_err)}")

                # Append the tool result
                messages_history.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call_id))

            # Ask the LLM to continue with the new context
            model_name = getattr(llm, 'model_name', 'unknown')
            start_time = log_llm_start("NEWS_ANALYST", model_name)
            try:
                result = chain.invoke(messages_history)
                log_llm_end("NEWS_ANALYST", model_name, start_time, result)
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"[LLM - NEWS_ANALYST] ❌ {model_name} failed after {elapsed:.2f}s: {str(e)}")
                raise
            tool_calls_list = getattr(result, "tool_calls", None) or getattr(result, "additional_kwargs", {}).get("tool_calls", [])

        # Check if we hit max iterations
        if iteration_count >= max_iterations:
            print(f"[NEWS] ⚠️ Reached max tool iterations ({max_iterations}). Proceeding with available data.")

        # Check if the result already contains FINAL TRANSACTION PROPOSAL
        if "FINAL TRANSACTION PROPOSAL:" not in result.content:
            # Create a simple prompt that includes the analysis content directly
            final_prompt = f"""Based on the following news analysis for {ticker}, please provide your final trading recommendation considering the overall news sentiment and implications.

Analysis:
{result.content}

You must conclude with: FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** followed by a brief justification."""

            # Use a simple chain without tools for the final recommendation
            final_chain = llm
            model_name = getattr(llm, 'model_name', 'unknown')
            start_time = log_llm_start("NEWS_ANALYST", model_name)
            try:
                final_result = final_chain.invoke(final_prompt)
                log_llm_end("NEWS_ANALYST", model_name, start_time, final_result)
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"[LLM - NEWS_ANALYST] ❌ {model_name} failed after {elapsed:.2f}s: {str(e)}")
                raise

            # Combine the analysis with the final proposal
            combined_content = result.content + "\n\n" + final_result.content
            result = AIMessage(content=combined_content)

        return {
            "messages": [result],
            "news_report": result.content,
        }

    return news_analyst_node
