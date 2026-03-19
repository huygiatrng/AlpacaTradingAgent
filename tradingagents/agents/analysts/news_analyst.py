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

        if toolkit.config["online_tools"]:
            if is_crypto:
                tools = [
                    toolkit.get_global_news_openai,
                    toolkit.get_google_news,
                    toolkit.get_coindesk_news,
                ]
            else:
                tools = [
                    toolkit.get_global_news_openai,
                    toolkit.get_google_news,
                    toolkit.get_finnhub_news_recent,
                ]
        else:
            if is_crypto:
                tools = [
                    toolkit.get_google_news,
                    toolkit.get_coindesk_news,
                ]
            else:
                tools = [
                    toolkit.get_google_news,
                    toolkit.get_finnhub_news_recent,
                ]

        source_guidance = (
            " **MUST** call the provided news tools to gather current information. "
            " ALWAYS call both OpenAI web search and direct news sources before providing analysis."
            " For stocks: MUST use `get_global_news_openai` + `get_google_news` + `get_finnhub_news_recent`."
            " For crypto: MUST use `get_global_news_openai` + `get_google_news` + `get_coindesk_news`."
            " For `get_finnhub_news_recent`, always pass ticker and curr_date from context."
            " Do not provide news analysis without first executing these tool calls to obtain current market-moving news."
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
            + f"**IMPORTANT:** When using get_global_news_openai, ALWAYS pass ticker_context='{ticker}' to get {('crypto-relevant global news (regulation, institutional adoption, DeFi developments)' if is_crypto else 'sector-relevant global news')} instead of generic macro news.\n"
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

        # First LLM response
        result = chain.invoke(messages_history)

        # Track iterations to prevent infinite loops
        max_iterations = 10
        iteration_count = 0

        # Handle iterative tool calls until the model stops requesting them
        while getattr(result, "additional_kwargs", {}).get("tool_calls") and iteration_count < max_iterations:
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
        
        # ========== LAYER 2: Content Quality Check ==========
        # Enhanced validation to ensure substantial analysis content
        analysis_content = result.content if result.content else ""
        
        # Check if we have substantial analysis content (not just final proposal)
        if len(analysis_content.strip()) < 100 or ("FINAL TRANSACTION PROPOSAL:" in analysis_content and len(analysis_content.replace("FINAL TRANSACTION PROPOSAL:", "").strip()) < 100):
            # Generate fallback analysis if content is too short
            fallback_prompt = f"""As a swing trading news analyst, provide a comprehensive news analysis for {ticker} on {current_date}.

Since detailed news data may not be available, provide a professional analysis covering:
1. **Recent Company News** for {ticker}
2. **Industry & Sector News** affecting {ticker}
3. **Earnings & Events** within 2-10 days
4. **Analyst Activity & Sentiment** shifts
5. **Market-Moving Catalysts** for swing traders
6. **Swing Trading Implications** for multi-day positions

Include the required news events table and conclude with swing trading implications.
Focus on actionable insights for multi-day (2-10 day) swing trading decisions."""
            
            fallback_result = llm.invoke(fallback_prompt)
            analysis_content = fallback_result.content if hasattr(fallback_result, 'content') else str(fallback_result)
        
        # ========== LAYER 3: Ensure Final Recommendation ==========
        # Ensure we have a final recommendation
        if "FINAL TRANSACTION PROPOSAL:" not in analysis_content:
            # Create a final recommendation based on the analysis
            final_prompt = f"""Based on the following news analysis for {ticker}, provide your final swing trading recommendation considering news catalysts and market implications.

Analysis:
{analysis_content}

Provide a brief justification and conclude with: FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**"""
            
            # Use a simple chain without tools for the final recommendation
            final_chain = llm
            final_result = final_chain.invoke(final_prompt)
            final_content = final_result.content if hasattr(final_result, 'content') else str(final_result)
            
            # Properly combine the analysis with the final proposal
            combined_content = analysis_content + "\n\n---\n\n## Final Recommendation\n\n" + final_content
            result = AIMessage(content=combined_content)
        else:
            # Analysis already contains final proposal
            result = AIMessage(content=analysis_content)

        return {
            "messages": [result],
            "news_report": result.content,
        }

    return news_analyst_node
