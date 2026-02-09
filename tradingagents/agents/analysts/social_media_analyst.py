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


def create_social_media_analyst(llm, toolkit):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]
        is_crypto = "/" in ticker or "USD" in ticker.upper() or "USDT" in ticker.upper()

        if toolkit.config["online_tools"]:
            reddit_tool = toolkit.get_reddit_news if is_crypto else toolkit.get_reddit_stock_info
            tools = [
                toolkit.get_stock_news_openai,
                reddit_tool,
            ]
        else:
            tools = [toolkit.get_reddit_news] if is_crypto else [toolkit.get_reddit_stock_info]

        source_guidance = (
            " When online tools are enabled, use both OpenAI web-search sentiment and Reddit-based social signals before concluding."
            + (" Use `get_reddit_news(curr_date)` for crypto context." if is_crypto else " Use `get_reddit_stock_info(ticker, curr_date)` for stock context.")
        )
        system_message = (
            "You are a SWING TRADING social media analyst specializing in identifying sentiment shifts and social catalysts that could drive multi-day price movements. "
            "Your role is to analyze social media posts, community sentiment, and social momentum indicators that affect swing trading positions (2-10 day holds).\n\n"
            "**SWING TRADING SOCIAL MEDIA FOCUS:**\n"
            "1. **Trending Sentiment:** Social sentiment trends that could sustain or reverse a multi-day price swing\n"
            "2. **Sustained Catalysts:** Influencer mentions, viral campaigns, or trending hashtags with multi-day momentum\n"
            "3. **Community Positioning:** Reddit, Twitter sentiment indicating retail trader conviction over the coming days\n"
            "4. **Buzz Persistence:** Volume and duration of social discussions suggesting sustained moves vs. short-lived spikes\n"
            "5. **Social Contrarian Signals:** Extreme social sentiment indicating potential multi-day reversals\n"
            "6. **Narrative Durability:** How long the current social narrative is likely to persist and drive price action\n\n"
            "**SWING TRADING ANALYSIS REQUIREMENTS:**\n"
            "- **Sentiment Direction:** Current bullish/bearish/neutral social sentiment with recent trend\n"
            "- **Momentum Indicators:** Social volume, engagement rates, viral potential for multi-day moves\n"
            "- **Key Influencers:** Important social media accounts or communities driving sentiment shifts\n"
            "- **Contrarian Opportunities:** Over-extended social sentiment suggesting mean reversion swings\n"
            "- **Event Catalysts:** Social media events or announcements with multi-day trading implications\n"
            "- **Risk Factors:** Social media risks that could impact swing positions negatively\n\n"
            "**AVOID:** Intraday noise, one-off viral spikes with no follow-through. Focus on social factors "
            "that create sustained multi-day moves relevant to swing trading.\n\n"
            "Provide comprehensive social media sentiment analysis that swing traders can use for entry/exit timing and "
            f"position sizing decisions. Always include specific social media examples and sentiment metrics when available.{source_guidance}"
            + """ 

**SWING TRADING SOCIAL SENTIMENT TABLE:**
Make sure to append a Markdown table organizing:
| Social Platform | Sentiment | Volume | Trend (Multi-Day) | Swing Trading Signal |
|-----------------|-----------|---------|-------------------|---------------------|
| [Platform] | [Bullish/Bearish/Neutral] | [High/Med/Low] | [Direction & Duration] | [Enter/Exit/Hold Strategy] |

Focus on actionable social sentiment insights for swing trading decisions."""
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
                    "For your reference, the current date is {current_date}. The current company we want to analyze is {ticker}",
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

For your reference, the current date is {current_date}. The current company we want to analyze is {ticker}"""
            
            capture_agent_prompt("sentiment_report", complete_prompt, ticker)
        except Exception as e:
            print(f"[SOCIAL] Warning: Could not capture complete prompt: {e}")
            # Fallback to system message only
            capture_agent_prompt("sentiment_report", system_message, ticker)

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
                    print(f"[SOCIAL] ⚠️ {tool_result}")
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
        
        # Enhanced validation and final proposal handling
        analysis_content = result.content if result.content else ""
        
        # Check if we have substantial analysis content (not just final proposal)
        if len(analysis_content.strip()) < 100 or "FINAL TRANSACTION PROPOSAL:" in analysis_content and len(analysis_content.replace("FINAL TRANSACTION PROPOSAL:", "").strip()) < 100:
            # Generate fallback analysis if content is too short
            fallback_prompt = f"""As a swing trading social media analyst, provide a comprehensive social sentiment analysis for {ticker} on {current_date}.

Since detailed social media data may not be available, provide a professional analysis covering:
1. **General Social Sentiment Trends** for {ticker}
2. **Social Media Momentum Indicators** 
3. **Community Positioning Analysis**
4. **Swing Trading Social Signals**
5. **Risk Factors from Social Sentiment**

Include the required social sentiment table and conclude with swing trading implications.
Focus on actionable insights for multi-day (2-10 day) swing trading decisions."""
            
            fallback_result = llm.invoke(fallback_prompt)
            analysis_content = fallback_result.content if hasattr(fallback_result, 'content') else str(fallback_result)
        
        # Ensure we have a final recommendation
        if "FINAL TRANSACTION PROPOSAL:" not in analysis_content:
            # Create a final recommendation based on the analysis
            final_prompt = f"""Based on the following social media and sentiment analysis for {ticker}, provide your final swing trading recommendation considering social momentum and sentiment indicators.

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
            "sentiment_report": result.content,
        }

    return social_media_analyst_node
