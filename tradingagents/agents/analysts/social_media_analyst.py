from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, ToolMessage
import time
import json
from tradingagents.prompts import load_prompt, render_prompt

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
        openai_available = toolkit.has_openai_web_search()

        reddit_tool = toolkit.get_reddit_news if is_crypto else toolkit.get_reddit_stock_info
        tools = [reddit_tool]
        if toolkit.config["online_tools"] and openai_available:
            tools.insert(0, toolkit.get_stock_news_openai)

        source_labels = ["Reddit"]
        if toolkit.config["online_tools"] and openai_available:
            source_labels.insert(0, "OpenAI web-search sentiment")

        source_guidance = (
            " Use all currently available social tools before concluding."
            f" Active sources: {', '.join(source_labels)}."
            + (" Use `get_reddit_news(curr_date)` for crypto context." if is_crypto else " Use `get_reddit_stock_info(ticker, curr_date)` for stock context.")
        )
        system_message = render_prompt(
            "analysts/social_system",
            source_guidance=source_guidance,
        )
        asset_context = f"The current company we want to analyze is {ticker}"

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    load_prompt("shared/analyst_tool_system"),
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)
        prompt = prompt.partial(asset_context=asset_context)

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
                complete_prompt = render_prompt(
                    "shared/analyst_tool_system",
                    tool_names=tool_names_str,
                    system_message=system_message,
                    current_date=current_date,
                    asset_context=asset_context,
                )
            
            capture_agent_prompt("sentiment_report", complete_prompt, ticker)
        except Exception as e:
            print(f"[SOCIAL] Warning: Could not capture complete prompt: {e}")
            # Fallback to system message only
            capture_agent_prompt("sentiment_report", system_message, ticker)

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
                    print(f"[SOCIAL] ⚠️ {tool_result}")
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
        
        analysis_content = (result.content or "").strip()
        if not analysis_content:
            analysis_content = (
                "The analyst did not return a full social narrative. "
                "Be explicit about that limitation in the final recommendation."
            )

        # Ensure we have a final recommendation without replacing tool-grounded evidence.
        if "FINAL TRANSACTION PROPOSAL:" not in analysis_content:
            # Create a final recommendation based on the analysis
            final_prompt = render_prompt(
                "analysts/social_final_recommendation",
                ticker=ticker,
                analysis_content=analysis_content,
            )
            
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
