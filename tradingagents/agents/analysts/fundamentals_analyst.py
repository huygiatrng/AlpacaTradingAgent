from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from langchain_core.messages import AIMessage, ToolMessage
from tradingagents.agents.utils.agent_utils import log_llm_start, log_llm_end

# Import prompt capture utility
try:
    from webui.utils.prompt_capture import capture_agent_prompt
except ImportError:
    # Fallback for when webui is not available
    def capture_agent_prompt(report_type, prompt_content, symbol=None):
        pass


def create_fundamentals_analyst(llm, toolkit):
    def fundamentals_analyst_node(state):
        # print(f"[FUNDAMENTALS] Starting fundamentals analysis for {state['company_of_interest']}")
        start_time = time.time()
        
        try:
            current_date = state["trade_date"]
            ticker = state["company_of_interest"]
            company_name = state["company_of_interest"]
            
            # print(f"[FUNDAMENTALS] Analyzing {ticker} on {current_date}")
            
            # Check if the ticker is a cryptocurrency
            is_crypto = "/" in ticker or "USD" in ticker.upper() or "USDT" in ticker.upper()
            # print(f"[FUNDAMENTALS] Detected asset type: {'Cryptocurrency' if is_crypto else 'Stock'}")
            
            # Extract base ticker for cryptocurrencies (BTC from BTC/USD or BTCUSDT)
            display_ticker = ticker
            if is_crypto:
                # Remove USD, USDT or anything after /
                if "/" in ticker:
                    display_ticker = ticker.split("/")[0]
                elif "USDT" in ticker.upper():
                    display_ticker = ticker.upper().replace("USDT", "")
                elif "USD" in ticker.upper():
                    display_ticker = ticker.upper().replace("USD", "")

            # All tools now use smart caching - no need for online_tools flag
            if is_crypto:
                tools = [
                    toolkit.get_defillama_fundamentals,  # DeFiLlama data (with caching)
                    toolkit.get_earnings_calendar  # For crypto events/announcements (with caching)
                ]
                # print(f"[FUNDAMENTALS] Using crypto tools: DeFiLlama + Events Calendar")
            else:
                tools = [
                    toolkit.get_fundamentals_openai,
                    toolkit.get_earnings_calendar,  # Earnings data (with caching)
                    toolkit.get_earnings_surprise_analysis,  # Earnings analysis (with caching)
                ]
                # print(f"[FUNDAMENTALS] Using stock tools: OpenAI Fundamentals + Earnings Analysis")

            system_message = (
                "You are an EOD TRADING fundamentals analyst focused on identifying fundamental catalysts and factors that could drive overnight and next-day price movements. "
                + ("Analyze DeFi metrics like TVL changes, protocol upgrades, token unlock schedules, yield farming opportunities, and major partnership announcements that could impact crypto prices overnight and next trading day. " if is_crypto else "Focus on after-hours earnings, analyst upgrades/downgrades, insider activity, overnight news, and fundamental shifts that could create EOD trading opportunities for next-day positioning. ")
                + "**EOD TRADING FUNDAMENTALS FOCUS:** \n"
                + "Look for overnight catalysts, not long-term value investing metrics. Identify events and data releases that could drive overnight gaps and next-day price movements. \n"
                + "**KEY AREAS FOR EOD TRADERS:** \n"
                + "1. **After-Hours Earnings:** Quarterly results released after market close, guidance changes, surprise potential \n"
                + "2. **Analyst Activity:** After-hours upgrades/downgrades, price target changes, overnight research reports \n"
                + "3. **Insider Trading:** Recent insider buying/selling patterns indicating overnight sentiment shifts \n"
                + "4. **Overnight Sector Trends:** Industry rotation, peer performance, relative strength for next day \n"
                + "5. **Event Calendar:** FDA approvals, contract announcements, product launches affecting next trading day \n"
                + "6. **Financial Health:** Any deteriorating metrics that could trigger overnight selling pressure \n"
                + "7. **Momentum Factors:** After-hours estimate revisions, sales trends, competitive positioning changes \n"
                + "**ANALYSIS REQUIREMENTS:** \n"
                + "- Identify specific times for overnight catalysts \n"
                + "- Assess probability and magnitude of potential overnight price impact \n"
                + "- Consider both positive and negative fundamental drivers for next day \n"
                + "- Focus on actionable insights for overnight trading and next-day positioning \n"
                + "- Avoid long-term valuation metrics unless they create immediate overnight catalysts \n"
                + "Provide detailed, actionable fundamental analysis that EOD traders can use to time entries and exits around overnight events and after-hours data releases."
                + " Make sure to append a Markdown table at the end organizing key overnight events, times, and potential price impact for EOD trading decisions."
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
                        "For your reference, the current date is {current_date}. " 
                        + ("The cryptocurrency we want to analyze is {ticker}" if is_crypto else "The company we want to look at is {ticker}"),
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )

            # print(f"[FUNDAMENTALS] Setting up prompt and chain...")
            prompt = prompt.partial(system_message=system_message)
            prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
            prompt = prompt.partial(current_date=current_date)
            prompt = prompt.partial(ticker=display_ticker if is_crypto else ticker)

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
                    ticker_display = display_ticker if is_crypto else ticker
                    asset_type_text = "The cryptocurrency we want to analyze is" if is_crypto else "The company we want to look at is"
                    complete_prompt = f"""You are a helpful AI assistant, collaborating with other assistants. Use the provided tools to progress towards answering the question. If you are unable to fully answer, that's OK; another assistant with different tools will help where you left off. Execute what you can to make progress. If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable, prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop. You have access to the following tools: {tool_names_str}.

{system_message}

For your reference, the current date is {current_date}. {asset_type_text} {ticker_display}"""
                
                capture_agent_prompt("fundamentals_report", complete_prompt, ticker)
            except Exception as e:
                print(f"[FUNDAMENTALS] Warning: Could not capture complete prompt: {e}")
                # Fallback to system message only
                capture_agent_prompt("fundamentals_report", system_message, ticker)

            chain = prompt | llm.bind_tools(tools)
            
            # print(f"[FUNDAMENTALS] Invoking LLM chain...")
            # Copy the incoming conversation history so we can append to it when the model makes tool calls
            messages_history = list(state["messages"])

            # Track tools that have failed to prevent LLM retries
            failed_tools = set()
            max_iterations = 5  # Prevent infinite loops
            iteration_count = 0

            # First LLM response
            model_name = getattr(llm, 'model_name', 'unknown')
            llm_start_time = log_llm_start("FUNDAMENTALS_ANALYST", model_name)
            try:
                result = chain.invoke(messages_history)
                log_llm_end("FUNDAMENTALS_ANALYST", model_name, llm_start_time, result)
            except Exception as e:
                elapsed = time.time() - llm_start_time
                print(f"[LLM - FUNDAMENTALS_ANALYST] ❌ {model_name} failed after {elapsed:.2f}s: {str(e)}")
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
                        print(f"[FUNDAMENTALS] ⚠️ Skipping retry of failed tool: {tool_name}")
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
                        print(f"[FUNDAMENTALS] ⚠️ {tool_result}")
                    else:
                        try:
                            print(f"[FUNDAMENTALS] 🔧 Starting tool '{tool_name}'...")
                            start_time = time.time()

                            # LangChain Tool objects expose `.run` (string IO) as well as `.invoke` (dict/kwarg IO)
                            if hasattr(tool_fn, "invoke"):
                                tool_result = tool_fn.invoke(tool_args)
                            else:
                                tool_result = tool_fn.run(**tool_args)

                            elapsed = time.time() - start_time
                            print(f"[FUNDAMENTALS] ✅ Tool '{tool_name}' completed (took {elapsed:.2f}s)")

                        except TimeoutError as timeout_err:
                            elapsed = time.time() - start_time
                            # Mark this tool as failed to prevent retries
                            failed_tools.add(tool_name)
                            tool_result = f"PERMANENT FAILURE: Tool '{tool_name}' timed out after {elapsed:.0f}s. This tool will not be retried. Proceed with analysis using other available data sources."
                            print(f"[FUNDAMENTALS] ⏰ TIMEOUT: {tool_result}")
                        except Exception as tool_err:
                            elapsed = time.time() - start_time
                            tool_result = f"Error running tool '{tool_name}': {str(tool_err)}"
                            print(f"[FUNDAMENTALS] ❌ Tool '{tool_name}' failed after {elapsed:.2f}s: {str(tool_err)}")

                    # Append the tool result message
                    tool_msg = ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call_id,
                    )
                    messages_history.append(tool_msg)

                # Ask the LLM to continue with the new context
                model_name = getattr(llm, 'model_name', 'unknown')
                llm_start_time = log_llm_start("FUNDAMENTALS_ANALYST", model_name)
                try:
                    result = chain.invoke(messages_history)
                    log_llm_end("FUNDAMENTALS_ANALYST", model_name, llm_start_time, result)
                except Exception as e:
                    elapsed = time.time() - llm_start_time
                    print(f"[LLM - FUNDAMENTALS_ANALYST] ❌ {model_name} failed after {elapsed:.2f}s: {str(e)}")
                    raise

                # Update tool_calls_list for next iteration
                tool_calls_list = getattr(result, "tool_calls", None) or getattr(result, "additional_kwargs", {}).get("tool_calls", [])
             
            # Check if we hit max iterations
            if iteration_count >= max_iterations:
                print(f"[FUNDAMENTALS] ⚠️ Reached max tool iterations ({max_iterations}). Proceeding with available data.")
             
            elapsed_time = time.time() - start_time
            # print(f"[FUNDAMENTALS] ✅ Analysis completed in {elapsed_time:.2f} seconds")
            # print(f"[FUNDAMENTALS] Generated report length: {len(result.content)} characters")

            # Check if the result already contains FINAL TRANSACTION PROPOSAL
            if "FINAL TRANSACTION PROPOSAL:" not in result.content:
                # Create a simple prompt that includes the analysis content directly
                final_prompt = f"""Based on the following fundamental analysis for {ticker}, please provide your final trading recommendation considering the financial health, valuation, and earnings outlook.

Analysis:
{result.content}

You must conclude with: FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** followed by a brief justification."""
                
                # Use a simple chain without tools for the final recommendation
                final_chain = llm
                model_name = getattr(llm, 'model_name', 'unknown')
                llm_start_time = log_llm_start("FUNDAMENTALS_ANALYST", model_name)
                try:
                    final_result = final_chain.invoke(final_prompt)
                    log_llm_end("FUNDAMENTALS_ANALYST", model_name, llm_start_time, final_result)
                except Exception as e:
                    elapsed = time.time() - llm_start_time
                    print(f"[LLM - FUNDAMENTALS_ANALYST] ❌ {model_name} failed after {elapsed:.2f}s: {str(e)}")
                    raise

                # Combine the analysis with the final proposal
                combined_content = result.content + "\n\n" + final_result.content
                result = AIMessage(content=combined_content)

            # Append final assistant response to history for downstream agents
            messages_history.append(result)

            # Ensure we have a report
            if not result.content or len(result.content) < 100:
                # Fallback: create a summary from tool results if final LLM call failed
                fallback_report = f"Fundamentals Analysis for {ticker}:\n\n"
                for msg in messages_history:
                    if hasattr(msg, '__class__') and msg.__class__.__name__ == 'ToolMessage':
                        fallback_report += f"{msg.content[:500]}\n\n"
                fundamentals_report = fallback_report if len(fallback_report) > 100 else "Error: Fundamentals analysis could not be completed"
                print(f"[FUNDAMENTALS] ⚠️ Using fallback report due to insufficient content from LLM")
            else:
                fundamentals_report = result.content

            return {
                "messages": messages_history,
                "fundamentals_report": fundamentals_report,
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"Error in fundamentals analysis for {state['company_of_interest']}: {str(e)}"
            print(f"[FUNDAMENTALS] ❌ {error_msg}")
            print(f"[FUNDAMENTALS] ❌ Failed after {elapsed_time:.2f} seconds")
            
            # Import traceback for detailed error logging
            import traceback
            print(f"[FUNDAMENTALS] ❌ Full traceback:")
            traceback.print_exc()
            
            # Return a minimal report with error information
            fallback_report = f"""
# Fundamentals Analysis Error

**Symbol:** {state['company_of_interest']}
**Date:** {state.get('trade_date', 'Unknown')}
**Error:** {str(e)}
**Duration:** {elapsed_time:.2f} seconds

## Error Details
The fundamentals analysis encountered an error and could not complete successfully. This may be due to:
- API rate limits or timeouts
- Network connectivity issues  
- Invalid ticker symbol
- Missing data for the requested symbol

## Recommendation
⚠️ **PROCEED WITH CAUTION** - Unable to perform fundamental analysis for this symbol.

| Metric | Status |
|--------|--------|
| Fundamental Data | ❌ Unavailable |
| Analysis Status | ❌ Failed |
| Recommendation | ⚠️ Incomplete Analysis |
"""
            
            return {
                "messages": [result if 'result' in locals() else None],
                "fundamentals_report": fallback_report,
            }

    return fundamentals_analyst_node
