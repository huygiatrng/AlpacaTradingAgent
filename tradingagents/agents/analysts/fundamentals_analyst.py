from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from langchain_core.messages import AIMessage, ToolMessage

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

            if toolkit.config["online_tools"]:
                if is_crypto:
                    tools = [
                        toolkit.get_defillama_fundamentals,
                        toolkit.get_fundamentals_openai,
                    ]
                    # print(f"[FUNDAMENTALS] Using online crypto tools: DeFiLlama + OpenAI fundamentals")
                else:
                    tools = [
                        toolkit.get_fundamentals_openai,
                        toolkit.get_finnhub_company_insider_sentiment,
                        toolkit.get_finnhub_company_insider_transactions,
                        toolkit.get_simfin_balance_sheet,
                        toolkit.get_simfin_cashflow,
                        toolkit.get_simfin_income_stmt,
                    ]
                    # print(f"[FUNDAMENTALS] Using online stock tools: OpenAI + Finnhub + SimFin")
            else:
                if is_crypto:
                    tools = [toolkit.get_defillama_fundamentals]
                else:
                    tools = [
                        toolkit.get_finnhub_company_insider_sentiment,
                        toolkit.get_finnhub_company_insider_transactions,
                        toolkit.get_simfin_balance_sheet,
                        toolkit.get_simfin_cashflow,
                        toolkit.get_simfin_income_stmt,
                    ]
                # print(f"[FUNDAMENTALS] Using offline tools: Finnhub + SimFin (stock) or DeFiLlama (crypto)")

            source_guidance = (
                " When online tools are enabled, combine OpenAI web-search fundamentals with structured data tools before concluding."
                " For stocks, use OpenAI + Finnhub + SimFin."
                " For crypto, use OpenAI + DeFiLlama."
            )
            system_message = (
                "You are a SWING TRADING fundamentals analyst focused on identifying fundamental catalysts and factors that could drive multi-day price movements (2-10 day swing horizon). "
                + ("Analyze DeFi metrics like TVL changes, protocol upgrades, token unlock schedules, yield farming opportunities, and major partnership announcements that could sustain multi-day crypto price trends. " if is_crypto else "Focus on earnings surprises, analyst upgrades/downgrades, insider activity, and fundamental shifts that could sustain multi-day swing moves. ")
                + "**SWING TRADING FUNDAMENTALS FOCUS:** \n"
                + "Look for catalysts that can sustain price movement across multiple days. Identify events within the 2-10 day holding window. \n"
                + "**KEY AREAS FOR SWING TRADERS:** \n"
                + "1. **Earnings & Guidance:** Recent or upcoming quarterly results, guidance changes, and surprise potential within the swing window \n"
                + "2. **Analyst Activity:** Upgrades/downgrades, price target changes, initiation of coverage that could drive multi-day re-rating \n"
                + "3. **Insider Trading:** Recent insider buying/selling patterns indicating multi-day directional conviction \n"
                + "4. **Sector Trends:** Industry rotation, peer performance, relative strength persisting across days \n"
                + "5. **Event Calendar:** FDA approvals, contract announcements, product launches within the swing holding period \n"
                + "6. **Financial Health:** Key metrics that could trigger sustained buying or selling pressure \n"
                + "7. **Momentum Factors:** Estimate revisions, revenue trends, and competitive positioning changes with multi-day impact \n"
                + "**ANALYSIS REQUIREMENTS:** \n"
                + "- Identify fundamental catalysts within the 2-10 day swing window \n"
                + "- Assess probability and magnitude of potential multi-day price impact \n"
                + "- Consider both positive and negative fundamental drivers over the swing period \n"
                + "- Focus on actionable insights for swing trade entries and exits \n"
                + "- Avoid long-term valuation metrics unless they create catalysts during the swing window \n"
                + f"Provide detailed, actionable fundamental analysis that swing traders can use to time entries and exits around multi-day catalysts.{source_guidance}"
                + " Make sure to append a Markdown table at the end organizing key events, dates, and potential price impact for swing trading decisions."
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
                        # print(f"[FUNDAMENTALS] ⚠️ {tool_result}")
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
                final_result = final_chain.invoke(final_prompt)
                
                # Combine the analysis with the final proposal
                combined_content = result.content + "\n\n" + final_result.content
                result = AIMessage(content=combined_content)

            # Append final assistant response to history for downstream agents
            messages_history.append(result)

            return {
                "messages": messages_history,
                "fundamentals_report": result.content,
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
