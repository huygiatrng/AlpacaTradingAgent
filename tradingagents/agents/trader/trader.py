import functools
import time
import json
from langchain_core.messages import AIMessage
from ..schemas import TraderProposal, render_trader_proposal
from ..utils.agent_trading_modes import (
    ensure_final_transaction_proposal,
    extract_recommendation,
    get_agent_specific_context,
    get_trading_mode_context,
)
from ..utils.memory import TradingMemoryLog
from ..utils.report_context import (
    get_agent_context_bundle,
    build_debate_digest,
)
from ..utils.structured import bind_structured, invoke_structured_or_freetext
from tradingagents.dataflows.alpaca_utils import AlpacaUtils
from tradingagents.prompts import render_prompt

# Import prompt capture utility
try:
    from webui.utils.prompt_capture import capture_agent_prompt
except ImportError:
    # Fallback for when webui is not available
    def capture_agent_prompt(report_type, prompt_content, symbol=None):
        pass


def create_trader(llm, memory, config=None):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")
    decision_log = TradingMemoryLog(config)

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        
        # Determine current position from live Alpaca account (fallback to state)
        current_position = AlpacaUtils.get_current_position_state(company_name)
        # Persist into state so downstream agents see an accurate picture
        state["current_position"] = current_position

        # ---------------------------------------------------------
        # NEW: Pull richer live account & position metrics from Alpaca
        # ---------------------------------------------------------
        positions_data = AlpacaUtils.get_positions_data()
        account_info = AlpacaUtils.get_account_info()

        # Build a user-friendly summary for the specific symbol the agent cares about
        position_stats_desc = ""
        symbol_key = company_name.upper().replace("/", "")
        for pos in positions_data:
            if pos["Symbol"].upper() == symbol_key:
                qty = pos["Qty"]
                avg_entry = pos["Avg Entry"]
                today_pl_dollars = pos["Today's P/L ($)"]
                today_pl_percent = pos["Today's P/L (%)"]
                total_pl_dollars = pos["Total P/L ($)"]
                total_pl_percent = pos["Total P/L (%)"]

                position_stats_desc = (
                    f"Position Details for {company_name}:\n"
                    f"- Quantity: {qty}\n"
                    f"- Average Entry Price: {avg_entry}\n"
                    f"- Today's P/L: {today_pl_dollars} ({today_pl_percent})\n"
                    f"- Total P/L: {total_pl_dollars} ({total_pl_percent})"
                )
                break
        if not position_stats_desc:
            position_stats_desc = "No open position details available for this symbol."

        buying_power = account_info.get("buying_power", 0.0)
        cash = account_info.get("cash", 0.0)
        daily_change_dollars = account_info.get("daily_change_dollars", 0.0)
        daily_change_percent = account_info.get("daily_change_percent", 0.0)
        account_status_desc = (
            "Account Status:\n"
            f"- Buying Power: ${buying_power:,.2f}\n"
            f"- Cash: ${cash:,.2f}\n"
            f"- Daily Change: ${daily_change_dollars:,.2f} ({daily_change_percent:.2f}%)"
        )
        # ---------------------------------------------------------
        # END NEW BLOCK
        # ---------------------------------------------------------

        # Human-readable description for the prompt
        open_pos_desc = (
            f"We currently have an open {current_position} position in {company_name}."
            if current_position != "NEUTRAL"
            else f"We do not have any open position in {company_name}."
        )
        
        # Get centralized trading mode context
        trading_context = get_trading_mode_context(config, current_position)
        agent_context = get_agent_specific_context("trader", trading_context)
        
        # Get mode-specific terms for the prompt
        actions = trading_context["actions"]
        mode_name = trading_context["mode_name"]
        decision_format = trading_context["decision_format"]
        final_format = trading_context["final_format"]
        output_language = (config or {}).get("output_language", "English")

        context_bundle = get_agent_context_bundle(
            state,
            agent_role="trader",
            objective=(
                f"Prepare a swing trading plan for {company_name}. "
                f"Current trader plan draft: {investment_plan}"
            ),
            config=config,
        )
        analysis_context = context_bundle.get("analysis_context_compact", context_bundle["analysis_context"])
        claim_matrix = context_bundle.get("decision_claim_matrix", "")
        debate_digest = build_debate_digest(state.get("investment_debate_state"), "investment", config=config)
        all_reports_text = context_bundle.get("all_reports_text", "")
        curr_situation = context_bundle["memory_context"]
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"
        decision_memory_str = decision_log.get_past_context(company_name)

        trader_context = render_prompt(
            "trader/trader_context",
            agent_context=agent_context,
            open_pos_desc=open_pos_desc,
            position_stats_desc=position_stats_desc,
            account_status_desc=account_status_desc,
            claim_matrix=claim_matrix,
            all_reports_text=all_reports_text,
            debate_digest=debate_digest,
            decision_format=decision_format,
            final_format=final_format,
            output_language=output_language,
        )

        # Enhanced content validation for investment plan
        plan_content = investment_plan if investment_plan else ""
        
        # Check if investment plan is substantial enough
        if len(plan_content.strip()) < 150 or ("FINAL TRANSACTION PROPOSAL:" in plan_content and len(plan_content.replace("FINAL TRANSACTION PROPOSAL:", "").strip()) < 100):
            # Generate enhanced analysis prompt when investment plan is insufficient
            enhanced_prompt = render_prompt(
                "trader/trader_enhanced_plan",
                company_name=company_name,
                analysis_context=analysis_context,
            )

            context = {
                "role": "user", 
                "content": enhanced_prompt
            }
        else:
            # Use original context with valid investment plan
            context = {
                "role": "user",
                "content": render_prompt(
                    "trader/trader_user_plan",
                    company_name=company_name,
                    investment_plan=investment_plan,
                ),
            }

        messages = [
            {
                "role": "system",
                "content": render_prompt(
                    "trader/trader_system",
                    trader_context=trader_context,
                    past_memory_str=past_memory_str,
                    decision_memory_str=decision_memory_str,
                ),
            },
            context,
        ]

        # Capture the COMPLETE prompt that gets sent to the LLM
        try:
            # Combine system and user messages into complete prompt
            system_message = messages[0]["content"]
            user_message = context["content"]
            complete_prompt = f"""SYSTEM MESSAGE:
{system_message}

USER MESSAGE:
{user_message}"""
            
            capture_agent_prompt("trader_investment_plan", complete_prompt, company_name)
        except Exception as e:
            print(f"[TRADER] Warning: Could not capture complete prompt: {e}")
            # Fallback to system message only
            capture_agent_prompt("trader_investment_plan", messages[0]["content"], company_name)

        analysis_content = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        # Enhanced validation and final proposal handling
        # Check if we have substantial analysis content
        if len(analysis_content.strip()) < 200 or ("FINAL TRANSACTION PROPOSAL:" in analysis_content and len(analysis_content.replace("FINAL TRANSACTION PROPOSAL:", "").strip()) < 150):
            # Generate fallback comprehensive analysis
            fallback_prompt = render_prompt(
                "trader/trader_fallback_plan",
                company_name=company_name,
            )
            
            analysis_content = invoke_structured_or_freetext(
                structured_llm,
                llm,
                fallback_prompt,
                render_trader_proposal,
                "Trader",
            )
        
        # Ensure we have a final recommendation
        if "FINAL TRANSACTION PROPOSAL:" not in analysis_content:
            # Create final recommendation based on analysis
            final_prompt = render_prompt(
                "trader/trader_final_decision",
                company_name=company_name,
                analysis_content=analysis_content,
                final_format=final_format,
            )
            
            final_content = invoke_structured_or_freetext(
                structured_llm,
                llm,
                final_prompt,
                render_trader_proposal,
                "Trader",
            )
            
            # Properly combine analysis with final proposal
            combined_content = analysis_content + "\n\n---\n\n## Final Trading Decision\n\n" + final_content
            analysis_content = combined_content

        result = AIMessage(content=analysis_content)

        # Extract the recommendation from the response
        trading_mode = trading_context["mode"]
        extracted_recommendation = extract_recommendation(result.content, trading_mode)
        if not extracted_recommendation:
            extracted_recommendation = "NEUTRAL" if trading_mode == "trading" else "HOLD"
        
        final_decision_content = ensure_final_transaction_proposal(
            result.content, extracted_recommendation, trading_mode
        )
        result = AIMessage(content=final_decision_content)

        return {
            "messages": [result],
            "trader_investment_plan": final_decision_content,
            "sender": name,
            "trading_mode": trading_mode,
            "current_position": current_position,
            "recommended_action": extracted_recommendation,
        }

    return functools.partial(trader_node, name="Trader")
