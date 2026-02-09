import time
import json
from ..utils.agent_trading_modes import (
    get_trading_mode_context,
    get_agent_specific_context,
    extract_recommendation,
    format_final_decision,
)
from ..utils.report_context import (
    get_agent_context_bundle,
    build_debate_digest,
    truncate_for_prompt,
)
from tradingagents.dataflows.alpaca_utils import AlpacaUtils

# Import prompt capture utility
try:
    from webui.utils.prompt_capture import capture_agent_prompt
except ImportError:
    # Fallback for when webui is not available
    def capture_agent_prompt(report_type, prompt_content, symbol=None):
        pass


def create_risk_manager(llm, memory, config=None):
    def risk_manager_node(state) -> dict:

        company_name = state["company_of_interest"]

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        trader_plan = state["investment_plan"]

        # Get trading mode from config
        allow_shorts = config.get("allow_shorts", False) if config else False

        # Determine live position from Alpaca
        current_position = AlpacaUtils.get_current_position_state(company_name)
        state["current_position"] = current_position

        # ---------------------------------------------------------
        # NEW: Fetch richer live account & position metrics from Alpaca
        # ---------------------------------------------------------
        positions_data = AlpacaUtils.get_positions_data()
        account_info = AlpacaUtils.get_account_info()

        # Build summary for specific symbol
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

        open_pos_desc = (
            f"We currently have an open {current_position} position in {company_name}."
            if current_position != "NEUTRAL"
            else f"We do not have any open position in {company_name}."
        )
        
        # Get centralized trading mode context
        trading_context = get_trading_mode_context(config, current_position)
        agent_context = get_agent_specific_context("manager", trading_context)
        agent_context = truncate_for_prompt(agent_context, 1600)
        
        # Get mode-specific terms for the prompt
        actions = trading_context["actions"]
        mode_name = trading_context["mode_name"]
        decision_format = trading_context["decision_format"]
        final_format = trading_context["final_format"]
        context_bundle = get_agent_context_bundle(
            state,
            agent_role="risk_manager",
            objective=(
                f"Judge risk debate and finalize risk-adjusted trade decision for {company_name}. "
                f"Trader plan: {truncate_for_prompt(trader_plan, 700)}"
            ),
            config=config,
        )
        claim_matrix = context_bundle.get("decision_claim_matrix", "")
        risk_debate_digest = build_debate_digest(risk_debate_state, "risk", config=config)
        all_reports_text = context_bundle.get("all_reports_text", "")

        curr_situation = context_bundle["memory_context"]
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""{agent_context}

You are the final swing-trading risk judge. Make a decisive {decision_format} call with strict downside controls.

Inputs:
- Current position status: {open_pos_desc}
- Position stats: {position_stats_desc}
- Account stats: {account_status_desc}
- Trader plan: {trader_plan}
- Decision claim matrix: {claim_matrix}
- Full untruncated analyst reports: {all_reports_text}
- Risk debate digest: {risk_debate_digest}
- Full risk debate history: {history}
- Past lessons: {truncate_for_prompt(past_memory_str, 1200)}

Decision constraints:
1. Reject proposals implying >3% account risk or unclear exits.
2. Require explicit invalidation/stop logic.
3. Prioritize capital preservation under elevated volatility/event risk.

Output format (concise):
- Recommendation: {actions} (with confidence high/medium/low)
- 4-6 concise bullets explaining risk rationale and required risk controls
- End exactly with: {final_format}

Keep response under 260 words."""

        # Capture the COMPLETE prompt that gets sent to the LLM
        capture_agent_prompt("final_trade_decision", prompt, company_name)

        response = llm.invoke(prompt)

        # Extract the recommendation from the response
        trading_mode = trading_context["mode"]
        extracted_recommendation = extract_recommendation(response.content, trading_mode)
        
        # Format the final decision if extraction was successful
        final_decision_content = response.content
        if extracted_recommendation:
            final_decision_content = format_final_decision(extracted_recommendation, trading_mode)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "risky_history": risk_debate_state["risky_history"],
            "safe_history": risk_debate_state["safe_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "risky_messages": risk_debate_state.get("risky_messages", []),
            "safe_messages": risk_debate_state.get("safe_messages", []),
            "neutral_messages": risk_debate_state.get("neutral_messages", []),
            "latest_speaker": "Judge",
            "current_risky_response": risk_debate_state["current_risky_response"],
            "current_safe_response": risk_debate_state["current_safe_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_decision_content,
            "trading_mode": trading_mode,
            "current_position": current_position,
            "recommended_action": extracted_recommendation,
        }

    return risk_manager_node
