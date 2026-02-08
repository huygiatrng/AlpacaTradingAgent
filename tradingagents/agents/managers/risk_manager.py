import time
import json
from ..utils.agent_trading_modes import (
    get_trading_mode_context,
    get_agent_specific_context,
    extract_recommendation,
    format_final_decision,
)
from ..utils.report_context import get_agent_context_bundle
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
                f"Trader plan: {trader_plan}"
            ),
            config=config,
        )
        analysis_context = context_bundle["analysis_context"]

        curr_situation = context_bundle["memory_context"]
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        # Use centralized trading mode context
        manager_context = f"""
{agent_context}

**SWING TRADING RISK MANAGEMENT:**
As the Swing Trading Risk Manager, you specialize in managing risks for multi-day position holds (2-10 days). Your focus areas:

**SWING TRADING RISK FACTORS:**
1. **Multi-Day Event Risk:** Positions exposed to earnings, FOMC, or major news during the holding period
2. **Position Sizing:** Never risk more than 1-3% of capital per swing trade
3. **Stop Loss Management:** Use ATR-based stops (1.5-2x ATR) at key swing levels
4. **Correlation Risk:** Avoid multiple correlated swing positions simultaneously
5. **Market Environment:** Adjust exposure based on overall market volatility (VIX)
6. **Trend Invalidation:** Monitor for change of character (CHOCH) or trend reversal signals

**RISK ASSESSMENT FRAMEWORK:**
- **Entry Risk:** Distance to ATR-based stop vs. account size (max 3% risk)
- **Holding Risk:** News/earnings events during the 2-10 day swing period
- **Exit Risk:** Liquidity concerns, gap risk on adverse catalysts
- **Portfolio Risk:** Total swing exposure across all positions (<20% of capital)

**POSITION SIZING CALCULATION:**
Position Size = (Risk Amount / Stop Distance) × Share Price
- Risk Amount: 1-3% of total capital
- Stop Distance: Entry price - ATR-based stop loss price
- Maximum position: Never exceed 10% of portfolio in a single swing trade

Current Alpaca Position Status:
{open_pos_desc}

{position_stats_desc}

Alpaca Account Status:
{account_status_desc}

Cross-Analyst Context Packet:
{analysis_context}

**RISK DECISION MATRIX:**
Consider the arguments from all three risk perspectives:
- **Aggressive:** High-reward swing setups, wider stops, larger positions
- **Conservative:** Tight stops, smaller positions, avoid volatile swing setups  
- **Neutral:** Balanced approach, standard position sizing, moderate targets

Your final {decision_format} decision should address:
1. **Position Size:** Exact dollar amount or share quantity based on ATR stop distance
2. **Risk/Reward Ratio:** Minimum 2:1, preferably 3:1 for swing trades
3. **Time Horizon:** Confirm 2-10 day swing hold with daily monitoring
4. **Risk Controls:** ATR-based stops, position limits, correlation checks
5. **Market Conditions:** Factor in VIX, multi-timeframe trend strength, volume patterns

Use the format: {final_format}

**CRITICAL:** Reject any proposal with >3% account risk or unclear exit strategy."""

        prompt = f"""{manager_context}

Strive for clarity and decisiveness.

Guidelines for Decision-Making:
1. **Summarize Key Arguments**: Extract the strongest points from each analyst, focusing on relevance to the context.
2. **Provide Rationale**: Support your recommendation with direct quotes and counterarguments from the debate.
3. **Refine the Trader's Plan**: Start with the trader's original plan, **{trader_plan}**, and adjust it based on the analysts' insights.
4. **Learn from Past Mistakes**: Use lessons from **{past_memory_str}** to address prior misjudgments and improve the decision you are making now to make sure you don't make a wrong recommendation that loses money.

Deliverables:
- A clear and actionable recommendation: {actions}.
- Detailed reasoning anchored in the debate and past reflections.
- Always conclude your response with '{final_format}' to confirm your recommendation.

---

**Analysts Debate History:**  
{history}

---

Focus on actionable insights and continuous improvement. Build on past lessons, critically evaluate all perspectives, and ensure each decision advances better outcomes."""

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
