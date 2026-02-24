import time
import json
from ..utils.agent_trading_modes import (
    get_trading_mode_context,
    get_agent_specific_context,
    extract_recommendation,
    format_final_decision,
)
from tradingagents.dataflows.alpaca_utils import AlpacaUtils
from tradingagents.agents.utils.agent_utils import log_llm_start, log_llm_end

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
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["news_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]
        macro_report = state["macro_report"]

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
        equity = account_info.get("equity", 0.0)
        daily_change_dollars = account_info.get("daily_change_dollars", 0.0)
        daily_change_percent = account_info.get("daily_change_percent", 0.0)
        account_status_desc = (
            "Account Status:\n"
            f"- Account Equity: ${equity:,.2f}\n"
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

        curr_situation = f"{macro_report}\n\n{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        # Build price format guidance depending on whether shorts are allowed
        if allow_shorts:
            price_format_guidance = """**CRITICAL FORMATTING RULES FOR APPROVED PRICES:**
1. Use the exact labels: "Entry Price:", "Stop Loss:", "Target 1:", "Target 2:"
2. Include the dollar sign and exactly 2 decimal places
3. **FOR LONG positions:** Stop Loss BELOW Entry Price, Targets ABOVE Entry Price
4. **FOR SHORT positions:** Stop Loss ABOVE Entry Price (limits upside risk), Targets BELOW Entry Price (lock profits)

**Example Format (LONG position):**
Entry Price: $337.20
Stop Loss: $327.00  (below entry - limits downside)
Target 1: $352.00  (above entry - profit taking)
Target 2: $367.00  (above entry - extended profit)

**Example Format (SHORT position):**
Entry Price: $337.20
Stop Loss: $350.00  (above entry - limits upside risk)
Target 1: $320.00  (below entry - profit taking)
Target 2: $310.00  (below entry - extended profit)"""
        else:
            price_format_guidance = """**Example Format (LONG position):**
Entry Price: $337.20
Stop Loss: $327.00  (below entry - limits downside)
Target 1: $352.00  (above entry - profit taking)
Target 2: $367.00  (above entry - extended profit)"""

        # Use centralized trading mode context
        manager_context = f"""
{agent_context}

**EOD TRADING RISK MANAGEMENT:**
As the EOD Trading Risk Manager, you specialize in managing risks for overnight position holds. Your focus areas:

**EOD TRADING RISK FACTORS:**
1. **Overnight Gap Risk:** Positions exposed to gap risk from overnight news/events
2. **Position Sizing:** Never risk more than 1-3% of capital per EOD trade
3. **Stop Loss Management:** Use daily technical levels, not arbitrary percentages
4. **Correlation Risk:** Avoid multiple correlated overnight positions simultaneously
5. **Market Environment:** Adjust exposure based on overall market volatility (VIX)
6. **Time Decay:** Consider theta decay for any options positions held overnight

**RISK ASSESSMENT FRAMEWORK:**
- **Entry Risk:** Distance to stop loss vs. account size (max 3% risk)
- **Holding Risk:** News/earnings events during overnight holding period
- **Exit Risk:** Gap risk, liquidity concerns, pre-market volatility
- **Portfolio Risk:** Total overnight exposure across all positions (<15% of capital)

**POSITION SIZING CALCULATION:**
Position Size = (Risk Amount / Stop Distance) × Share Price
- Risk Amount: 1-3% of total capital
- Stop Distance: Entry price - daily stop loss price
- Maximum position: Never exceed 8% of portfolio in single overnight hold

Current Alpaca Position Status:
{open_pos_desc}

{position_stats_desc}

Alpaca Account Status:
{account_status_desc}

**RISK DECISION MATRIX:**
Consider the arguments from all three risk perspectives:
- **Aggressive:** High-reward EOD setups, wider stops, larger positions
- **Conservative:** Tight stops, smaller positions, avoid volatile overnight setups  
- **Neutral:** Balanced approach, standard position sizing, moderate targets

Your final {decision_format} decision should address:
1. **Position Size:** Exact dollar amount or share quantity based on daily stop distance
2. **Risk/Reward Ratio:** Minimum 2:1, preferably 3:1 for EOD trades
3. **Time Horizon:** Confirm overnight hold with daily reassessment
4. **Risk Controls:** Daily stop loss, position limits, correlation checks
5. **Market Conditions:** Factor in VIX, daily trend strength, volume patterns

**POSITION SIZE VALIDATION:**
Review the trader's recommended position size and validate it meets risk parameters:
- Account Equity: ${equity:,.2f}
- Buying Power: ${buying_power:,.2f}
- Trader's Recommendation: [extracted from trader plan above]

Validate that position size:
- Does not exceed 3% account risk
- Is appropriate for the volatility and stop loss distance
- Stays within available buying power
- Fits overall portfolio risk limits

**APPROVED TRADING PRICES OUTPUT REQUIREMENT:**
After validating the trader's recommendation, you MUST provide approved price levels in this EXACT format:

Entry Price: $XXX.XX
Stop Loss: $XXX.XX
Target 1: $XXX.XX
Target 2: $XXX.XX

**RISK MANAGER PRICE VALIDATION:**
1. Review the trader's suggested prices (if provided)
2. Validate they meet risk management criteria:
   - Risk per trade ≤ 3% of account equity
   - Stop loss not too tight (<0.5%) or too wide (>20%)
   - Risk/Reward ratio ≥ 2:1 (preferably 3:1)
   - Prices within ±20% of current market price
3. Adjust prices if needed to meet risk parameters
4. Output the APPROVED prices in the exact format above

{price_format_guidance}

**CRITICAL:** These prices will be used to place ACTUAL stop loss and take profit orders with Alpaca.
This section must appear BEFORE your final APPROVED POSITION SIZE conclusion.

You MUST conclude with:
APPROVED POSITION SIZE: $X,XXX
(If adjusted from trader's recommendation, explain why. If approved as-is, confirm the reasoning.)

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

        model_name = getattr(llm, 'model_name', 'unknown')
        start_time = log_llm_start("RISK_MANAGER", model_name)
        try:
            response = llm.invoke(prompt)
            log_llm_end("RISK_MANAGER", model_name, start_time, response)
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[LLM - RISK_MANAGER] ❌ {model_name} failed after {elapsed:.2f}s: {str(e)}")
            raise

        # Extract the recommendation from the response
        trading_mode = trading_context["mode"]
        extracted_recommendation = extract_recommendation(response.content, trading_mode)

        # Extract approved position size from risk manager's analysis
        from tradingagents.agents.utils.position_size_extractor import extract_position_size
        from tradingagents.agents.utils.price_extractor import (
            extract_trading_prices,
            validate_trading_prices
        )

        approved_position_size = extract_position_size(
            response.content,
            account_info={"equity": equity, "buying_power": buying_power, "cash": cash}
        )

        # Get current price for validation
        current_price = None
        try:
            quote = AlpacaUtils.get_latest_quote(company_name)
            current_price = quote.get("ask_price") or quote.get("bid_price")
            print(f"[RISK MANAGER] Current market price for {company_name}: ${current_price:.2f}")
        except Exception as e:
            print(f"[RISK MANAGER] ⚠️ Could not get current price for {company_name}: {e}")

        # Extract prices from risk manager's analysis
        print(f"[RISK MANAGER] Extracting trading prices from risk manager analysis for {company_name}...")
        risk_manager_prices = extract_trading_prices(
            response.content,
            current_price=current_price
        )

        # Get trader's recommended prices from state
        trader_prices = state.get("recommended_trading_prices", {})
        print(f"[RISK MANAGER] Trader prices from state: {trader_prices}")

        # Priority: Use risk manager's prices if present, else trader's
        if not risk_manager_prices.get("fallback_used"):
            final_prices = risk_manager_prices
            print(f"[RISK MANAGER] ✅ Using risk manager's prices")
        else:
            final_prices = trader_prices
            print(f"[RISK MANAGER] ⚠️ Risk manager didn't specify prices, using trader's prices")

        # Validate the prices
        validated_prices = None
        if final_prices and not final_prices.get("fallback_used"):
            print(f"[RISK MANAGER] Validating prices for {company_name}...")
            # Determine position type from multiple sources to handle extraction failures:
            # 1. Risk manager's own recommendation (primary)
            # 2. Trader's recommendation already stored in state (fallback)
            if extracted_recommendation and extracted_recommendation.upper() == "SHORT":
                position_type = "short"
            else:
                trader_action = state.get("recommended_action") or ""
                position_type = "short" if trader_action.upper() == "SHORT" else "long"
                if not extracted_recommendation:
                    print(f"[RISK MANAGER] ⚠️ Extraction failed (returned None), using trader action '{trader_action}' for position_type")
            print(f"[RISK MANAGER] Position type for validation: {position_type.upper()} (risk_mgr={extracted_recommendation}, trader={state.get('recommended_action')})")
            validated_prices = validate_trading_prices(
                entry=final_prices.get("entry_price"),
                stop=final_prices.get("stop_loss"),
                targets=final_prices.get("targets", []),
                current_price=current_price,
                symbol=company_name,
                position_type=position_type
            )

            if validated_prices:
                print(f"[RISK MANAGER] ✅ Prices validated successfully")
            else:
                print(f"[RISK MANAGER] ❌ Price validation failed")
        else:
            print(f"[RISK MANAGER] ❌ No prices to validate (both trader and risk manager extraction failed)")

        # Format the final decision while preserving full risk analysis
        if extracted_recommendation:
            # Pass full risk analysis to preserve it
            final_decision_content = format_final_decision(
                extracted_recommendation,
                trading_mode,
                full_analysis=response.content  # Preserve the full risk manager analysis
            )
        else:
            # No recommendation extracted, keep original content
            final_decision_content = response.content

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
            "approved_position_size": approved_position_size,
            "approved_trading_prices": validated_prices,  # NEW
        }

    return risk_manager_node
