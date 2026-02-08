from langchain_core.messages import AIMessage
import time
import json
from ..utils.agent_trading_modes import get_trading_mode_context, get_agent_specific_context
from ..utils.report_context import get_agent_context_bundle

# Import prompt capture utility
try:
    from webui.utils.prompt_capture import capture_agent_prompt
except ImportError:
    # Fallback for when webui is not available
    def capture_agent_prompt(report_type, prompt_content, symbol=None):
        pass


def create_safe_debator(llm, config=None):
    def safe_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        safe_history = risk_debate_state.get("safe_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        trader_decision = state["trader_investment_plan"]
        
        # Get trading mode from config
        current_position = state.get("current_position", "NEUTRAL")
        
        # Get centralized trading mode context
        trading_context = get_trading_mode_context(config, current_position)
        agent_context = get_agent_specific_context("risk_mgmt", trading_context)
        
        # Get mode-specific terms for the prompt
        actions = trading_context["actions"]
        mode_name = trading_context["mode_name"]
        decision_format = trading_context["decision_format"]
        context_bundle = get_agent_context_bundle(
            state,
            agent_role="safe_debator",
            objective=(
                f"Build conservative risk argument for {state.get('company_of_interest', '')} "
                f"from trader plan: {trader_decision}"
            ),
            config=config,
        )
        analysis_context = context_bundle["analysis_context"]

        # Use centralized trading mode context with conservative risk bias
        risk_specific_context = f"""
{agent_context}

**CONSERVATIVE SWING TRADING APPROACH:**
As the Conservative Risk Analyst for swing trading, you prioritize capital preservation while capturing controlled multi-day moves:

**CONSERVATIVE SWING PRINCIPLES:**
- **Position Sizing:** Never risk more than 1.5% per swing trade (vs. aggressive 3%)
- **Entry Timing:** Wait for clear multi-timeframe confirmation (1h/4h/1d alignment) before entering
- **Stop Losses:** Tight stops at 1.5x ATR below entry, placed at key swing levels
- **Target Profits:** Take partial profits (50%) at first swing target; trail the remainder
- **Market Selection:** Favor liquid, established stocks over volatile small-caps for swing holds
- **Risk/Reward:** Minimum 2.5:1 R/R ratio required, preferably 3:1

**CONSERVATIVE RISK ASSESSMENT:**
1. **Event Risk:** Minimize exposure to earnings/major news during the 2-10 day holding period
2. **Gap Risk:** Avoid swing positions in stocks prone to large gaps without clear catalysts
3. **Liquidity Risk:** Only swing trade stocks with >1M average daily volume
4. **Market Environment:** Reduce swing exposure during high VIX periods (>25)
5. **Position Limits:** Maximum 15% of portfolio in swing positions total
6. **Time Limits:** Exit swing trades if thesis is invalidated or max holding period (10 days) is reached

**CONSERVATIVE SWING SIGNALS:**
- Require multi-timeframe confluence (1h/4h/1d) before entry
- Prefer buying at pullbacks to support rather than chasing breakouts
- Exit immediately if stop loss is hit, no second chances
- Take 50% profits at first swing target to lock in gains
- Avoid initiating new swings during major earnings weeks or FOMC events

Focus on preserving capital first, generating returns second. Challenge aggressive proposals that exceed conservative risk limits."""

        prompt = f"""As the Safe/Conservative Risk Analyst, your primary objective is to protect assets, minimize volatility, and ensure steady, reliable growth. You prioritize stability, security, and risk mitigation, carefully assessing potential losses, economic downturns, and market volatility. {risk_specific_context}

When evaluating the trader's decision or plan, critically examine high-risk elements, pointing out where the decision may expose the firm to undue risk and where more cautious alternatives could secure long-term gains.

Here is the trader's decision:
{trader_decision}

Your task is to actively counter the arguments of the Risky and Neutral Analysts, advocating for conservative {actions} and highlighting where their views may overlook potential threats or fail to prioritize sustainability. Respond directly to their points, drawing from the following data sources to build a convincing case for a low-risk approach adjustment to the trader's decision:

Cross-analyst context packet: {analysis_context}

Here is the current conversation history: {history} 
Here is the last response from the risky analyst: {current_risky_response} 
Here is the last response from the neutral analyst: {current_neutral_response}. 

If there are no responses from the other viewpoints, do not hallucinate and just present your point.

Engage by questioning their optimism and emphasizing the potential downsides they may have overlooked. Address each of their counterpoints to showcase why a conservative stance is ultimately the safest path for the firm's assets. Focus on debating and critiquing their arguments to demonstrate the strength of a low-risk strategy over their approaches. 

Always conclude with your recommendation using the format: {decision_format}

Output conversationally as if you are speaking without any special formatting."""

        # Capture the COMPLETE prompt that gets sent to the LLM
        ticker = state.get("company_of_interest", "")
        capture_agent_prompt("conservative_report", prompt, ticker)

        response = llm.invoke(prompt)

        argument = f"Safe Analyst: {response.content}"

        # Store safe messages as a list for proper conversation display
        safe_messages = risk_debate_state.get("safe_messages", [])
        safe_messages.append(argument)
        
        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "risky_messages": risk_debate_state.get("risky_messages", []),
            "safe_history": safe_history + "\n" + argument,
            "safe_messages": safe_messages,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "neutral_messages": risk_debate_state.get("neutral_messages", []),
            "latest_speaker": "Safe",
            "current_risky_response": risk_debate_state.get(
                "current_risky_response", ""
            ),
            "current_safe_response": argument,
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return safe_node
