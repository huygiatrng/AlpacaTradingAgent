import time
import json
from ..utils.agent_trading_modes import get_trading_mode_context, get_agent_specific_context
from ..utils.report_context import (
    get_agent_context_bundle,
    build_debate_digest,
    truncate_for_prompt,
)

# Import prompt capture utility
try:
    from webui.utils.prompt_capture import capture_agent_prompt
except ImportError:
    # Fallback for when webui is not available
    def capture_agent_prompt(report_type, prompt_content, symbol=None):
        pass


def create_neutral_debator(llm, config=None):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_safe_response = risk_debate_state.get("current_safe_response", "")

        trader_decision = state["trader_investment_plan"]
        
        # Get trading mode from config
        current_position = state.get("current_position", "NEUTRAL")
        
        # Get centralized trading mode context
        trading_context = get_trading_mode_context(config, current_position)
        agent_context = get_agent_specific_context("risk_mgmt", trading_context)
        agent_context = truncate_for_prompt(agent_context, 1400)
        
        # Get mode-specific terms for the prompt
        actions = trading_context["actions"]
        mode_name = trading_context["mode_name"]
        decision_format = trading_context["decision_format"]
        context_bundle = get_agent_context_bundle(
            state,
            agent_role="neutral_debator",
            objective=(
                f"Build balanced risk argument for {state.get('company_of_interest', '')} "
                f"from trader plan: {truncate_for_prompt(trader_decision, 700)}"
            ),
            config=config,
        )
        claim_matrix = context_bundle.get("decision_claim_matrix", "")
        debate_digest = build_debate_digest(risk_debate_state, "risk", config=config)
        all_reports_text = context_bundle.get("all_reports_text", "")

        # Use centralized trading mode context with balanced risk bias
        risk_specific_context = f"""
{agent_context}

BALANCED RISK APPROACH:
- Balance growth opportunities with risk management
- Target {actions} that offer reasonable risk-adjusted returns
- Focus on strategic positioning that adapts to market conditions
- Advocate for measured approaches that avoid both excessive risk and excessive caution
"""

        prompt = f"""As the Neutral Risk Analyst, your role is to provide a balanced perspective, weighing both the potential benefits and risks of the trader's decision or plan. You prioritize a well-rounded approach, evaluating the upsides and downsides while factoring in broader market trends, potential economic shifts, and diversification strategies. {risk_specific_context}

Here is the trader's decision:
{trader_decision}

Your task is to challenge both the Risky and Safe Analysts, pointing out where each perspective may be overly optimistic or overly cautious. Use insights from the following data sources to support a moderate, sustainable strategy for {actions} to adjust the trader's decision:

Decision claim matrix: {claim_matrix}
Full untruncated analyst reports: {all_reports_text}
Risk debate digest: {debate_digest}
Full conversation history: {history}

Last risky response: {current_risky_response} 
Last safe response: {current_safe_response}. 

If there are no responses from the other viewpoints, do not hallucinate and just present your point.

Engage actively by analyzing both sides critically, addressing weaknesses in the risky and conservative arguments to advocate for a more balanced approach. Challenge each of their points to illustrate why a balanced view can lead to the most reliable outcomes. Focus on debating rather than simply presenting data, aiming to show that a balanced view can lead to the most reliable outcomes. 

Always conclude with your recommendation using the format: {decision_format}

Output conversationally as if you are speaking without any special formatting.
Keep your response concise (max 300 words)."""

        # Capture the COMPLETE prompt that gets sent to the LLM
        ticker = state.get("company_of_interest", "")
        capture_agent_prompt("neutral_report", prompt, ticker)

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        # Store neutral messages as a list for proper conversation display
        neutral_messages = risk_debate_state.get("neutral_messages", [])
        neutral_messages.append(argument)
        
        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "risky_messages": risk_debate_state.get("risky_messages", []),
            "safe_history": risk_debate_state.get("safe_history", ""),
            "safe_messages": risk_debate_state.get("safe_messages", []),
            "neutral_history": neutral_history + "\n" + argument,
            "neutral_messages": neutral_messages,
            "latest_speaker": "Neutral",
            "current_risky_response": risk_debate_state.get(
                "current_risky_response", ""
            ),
            "current_safe_response": risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
