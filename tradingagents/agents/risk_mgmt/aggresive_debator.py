import time
import json
from ..utils.agent_trading_modes import get_trading_mode_context, get_agent_specific_context
from ..utils.report_context import (
    get_agent_context_bundle,
    build_debate_digest,
)

# Import prompt capture utility
try:
    from webui.utils.prompt_capture import capture_agent_prompt
except ImportError:
    # Fallback for when webui is not available
    def capture_agent_prompt(report_type, prompt_content, symbol=None):
        pass


def create_risky_debator(llm, config=None):
    def risky_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        risky_history = risk_debate_state.get("risky_history", "")

        current_safe_response = risk_debate_state.get("current_safe_response", "")
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
            agent_role="risky_debator",
            objective=(
                f"Build aggressive risk argument for {state.get('company_of_interest', '')} "
                f"from trader plan: {trader_decision}"
            ),
            config=config,
        )
        claim_matrix = context_bundle.get("decision_claim_matrix", "")
        debate_digest = build_debate_digest(risk_debate_state, "risk", config=config)
        all_reports_text = context_bundle.get("all_reports_text", "")

        # Use centralized trading mode context with aggressive risk bias
        risk_specific_context = f"""
{agent_context}

AGGRESSIVE RISK APPROACH:
- Favor high-reward opportunities with calculated risks
- Target {actions} that maximize profit potential
- Focus on growth over safety when conditions are favorable
- Take decisive action when market signals are strong
"""

        prompt = f"""As the Risky Risk Analyst, your role is to actively champion high-reward, high-risk opportunities, emphasizing bold strategies and competitive advantages.

{risk_specific_context}

When evaluating the trader's decision or plan, focus intently on the potential upside, growth potential, and innovative benefits—even when these come with elevated risk. Use the provided market data and sentiment analysis to strengthen your arguments and challenge the opposing views. 

Here is the trader's decision:
{trader_decision}

Your task is to create a compelling case for aggressive {actions} by questioning and critiquing the conservative and neutral stances to demonstrate why your high-reward perspective offers the best path forward. Specifically, respond directly to each point made by the conservative and neutral analysts, countering with data-driven rebuttals and persuasive reasoning. Highlight where their caution might miss critical opportunities or where their assumptions may be overly conservative.

Incorporate insights from the following sources into your arguments:

Decision claim matrix: {claim_matrix}
Full untruncated analyst reports: {all_reports_text}
Risk debate digest: {debate_digest}
Full conversation history: {history}

Last conservative argument: {current_safe_response} 
Last neutral argument: {current_neutral_response}. 

If there are no responses from the other viewpoints, do not hallucinate and just present your point.

Engage actively by addressing any specific concerns raised, refuting the weaknesses in their logic, and asserting the benefits of risk-taking to outpace market norms. Maintain a focus on debating and persuading, not just presenting data. Challenge each counterpoint to underscore why a high-risk approach is optimal. 

Always conclude with your recommendation using the format: {decision_format}

Output conversationally as if you are speaking without any special formatting.
Keep your response concise (max 300 words)."""

        # Capture the COMPLETE prompt that gets sent to the LLM
        ticker = state.get("company_of_interest", "")
        capture_agent_prompt("aggressive_report", prompt, ticker)

        response = llm.invoke(prompt)

        argument = f"Risky Analyst: {response.content}"

        # Store risky messages as a list for proper conversation display
        risky_messages = risk_debate_state.get("risky_messages", [])
        risky_messages.append(argument)
        
        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risky_history + "\n" + argument,
            "risky_messages": risky_messages,
            "safe_history": risk_debate_state.get("safe_history", ""),
            "safe_messages": risk_debate_state.get("safe_messages", []),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "neutral_messages": risk_debate_state.get("neutral_messages", []),
            "latest_speaker": "Risky",
            "current_risky_response": argument,
            "current_safe_response": risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return risky_node
