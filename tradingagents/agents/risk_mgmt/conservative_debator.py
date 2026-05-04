from langchain_core.messages import AIMessage
import time
import json
from ..utils.agent_trading_modes import get_trading_mode_context, get_agent_specific_context
from ..utils.report_context import (
    get_agent_context_bundle,
    build_debate_digest,
)
from tradingagents.prompts import render_prompt

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
        claim_matrix = context_bundle.get("decision_claim_matrix", "")
        debate_digest = build_debate_digest(risk_debate_state, "risk", config=config)
        all_reports_text = context_bundle.get("all_reports_text", "")

        risk_specific_context = render_prompt(
            "risk/conservative_context",
            agent_context=agent_context,
        )

        prompt = render_prompt(
            "risk/conservative_debator",
            risk_specific_context=risk_specific_context,
            trader_decision=trader_decision,
            actions=actions,
            claim_matrix=claim_matrix,
            all_reports_text=all_reports_text,
            debate_digest=debate_digest,
            history=history,
            current_risky_response=current_risky_response,
            current_neutral_response=current_neutral_response,
            decision_format=decision_format,
        )

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
