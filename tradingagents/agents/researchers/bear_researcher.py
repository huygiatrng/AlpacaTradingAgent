from langchain_core.messages import AIMessage
import time
import json
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


def create_bear_researcher(llm, memory):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        context_bundle = get_agent_context_bundle(
            state,
            agent_role="bear_researcher",
            objective=(
                f"Build a bearish thesis for {state.get('company_of_interest', '')}. "
                f"Counter the latest bull argument: {current_response}"
            ),
        )
        claim_matrix = context_bundle.get("decision_claim_matrix", "")
        debate_digest = build_debate_digest(investment_debate_state, "investment")
        all_reports_text = context_bundle.get("all_reports_text", "")
        curr_situation = context_bundle["memory_context"]
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""You are a Bear Analyst making the case against investing in the stock. Your goal is to present a well-reasoned argument emphasizing risks, challenges, and negative indicators. Leverage the provided research and data to highlight potential downsides and counter bullish arguments effectively.

Key points to focus on:

- Risks and Challenges: Highlight factors like market saturation, financial instability, or macroeconomic threats that could hinder the stock's performance.
- Competitive Weaknesses: Emphasize vulnerabilities such as weaker market positioning, declining innovation, or threats from competitors.
- Negative Indicators: Use evidence from financial data, market trends, or recent adverse news to support your position.
- Bull Counterpoints: Critically analyze the bull argument with specific data and sound reasoning, exposing weaknesses or over-optimistic assumptions.
- Engagement: Present your argument in a conversational style, directly engaging with the bull analyst's points and debating effectively rather than simply listing facts.

Resources available:

Decision claim matrix: {claim_matrix}
Full untruncated analyst reports: {all_reports_text}
Debate digest: {debate_digest}
Conversation history of the debate: {history}
Last bull argument: {current_response}
Reflections from similar situations and lessons learned: {past_memory_str}
Use this information to deliver a compelling bear argument, refute the bull's claims, and engage in a dynamic debate that demonstrates the risks and weaknesses of investing in the stock. You must also address reflections and learn from lessons and mistakes you made in the past.
Keep your response concise (max 320 words).
"""

        # Capture the COMPLETE prompt that gets sent to the LLM (including all dynamic content)
        ticker = state.get("company_of_interest", "")
        capture_agent_prompt("bear_report", prompt, ticker)

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        # Store bear messages as a list for proper conversation display
        bear_messages = investment_debate_state.get("bear_messages", [])
        bear_messages.append(argument)
        
        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bear_messages": bear_messages,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "bull_messages": investment_debate_state.get("bull_messages", []),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
