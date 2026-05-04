from langchain_core.messages import AIMessage
import time
import json
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


def create_bull_researcher(llm, memory):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = investment_debate_state.get("current_response", "")
        context_bundle = get_agent_context_bundle(
            state,
            agent_role="researchers/bull_researcher",
            objective=(
                f"Build a bullish thesis for {state.get('company_of_interest', '')}. "
                f"Counter the latest bear argument: {current_response}"
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

        prompt = render_prompt(
            "researchers/bull_researcher",
            claim_matrix=claim_matrix,
            all_reports_text=all_reports_text,
            debate_digest=debate_digest,
            history=history,
            current_response=current_response,
            past_memory_str=past_memory_str,
        )

        # Capture the COMPLETE prompt that gets sent to the LLM (including all dynamic content)
        ticker = state.get("company_of_interest", "")
        capture_agent_prompt("bull_report", prompt, ticker)

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        # Store bull messages as a list for proper conversation display
        bull_messages = investment_debate_state.get("bull_messages", [])
        bull_messages.append(argument)
        
        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bull_messages": bull_messages,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bear_messages": investment_debate_state.get("bear_messages", []),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
