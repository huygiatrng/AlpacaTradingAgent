import time
import json
from ..utils.report_context import get_agent_context_bundle

# Import prompt capture utility
try:
    from webui.utils.prompt_capture import capture_agent_prompt
except ImportError:
    # Fallback for when webui is not available
    def capture_agent_prompt(report_type, prompt_content, symbol=None):
        pass


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        history = state["investment_debate_state"].get("history", "")
        investment_debate_state = state["investment_debate_state"]

        context_bundle = get_agent_context_bundle(
            state,
            agent_role="research_manager",
            objective=(
                f"Adjudicate bull/bear debate for {state.get('company_of_interest', '')} "
                "and produce a decisive investment plan."
            ),
        )
        analysis_context = context_bundle["analysis_context"]

        curr_situation = context_bundle["memory_context"]
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""As the portfolio manager and debate facilitator, your role is to critically evaluate this round of debate and make a definitive decision: align with the bear analyst, the bull analyst, or choose Hold only if it is strongly justified based on the arguments presented.

Summarize the key points from both sides concisely, focusing on the most compelling evidence or reasoning. Your recommendation—Buy, Sell, or Hold—must be clear and actionable. Avoid defaulting to Hold simply because both sides have valid points; commit to a stance grounded in the debate's strongest arguments.

Additionally, develop a detailed investment plan for the trader. This should include:

Your Recommendation: A decisive stance supported by the most convincing arguments.
Rationale: An explanation of why these arguments lead to your conclusion.
Strategic Actions: Concrete steps for implementing the recommendation.
Take into account your past mistakes on similar situations. Use these insights to refine your decision-making and ensure you are learning and improving. Present your analysis conversationally, as if speaking naturally, without special formatting. 

Here are your past reflections on mistakes:
\"{past_memory_str}\"

Cross-analyst context packet:
{analysis_context}

Here is the debate:
Debate History:
{history}"""

        # Capture the COMPLETE prompt that gets sent to the LLM
        ticker = state.get("company_of_interest", "")
        capture_agent_prompt("research_manager_report", prompt, ticker)

        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "bull_messages": investment_debate_state.get("bull_messages", []),
            "bear_messages": investment_debate_state.get("bear_messages", []),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node
