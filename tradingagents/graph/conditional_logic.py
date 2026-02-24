# TradingAgents/graph/conditional_logic.py

from tradingagents.agents.utils.agent_states import AgentState


class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""

    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        """Initialize with configuration parameters."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    def should_continue_market(self, state: AgentState):
        """Determine if market analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_market"
        return "Msg Clear Market"

    def should_continue_social(self, state: AgentState):
        """Determine if social media analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_social"
        return "Msg Clear Social"

    def should_continue_news(self, state: AgentState):
        """Determine if news analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_news"
        return "Msg Clear News"

    def should_continue_fundamentals(self, state: AgentState):
        """Determine if fundamentals analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_fundamentals"
        return "Msg Clear Fundamentals"

    def should_continue_macro(self, state: AgentState):
        """Determine if macro analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_macro"
        return "Msg Clear Macro"

    def should_continue_debate(self, state: AgentState) -> str:
        """Determine if debate should continue."""

        if (
            state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds
        ):  # 3 rounds of back-and-forth between 2 agents
            return "Research Manager"
        if state["investment_debate_state"]["current_response"].startswith("Bull"):
            return "Bear Researcher"
        return "Bull Researcher"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """Determine if risk analysis should continue."""
        count = state["risk_debate_state"]["count"]
        max_count = 3 * self.max_risk_discuss_rounds
        latest_speaker = state["risk_debate_state"].get("latest_speaker", "Unknown")

        print(f"[RISK CONDITIONAL] count={count}, max={max_count}, latest_speaker={latest_speaker}")

        if count >= max_count:
            print(f"[RISK CONDITIONAL] ✅ Debate complete ({count} >= {max_count}), routing to Risk Judge")
            return "Risk Judge"

        # Check if latest_speaker exists in the state, if not initialize it
        if "latest_speaker" not in state["risk_debate_state"]:
            # Default to Risky Analyst as the first speaker
            state["risk_debate_state"]["latest_speaker"] = "Risky"
            latest_speaker = "Risky"

        if latest_speaker.startswith("Risky"):
            print(f"[RISK CONDITIONAL] Risky completed, routing to Safe Analyst")
            return "Safe Analyst"
        if latest_speaker.startswith("Safe"):
            print(f"[RISK CONDITIONAL] Safe completed, routing to Neutral Analyst")
            return "Neutral Analyst"
        print(f"[RISK CONDITIONAL] Neutral completed, routing to Risky Analyst (round continues)")
        return "Risky Analyst"
