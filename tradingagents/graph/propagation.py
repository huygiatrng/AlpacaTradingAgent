# TradingAgents/graph/propagation.py

from typing import Dict, Any
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)


class Propagator:
    """Handles state initialization and propagation through the graph."""

    def __init__(self, max_recur_limit=200):
        """Initialize with configuration parameters."""
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self, company_name: str, trade_date: str
    ) -> Dict[str, Any]:
        """Create the initial state for the agent graph."""
        # Preserve the original ticker format for cryptocurrencies and other symbols
        ticker_symbol = company_name
        
        return {
            "messages": [("human", ticker_symbol)],
            "company_of_interest": ticker_symbol,
            "trade_date": str(trade_date),
            "investment_debate_state": InvestDebateState(
                {
                    "history": "", 
                    "current_response": "", 
                    "count": 0,
                    "bull_history": "",
                    "bear_history": "",
                    "bull_messages": [],
                    "bear_messages": [],
                    "judge_decision": ""
                }
            ),
            "risk_debate_state": RiskDebateState(
                {
                    "history": "",
                    "current_risky_response": "",
                    "current_safe_response": "",
                    "current_neutral_response": "",
                    "latest_speaker": "Risky",  # Initialize latest speaker
                    "count": 0,
                    "risky_history": "",
                    "safe_history": "",
                    "neutral_history": "",
                    "risky_messages": [],
                    "safe_messages": [],
                    "neutral_messages": [],
                    "judge_decision": ""
                }
            ),
            "market_report": "",
            "fundamentals_report": "",
            "sentiment_report": "",
            "news_report": "",
            "macro_report": "",
            "report_context": {},
        }

    def get_graph_args(self) -> Dict[str, Any]:
        """Get arguments for the graph invocation."""
        return {
            "stream_mode": "values",
            "config": {"recursion_limit": self.max_recur_limit},
        }
