# TradingAgents/graph/signal_processing.py

from langchain_openai import ChatOpenAI
from tradingagents.prompts import load_prompt


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """Initialize with an LLM for processing."""
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """
        Process a full trading signal to extract the core decision.

        Args:
            full_signal: Complete trading signal text

        Returns:
            Extracted decision (BUY, SELL, or HOLD)
        """
        # First try deterministic extraction to avoid unnecessary LLM calls
        content = full_signal.upper()

        # Check for trading-mode keywords first (LONG / SHORT / NEUTRAL)
        for action in ["LONG", "SHORT", "NEUTRAL"]:
            pattern = f"FINAL TRANSACTION PROPOSAL: **{action}**"
            if pattern in content:
                return action

        # Check for investment-mode keywords (BUY / SELL / HOLD)
        for action in ["BUY", "SELL", "HOLD"]:
            pattern = f"FINAL TRANSACTION PROPOSAL: **{action}**"
            if pattern in content:
                return action

        # Fallback: simple keyword search in the last 100 characters
        tail = content[-100:]
        for action in ["LONG", "SHORT", "NEUTRAL", "BUY", "SELL", "HOLD"]:
            if action in tail:
                return action

        # If deterministic parsing fails, let the LLM infer (default to BUY/SELL/HOLD)
        messages = [
            (
                "system",
                load_prompt("graph/signal_extraction_system"),
            ),
            ("human", full_signal),
        ]

        return self.quick_thinking_llm.invoke(messages).content.strip().upper()
