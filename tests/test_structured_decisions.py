import unittest

from tradingagents.agents.schemas import (
    AdvisoryRating,
    ExecutableAction,
    ResearchPlan,
    RiskDecision,
    TraderProposal,
    render_research_plan,
    render_risk_decision,
    render_trader_proposal,
)
from tradingagents.agents.utils.agent_trading_modes import extract_recommendation
from tradingagents.agents.utils.structured import invoke_structured_or_freetext


class Message:
    def __init__(self, content):
        self.content = content


class PlainLLM:
    def invoke(self, _prompt):
        return Message("plain fallback\nFINAL TRANSACTION PROPOSAL: **HOLD**")


class BrokenStructuredLLM:
    def invoke(self, _prompt):
        raise RuntimeError("structured unavailable")


class StructuredDecisionTests(unittest.TestCase):
    def test_renderers_preserve_exact_executable_action_line(self):
        research = render_research_plan(
            ResearchPlan(
                recommendation=ExecutableAction.BUY,
                confidence="medium",
                advisory_rating=AdvisoryRating.OVERWEIGHT,
                rationale="Evidence supports upside.",
                strategic_actions="Enter on confirmation.",
            )
        )
        trader = render_trader_proposal(
            TraderProposal(
                action=ExecutableAction.LONG,
                confidence="high",
                reasoning="Trend and macro align.",
            )
        )
        risk = render_risk_decision(
            RiskDecision(
                action=ExecutableAction.SELL,
                confidence="low",
                risk_rationale="Downside exceeds reward.",
                required_controls="Do not re-enter without reversal.",
            )
        )

        self.assertIn("**Advisory Rating**: Overweight", research)
        self.assertIn("FINAL TRANSACTION PROPOSAL: **BUY**", research)
        self.assertIn("FINAL TRANSACTION PROPOSAL: **LONG**", trader)
        self.assertIn("FINAL TRANSACTION PROPOSAL: **SELL**", risk)
        self.assertEqual(extract_recommendation(research, "investment"), "BUY")

    def test_structured_failure_falls_back_to_plain_text(self):
        content = invoke_structured_or_freetext(
            BrokenStructuredLLM(),
            PlainLLM(),
            "prompt",
            lambda value: value,
            "Unit Agent",
        )

        self.assertIn("FINAL TRANSACTION PROPOSAL: **HOLD**", content)


if __name__ == "__main__":
    unittest.main()
