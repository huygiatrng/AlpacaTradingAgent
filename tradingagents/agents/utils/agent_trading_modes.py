"""
agent_trading_modes.py - Centralized trading mode utilities for all agents

This module provides consistent trading mode handling across all agent types:
- Analysts, Researchers, Risk Management, Managers, and Traders

Two trading modes supported:
1. Investment Mode (allow_shorts=False): BUY/HOLD/SELL actions
2. Trading Mode (allow_shorts=True): LONG/NEUTRAL/SHORT actions with position logic
"""

from typing import Dict, Any, Optional, Tuple
from tradingagents.prompts import render_prompt


class TradingModeConfig:
    """Configuration class for trading modes"""

    # Investment mode actions
    INVESTMENT_ACTIONS = ["BUY", "HOLD", "SELL"]

    # Trading mode actions
    TRADING_ACTIONS = ["LONG", "NEUTRAL", "SHORT"]

    # Position states
    POSITION_LONG = "LONG"
    POSITION_SHORT = "SHORT"
    POSITION_NEUTRAL = "NEUTRAL"


def get_trading_mode_context(config: Optional[Dict[str, Any]] = None,
                           current_position: str = "NEUTRAL") -> Dict[str, str]:
    """
    Get trading mode context information for agent prompts

    Args:
        config: Configuration dictionary containing 'allow_shorts' flag
        current_position: Current position state (LONG/SHORT/NEUTRAL)

    Returns:
        Dict containing trading mode context information
    """
    allow_shorts = config.get("allow_shorts", False) if config else False

    if allow_shorts:
        return _get_trading_context(current_position)
    else:
        return _get_investment_context()


def _get_investment_context() -> Dict[str, str]:
    """Get context for investment mode (BUY/HOLD/SELL) optimized for swing trading"""
    return {
        "mode": "investment",
        "mode_name": "SWING TRADING INVESTMENT MODE",
        "actions": "BUY, HOLD, or SELL",
        "action_list": TradingModeConfig.INVESTMENT_ACTIONS,
        "allow_shorts": False,
        "instructions": render_prompt("trading_modes/investment"),
        "decision_format": "BUY/HOLD/SELL",
        "final_format": "FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**"
    }


def _get_trading_context(current_position: str = "NEUTRAL") -> Dict[str, str]:
    """Get context for trading mode (LONG/NEUTRAL/SHORT) optimized for swing trading"""

    position_logic = render_prompt(
        "trading_modes/position_logic",
        current_position=current_position,
    )

    return {
        "mode": "trading",
        "mode_name": "SWING TRADING MODE",
        "actions": "LONG, NEUTRAL, or SHORT",
        "action_list": TradingModeConfig.TRADING_ACTIONS,
        "allow_shorts": True,
        "current_position": current_position,
        "position_logic": position_logic,
        "instructions": render_prompt(
            "trading_modes/trading",
            position_logic=position_logic,
        ),
        "decision_format": "LONG/NEUTRAL/SHORT",
        "final_format": "FINAL TRANSACTION PROPOSAL: **LONG/NEUTRAL/SHORT**"
    }


def get_agent_specific_context(agent_type: str, trading_context: Dict[str, str]) -> str:
    """
    Get agent-specific trading mode instructions

    Args:
        agent_type: Type of agent (analyst, researcher, trader, risk_mgmt, manager)
        trading_context: Trading context from get_trading_mode_context()

    Returns:
        Agent-specific instruction string
    """

    base_context = trading_context["instructions"]
    mode_name = trading_context["mode_name"]
    actions = trading_context["actions"]

    agent_contexts = {
        agent_type: render_prompt(
            f"trading_modes/agent_context_{agent_type}",
            mode_name=mode_name,
            actions=actions,
            base_context=base_context,
        )
        for agent_type in ("analyst", "researcher", "trader", "risk_mgmt", "manager")
    }

    return agent_contexts.get(agent_type, base_context)


def extract_recommendation(response_content: str, trading_mode: str) -> Optional[str]:
    """
    Extract trading recommendation from agent response

    Args:
        response_content: The agent's response content
        trading_mode: 'investment' or 'trading'

    Returns:
        Extracted recommendation or None if not found
    """
    content = response_content.upper()

    if trading_mode == "investment":
        # Look for BUY/HOLD/SELL patterns
        patterns = [
            "FINAL TRANSACTION PROPOSAL: **BUY**",
            "FINAL TRANSACTION PROPOSAL: **HOLD**",
            "FINAL TRANSACTION PROPOSAL: **SELL**",
            "FINAL INVESTMENT DECISION: **BUY**",
            "FINAL INVESTMENT DECISION: **HOLD**",
            "FINAL INVESTMENT DECISION: **SELL**",
            "FINAL DECISION: **BUY**",
            "FINAL DECISION: **HOLD**",
            "FINAL DECISION: **SELL**"
        ]

        for pattern in patterns:
            if pattern in content:
                return pattern.split("**")[1]

        # Fallback - look for standalone actions at end
        for action in TradingModeConfig.INVESTMENT_ACTIONS:
            if f"**{action}**" in content[-100:]:  # Check last 100 chars
                return action

    else:  # trading mode
        # Look for LONG/NEUTRAL/SHORT patterns
        patterns = [
            "FINAL TRANSACTION PROPOSAL: **LONG**",
            "FINAL TRANSACTION PROPOSAL: **NEUTRAL**",
            "FINAL TRANSACTION PROPOSAL: **SHORT**",
            "FINAL TRADING DECISION: **LONG**",
            "FINAL TRADING DECISION: **NEUTRAL**",
            "FINAL TRADING DECISION: **SHORT**",
            "FINAL RISK MANAGEMENT DECISION: **LONG**",
            "FINAL RISK MANAGEMENT DECISION: **NEUTRAL**",
            "FINAL RISK MANAGEMENT DECISION: **SHORT**",
            "FINAL DECISION: **LONG**",
            "FINAL DECISION: **NEUTRAL**",
            "FINAL DECISION: **SHORT**"
        ]

        for pattern in patterns:
            if pattern in content:
                return pattern.split("**")[1]

        # Fallback - look for standalone actions at end
        for action in TradingModeConfig.TRADING_ACTIONS:
            if f"**{action}**" in content[-100:]:  # Check last 100 chars
                return action

    return None


def validate_recommendation(recommendation: str, trading_mode: str) -> bool:
    """
    Validate if recommendation is valid for the trading mode

    Args:
        recommendation: The recommendation to validate
        trading_mode: 'investment' or 'trading'

    Returns:
        True if valid, False otherwise
    """
    if not recommendation:
        return False

    recommendation = recommendation.upper()

    if trading_mode == "investment":
        return recommendation in TradingModeConfig.INVESTMENT_ACTIONS
    else:  # trading mode
        return recommendation in TradingModeConfig.TRADING_ACTIONS


def get_position_transition(current_position: str, new_signal: str) -> Dict[str, str]:
    """
    Get position transition information for trading mode

    Args:
        current_position: Current position (LONG/SHORT/NEUTRAL)
        new_signal: New signal (LONG/SHORT/NEUTRAL)

    Returns:
        Dict with transition information
    """
    current = current_position.upper()
    signal = new_signal.upper()

    transitions = {
        ("LONG", "LONG"): {
            "action": "HOLD",
            "description": "Keep existing LONG position",
            "new_position": "LONG"
        },
        ("LONG", "NEUTRAL"): {
            "action": "CLOSE_LONG",
            "description": "Close LONG position, exit to neutral",
            "new_position": "NEUTRAL"
        },
        ("LONG", "SHORT"): {
            "action": "REVERSE_TO_SHORT",
            "description": "Close LONG position and open SHORT position",
            "new_position": "SHORT"
        },
        ("SHORT", "SHORT"): {
            "action": "HOLD",
            "description": "Keep existing SHORT position",
            "new_position": "SHORT"
        },
        ("SHORT", "NEUTRAL"): {
            "action": "CLOSE_SHORT",
            "description": "Close SHORT position, exit to neutral",
            "new_position": "NEUTRAL"
        },
        ("SHORT", "LONG"): {
            "action": "REVERSE_TO_LONG",
            "description": "Close SHORT position and open LONG position",
            "new_position": "LONG"
        },
        ("NEUTRAL", "LONG"): {
            "action": "OPEN_LONG",
            "description": "Open LONG position",
            "new_position": "LONG"
        },
        ("NEUTRAL", "SHORT"): {
            "action": "OPEN_SHORT",
            "description": "Open SHORT position",
            "new_position": "SHORT"
        },
        ("NEUTRAL", "NEUTRAL"): {
            "action": "STAY_NEUTRAL",
            "description": "Stay in neutral position",
            "new_position": "NEUTRAL"
        }
    }

    return transitions.get((current, signal), {
        "action": "UNKNOWN",
        "description": f"Unknown transition from {current} to {signal}",
        "new_position": signal
    })


def format_final_decision(recommendation: str, trading_mode: str) -> str:
    """
    Format the final decision string consistently

    Args:
        recommendation: The recommendation (BUY/HOLD/SELL or LONG/NEUTRAL/SHORT)
        trading_mode: 'investment' or 'trading'

    Returns:
        Formatted final decision string
    """
    if not recommendation:
        return "FINAL DECISION: **NO_RECOMMENDATION**"

    recommendation = recommendation.upper()

    if trading_mode == "investment":
        return f"FINAL TRANSACTION PROPOSAL: **{recommendation}**"
    else:  # trading mode
        return f"FINAL TRANSACTION PROPOSAL: **{recommendation}**"


def ensure_final_transaction_proposal(
    response_content: str,
    recommendation: str,
    trading_mode: str,
) -> str:
    """Append the exact executable final proposal line without discarding analysis."""
    final_line = format_final_decision(recommendation, trading_mode)
    content = str(response_content or "").rstrip()
    if final_line in content:
        return content
    if not content:
        return final_line
    return f"{content}\n\n{final_line}"
