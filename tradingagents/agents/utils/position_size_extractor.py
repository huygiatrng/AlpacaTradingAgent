"""
Position Size Extraction Module

Extracts AI-recommended position sizes from agent text outputs and applies
safety validations to ensure responsible risk management.
"""

import re
from typing import Dict, Optional


def extract_position_size(text: str, account_info: dict) -> dict:
    """
    Extract position size recommendation from AI agent text.

    Supports multiple formats:
    - Dollar amounts: $1,000 or $1000 or $1k
    - Percentages: "3% of account" or "risk 3%"
    - Share quantities: "50 shares"
    - Risk statements: "risk $500"

    Args:
        text: The AI agent's analysis text
        account_info: Dict with keys: equity, buying_power, cash

    Returns:
        Dict with:
        - recommended_size_dollars: float (the recommended position size)
        - recommended_shares: Optional[int] (if shares were specified)
        - extraction_method: str (how we extracted the value)
        - confidence: str (high/medium/low)
        - original_text: str (the matched text snippet)
        - fallback_used: bool (whether extraction failed and fallback was used)
    """

    result = {
        "recommended_size_dollars": None,
        "recommended_shares": None,
        "extraction_method": None,
        "confidence": "low",
        "original_text": None,
        "fallback_used": False
    }

    if not text:
        result["fallback_used"] = True
        return result

    # Pattern 1: Explicit dollar amounts with "RECOMMENDED POSITION SIZE" or similar
    # Match: "RECOMMENDED POSITION SIZE: $2,500" or "Position Size: $1000"
    pattern_explicit_dollar = r"(?:RECOMMENDED POSITION SIZE|APPROVED POSITION SIZE|Position Size|Trade Size):\s*\$?([\d,]+(?:\.\d{2})?)\s*(?:USD|dollars?)?"
    match = re.search(pattern_explicit_dollar, text, re.IGNORECASE)
    if match:
        amount_str = match.group(1).replace(',', '')
        result["recommended_size_dollars"] = float(amount_str)
        result["extraction_method"] = "explicit_dollar"
        result["confidence"] = "high"
        result["original_text"] = match.group(0)
        return result

    # Pattern 2: Percentage of account/buying power
    # Match: "3% of buying power" or "risk 2.5% of account"
    pattern_percentage = r"(?:risk|allocate|use)\s*([\d.]+)%\s*(?:of\s*)?(?:account|buying power|equity|portfolio)"
    match = re.search(pattern_percentage, text, re.IGNORECASE)
    if match:
        percentage = float(match.group(1))
        # Convert percentage to dollar amount based on buying power
        buying_power = account_info.get("buying_power", 0)
        result["recommended_size_dollars"] = (percentage / 100) * buying_power
        result["extraction_method"] = "percentage_of_account"
        result["confidence"] = "medium"
        result["original_text"] = match.group(0)
        return result

    # Pattern 3: Dollar amounts with context (anywhere in text)
    # Match: "allocate $2,500" or "risk $1000" or "buy $1.5k worth"
    pattern_contextual_dollar = r"(?:allocate|risk|buy|invest|trade)\s*\$?([\d,]+(?:\.\d{2})?)\s*(?:k|K)?\s*(?:USD|dollars?|worth)?"
    match = re.search(pattern_contextual_dollar, text, re.IGNORECASE)
    if match:
        amount_str = match.group(1).replace(',', '').strip()
        # Validate that we got a non-empty string and can convert to float
        if amount_str:
            try:
                amount = float(amount_str)
                # Check if 'k' or 'K' suffix (multiply by 1000)
                if 'k' in match.group(0).lower():
                    amount *= 1000

                result["recommended_size_dollars"] = amount
                result["extraction_method"] = "contextual_dollar"
                result["confidence"] = "medium"
                result["original_text"] = match.group(0)
                return result
            except (ValueError, AttributeError):
                # Failed to parse, continue to next pattern
                pass

    # Pattern 4: Share quantities
    # Match: "50 shares" or "buy 100 shares"
    pattern_shares = r"(?:buy|trade|purchase)\s*([\d,]+)\s*shares?"
    match = re.search(pattern_shares, text, re.IGNORECASE)
    if match:
        shares_str = match.group(1).replace(',', '')
        result["recommended_shares"] = int(shares_str)
        result["extraction_method"] = "share_quantity"
        result["confidence"] = "medium"
        result["original_text"] = match.group(0)
        # Note: dollar amount will be calculated later based on current price
        return result

    # Pattern 5: Generic dollar amounts (last resort)
    # Match any standalone dollar amount: "$2,500"
    pattern_generic_dollar = r"\$([\d,]+(?:\.\d{2})?)"
    matches = re.findall(pattern_generic_dollar, text)
    if matches:
        # Take the first substantial amount (> $100)
        for amount_str in matches:
            amount = float(amount_str.replace(',', ''))
            if amount >= 100:
                result["recommended_size_dollars"] = amount
                result["extraction_method"] = "generic_dollar"
                result["confidence"] = "low"
                result["original_text"] = f"${amount_str}"
                return result

    # No pattern matched - extraction failed
    result["fallback_used"] = True
    return result


def validate_position_size(
    size_dollars: float,
    account_info: dict,
    limits: dict,
    ticker: str = ""
) -> float:
    """
    Apply safety validations to position size.

    Safety mechanisms:
    1. Max 30% of buying power per trade (configurable)
    2. Max 3-5% account equity risk per trade (configurable)
    3. Minimum $100 position size
    4. Never exceed available cash/buying power

    Args:
        size_dollars: Recommended position size
        account_info: Dict with equity, buying_power, cash
        limits: Dict with max_position_pct_of_buying_power, max_risk_pct_per_trade, min_position_size
        ticker: Symbol (for logging purposes)

    Returns:
        Validated position size (may be adjusted down)
    """

    if size_dollars is None or size_dollars <= 0:
        return 0

    # Extract account info
    buying_power = account_info.get("buying_power", 0)
    equity = account_info.get("equity", 0)
    cash = account_info.get("cash", 0)

    # Extract limits with defaults
    max_pct_buying_power = limits.get("max_position_pct_of_buying_power", 30)
    max_risk_pct = limits.get("max_risk_pct_per_trade", 3)
    min_size = limits.get("min_position_size", 100)

    original_size = size_dollars

    # Validation 1: Minimum position size
    if size_dollars < min_size:
        print(f"[POSITION SIZE] {ticker} position ${size_dollars:.2f} below minimum ${min_size}, skipping trade")
        return 0

    # Validation 2: Max percentage of buying power
    max_from_buying_power = (max_pct_buying_power / 100) * buying_power
    if size_dollars > max_from_buying_power:
        print(f"[POSITION SIZE] {ticker} capping ${size_dollars:,.2f} to {max_pct_buying_power}% of buying power: ${max_from_buying_power:,.2f}")
        size_dollars = max_from_buying_power

    # Validation 3: Max risk percentage of account equity
    max_from_risk = (max_risk_pct / 100) * equity
    if size_dollars > max_from_risk:
        print(f"[POSITION SIZE] {ticker} capping ${size_dollars:,.2f} to {max_risk_pct}% of equity: ${max_from_risk:,.2f}")
        size_dollars = max_from_risk

    # Validation 4: Never exceed available cash/buying power
    if size_dollars > buying_power:
        print(f"[POSITION SIZE] {ticker} capping ${size_dollars:,.2f} to available buying power: ${buying_power:,.2f}")
        size_dollars = buying_power

    if size_dollars > cash:
        print(f"[POSITION SIZE] {ticker} capping ${size_dollars:,.2f} to available cash: ${cash:,.2f}")
        size_dollars = cash

    # Log if adjusted
    if abs(size_dollars - original_size) > 0.01:
        print(f"[POSITION SIZE] {ticker} adjusted from ${original_size:,.2f} to ${size_dollars:,.2f}")

    return size_dollars


def convert_percentage_to_dollars(percentage: float, account_info: dict) -> float:
    """
    Convert percentage-based sizing to dollar amount.

    Args:
        percentage: Percentage of account to allocate (e.g., 3.0 for 3%)
        account_info: Dict with buying_power

    Returns:
        Dollar amount
    """
    buying_power = account_info.get("buying_power", 0)
    return (percentage / 100) * buying_power
