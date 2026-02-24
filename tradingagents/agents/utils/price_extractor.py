"""
Price Extraction Module

Extracts stop loss and take profit prices from AI agent text outputs and applies
safety validations to ensure responsible risk management.
"""

import re
from typing import Dict, List, Optional


def extract_trading_prices(text: str, current_price: float = None) -> dict:
    """
    Extract entry, stop loss, and target prices from AI text.

    Patterns to support:
    - "Entry Price: $160.00" or "Entry: $160"
    - "Stop Loss: $150.00" or "Stop at $150"
    - "Target 1: $175.00" or "First Target: $175"
    - "Target 2: $185.00" or "Second Target: $185"
    - "Profit Target: $180" (single target)
    - Table format: | Stop Loss | $150.00 |

    Args:
        text: The AI agent's analysis text
        current_price: Current market price (for validation)

    Returns:
        Dict with:
        - entry_price: float (recommended entry price)
        - stop_loss: float (stop loss price)
        - targets: list (list of target prices)
        - extraction_method: str (how we extracted the values)
        - confidence: str (high/medium/low)
        - fallback_used: bool (whether extraction failed)
    """

    print(f"[PRICE EXTRACTOR] Starting price extraction (text length: {len(text)} chars)")
    print(f"[PRICE EXTRACTOR] Current market price: ${current_price:.2f}" if current_price else "[PRICE EXTRACTOR] No current price provided")

    result = {
        "entry_price": None,
        "stop_loss": None,
        "targets": [],
        "extraction_method": None,
        "confidence": "low",
        "fallback_used": False
    }

    if not text:
        print("[PRICE EXTRACTOR] ❌ No text provided, extraction failed")
        result["fallback_used"] = True
        return result

    # Show sample of text for debugging
    print("[PRICE EXTRACTOR] Sample of text to extract from (first 500 chars):")
    print(text[:500] + "..." if len(text) > 500 else text)
    print("[PRICE EXTRACTOR] Looking for patterns like: 'Entry Price: $X.XX', 'Stop Loss: $X.XX', etc.")

    # Pattern 1: Entry Price
    # Match: "Entry Price: $160.00" or "Entry: $160" or "Entry Point: 160"
    print("[PRICE EXTRACTOR] Searching for entry price...")
    pattern_entry = r"(?:Entry Price|Entry Point|Entry Level|Entry):\s*\$?([\d,]+(?:\.\d{2,4})?)"
    match = re.search(pattern_entry, text, re.IGNORECASE)
    if match:
        entry_str = match.group(1).replace(',', '')
        result["entry_price"] = float(entry_str)
        result["extraction_method"] = "explicit_prices"
        result["confidence"] = "high"
        print(f"[PRICE EXTRACTOR] ✅ Found entry price: ${result['entry_price']:.2f} (matched: '{match.group(0)}')")
    else:
        print("[PRICE EXTRACTOR] ⚠️ No entry price found")

    # Pattern 2: Stop Loss
    # Match: "Stop Loss: $150.00" or "Stop Price: $150" or "Stop at $150"
    print("[PRICE EXTRACTOR] Searching for stop loss...")
    pattern_stop = r"(?:Stop Loss|Stop Price|Stop Level|Stop at|Stop):\s*\$?([\d,]+(?:\.\d{2,4})?)"
    match = re.search(pattern_stop, text, re.IGNORECASE)
    if match:
        stop_str = match.group(1).replace(',', '')
        result["stop_loss"] = float(stop_str)
        if result["extraction_method"] is None:
            result["extraction_method"] = "explicit_prices"
        result["confidence"] = "high"
        print(f"[PRICE EXTRACTOR] ✅ Found stop loss: ${result['stop_loss']:.2f} (matched: '{match.group(0)}')")
    else:
        print("[PRICE EXTRACTOR] ⚠️ No stop loss found with primary pattern")

    # Pattern 3: Target Prices (multiple formats)
    print("[PRICE EXTRACTOR] Searching for target prices...")
    # Match: "Target 1: $175" or "First Target: $175.00" or "T1: 175"
    pattern_target1 = r"(?:Target 1|First Target|Target Price 1|T1|Initial Target):\s*\$?([\d,]+(?:\.\d{2,4})?)"
    match = re.search(pattern_target1, text, re.IGNORECASE)
    if match:
        target_str = match.group(1).replace(',', '')
        result["targets"].append(float(target_str))
        if result["extraction_method"] is None:
            result["extraction_method"] = "explicit_prices"
        result["confidence"] = "high"
        print(f"[PRICE EXTRACTOR] ✅ Found target 1: ${result['targets'][0]:.2f} (matched: '{match.group(0)}')")
    else:
        print("[PRICE EXTRACTOR] ⚠️ No target 1 found")

    # Match: "Target 2: $185" or "Second Target: $185.00" or "T2: 185"
    pattern_target2 = r"(?:Target 2|Second Target|Target Price 2|T2|Final Target):\s*\$?([\d,]+(?:\.\d{2,4})?)"
    match = re.search(pattern_target2, text, re.IGNORECASE)
    if match:
        target_str = match.group(1).replace(',', '')
        result["targets"].append(float(target_str))
        if result["extraction_method"] is None:
            result["extraction_method"] = "explicit_prices"
        result["confidence"] = "high"
        print(f"[PRICE EXTRACTOR] ✅ Found target 2: ${result['targets'][-1]:.2f} (matched: '{match.group(0)}')")
    else:
        print("[PRICE EXTRACTOR] ⚠️ No target 2 found")

    # Match single target: "Profit Target: $180" or "Take Profit: $180"
    if len(result["targets"]) == 0:
        print("[PRICE EXTRACTOR] Searching for single target...")
        pattern_single_target = r"(?:Profit Target|Take Profit|Target Price|Target):\s*\$?([\d,]+(?:\.\d{2,4})?)"
        match = re.search(pattern_single_target, text, re.IGNORECASE)
        if match:
            target_str = match.group(1).replace(',', '')
            result["targets"].append(float(target_str))
            if result["extraction_method"] is None:
                result["extraction_method"] = "explicit_prices"
            result["confidence"] = "high"
            print(f"[PRICE EXTRACTOR] ✅ Found single target: ${result['targets'][0]:.2f} (matched: '{match.group(0)}')")
        else:
            print("[PRICE EXTRACTOR] ⚠️ No single target found")

    # Pattern 4: Table format
    # Match: | Stop Loss | $150.00 |
    if result["stop_loss"] is None:
        print("[PRICE EXTRACTOR] Trying table format for stop loss...")
        pattern_table_stop = r"\|\s*(?:Stop Loss|Stop)\s*\|\s*\$?([\d,]+(?:\.\d{2,4})?)\s*\|"
        match = re.search(pattern_table_stop, text, re.IGNORECASE)
        if match:
            stop_str = match.group(1).replace(',', '')
            result["stop_loss"] = float(stop_str)
            result["extraction_method"] = "table_format"
            result["confidence"] = "medium"
            print(f"[PRICE EXTRACTOR] ✅ Found stop loss in table: ${result['stop_loss']:.2f}")
        else:
            print("[PRICE EXTRACTOR] ⚠️ No stop loss in table format")

    # Match: | Target 1 | $175.00 |
    if len(result["targets"]) == 0:
        print("[PRICE EXTRACTOR] Trying table format for targets...")
        pattern_table_target = r"\|\s*(?:Target 1|T1|Target)\s*\|\s*\$?([\d,]+(?:\.\d{2,4})?)\s*\|"
        match = re.search(pattern_table_target, text, re.IGNORECASE)
        if match:
            target_str = match.group(1).replace(',', '')
            result["targets"].append(float(target_str))
            result["extraction_method"] = "table_format"
            result["confidence"] = "medium"
            print(f"[PRICE EXTRACTOR] ✅ Found target in table: ${result['targets'][0]:.2f}")
        else:
            print("[PRICE EXTRACTOR] ⚠️ No target in table format")

    # Pattern 5: Percentage-based stop loss
    # Match: "2% stop loss" or "stop loss of 3%"
    if result["stop_loss"] is None and current_price:
        print("[PRICE EXTRACTOR] Trying percentage-based stop loss...")
        pattern_pct_stop = r"([\d.]+)%\s*(?:stop|stop loss)"
        match = re.search(pattern_pct_stop, text, re.IGNORECASE)
        if match:
            pct = float(match.group(1))
            # NOTE: Position type is unknown here; always calculates as "long" (below entry).
            # For SHORT positions, the extracted stop will be in the wrong direction and will
            # correctly fail validate_trading_prices() — this is intentional to avoid placing
            # incorrect stops. SHORT stops must come from explicit prices in the AI output.
            result["stop_loss"] = calculate_stop_loss_from_percent(current_price, pct, "long")
            result["extraction_method"] = "percentage_based"
            result["confidence"] = "medium"
            print(f"[PRICE EXTRACTOR] ✅ Found percentage-based stop: {pct}% → ${result['stop_loss']:.2f}")
        else:
            print("[PRICE EXTRACTOR] ⚠️ No percentage-based stop found")

    # Pattern 6: ATR-based stop loss
    # Match: "2x ATR stop" or "stop at 2 ATR"
    if result["stop_loss"] is None:
        pattern_atr_stop = r"(?:stop at|stop of|stop loss of)\s*([\d.]+)\s*(?:x\s*)?ATR"
        match = re.search(pattern_atr_stop, text, re.IGNORECASE)
        if match:
            # Note: We can't calculate without ATR value, so mark as extraction method but leave None
            result["extraction_method"] = "atr_based"
            result["confidence"] = "low"

    # Determine if extraction was successful
    if result["stop_loss"] is None and len(result["targets"]) == 0:
        result["fallback_used"] = True
        result["confidence"] = "low"
        print("[PRICE EXTRACTOR] ❌ EXTRACTION FAILED - No stop loss or targets found")
    else:
        print("[PRICE EXTRACTOR] ✅ EXTRACTION SUCCESSFUL")
        print(f"[PRICE EXTRACTOR] Summary:")
        print(f"  - Entry: ${result['entry_price']:.2f}" if result['entry_price'] else "  - Entry: Not found")
        print(f"  - Stop Loss: ${result['stop_loss']:.2f}" if result['stop_loss'] else "  - Stop Loss: Not found")
        print(f"  - Targets: {[f'${t:.2f}' for t in result['targets']]}" if result['targets'] else "  - Targets: Not found")
        print(f"  - Method: {result['extraction_method']}")
        print(f"  - Confidence: {result['confidence']}")

    return result


def validate_trading_prices(entry: float, stop: float, targets: list,
                           current_price: float, symbol: str, position_type: str = "long") -> Optional[dict]:
    """
    Validate price levels make sense for both LONG and SHORT positions.

    Checks:
    - LONG: Stop loss below entry, targets above entry
    - SHORT: Stop loss above entry, targets below entry
    - Risk/reward ratio >= 2:1
    - Prices within reasonable range (±20% from current)
    - Stop not too tight (>0.5% from entry)

    Args:
        entry: Entry price (can be None, will use current_price)
        stop: Stop loss price
        targets: List of target prices
        current_price: Current market price
        symbol: Ticker symbol (for logging)
        position_type: "long" or "short" (default: "long")

    Returns:
        Dict with validated prices or None if invalid
    """

    position_type = position_type.lower()
    print(f"[PRICE VALIDATION] Starting validation for {symbol} ({position_type.upper()} position)")
    print(f"[PRICE VALIDATION]   Entry: ${entry:.2f}" if entry else "[PRICE VALIDATION]   Entry: None")
    print(f"[PRICE VALIDATION]   Stop: ${stop:.2f}" if stop else "[PRICE VALIDATION]   Stop: None")
    print(f"[PRICE VALIDATION]   Targets: {[f'${t:.2f}' for t in targets]}" if targets else "[PRICE VALIDATION]   Targets: []")
    print(f"[PRICE VALIDATION]   Current Price: ${current_price:.2f}" if current_price else "[PRICE VALIDATION]   Current Price: None")

    # Use current price as entry if not specified
    if entry is None:
        entry = current_price
        print(f"[PRICE VALIDATION] Using current price as entry: ${entry:.2f}")

    # Validate we have at least stop or targets
    if stop is None and len(targets) == 0:
        print(f"[PRICE VALIDATION] {symbol}: ❌ No stop loss or targets provided, skipping validation")
        return None

    validated = {
        "entry_price": entry,
        "stop_loss": stop,
        "targets": targets,
        "validation_passed": True,
        "warnings": []
    }

    # Validation 1: Stop loss position validation (direction depends on position type)
    if stop is not None:
        if position_type == "long":
            # For LONG: stop must be BELOW entry
            if stop >= entry:
                warning = f"Stop loss ${stop:.2f} must be below entry ${entry:.2f} (LONG position)"
                print(f"[PRICE VALIDATION] {symbol}: {warning}")
                validated["warnings"].append(warning)
                validated["validation_passed"] = False
                return None

            # Check if stop is too tight (< 0.5% from entry)
            stop_pct = ((entry - stop) / entry) * 100
            if stop_pct < 0.5:
                warning = f"Stop loss too tight: {stop_pct:.2f}% from entry (minimum 0.5%)"
                print(f"[PRICE VALIDATION] {symbol}: {warning}")
                validated["warnings"].append(warning)

            # Check if stop is too wide (> 20% from entry)
            if stop_pct > 20:
                warning = f"Stop loss very wide: {stop_pct:.2f}% from entry (maximum recommended 20%)"
                print(f"[PRICE VALIDATION] {symbol}: WARNING - {warning}")
                validated["warnings"].append(warning)
        else:
            # For SHORT: stop must be ABOVE entry
            if stop <= entry:
                warning = f"Stop loss ${stop:.2f} must be above entry ${entry:.2f} (SHORT position)"
                print(f"[PRICE VALIDATION] {symbol}: {warning}")
                validated["warnings"].append(warning)
                validated["validation_passed"] = False
                return None

            # Check if stop is too tight (< 0.5% from entry)
            stop_pct = ((stop - entry) / entry) * 100
            if stop_pct < 0.5:
                warning = f"Stop loss too tight: {stop_pct:.2f}% from entry (minimum 0.5%)"
                print(f"[PRICE VALIDATION] {symbol}: {warning}")
                validated["warnings"].append(warning)

            # Check if stop is too wide (> 20% from entry)
            if stop_pct > 20:
                warning = f"Stop loss very wide: {stop_pct:.2f}% from entry (maximum recommended 20%)"
                print(f"[PRICE VALIDATION] {symbol}: WARNING - {warning}")
                validated["warnings"].append(warning)

    # Validation 2: Target position validation (direction depends on position type)
    valid_targets = []
    for i, target in enumerate(targets):
        if position_type == "long":
            # For LONG: targets must be ABOVE entry
            if target <= entry:
                warning = f"Target {i+1} ${target:.2f} must be above entry ${entry:.2f} (LONG position)"
                print(f"[PRICE VALIDATION] {symbol}: {warning}")
                validated["warnings"].append(warning)
                continue  # Skip this target
        else:
            # For SHORT: targets must be BELOW entry
            if target >= entry:
                warning = f"Target {i+1} ${target:.2f} must be below entry ${entry:.2f} (SHORT position)"
                print(f"[PRICE VALIDATION] {symbol}: {warning}")
                validated["warnings"].append(warning)
                continue  # Skip this target
        valid_targets.append(target)

    validated["targets"] = valid_targets

    # Validation 3: Risk/Reward ratio >= 2:1 (for first target)
    if stop is not None and len(valid_targets) > 0:
        if position_type == "long":
            risk = entry - stop
            reward = valid_targets[0] - entry
        else:
            risk = stop - entry
            reward = entry - valid_targets[0]

        rr_ratio = reward / risk if risk > 0 else 0

        if rr_ratio < 2.0:
            warning = f"Risk/Reward ratio {rr_ratio:.2f}:1 below recommended 2:1"
            print(f"[PRICE VALIDATION] {symbol}: WARNING - {warning}")
            validated["warnings"].append(warning)
            # Don't fail, just warn
        else:
            print(f"[PRICE VALIDATION] {symbol}: Risk/Reward ratio {rr_ratio:.2f}:1 ✓")

    # Validation 4: Prices within reasonable range (±20% from current)
    if current_price:
        max_price = current_price * 1.20
        min_price = current_price * 0.80

        if stop is not None and (stop > max_price or stop < min_price):
            warning = f"Stop loss ${stop:.2f} outside ±20% range from current ${current_price:.2f}"
            print(f"[PRICE VALIDATION] {symbol}: {warning}")
            validated["warnings"].append(warning)
            validated["validation_passed"] = False
            return None

        for target in valid_targets:
            if target > max_price or target < min_price:
                warning = f"Target ${target:.2f} outside ±20% range from current ${current_price:.2f}"
                print(f"[PRICE VALIDATION] {symbol}: WARNING - {warning}")
                validated["warnings"].append(warning)

    # If we have no valid targets and no valid stop, fail validation
    if stop is None and len(valid_targets) == 0:
        print(f"[PRICE VALIDATION] {symbol}: No valid stop loss or targets after validation")
        validated["validation_passed"] = False
        return None

    print(f"[PRICE VALIDATION] {symbol}: Validation passed ✓")
    return validated


def calculate_stop_loss_from_percent(entry: float, percent: float,
                                     direction: str = "long") -> float:
    """
    Convert percentage stop to absolute price.

    Args:
        entry: Entry price
        percent: Stop loss percentage (e.g., 2.0 for 2%)
        direction: "long" or "short"

    Returns:
        Stop loss price
    """
    if direction == "long":
        return entry * (1 - percent / 100)
    else:  # short
        return entry * (1 + percent / 100)


def extract_stop_from_atr(text: str, entry: float, atr: float) -> Optional[float]:
    """
    Extract ATR-based stops like '2x ATR stop'.

    Args:
        text: AI agent text
        entry: Entry price
        atr: ATR value

    Returns:
        Stop loss price or None
    """
    pattern = r"(?:stop at|stop of|stop loss of)\s*([\d.]+)\s*(?:x\s*)?ATR"
    match = re.search(pattern, text, re.IGNORECASE)
    if match and atr:
        multiplier = float(match.group(1))
        return entry - (multiplier * atr)
    return None
