# AI-Determined Position Sizing Implementation

## Overview

This implementation enables the AI agents (Trader and Risk Manager) to determine actual position sizes based on risk analysis, account status, and market conditions. The user-configured trade amount now acts as a **safety cap** rather than a fixed trade size.

## Features

### 1. AI Position Size Determination
- **Trader Agent**: Analyzes risk, volatility, and stop loss distance to recommend position size
- **Risk Manager**: Validates and approves (or adjusts) the trader's recommendation
- **Priority**: Risk Manager > Trader > Fallback to user amount

### 2. Safety Validations
All AI-recommended sizes are validated against multiple safety constraints:
- **Max 30% of buying power** per trade (configurable)
- **Max 3% account equity risk** per trade (configurable)
- **Minimum $100** position size
- **Never exceed available cash/buying power**
- **User-configured amount acts as ultimate safety cap**

### 3. Extraction Patterns
The system recognizes multiple formats in AI agent text:
- Explicit dollar amounts: `RECOMMENDED POSITION SIZE: $2,500`
- Percentages: `risk 3% of buying power`
- Contextual amounts: `allocate $1,000`
- Share quantities: `buy 50 shares`

### 4. User Controls
- **AI Position Sizing Toggle**: Enable/disable AI-determined sizing (Web UI)
- **Max Order Amount**: Acts as cap when AI sizing enabled, or fixed amount when disabled
- **Configuration Persistence**: Settings saved to localStorage

## Implementation Details

### Files Modified

#### Core Logic
1. **`tradingagents/agents/utils/position_size_extractor.py`** (NEW)
   - `extract_position_size()`: Parses AI text for position size recommendations
   - `validate_position_size()`: Applies safety validations
   - `convert_percentage_to_dollars()`: Converts percentage-based sizing

2. **`tradingagents/agents/trader/trader.py`**
   - Lines 60-70: Added equity to account_status_desc
   - Lines 135-155: Added POSITION SIZING DECISION section to prompt
   - Lines 274-299: Extract and return recommended_position_size in state

3. **`tradingagents/agents/managers/risk_manager.py`**
   - Lines 70-80: Added equity to account_status_desc
   - Lines 148-165: Added POSITION SIZE VALIDATION section to prompt
   - Lines 186-230: Extract and return approved_position_size in state

4. **`webui/components/analysis.py`**
   - Lines 14-107: Modified `execute_trade_after_analysis()` to support AI sizing
   - Lines 49-87: Position size extraction, validation, and capping logic
   - Lines 193-207: Pass `use_ai_sizing` parameter

#### User Interface
5. **`webui/components/config_panel.py`**
   - Lines 148-172: Added AI Position Sizing toggle, updated labels

6. **`webui/callbacks/control_callbacks.py`**
   - Lines 501-522: Added `ai-position-sizing` state to callback inputs
   - Lines 582-584: Store `use_ai_sizing` in app_state

7. **`webui/callbacks/storage_callbacks.py`**
   - Lines 12-71: Added `ai-position-sizing` output to load callback
   - Lines 73-131: Added `ai-position-sizing` input to save callback

8. **`webui/utils/storage.py`**
   - Line 23: Added `ai_position_sizing: True` to DEFAULT_SETTINGS

#### Configuration
9. **`tradingagents/default_config.py`**
   - Lines 22-26: Added position sizing configuration parameters

## Usage

### For Users

1. **Enable AI Position Sizing** (default: enabled)
   - Toggle "AI-Determined Position Sizing" in Web UI config panel
   - Set "Max Order Amount" as your safety cap (e.g., $4500)

2. **AI determines position size based on:**
   - Account equity and buying power
   - Stop loss distance
   - Daily ATR (volatility)
   - Risk tolerance (1-3% recommended)

3. **Safety mechanisms apply:**
   - AI recommends $2,500, validated to comply with risk limits
   - Final amount capped at user's Max Order Amount
   - Position skipped if below $100 minimum

### For Developers

**Extract position size from AI text:**
```python
from tradingagents.agents.utils.position_size_extractor import extract_position_size

result = extract_position_size(
    text=agent_output,
    account_info={"equity": 100000, "buying_power": 200000, "cash": 50000}
)

# Returns:
{
    "recommended_size_dollars": 2500.0,
    "extraction_method": "explicit_dollar",
    "confidence": "high",
    "fallback_used": False
}
```

**Validate position size:**
```python
from tradingagents.agents.utils.position_size_extractor import validate_position_size

validated = validate_position_size(
    size_dollars=2500.0,
    account_info={"equity": 100000, "buying_power": 200000, "cash": 50000},
    limits={
        "max_position_pct_of_buying_power": 30,
        "max_risk_pct_per_trade": 3,
        "min_position_size": 100
    },
    ticker="NVDA"
)
# Returns: 2500.0 (or adjusted amount if validations fail)
```

## Configuration Parameters

In `tradingagents/default_config.py`:

```python
"ai_position_sizing": True,  # Enable AI-determined sizing
"max_position_pct_of_buying_power": 30,  # Max % of buying power per trade
"max_risk_pct_per_trade": 3,  # Max % account risk per trade
"min_position_size": 100,  # Minimum position size in dollars
```

## Logging

Position sizing decisions are logged with `[POSITION SIZE]` prefix:

```
[POSITION SIZE] NVDA AI recommended $2,500.00, validated to $2,500.00, capped at $4,500.00
[POSITION SIZE] Using Risk Manager's approved size
[POSITION SIZE] NVDA position $2,500.00 executing
```

## Testing Scenarios

1. **AI recommends $2,000, user max is $4,500**
   - Result: $2,000 position (AI recommendation used)

2. **AI recommends $5,000, user max is $4,500**
   - Result: $4,500 position (capped at user maximum)

3. **AI recommends 50% of account (buying power $10,000)**
   - Result: $3,000 position (capped at 30% max buying power)

4. **AI extraction fails**
   - Result: $4,500 position (fallback to user amount)

5. **AI recommends $50 (below minimum)**
   - Result: Trade skipped (below $100 minimum)

## Benefits

1. **Dynamic Risk Management**: Position sizes adjust to market volatility and account size
2. **AI Integration**: Leverages AI's risk analysis for actual execution
3. **Safety Preserved**: Multiple validation layers and user-controlled maximum
4. **Transparency**: Users see AI's reasoning in analysis text
5. **Fallback Protection**: Uses fixed amount if extraction fails

## Future Enhancements

### Optional: Stop Loss & Take Profit Orders

While position sizing is now AI-determined, stop loss and take profit levels are still only mentioned in analysis text but **not executed as actual orders**. To add this:

1. Import stop/limit order types from Alpaca SDK
2. Extend position_size_extractor.py to also extract stop/target prices
3. Modify order placement to use bracket orders or OCO orders
4. Add order monitoring for multi-leg positions

See plan document for detailed implementation steps.

## Troubleshooting

**Issue**: AI sizing not being applied
- Check: `use_ai_sizing` toggle is enabled in UI
- Check: Logs show `[POSITION SIZE]` extraction messages
- Check: `approved_position_size` or `recommended_position_size` in state

**Issue**: Always using fallback amount
- Check: Agent prompts include position sizing sections
- Check: Regex patterns in `position_size_extractor.py` match agent output
- Review: Agent analysis text for position size recommendations

**Issue**: Position size seems wrong
- Check: Validation limits in config (30% buying power, 3% risk)
- Check: Account equity and buying power values
- Review: Logs for validation adjustments

## Summary

This implementation bridges the gap between AI risk analysis and actual trade execution. The AI agents now control position sizing based on their risk assessment, while multiple safety mechanisms ensure responsible risk management. The user maintains ultimate control through the max amount safety cap and the ability to disable AI sizing entirely.

**Default Behavior**: AI determines size, user amount acts as safety cap, enabled by default.
