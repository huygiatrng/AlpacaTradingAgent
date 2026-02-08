"""
Pydantic schema for the standardized Technical Brief.

Defines the fixed JSON contract between Tier 1 (deterministic quant) and
Tier 2 (LLM market analyst).  Every field is intentionally compact so the
serialized brief stays well under 2 000 tokens.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────

class Direction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class Strength(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


# ── Per-indicator state models ───────────────────────────────────────────

class TrendState(BaseModel):
    direction: Direction
    strength: Strength
    ema_slope: float = Field(
        description="Normalized slope of fast EMA (8-period) as percent change over last 5 bars"
    )
    higher_highs: bool = Field(description="Higher-high pattern detected in recent swings")
    higher_lows: bool = Field(description="Higher-low pattern detected in recent swings")
    adx: float = Field(description="ADX trend strength (0-100)", default=0.0)
    trend_strength_adx: Literal["weak", "strong", "very_strong"] = Field(
        description="Qualitative strength based on ADX", default="weak"
    )
    sma_200: float = Field(description="200-period Simple Moving Average", default=0.0)
    sma_200_dist: float = Field(description="Distance from SMA 200 (%)", default=0.0)


class MomentumState(BaseModel):
    rsi_value: float = Field(description="Current RSI-14 value (0-100)")
    rsi_zone: Literal["oversold", "neutral", "overbought"] = Field(
        description="RSI zone: <30 oversold, 30-70 neutral, >70 overbought"
    )
    rsi_divergence: bool = Field(
        description="True if price/RSI divergence detected (bullish or bearish)"
    )
    macd_cross: Literal["bullish", "bearish", "none"] = Field(
        description="Recent MACD/signal line crossover direction"
    )
    macd_histogram_trend: Literal["expanding", "contracting", "flat"] = Field(
        description="Whether MACD histogram bars are growing or shrinking"
    )
    stoch_k: float = Field(description="Stochastic RSI %K (0-100)", default=50.0)
    stoch_d: float = Field(description="Stochastic RSI %D (0-100)", default=50.0)
    stoch_state: Literal["oversold", "neutral", "overbought"] = Field(
        description="Stoch RSI state (<20 oversold, >80 overbought)", default="neutral"
    )


class VWAPState(BaseModel):
    position: Literal["above", "below", "at"] = Field(
        description="Current close relative to VWAP"
    )
    zscore_distance: float = Field(
        description="Distance from VWAP expressed as z-score units"
    )


class VolatilityState(BaseModel):
    atr_value: float = Field(description="Current ATR-14 value in price units")
    atr_percentile: float = Field(
        description="Where current ATR sits within the 90-day range (0-100)"
    )
    squeeze: bool = Field(
        description="Bollinger Band squeeze detected (bandwidth below 20th percentile)"
    )
    breakout: bool = Field(
        description="Volatility breakout detected (bandwidth expanding above 80th percentile)"
    )
    gap_percent: float = Field(
        description="Percentage gap between prev close and current open", default=0.0
    )


class VolumeState(BaseModel):
    vol_ma_ratio: float = Field(description="Current volume / 20-period Volume MA")
    vol_trend: Literal["up", "down", "flat"] = Field(description="Volume trend (slope of Volume MA)")
    obv_slope: float = Field(description="Slope of On-Balance Volume over last 5 bars")


class MarketStructure(BaseModel):
    bos: bool = Field(description="Break of Structure detected")
    choch: bool = Field(description="Change of Character detected")
    last_swing_high: float = Field(description="Price of most recent swing high")
    last_swing_low: float = Field(description="Price of most recent swing low")


# ── Key levels ───────────────────────────────────────────────────────────

class KeyLevel(BaseModel):
    label: str = Field(description="e.g. 'VWAP', 'Yesterday High', 'Pivot R1'")
    price: float
    type: Literal["support", "resistance", "pivot"]


# ── Signal summary ───────────────────────────────────────────────────────

class SignalSummary(BaseModel):
    setup: Literal[
        "breakout", "pullback", "mean_reversion",
        "trend_continuation", "none"
    ] = Field(description="Classified setup type")
    confidence: Literal["high", "medium", "low"]
    description: str = Field(
        description="One-sentence summary of the current setup"
    )


# ── Per-timeframe brief ─────────────────────────────────────────────────

class TimeframeBrief(BaseModel):
    timeframe: str = Field(description="One of '1h', '4h', '1d'")
    trend: TrendState
    momentum: MomentumState
    vwap_state: VWAPState
    volatility: VolatilityState
    volume: VolumeState = Field(
        description="Volume analysis state",
        default_factory=lambda: VolumeState(vol_ma_ratio=1.0, vol_trend="flat", obv_slope=0.0)
    )
    market_structure: MarketStructure


# ── Top-level Technical Brief ────────────────────────────────────────────

class TechnicalBrief(BaseModel):
    symbol: str
    generated_at: str = Field(description="ISO-8601 timestamp of generation")
    timeframes: List[TimeframeBrief] = Field(
        description="Analysis for each of the 3 timeframes: 1h, 4h, 1d"
    )
    key_levels: List[KeyLevel] = Field(
        description="3-5 most important cross-timeframe price levels"
    )
    signal_summary: SignalSummary = Field(
        description="Aggregate signal across all timeframes"
    )
    raw_prices: dict = Field(
        description="Snapshot: last_close, prev_close, daily_change_pct"
    )
