"""
Tier 1 -- Deterministic / Quant Technical Analysis Engine.

Computes indicators, detects regime, extracts key levels, and produces a
standardized ``TechnicalBrief`` JSON.  **No LLM calls** happen in this module.

Timeframes: 1h, 4h, 1d (fixed set).
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .alpaca_utils import AlpacaUtils
from .ta_schema import (
    Direction,
    KeyLevel,
    MarketStructure,
    MomentumState,
    SignalSummary,
    Strength,
    TechnicalBrief,
    TimeframeBrief,
    TrendState,
    VolatilityState,
    VolumeState,
    VWAPState,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Timeframe config ─────────────────────────────────────────────────────

# (alpaca_tf_string, lookback_calendar_days)  -- enough bars for 50-period
# indicators plus some buffer.
TIMEFRAMES: Dict[str, Tuple[str, int]] = {
    "1h": ("1Hour", 30),      # ~30 days of hourly bars ≈ 200 bars
    "4h": ("4Hour", 90),      # ~90 days of 4h bars ≈ 540 bars
    "1d": ("1Day", 200),      # 200 calendar days ≈ 140 trading days
}


# ═════════════════════════════════════════════════════════════════════════
#  Indicator computation helpers
# ═════════════════════════════════════════════════════════════════════════

def _ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _stoch_rsi(close: pd.Series, period: int = 14, k: int = 3, d: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Stochastic RSI. Returns (K, D)."""
    rsi = _rsi(close, period)
    rsi_min = rsi.rolling(window=period).min()
    rsi_max = rsi.rolling(window=period).max()
    stoch = ((rsi - rsi_min) / (rsi_max - rsi_min)) * 100
    k_line = stoch.rolling(window=k).mean()
    d_line = k_line.rolling(window=d).mean()
    return k_line, d_line


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr = _atr(high, low, close, period=1) # TR for 1 period
    atr = _atr(high, low, close, period=period)
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
    minus_di = 100 * (minus_dm.abs().ewm(alpha=1/period).mean() / atr)
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    return dx.ewm(alpha=1/period).mean()


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd_line, signal_line, histogram)."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def _bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    """Return (upper_band, lower_band, bandwidth)."""
    sma = _sma(close, period)
    std = close.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    bandwidth = (upper - lower) / sma
    return upper, lower, bandwidth


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# ═════════════════════════════════════════════════════════════════════════
#  Core: fetch OHLCV and compute all indicators for one timeframe
# ═════════════════════════════════════════════════════════════════════════

def compute_indicators(
    symbol: str,
    curr_date: str,
    timeframe_key: str,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV from Alpaca and add derived indicator columns.

    Returns a DataFrame with at least columns:
        open, high, low, close, volume, vwap,
        ema_8, ema_21, sma_50, rsi_14, macd, macds, macdh,
        atr_14, boll_ub, boll_lb, boll_bw, obv
    or ``None`` if data is unavailable.
    """
    alpaca_tf, lookback_days = TIMEFRAMES[timeframe_key]
    curr_dt = pd.to_datetime(curr_date)
    start_dt = curr_dt - timedelta(days=lookback_days)

    df = AlpacaUtils.get_stock_data(
        symbol=symbol,
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=curr_date,
        timeframe=alpaca_tf,
    )

    if df is None or df.empty or len(df) < 30:
        print(f"[TA-BRIEF] Insufficient data for {symbol} @ {timeframe_key} "
              f"(got {0 if df is None else len(df)} bars)")
        return None

    # Ensure lowercase column names
    df.columns = [c.lower() for c in df.columns]

    # Guarantee required columns
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            print(f"[TA-BRIEF] Missing column '{col}' in {symbol} @ {timeframe_key}")
            return None

    df = df.sort_values("timestamp" if "timestamp" in df.columns else df.columns[0]).reset_index(drop=True)

    # ── Trend indicators ──
    df["ema_8"] = _ema(df["close"], 8)
    df["ema_21"] = _ema(df["close"], 21)
    df["sma_50"] = _sma(df["close"], 50)
    df["sma_200"] = _sma(df["close"], 200)

    df["adx_14"] = _adx(df["high"], df["low"], df["close"], 14)

    # ── Momentum ──
    df["rsi_14"] = _rsi(df["close"], 14)
    df["stoch_k"], df["stoch_d"] = _stoch_rsi(df["close"])
    df["macd"], df["macds"], df["macdh"] = _macd(df["close"])

    # ── Volatility ──
    df["atr_14"] = _atr(df["high"], df["low"], df["close"], 14)
    df["boll_ub"], df["boll_lb"], df["boll_bw"] = _bollinger(df["close"])

    # ── Volume ──
    df["obv"] = _obv(df["close"], df["volume"])
    df["vol_sma_20"] = _sma(df["volume"], 20)

    # ── VWAP (use daily if available, else recalc) ──
    if "vwap" not in df.columns:
        # Approximate intra-period VWAP
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    return df


# ═════════════════════════════════════════════════════════════════════════
#  Detectors -- each returns a schema model
# ═════════════════════════════════════════════════════════════════════════

def detect_trend(df: pd.DataFrame) -> TrendState:
    """Classify trend from EMA alignment & swing structure."""
    close = df["close"].iloc[-1]
    ema8 = df["ema_8"].iloc[-1]
    ema21 = df["ema_21"].iloc[-1]
    sma50 = df["sma_50"].iloc[-1]
    sma200 = float(df["sma_200"].iloc[-1]) if "sma_200" in df.columns and not pd.isna(df["sma_200"].iloc[-1]) else 0.0

    # Normalized EMA-8 slope (pct change over last 5 bars)
    ema8_recent = df["ema_8"].iloc[-6:]
    ema_slope = (ema8_recent.iloc[-1] / ema8_recent.iloc[0] - 1) * 100 if len(ema8_recent) >= 2 and ema8_recent.iloc[0] != 0 else 0.0

    # Alignment score: how many of (ema8 > ema21 > sma50) hold
    bullish_align = int(ema8 > ema21) + int(ema21 > sma50) + int(close > ema8)
    bearish_align = int(ema8 < ema21) + int(ema21 < sma50) + int(close < ema8)

    if bullish_align >= 2:
        direction = Direction.BULLISH
    elif bearish_align >= 2:
        direction = Direction.BEARISH
    else:
        direction = Direction.NEUTRAL

    # Strength from slope magnitude
    abs_slope = abs(ema_slope)
    if abs_slope > 1.5:
        strength = Strength.STRONG
    elif abs_slope > 0.5:
        strength = Strength.MODERATE
    else:
        strength = Strength.WEAK

    # HH / HL detection (simple: last 20 bars)
    hh, hl = _detect_hh_hl(df, lookback=20)
    
    # ADX Strength
    adx_val = float(df["adx_14"].iloc[-1]) if "adx_14" in df.columns and not pd.isna(df["adx_14"].iloc[-1]) else 0.0
    if adx_val > 40:
        adx_strength = "very_strong"
    elif adx_val > 25:
        adx_strength = "strong"
    else:
        adx_strength = "weak"
    
    # SMA 200 Distance
    sma200_dist = ((close - sma200) / sma200) * 100 if sma200 > 0 else 0.0

    return TrendState(
        direction=direction,
        strength=strength,
        ema_slope=round(ema_slope, 4),
        higher_highs=hh,
        higher_lows=hl,
        adx=round(adx_val, 2),
        trend_strength_adx=adx_strength,
        sma_200=round(sma200, 2),
        sma_200_dist=round(sma200_dist, 2),
    )


def _detect_hh_hl(df: pd.DataFrame, lookback: int = 20) -> Tuple[bool, bool]:
    """Simple swing-point HH / HL detection."""
    recent = df.tail(lookback)
    if len(recent) < 10:
        return False, False

    swing_highs = []
    swing_lows = []
    highs = recent["high"].values
    lows = recent["low"].values

    for i in range(2, len(recent) - 2):
        if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and highs[i] > highs[i + 1] and highs[i] > highs[i + 2]:
            swing_highs.append(highs[i])
        if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
            swing_lows.append(lows[i])

    hh = len(swing_highs) >= 2 and swing_highs[-1] > swing_highs[-2]
    hl = len(swing_lows) >= 2 and swing_lows[-1] > swing_lows[-2]
    return hh, hl


def detect_momentum(df: pd.DataFrame) -> MomentumState:
    """RSI zone, divergence flag, MACD cross."""
    rsi_val = float(df["rsi_14"].iloc[-1])
    if pd.isna(rsi_val):
        rsi_val = 50.0

    # RSI zone
    if rsi_val < 30:
        rsi_zone = "oversold"
    elif rsi_val > 70:
        rsi_zone = "overbought"
    else:
        rsi_zone = "neutral"

    # RSI divergence: price making new high but RSI not (bearish) or vice versa
    rsi_divergence = _detect_rsi_divergence(df)

    # MACD cross
    macd_curr = df["macd"].iloc[-1]
    macds_curr = df["macds"].iloc[-1]
    macd_prev = df["macd"].iloc[-2] if len(df) > 1 else macd_curr
    macds_prev = df["macds"].iloc[-2] if len(df) > 1 else macds_curr

    if pd.isna(macd_curr) or pd.isna(macds_curr):
        macd_cross = "none"
    elif macd_prev <= macds_prev and macd_curr > macds_curr:
        macd_cross = "bullish"
    elif macd_prev >= macds_prev and macd_curr < macds_curr:
        macd_cross = "bearish"
    else:
        macd_cross = "none"

    # MACD histogram trend (last 3 bars)
    hist_recent = df["macdh"].iloc[-3:]
    if len(hist_recent.dropna()) >= 2:
        diffs = hist_recent.diff().dropna()
        if (diffs.abs() < 0.001).all():
            hist_trend = "flat"
        elif (diffs > 0).all() or (hist_recent.iloc[-1] > 0 and abs(hist_recent.iloc[-1]) > abs(hist_recent.iloc[-2])):
            hist_trend = "expanding"
        elif (diffs < 0).all() or (hist_recent.iloc[-1] < 0 and abs(hist_recent.iloc[-1]) > abs(hist_recent.iloc[-2])):
            # Contracting if magnitude is shrinking toward zero, OR expanding in bearish direction
            hist_trend = "contracting" if abs(hist_recent.iloc[-1]) < abs(hist_recent.iloc[-2]) else "expanding"
        else:
            hist_trend = "flat"
    else:
        hist_trend = "flat"

    # Stoch RSI
    stoch_k = float(df["stoch_k"].iloc[-1]) if "stoch_k" in df.columns and not pd.isna(df["stoch_k"].iloc[-1]) else 50.0
    stoch_d = float(df["stoch_d"].iloc[-1]) if "stoch_d" in df.columns and not pd.isna(df["stoch_d"].iloc[-1]) else 50.0
    
    if stoch_k > 80:
        stoch_state = "overbought"
    elif stoch_k < 20:
        stoch_state = "oversold"
    else:
        stoch_state = "neutral"

    return MomentumState(
        rsi_value=round(rsi_val, 1),
        rsi_zone=rsi_zone,
        rsi_divergence=rsi_divergence,
        macd_cross=macd_cross,
        macd_histogram_trend=hist_trend,
        stoch_k=round(stoch_k, 2),
        stoch_d=round(stoch_d, 2),
        stoch_state=stoch_state,
    )


def _detect_rsi_divergence(df: pd.DataFrame, lookback: int = 14) -> bool:
    """Simple RSI divergence: price new high + RSI lower (bearish), or vice versa."""
    if len(df) < lookback + 1:
        return False
    recent = df.tail(lookback)
    price_high = recent["close"].iloc[-1] >= recent["close"].max() * 0.99
    rsi_lower = recent["rsi_14"].iloc[-1] < recent["rsi_14"].iloc[: len(recent) // 2].max()

    price_low = recent["close"].iloc[-1] <= recent["close"].min() * 1.01
    rsi_higher = recent["rsi_14"].iloc[-1] > recent["rsi_14"].iloc[: len(recent) // 2].min()

    return bool((price_high and rsi_lower) or (price_low and rsi_higher))


def detect_vwap_state(df: pd.DataFrame) -> VWAPState:
    """Position relative to VWAP with z-score distance."""
    close = df["close"].iloc[-1]
    vwap = df["vwap"].iloc[-1]

    if pd.isna(vwap) or vwap == 0:
        return VWAPState(position="at", zscore_distance=0.0)

    diff = close - vwap
    # z-score based on recent close std
    std = df["close"].iloc[-20:].std()
    zscore = diff / std if std and std > 0 else 0.0

    if abs(zscore) < 0.3:
        position = "at"
    elif zscore > 0:
        position = "above"
    else:
        position = "below"

    return VWAPState(
        position=position,
        zscore_distance=round(zscore, 2),
    )


def detect_volatility(df: pd.DataFrame) -> VolatilityState:
    """ATR percentile, Bollinger squeeze / breakout."""
    atr_val = float(df["atr_14"].iloc[-1]) if not pd.isna(df["atr_14"].iloc[-1]) else 0.0

    # ATR percentile within the available data (up to 90 bars)
    atr_series = df["atr_14"].dropna().tail(90)
    if len(atr_series) > 5:
        atr_pct = float((atr_series < atr_val).sum() / len(atr_series) * 100)
    else:
        atr_pct = 50.0

    # Bollinger bandwidth percentile
    bw_series = df["boll_bw"].dropna().tail(90)
    bw_curr = df["boll_bw"].iloc[-1] if not pd.isna(df["boll_bw"].iloc[-1]) else 0.0

    if len(bw_series) > 5:
        bw_pct = float((bw_series < bw_curr).sum() / len(bw_series) * 100)
    else:
        bw_pct = 50.0

    squeeze = bw_pct < 20
    breakout = bw_pct > 80
    
    # Gap % (only relevant for daily, but we can compute for all)
    # Gap = (Open - Prev Close) / Prev Close
    if len(df) >= 2:
        curr_open = df["open"].iloc[-1]
        prev_close = df["close"].iloc[-2]
        gap_pct = ((curr_open - prev_close) / prev_close) * 100
    else:
        gap_pct = 0.0

    return VolatilityState(
        atr_value=round(atr_val, 4),
        atr_percentile=round(atr_pct, 1),
        squeeze=squeeze,
        breakout=breakout,
        gap_percent=round(gap_pct, 2),
    )


def detect_volume(df: pd.DataFrame) -> VolumeState:
    """Analyze volume trends and OBV slope."""
    vol_curr = float(df["volume"].iloc[-1])
    vol_sma = float(df["vol_sma_20"].iloc[-1]) if "vol_sma_20" in df.columns and not pd.isna(df["vol_sma_20"].iloc[-1]) else 1.0
    
    vol_ratio = vol_curr / vol_sma if vol_sma > 0 else 1.0
    
    # Volume trend (slope of SMA 20 over last 5 bars)
    sma_recent = df["vol_sma_20"].tail(5)
    if len(sma_recent) >= 5:
        slope = (sma_recent.iloc[-1] / sma_recent.iloc[0]) - 1
        if slope > 0.05:
            vol_trend = "up"
        elif slope < -0.05:
            vol_trend = "down"
        else:
            vol_trend = "flat"
    else:
        vol_trend = "flat"

    # OBV slope (last 5 bars)
    obv_recent = df["obv"].tail(5)
    if len(obv_recent) >= 5:
        # Simple linear regression slope or just pt-to-pt
        obv_slope = float(obv_recent.iloc[-1] - obv_recent.iloc[0])
        # Normalize? OBV is absolute, so raw slope is hard to interpret universally without context.
        # For now, just raw change.
    else:
        obv_slope = 0.0

    return VolumeState(
        vol_ma_ratio=round(vol_ratio, 2),
        vol_trend=vol_trend,
        obv_slope=round(obv_slope, 2),
    )


def detect_market_structure(df: pd.DataFrame) -> MarketStructure:
    """Detect BOS / CHOCH and last swing points."""
    swing_highs, swing_lows = _get_swing_points(df, lookback=40)

    # Defaults
    last_sh = float(df["high"].iloc[-1])
    last_sl = float(df["low"].iloc[-1])
    bos = False
    choch = False

    if len(swing_highs) >= 2:
        last_sh = float(swing_highs[-1][1])
        prev_sh = float(swing_highs[-2][1])
        # BOS: price closed above previous swing high
        if df["close"].iloc[-1] > prev_sh:
            bos = True
    if len(swing_lows) >= 2:
        last_sl = float(swing_lows[-1][1])
        prev_sl = float(swing_lows[-2][1])
    elif len(swing_lows) >= 1:
        last_sl = float(swing_lows[-1][1])

    # CHOCH: trend was making HH/HL but last swing broke the pattern
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        hh = swing_highs[-1][1] > swing_highs[-2][1]
        hl = swing_lows[-1][1] > swing_lows[-2][1]
        lh = swing_highs[-1][1] < swing_highs[-2][1]
        ll = swing_lows[-1][1] < swing_lows[-2][1]
        # Change of character = trend was bullish (HH+HL) but last swing is LH or LL
        # or trend was bearish (LH+LL) but last swing is HH or HL
        if (hh and ll) or (lh and hl):
            choch = True

    return MarketStructure(
        bos=bos,
        choch=choch,
        last_swing_high=round(last_sh, 2),
        last_swing_low=round(last_sl, 2),
    )


def _get_swing_points(
    df: pd.DataFrame, lookback: int = 40, order: int = 3
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Return lists of (index, price) for swing highs and swing lows."""
    recent = df.tail(lookback)
    highs_arr = recent["high"].values
    lows_arr = recent["low"].values
    idx_start = len(df) - lookback

    swing_highs: List[Tuple[int, float]] = []
    swing_lows: List[Tuple[int, float]] = []

    for i in range(order, len(recent) - order):
        # Check if it's a swing high
        is_sh = all(highs_arr[i] > highs_arr[i - j] for j in range(1, order + 1)) and \
                all(highs_arr[i] > highs_arr[i + j] for j in range(1, order + 1))
        if is_sh:
            swing_highs.append((idx_start + i, float(highs_arr[i])))

        # Check if it's a swing low
        is_sl = all(lows_arr[i] < lows_arr[i - j] for j in range(1, order + 1)) and \
                all(lows_arr[i] < lows_arr[i + j] for j in range(1, order + 1))
        if is_sl:
            swing_lows.append((idx_start + i, float(lows_arr[i])))

    return swing_highs, swing_lows


# ═════════════════════════════════════════════════════════════════════════
#  Key levels (cross-timeframe)
# ═════════════════════════════════════════════════════════════════════════

def extract_key_levels(
    dfs_by_tf: Dict[str, pd.DataFrame],
) -> List[KeyLevel]:
    """
    Merge VWAP, previous-day high/low, and pivot levels from all timeframes.
    Return the 3-5 most important (de-duplicated, sorted by proximity to close).
    """
    levels: List[KeyLevel] = []

    # Use the daily frame for reference close
    daily_df = dfs_by_tf.get("1d")
    if daily_df is None or daily_df.empty:
        # Fallback to any available frame
        for k, v in dfs_by_tf.items():
            if v is not None and not v.empty:
                daily_df = v
                break
    if daily_df is None or daily_df.empty:
        return levels

    current_close = float(daily_df["close"].iloc[-1])

    # ── VWAP from 1d ──
    if "vwap" in daily_df.columns and not pd.isna(daily_df["vwap"].iloc[-1]):
        vwap_price = float(daily_df["vwap"].iloc[-1])
        ltype = "support" if vwap_price < current_close else "resistance"
        levels.append(KeyLevel(label="VWAP (daily)", price=round(vwap_price, 2), type=ltype))

    # ── Previous day high / low ──
    if len(daily_df) >= 2:
        prev_high = float(daily_df["high"].iloc[-2])
        prev_low = float(daily_df["low"].iloc[-2])
        levels.append(KeyLevel(
            label="Yesterday High",
            price=round(prev_high, 2),
            type="resistance" if prev_high > current_close else "support",
        ))
        levels.append(KeyLevel(
            label="Yesterday Low",
            price=round(prev_low, 2),
            type="support" if prev_low < current_close else "resistance",
        ))

    # ── Classic pivot points (from daily) ──
    if len(daily_df) >= 2:
        ph = float(daily_df["high"].iloc[-2])
        pl = float(daily_df["low"].iloc[-2])
        pc = float(daily_df["close"].iloc[-2])
        pivot = (ph + pl + pc) / 3
        r1 = 2 * pivot - pl
        s1 = 2 * pivot - ph
        levels.append(KeyLevel(label="Pivot", price=round(pivot, 2), type="pivot"))
        levels.append(KeyLevel(label="R1", price=round(r1, 2), type="resistance"))
        levels.append(KeyLevel(label="S1", price=round(s1, 2), type="support"))

    # ── Bollinger bands from 4h or 1h ──
    for tf_key in ("4h", "1h"):
        tf_df = dfs_by_tf.get(tf_key)
        if tf_df is not None and not tf_df.empty:
            if "boll_ub" in tf_df.columns and not pd.isna(tf_df["boll_ub"].iloc[-1]):
                levels.append(KeyLevel(
                    label=f"Boll Upper ({tf_key})",
                    price=round(float(tf_df["boll_ub"].iloc[-1]), 2),
                    type="resistance",
                ))
            if "boll_lb" in tf_df.columns and not pd.isna(tf_df["boll_lb"].iloc[-1]):
                levels.append(KeyLevel(
                    label=f"Boll Lower ({tf_key})",
                    price=round(float(tf_df["boll_lb"].iloc[-1]), 2),
                    type="support",
                ))
            break  # only need one sub-daily

    # ── Fibonacci Retracements (from daily high/low of last 200 days) ──
    if daily_df is not None and len(daily_df) >= 30:
        # Find major swing high/low in the lookback
        lookback_max = daily_df["high"].max()
        lookback_min = daily_df["low"].min()
        
        # Determine trend direction to apply fibs correctly? 
        # For simplicity, we just provide the levels between min and max
        diff = lookback_max - lookback_min
        if diff > 0:
            levels.append(KeyLevel(label="Fib 0.236", price=round(lookback_max - 0.236 * diff, 2), type="support" if current_close > lookback_max - 0.236 * diff else "resistance"))
            levels.append(KeyLevel(label="Fib 0.382", price=round(lookback_max - 0.382 * diff, 2), type="support" if current_close > lookback_max - 0.382 * diff else "resistance"))
            levels.append(KeyLevel(label="Fib 0.5", price=round(lookback_max - 0.5 * diff, 2), type="support" if current_close > lookback_max - 0.5 * diff else "resistance"))
            levels.append(KeyLevel(label="Fib 0.618", price=round(lookback_max - 0.618 * diff, 2), type="support" if current_close > lookback_max - 0.618 * diff else "resistance"))

    # ── Multi-month Highs/Lows (Resistance/Support) ──
    if daily_df is not None and len(daily_df) >= 60:
        # 3-Month High (approx 63 trading days)
        lookback_3m = min(len(daily_df), 63)
        high_3m = daily_df["high"].tail(lookback_3m).max()
        levels.append(KeyLevel(label="3-Month High", price=round(high_3m, 2), type="resistance"))
        
        # 6-Month High (approx 126 trading days)
        if len(daily_df) >= 120:
             lookback_6m = min(len(daily_df), 126)
             high_6m = daily_df["high"].tail(lookback_6m).max()
             levels.append(KeyLevel(label="6-Month High", price=round(high_6m, 2), type="resistance"))

    # De-duplicate levels that are within 0.3 % of each other
    levels = _deduplicate_levels(levels, current_close, tolerance_pct=0.3)

    # Sort by distance to current close, keep top 5
    levels.sort(key=lambda lv: abs(lv.price - current_close))
    return levels[:5]


def _deduplicate_levels(
    levels: List[KeyLevel], ref_price: float, tolerance_pct: float = 0.3
) -> List[KeyLevel]:
    """Remove levels whose prices are within *tolerance_pct* of each other."""
    if not levels:
        return levels
    levels_sorted = sorted(levels, key=lambda lv: lv.price)
    result = [levels_sorted[0]]
    for lv in levels_sorted[1:]:
        if ref_price > 0 and abs(lv.price - result[-1].price) / ref_price * 100 < tolerance_pct:
            # Keep the one with a more descriptive label
            if len(lv.label) > len(result[-1].label):
                result[-1] = lv
        else:
            result.append(lv)
    return result


# ═════════════════════════════════════════════════════════════════════════
#  Signal summary (aggregate across timeframes)
# ═════════════════════════════════════════════════════════════════════════

def generate_signal_summary(
    tf_briefs: List[TimeframeBrief],
    levels: List[KeyLevel],
) -> SignalSummary:
    """
    Classify the aggregate setup and confidence from multi-TF briefs.
    """
    if not tf_briefs:
        return SignalSummary(setup="none", confidence="low", description="Insufficient data for analysis.")

    # Count bullish / bearish / neutral across timeframes
    directions = [b.trend.direction for b in tf_briefs]
    bull_count = sum(1 for d in directions if d == Direction.BULLISH)
    bear_count = sum(1 for d in directions if d == Direction.BEARISH)

    # Determine overall bias
    if bull_count >= 2:
        bias = "bullish"
    elif bear_count >= 2:
        bias = "bearish"
    else:
        bias = "neutral"

    # Detect setup type
    setup = "none"
    description_parts = []

    # Check for squeeze -> breakout potential
    any_squeeze = any(b.volatility.squeeze for b in tf_briefs)
    any_breakout = any(b.volatility.breakout for b in tf_briefs)
    any_bos = any(b.market_structure.bos for b in tf_briefs)

    # Check momentum alignment
    rsi_zones = [b.momentum.rsi_zone for b in tf_briefs]
    any_oversold = "oversold" in rsi_zones
    any_overbought = "overbought" in rsi_zones

    if any_breakout and any_bos:
        setup = "breakout"
        description_parts.append(f"Volatility breakout with BOS confirmation, {bias} bias")
    elif any_squeeze:
        setup = "breakout"
        description_parts.append(f"Bollinger squeeze detected, potential breakout setup, {bias} bias")
    elif bias == "bullish" and any_oversold:
        setup = "mean_reversion"
        description_parts.append("Bullish trend with oversold RSI, mean reversion opportunity")
    elif bias == "bearish" and any_overbought:
        setup = "mean_reversion"
        description_parts.append("Bearish trend with overbought RSI, mean reversion opportunity")
    elif bias != "neutral":
        # Check for pullback vs continuation
        # Pullback: trend is clear but short-term momentum is against it
        short_tf = tf_briefs[0]  # 1h is typically first
        if bias == "bullish" and short_tf.momentum.rsi_zone == "neutral" and short_tf.trend.direction != Direction.BULLISH:
            setup = "pullback"
            description_parts.append("Bullish higher-TF trend with short-term pullback")
        elif bias == "bearish" and short_tf.momentum.rsi_zone == "neutral" and short_tf.trend.direction != Direction.BEARISH:
            setup = "pullback"
            description_parts.append("Bearish higher-TF trend with short-term pullback")
        else:
            setup = "trend_continuation"
            description_parts.append(f"Multi-timeframe {bias} trend continuation")

    # Confidence
    aligned = bull_count == len(tf_briefs) or bear_count == len(tf_briefs)
    macd_confirms = sum(
        1 for b in tf_briefs
        if (bias == "bullish" and b.momentum.macd_cross == "bullish")
        or (bias == "bearish" and b.momentum.macd_cross == "bearish")
    )
    if aligned and macd_confirms >= 1:
        confidence = "high"
    elif bull_count >= 2 or bear_count >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    description = description_parts[0] if description_parts else f"Mixed signals, {bias} lean"

    return SignalSummary(
        setup=setup,
        confidence=confidence,
        description=description,
    )


# ═════════════════════════════════════════════════════════════════════════
#  Orchestrator
# ═════════════════════════════════════════════════════════════════════════

def build_technical_brief(symbol: str, curr_date: str) -> TechnicalBrief:
    """
    Master function: compute indicators for 1h / 4h / 1d, detect regimes,
    extract levels, and return a ``TechnicalBrief``.
    """
    dfs_by_tf: Dict[str, pd.DataFrame] = {}
    tf_briefs: List[TimeframeBrief] = []

    for tf_key in ("1h", "4h", "1d"):
        print(f"[TA-BRIEF] Computing indicators for {symbol} @ {tf_key} ...")
        df = compute_indicators(symbol, curr_date, tf_key)
        if df is None:
            continue
        dfs_by_tf[tf_key] = df

        brief = TimeframeBrief(
            timeframe=tf_key,
            trend=detect_trend(df),
            momentum=detect_momentum(df),
            vwap_state=detect_vwap_state(df),
            volatility=detect_volatility(df),
            volume=detect_volume(df),
            market_structure=detect_market_structure(df),
        )
        tf_briefs.append(brief)
        print(f"[TA-BRIEF]   {tf_key}: trend={brief.trend.direction.value} "
              f"({brief.trend.strength.value}), RSI={brief.momentum.rsi_value}")

    # Key levels
    levels = extract_key_levels(dfs_by_tf)

    # Signal summary
    signal = generate_signal_summary(tf_briefs, levels)
    print(f"[TA-BRIEF] Signal: {signal.setup} ({signal.confidence}) -- {signal.description}")

    # Raw prices snapshot
    raw_prices = {"last_close": 0.0, "prev_close": 0.0, "daily_change_pct": 0.0}
    daily_df = dfs_by_tf.get("1d")
    if daily_df is not None and len(daily_df) >= 2:
        last_c = float(daily_df["close"].iloc[-1])
        prev_c = float(daily_df["close"].iloc[-2])
        raw_prices = {
            "last_close": round(last_c, 2),
            "prev_close": round(prev_c, 2),
            "daily_change_pct": round((last_c / prev_c - 1) * 100, 2) if prev_c else 0.0,
        }

    return TechnicalBrief(
        symbol=symbol,
        generated_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        timeframes=tf_briefs,
        key_levels=levels,
        signal_summary=signal,
        raw_prices=raw_prices,
    )
