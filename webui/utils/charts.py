# -------------------------------- charts.py -----------------------

import random
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import traceback
import pytz
from tradingagents.dataflows.alpaca_utils import AlpacaUtils
from tradingagents.dataflows.config import get_alpaca_api_key, get_alpaca_secret_key
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from typing import Union

def create_chart(ticker: str, period: str = "1y", end_date: Union[str, datetime] = None):
    """
    Create a Plotly candlestick+volume chart for a given ticker and period.
    Falls back to demo data if API fails or no bars are returned.
    """
    # determine end and start datetimes (in UTC for the API)
    now_utc = datetime.now(pytz.UTC)
    if end_date:
        end_dt = pd.to_datetime(end_date)
        end_dt = end_dt.tz_localize(pytz.UTC) if end_dt.tzinfo is None else end_dt
    else:
        end_dt = now_utc

    # Updated period mapping with proper differentiation
    period_map = {
        "15m": ("15Min", timedelta(days=5)),      # 5 days of 15-minute data for 15m view (covers weekends)
        "1d": ("5Min", timedelta(days=2)),        # 2 days of 5-minute data for 1d view (more detailed than 15m)
        "1w": ("30Min", timedelta(days=10)),      # 10 days of 30-minute data for 1w view (more data, less gaps)
        "1mo": ("1Hour", timedelta(days=45)),     # 45 days of 1-hour data for 1mo view (more data, less gaps)
        "1y": ("1Day", timedelta(days=365)),
    }
    tf_str, delta = period_map.get(period, period_map["1y"])
    start_dt = end_dt - delta

    # fetch data
    df = AlpacaUtils.get_stock_data(
        symbol=ticker,
        start_date=start_dt,
        end_date=end_dt,
        timeframe=tf_str
    )

    # if we got no data, make a demo chart
    if df.empty:
        return create_demo_chart(ticker, period, end_date, error_msg="No data returned from Alpaca API.")

    # build chart
    fig = go.Figure()
    # Add volume bars first with lower opacity
    fig.add_trace(go.Bar(
        x=df['timestamp'], y=df['volume'], name='Volume', yaxis='y2', opacity=0.3
    ))
    # Add candlestick trace on top
    fig.add_trace(go.Candlestick(
        x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'
    ))
    title = f"{ticker} - {period.upper()} Chart"
    if end_date:
        title += f" (as of {pd.to_datetime(end_date).date()})"
    
    # Improved layout with better gap handling - different rangebreaks for different timeframes
    rangebreaks = []
    if "/" not in ticker:  # Only apply to stocks, not crypto
        if period in ["15m", "1d"]:
            # For intraday charts, hide non-trading hours
            rangebreaks = [
                dict(bounds=["sat", "mon"]),  # Hide weekends
                dict(bounds=[20, 9.5], pattern="hour"),  # Hide non-trading hours (8PM to 9:30AM)
            ]
        elif period in ["1w", "1mo"]:
            # For weekly/monthly charts, only hide weekends
            rangebreaks = [
                dict(bounds=["sat", "mon"]),  # Hide weekends
            ]
        # For 1y charts, no rangebreaks to avoid issues with daily data
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#1E293B",
        plot_bgcolor="#1E293B",
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            type='date',
            rangeslider_visible=False,
            rangebreaks=rangebreaks
        ),
        yaxis_title='Price',
        yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False),
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        autosize=True
    )
    return fig


def create_demo_chart(ticker, period="1y", end_date=None, error_msg=None):
    """Create a demo chart with random walk data"""
    # Updated points mapping to reflect new timeframes
    points_map = {"15m":160, "1d":96, "1w":48, "1mo":90, "1y":252}  # Updated 1d to reflect 5Min data
    points = points_map.get(period, 252)
    title_map = {
        "15m": "15 Minutes", "1d": "1 Day",
        "1w": "1 Week", "1mo": "1 Month", "1y": "1 Year"
    }
    title = f"{ticker} - {title_map.get(period, period)} Chart (Demo)"
    if end_date:
        title += f" (as of {end_date})"
    # Dates and prices
    end_dt = pd.to_datetime(end_date) if end_date else datetime.now()
    dates = pd.date_range(end=end_dt, periods=points)
    prices = [100 + random.uniform(-20,20)]
    for _ in range(1, points):
        delta = random.uniform(-2,2) + random.uniform(-0.5,0.7)
        prices.append(max(5, prices[-1] + delta))
    opens, highs, lows, closes, vols = [], [], [], prices.copy(), []
    for i, close in enumerate(closes):
        opens.append(closes[i-1] if i>0 else close)
        high = max(opens[i], close) + random.uniform(0.1,1)
        low  = min(opens[i], close) - random.uniform(0.1,1)
        vols.append(random.randint(100000,10000000))
        highs.append(high); lows.append(low)
    fig = go.Figure()
    # Add volume bars first with lower opacity
    fig.add_trace(go.Bar(x=dates, y=vols, name='Volume', yaxis='y2', opacity=0.3))
    # Add candlestick trace on top
    fig.add_trace(go.Candlestick(x=dates, open=opens, high=highs, low=lows, close=closes, name='Price'))
    
    # Apply the same improved layout with gap handling - different rangebreaks for different timeframes
    rangebreaks = []
    if "/" not in ticker:  # Only apply to stocks, not crypto
        if period in ["15m", "1d"]:
            # For intraday charts, hide non-trading hours
            rangebreaks = [
                dict(bounds=["sat", "mon"]),  # Hide weekends
                dict(bounds=[20, 9.5], pattern="hour"),  # Hide non-trading hours (8PM to 9:30AM)
            ]
        elif period in ["1w", "1mo"]:
            # For weekly/monthly charts, only hide weekends
            rangebreaks = [
                dict(bounds=["sat", "mon"]),  # Hide weekends
            ]
        # For 1y charts, no rangebreaks to avoid issues with daily data
    
    fig.update_layout(
        title=title, 
        template="plotly_dark",
        paper_bgcolor="#1E293B",
        plot_bgcolor="#1E293B",
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            type='date',
            rangeslider_visible=False,
            rangebreaks=rangebreaks
        ),
        yaxis_title='Price', 
        yaxis2=dict(title='Volume', overlaying='y', side='right'),
        height=400, 
        margin=dict(l=40,r=40,t=40,b=40), 
        autosize=True
    )
    if error_msg:
        fig.add_annotation(x=0.5,y=0.1,xref='paper',yref='paper',text=f"DEMO DATA: {error_msg}",
                           showarrow=False,font=dict(color='red',size=12),
                           bgcolor='rgba(255,255,255,0.7)',bordercolor='red',borderwidth=1)
    return fig


def create_welcome_chart():
    """Show a quiet empty state until an analysis supplies chart data."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1E293B",
        plot_bgcolor="#1E293B",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="Your chart will appear here<br>after you start an analysis.",
            showarrow=False, font=dict(color="#94A3B8", size=14),
        )],
        height=400, margin=dict(l=20, r=20, t=20, b=20), autosize=True,
    )
    return fig
