import requests
import json
from datetime import datetime, timedelta
from typing import Annotated, Dict, List, Optional
from .config import get_api_key, DATA_DIR
import os
import pandas as pd


def get_finnhub_earnings_calendar(
    ticker: str,
    start_date: str,
    end_date: str
) -> str:
    """
    Get earnings calendar data from Finnhub API
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        Formatted string with earnings calendar data
    """
    try:
        from .finnhub_utils import get_finnhub_client
        client = get_finnhub_client()

        print(f"[EARNINGS] Fetching earnings calendar for {ticker} from {start_date} to {end_date}")

        # Convert dates to timestamps
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Get earnings calendar
        earnings_calendar = client.earnings_calendar(
            _from=start_date,
            to=end_date,
            symbol=ticker
        )

        print(f"[EARNINGS] API Response type: {type(earnings_calendar)}")
        print(f"[EARNINGS] API Response keys: {earnings_calendar.keys() if isinstance(earnings_calendar, dict) else 'N/A'}")

        if not isinstance(earnings_calendar, dict):
            error_msg = f"ERROR: Invalid API response type for {ticker}: {type(earnings_calendar)}"
            print(f"[EARNINGS] {error_msg}")
            return error_msg

        if 'error' in earnings_calendar:
            error_msg = f"ERROR: Finnhub API error for {ticker}: {earnings_calendar['error']}"
            print(f"[EARNINGS] {error_msg}")
            return error_msg

        if not earnings_calendar.get('earningsCalendar'):
            msg = f"No earnings data found for {ticker} between {start_date} and {end_date} - Finnhub free tier only provides 1 month of data"
            print(f"[EARNINGS] {msg}")
            return msg
        
        result = f"## {ticker} Earnings Calendar ({start_date} to {end_date})\n\n"
        
        for earning in earnings_calendar['earningsCalendar']:
            date = earning.get('date', 'N/A')
            eps_estimate = earning.get('epsEstimate', 'N/A')
            eps_actual = earning.get('epsActual', 'N/A')
            hour = earning.get('hour', 'N/A')
            quarter = earning.get('quarter', 'N/A')
            revenue_estimate = earning.get('revenueEstimate', 'N/A')
            revenue_actual = earning.get('revenueActual', 'N/A')
            year = earning.get('year', 'N/A')
            
            # Calculate surprise if both estimate and actual are available
            eps_surprise = "N/A"
            revenue_surprise = "N/A"
            
            if eps_actual != 'N/A' and eps_estimate != 'N/A' and eps_actual is not None and eps_estimate is not None:
                try:
                    eps_surprise_val = float(eps_actual) - float(eps_estimate)
                    eps_surprise_pct = (eps_surprise_val / float(eps_estimate)) * 100 if float(eps_estimate) != 0 else 0
                    eps_surprise = f"{eps_surprise_val:.4f} ({eps_surprise_pct:.2f}%)"
                except (ValueError, TypeError):
                    eps_surprise = "N/A"
            
            if revenue_actual != 'N/A' and revenue_estimate != 'N/A' and revenue_actual is not None and revenue_estimate is not None:
                try:
                    revenue_surprise_val = float(revenue_actual) - float(revenue_estimate)
                    revenue_surprise_pct = (revenue_surprise_val / float(revenue_estimate)) * 100 if float(revenue_estimate) != 0 else 0
                    revenue_surprise = f"{revenue_surprise_val:.0f} ({revenue_surprise_pct:.2f}%)"
                except (ValueError, TypeError):
                    revenue_surprise = "N/A"
            
            result += f"### {date} - Q{quarter} {year} ({hour})\n"
            result += f"- **EPS Estimate**: {eps_estimate}\n"
            result += f"- **EPS Actual**: {eps_actual}\n"
            result += f"- **EPS Surprise**: {eps_surprise}\n"
            result += f"- **Revenue Estimate**: {revenue_estimate}\n"
            result += f"- **Revenue Actual**: {revenue_actual}\n"
            result += f"- **Revenue Surprise**: {revenue_surprise}\n\n"

        print(f"[EARNINGS] Found {len(earnings_calendar['earningsCalendar'])} earnings entries")
        return result

    except Exception as e:
        error_msg = f"ERROR: Failed to fetch earnings data for {ticker}: {type(e).__name__}: {str(e)}"
        print(f"[EARNINGS] {error_msg}")
        import traceback
        print(f"[EARNINGS] Traceback: {traceback.format_exc()}")
        return error_msg


def get_earnings_calendar_yfinance(
    ticker: str,
    start_date: str,
    end_date: str
) -> str:
    """
    Get earnings calendar using yfinance as fallback

    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        Formatted earnings data string
    """
    try:
        import yfinance as yf
        from datetime import datetime

        print(f"[EARNINGS] Fetching earnings via yfinance for {ticker}")

        ticker_obj = yf.Ticker(ticker)

        # Get earnings dates
        earnings_dates = ticker_obj.earnings_dates

        if earnings_dates is None or earnings_dates.empty:
            return f"No earnings data found for {ticker} via yfinance"

        # Filter by date range
        # Handle timezone-aware indices from yfinance
        import pandas as pd
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)

        # Make timezone-aware if earnings_dates index is timezone-aware
        if earnings_dates.index.tz is not None:
            start_dt = start_dt.tz_localize(earnings_dates.index.tz)
            end_dt = end_dt.tz_localize(earnings_dates.index.tz)

        mask = (earnings_dates.index >= start_dt) & (earnings_dates.index <= end_dt)
        filtered = earnings_dates[mask]

        if filtered.empty:
            return f"No earnings data found for {ticker} between {start_date} and {end_date}"

        # Format results
        result = f"## {ticker} Earnings Calendar ({start_date} to {end_date})\n\n"
        result += f"**Data Source:** Yahoo Finance (via yfinance)\n\n"

        for date_idx, row in filtered.iterrows():
            date_str = date_idx.strftime("%Y-%m-%d")

            # Extract available data
            eps_estimate = row.get('EPS Estimate', 'N/A')
            eps_actual = row.get('Reported EPS', 'N/A')

            # Calculate surprise if both available
            surprise = "N/A"
            if eps_estimate != 'N/A' and eps_actual != 'N/A':
                try:
                    surprise_val = float(eps_actual) - float(eps_estimate)
                    surprise_pct = (surprise_val / float(eps_estimate)) * 100 if float(eps_estimate) != 0 else 0
                    surprise = f"{surprise_val:.4f} ({surprise_pct:+.2f}%)"
                except (ValueError, TypeError):
                    surprise = "N/A"

            result += f"### {date_str}\n"
            result += f"- **EPS Estimate**: {eps_estimate}\n"
            result += f"- **EPS Reported**: {eps_actual}\n"
            result += f"- **Surprise**: {surprise}\n\n"

        print(f"[EARNINGS] Found {len(filtered)} earnings entries via yfinance")
        return result

    except ImportError:
        return "ERROR: yfinance not installed. Run: pip install yfinance"
    except Exception as e:
        error_msg = f"ERROR: Failed to fetch earnings via yfinance for {ticker}: {type(e).__name__}: {str(e)}"
        print(f"[EARNINGS] {error_msg}")
        import traceback
        print(f"[EARNINGS] Traceback: {traceback.format_exc()}")
        return error_msg


def get_crypto_earnings_equivalent(
    ticker: str,
    start_date: str,
    end_date: str
) -> str:
    """
    Get crypto "earnings equivalent" data - major announcements, protocol upgrades, etc.
    
    Args:
        ticker: Crypto ticker symbol
        start_date: Start date in YYYY-MM-DD format  
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        Formatted string with crypto major events data
    """
    # Clean ticker (remove USD/USDT suffixes)
    crypto_symbol = ticker.upper()
    if "/" in crypto_symbol:
        crypto_symbol = crypto_symbol.split('/')[0]
    else:
        crypto_symbol = crypto_symbol.replace("USDT", "").replace("USD", "")
    
    # For crypto, we'll focus on major protocol events, partnerships, and developments
    # This is a simplified implementation - in production you'd want to integrate with
    # crypto-specific APIs like CoinGecko, CryptoCompare, or specialized crypto calendar APIs
    
    result = f"## {crypto_symbol} Major Events Calendar ({start_date} to {end_date})\n\n"
    result += "**Note**: Crypto assets don't have traditional earnings. This shows major protocol events, partnerships, and developments that could impact price.\n\n"
    
    # Common crypto events to look for
    crypto_events = {
        "BTC": [
            "Bitcoin halving events",
            "Major institutional adoptions",
            "Regulatory developments",
            "Lightning Network updates"
        ],
        "ETH": [
            "Ethereum network upgrades",
            "EIP implementations", 
            "Staking developments",
            "Layer 2 integrations"
        ],
        "ADA": [
            "Cardano protocol upgrades",
            "Smart contract developments",
            "Governance proposals"
        ],
        "SOL": [
            "Solana network upgrades",
            "DeFi protocol launches",
            "NFT marketplace developments"
        ]
    }
    
    if crypto_symbol in crypto_events:
        result += f"### Key Event Categories for {crypto_symbol}:\n"
        for event in crypto_events[crypto_symbol]:
            result += f"- {event}\n"
        result += "\n"
    
    result += "### Recommendation:\n"
    result += f"For detailed {crypto_symbol} events and announcements, monitor:\n"
    result += f"- Official {crypto_symbol} social media and blogs\n"
    result += "- CoinGecko events calendar\n"
    result += "- CryptoCompare events\n"
    result += "- Protocol-specific governance forums\n\n"
    
    return result


def get_earnings_calendar_data(
    ticker: str,
    start_date: str,
    end_date: str,
    lookback_days: int = 90
) -> str:
    """
    Get comprehensive earnings calendar data for stocks or crypto events.
    Uses yfinance for stocks (full historical data, no API key needed).

    Args:
        ticker: Stock or crypto ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        lookback_days: How many days to look back for historical data

    Returns:
        Formatted string with earnings/events data
    """
    # Determine if this is a crypto or stock ticker
    crypto_indicators = ["BTC", "ETH", "ADA", "SOL", "DOGE", "MATIC", "AVAX", "DOT", "LINK", "UNI"]
    is_crypto = any(indicator in ticker.upper() for indicator in crypto_indicators) or "USD" in ticker.upper()

    if is_crypto:
        return get_crypto_earnings_equivalent(ticker, start_date, end_date)

    # Use yfinance as primary source for stocks
    print(f"[EARNINGS] Fetching earnings data for {ticker} using yfinance")
    return get_earnings_calendar_yfinance(ticker, start_date, end_date)


def get_earnings_surprises_analysis(
    ticker: str,
    curr_date: str,
    lookback_quarters: int = 8
) -> str:
    """
    Analyze historical earnings surprises to identify patterns

    Args:
        ticker: Stock ticker symbol
        curr_date: Current date in YYYY-MM-DD format
        lookback_quarters: Number of quarters to analyze

    Returns:
        Analysis of earnings surprise patterns
    """
    try:
        print(f"[EARNINGS] Analyzing earnings surprises for {ticker} with {lookback_quarters} quarters lookback")

        # Calculate date range (approximately 2 years for 8 quarters)
        current_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        start_dt = current_dt - timedelta(days=lookback_quarters * 90)  # Rough quarter approximation
        start_date = start_dt.strftime("%Y-%m-%d")

        # Get earnings data using yfinance
        earnings_data = get_earnings_calendar_yfinance(ticker, start_date, curr_date)

        if "No earnings data found" in earnings_data or "ERROR:" in earnings_data:
            print(f"[EARNINGS] No valid earnings data found for analysis")
            return earnings_data
        
        result = f"## {ticker} Earnings Surprise Analysis\n\n"
        result += f"**Analysis Period**: {start_date} to {curr_date} (Last {lookback_quarters} quarters)\n\n"
        
        # Add pattern analysis
        result += "### Key Patterns to Monitor:\n"
        result += "- **Consistency**: Does the company consistently beat/miss estimates?\n"
        result += "- **Magnitude**: How significant are the surprises (>5% is typically material)?\n"
        result += "- **Revenue vs EPS**: Which metric shows more volatility?\n"
        result += "- **Seasonal Patterns**: Are certain quarters historically stronger?\n\n"
        
        result += "### Trading Implications:\n"
        result += "- **Pre-earnings**: High surprise history = higher IV and option premiums\n"
        result += "- **Post-earnings**: Surprise magnitude often correlates with price movement\n"
        result += "- **Guidance**: Forward guidance often more important than backward-looking results\n\n"
        
        result += earnings_data

        print(f"[EARNINGS] Successfully completed earnings surprise analysis")
        return result

    except Exception as e:
        error_msg = f"ERROR: Failed to analyze earnings surprises for {ticker}: {type(e).__name__}: {str(e)}"
        print(f"[EARNINGS] {error_msg}")
        import traceback
        print(f"[EARNINGS] Traceback: {traceback.format_exc()}")
        return error_msg 