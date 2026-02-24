import pandas as pd
from stockstats import wrap
from typing import Annotated
import os
from .config import get_config
from .alpaca_utils import AlpacaUtils


class StockstatsUtils:
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[
            str, "quantitative indicators based off of the stock data for the company"
        ],
        curr_date: Annotated[
            str, "curr date for retrieving stock price data, YYYY-mm-dd"
        ],
        data_dir: Annotated[
            str,
            "directory where the stock data is stored.",
        ],
        online: Annotated[
            bool,
            "whether to use online tools to fetch data or offline tools. If True, will use online tools.",
        ] = False,
    ):
        df = None
        data = None

        # Sanitize symbol for filename (replace / with _)
        safe_symbol = symbol.replace('/', '_')

        # Unified cache-first API fallback approach (no online/offline distinction)
        # Parse the current date
        curr_date_dt = pd.to_datetime(curr_date)

        # Get more historical data to ensure proper technical indicator calculations
        # Technical indicators like 50 SMA need at least 50+ days of data
        end_date = pd.Timestamp.today()
        start_date = end_date - pd.DateOffset(days=365)  # Get 1 year of data for reliable indicators

        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        # Get config and ensure cache directory exists
        config = get_config()
        os.makedirs(config["data_cache_dir"], exist_ok=True)

        data_file = os.path.join(
            config["data_cache_dir"],
            f"{safe_symbol}-Alpaca-data-{start_date_str}-{end_date_str}.csv",
        )

        try:
            if os.path.exists(data_file):
                # Load cached data
                print(f"[STOCKSTATS] Loading cached data for {symbol}")
                data = pd.read_csv(data_file)
                if 'Date' in data.columns:
                    data["Date"] = pd.to_datetime(data["Date"])

                # Ensure lowercase aliases exist for cached data too
                required_cols_map = {
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                }
                for cap, low in required_cols_map.items():
                    if cap in data.columns and low not in data.columns:
                        data[low] = data[cap]
            else:
                # Cache miss - fetch from Alpaca API
                print(f"[STOCKSTATS] Cache miss, fetching {symbol} data from Alpaca API...")
                data = AlpacaUtils.get_stock_data(
                    symbol=symbol,  # Use original symbol for API call
                    start_date=start_date_str,
                    end_date=end_date_str,
                    timeframe="1Day"
                )

                # Ensure we have data
                if data.empty:
                    return f"N/A: No data available for {symbol}"

                # Clean data and handle duplicates to prevent reindex errors
                data = data.dropna()
                if 'date' in data.columns:
                    data = data.drop_duplicates(subset=['date'])
                data = data.reset_index(drop=True)

                # Standardize column names for stockstats
                if 'timestamp' in data.columns:
                    data = data.rename(columns={
                        'timestamp': 'Date',
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume'
                    })

                # -----------------------------------------------------------------
                # Ensure lowercase aliases exist for Stockstats calculations.
                # Stockstats expects lowercase column names like 'close' and 'volume'.
                # When we rename to capitalized versions for display purposes, the
                # original lowercase columns disappear, causing certain indicators
                # (e.g., OBV that relies on 'volume') to return NaN. We therefore
                # create lowercase duplicates without altering existing display columns.
                # -----------------------------------------------------------------
                required_cols_map = {
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                }
                for cap, low in required_cols_map.items():
                    if cap in data.columns and low not in data.columns:
                        data[low] = data[cap]

                # Ensure Date column is datetime
                if 'Date' in data.columns:
                    data["Date"] = pd.to_datetime(data["Date"])

                # Sort by date to ensure proper chronological order for indicators
                data = data.sort_values('Date').reset_index(drop=True)

                # Save to cache for future use
                print(f"[STOCKSTATS] Saving {symbol} data to cache...")
                data.to_csv(data_file, index=False)

            # Ensure we have sufficient data for technical indicators
            if len(data) < 100:
                return f"N/A: Insufficient data for {indicator} calculation (need at least 100 days, got {len(data)})"

            # Wrap with stockstats for technical indicator calculations
            df = wrap(data)

            # Validate data structure after wrap
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                print(f"[STOCKSTATS] WARNING: Missing required columns after wrap: {missing}")
                print(f"[STOCKSTATS] Available columns: {list(df.columns)}")

                # Trigger the indicator calculation
                # Handle problematic indicators that have issues with stockstats
                if indicator == 'obv':
                    try:
                        # Calculate OBV manually
                        obv_values = []
                        obv = 0
                        for i in range(len(data)):
                            if i == 0:
                                obv_values.append(0)
                            else:
                                if data['close'].iloc[i] > data['close'].iloc[i-1]:
                                    obv += data['volume'].iloc[i]
                                elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                                    obv -= data['volume'].iloc[i]
                                # If close == prev close, OBV stays the same
                                obv_values.append(obv)
                        
                        # Add OBV to the dataframe
                        df['obv'] = obv_values
                        indicator_series = df['obv']
                    except Exception as manual_error:
                        return f"N/A: Error calculating OBV manually: {str(manual_error)}"
                elif indicator == 'atr_14':
                    try:
                        # Calculate ATR manually
                        import numpy as np
                        
                        # Calculate True Range
                        tr_values = []
                        for i in range(len(data)):
                            if i == 0:
                                tr_values.append(data['high'].iloc[i] - data['low'].iloc[i])
                            else:
                                tr1 = data['high'].iloc[i] - data['low'].iloc[i]
                                tr2 = abs(data['high'].iloc[i] - data['close'].iloc[i-1])
                                tr3 = abs(data['low'].iloc[i] - data['close'].iloc[i-1])
                                tr_values.append(max(tr1, tr2, tr3))
                        
                        # Calculate 14-period ATR using simple moving average
                        atr_values = []
                        for i in range(len(tr_values)):
                            if i < 13:  # Not enough data for 14-period ATR
                                atr_values.append(np.nan)
                            else:
                                atr = np.mean(tr_values[i-13:i+1])
                                atr_values.append(atr)
                        
                        # Add ATR to the dataframe
                        df['atr_14'] = atr_values
                        indicator_series = df['atr_14']
                    except Exception as manual_error:
                        return f"N/A: Error calculating ATR manually: {str(manual_error)}"
                elif indicator.endswith('_ema'):
                    try:
                        # Parse EMA indicator (e.g., 'close_8_ema')
                        parts = indicator.split('_')
                        column = parts[0]
                        window = int(parts[1])
                        
                        # Calculate EMA manually
                        import numpy as np
                        alpha = 2.0 / (window + 1)
                        ema_values = []
                        
                        for i in range(len(data)):
                            if i == 0:
                                ema_values.append(data[column].iloc[i])
                            else:
                                ema = alpha * data[column].iloc[i] + (1 - alpha) * ema_values[i-1]
                                ema_values.append(ema)
                        
                        # Add EMA to the dataframe
                        df[indicator] = ema_values
                        indicator_series = df[indicator]
                    except Exception as manual_error:
                        return f"N/A: Error calculating EMA manually: {str(manual_error)}"
                elif indicator.endswith('_sma'):
                    try:
                        # Parse SMA indicator (e.g., 'close_50_sma')
                        parts = indicator.split('_')
                        column = parts[0]
                        window = int(parts[1])
                        
                        # Calculate SMA manually
                        import numpy as np
                        sma_values = []
                        
                        for i in range(len(data)):
                            if i < window - 1:
                                sma_values.append(np.nan)
                            else:
                                sma = np.mean(data[column].iloc[i-window+1:i+1])
                                sma_values.append(sma)
                        
                        # Add SMA to the dataframe
                        df[indicator] = sma_values
                        indicator_series = df[indicator]
                    except Exception as manual_error:
                        return f"N/A: Error calculating SMA manually: {str(manual_error)}"
                else:
                    # Try stockstats for other indicators
                    try:
                        indicator_series = df[indicator]
                    except KeyError:
                        return f"N/A: Invalid indicator '{indicator}'"
                    except Exception as e:
                        return f"N/A: Error calculating {indicator}: {str(e)}"

                # Convert date column to string for matching
                df["date_str"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
                curr_date_str = curr_date_dt.strftime("%Y-%m-%d")

                # Find the most recent trading day on or before the requested date
                available_dates = df["date_str"].tolist()
                matching_rows = df[df["date_str"] == curr_date_str]

                # Debug: Check if indicator column exists
                if indicator not in df.columns:
                    print(f"[STOCKSTATS] ERROR: Indicator '{indicator}' not found in DataFrame columns")
                    print(f"[STOCKSTATS] Available indicator columns: {[c for c in df.columns if c not in ['Date', 'date_str', 'open', 'high', 'low', 'close', 'volume']]}")
                    return f"N/A: Indicator '{indicator}' not calculated by stockstats"

                if not matching_rows.empty:
                    indicator_value = matching_rows[indicator].iloc[-1]  # Get the last (most recent) value
                    # Handle NaN values
                    if pd.isna(indicator_value):
                        return f"N/A: {indicator} not calculable for {curr_date_str}"
                    return float(indicator_value)
                else:
                    # If exact date not found, try to find the most recent trading day before the requested date
                    df_filtered = df[df["date_str"] <= curr_date_str]
                    if not df_filtered.empty:
                        most_recent = df_filtered.iloc[-1]
                        indicator_value = most_recent[indicator]
                        if pd.isna(indicator_value):
                            return f"N/A: {indicator} not calculable for most recent trading day"
                        actual_date = most_recent["date_str"]
                        return f"{float(indicator_value)} (as of {actual_date})"
                    else:
                        return f"N/A: No trading data available on or before {curr_date_str}"

        except Exception as e:
            print(f"[STOCKSTATS] Error fetching/processing data for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return f"N/A: Error processing data for {symbol}: {str(e)}" 