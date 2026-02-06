# alpaca_utils.py

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Annotated, Union, Optional, List
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest, StockLatestQuoteRequest, CryptoLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest, GetOrdersRequest, MarketOrderRequest, ClosePositionRequest
from alpaca.trading.enums import AssetClass, OrderSide, TimeInForce
from .config import get_api_key


# Fallback dictionary for company names
ticker_to_company_fallback = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "NVDA": "Nvidia",
    "TSM": "Taiwan Semiconductor Manufacturing Company OR TSMC",
    "JPM": "JPMorgan Chase OR JP Morgan",
    "JNJ": "Johnson & Johnson OR JNJ",
    "V": "Visa",
    "WMT": "Walmart",
    "META": "Meta OR Facebook",
    "AMD": "AMD",
    "INTC": "Intel",
    "QCOM": "Qualcomm",
    "BABA": "Alibaba",
    "ADBE": "Adobe",
    "NFLX": "Netflix",
    "CRM": "Salesforce",
    "PYPL": "PayPal",
    "PLTR": "Palantir",
    "MU": "Micron",
    "SQ": "Block OR Square",
    "ZM": "Zoom",
    "CSCO": "Cisco",
    "SHOP": "Shopify",
    "ORCL": "Oracle",
    "X": "Twitter OR X",
    "SPOT": "Spotify",
    "AVGO": "Broadcom",
    "ASML": "ASML ",
    "TWLO": "Twilio",
    "SNAP": "Snap Inc.",
    "TEAM": "Atlassian",
    "SQSP": "Squarespace",
    "UBER": "Uber",
    "ROKU": "Roku",
    "PINS": "Pinterest",
}


def get_alpaca_stock_client() -> StockHistoricalDataClient:
    api_key = get_api_key("alpaca_api_key", "ALPACA_API_KEY")
    api_secret = get_api_key("alpaca_secret_key", "ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        print(f"Warning: Missing Alpaca API credentials. API key: {'present' if api_key else 'missing'}, Secret: {'present' if api_secret else 'missing'}")
        raise ValueError("Alpaca API key or secret not found. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY.")
    try:
        return StockHistoricalDataClient(api_key, api_secret)
    except Exception as e:
        print(f"Error creating Alpaca stock client: {e}")
        raise


def get_alpaca_crypto_client() -> CryptoHistoricalDataClient:
    api_key = get_api_key("alpaca_api_key", "ALPACA_API_KEY")
    api_secret = get_api_key("alpaca_secret_key", "ALPACA_SECRET_KEY")
    # Crypto calls work without keys, but keys raise rate limits
    if api_key and api_secret:
        return CryptoHistoricalDataClient(api_key, api_secret)
    else:
        return CryptoHistoricalDataClient()


def get_alpaca_trading_client() -> TradingClient:
    api_key = get_api_key("alpaca_api_key", "ALPACA_API_KEY")
    api_secret = get_api_key("alpaca_secret_key", "ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        raise ValueError("Alpaca API key or secret not found. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY.")
    return TradingClient(api_key, api_secret, paper=True)


def _parse_timeframe(tf: Union[str, TimeFrame]) -> TimeFrame:
    """Convert a string like '5Min' or a TimeFrame instance into a TimeFrame."""
    if isinstance(tf, TimeFrame):
        return tf

    tf = tf.strip()
    low = tf.lower()
    
    # mapping common strings
    if low == "1min":
        result = TimeFrame.Minute
    elif low.endswith("min"):
        # e.g. "5Min", "15min"
        amount = int(tf[:-3])
        result = TimeFrame(amount, TimeFrameUnit.Minute)
    elif low == "1hour" or low == "1h":
        result = TimeFrame.Hour
    elif low.endswith("hour"):
        amount = int(tf[:-4])
        result = TimeFrame(amount, TimeFrameUnit.Hour)
    elif low.endswith("h") and low[:-1].isdigit():
        # shorthand: "4h", "2h", etc.
        amount = int(low[:-1])
        result = TimeFrame(amount, TimeFrameUnit.Hour)
    elif low == "1day" or low == "1d":
        result = TimeFrame.Day
    elif low.endswith("day"):
        amount = int(tf[:-3])
        result = TimeFrame(amount, TimeFrameUnit.Day)
    elif low.endswith("d") and low[:-1].isdigit():
        # shorthand: "2d", "3d", etc.
        amount = int(low[:-1])
        result = TimeFrame(amount, TimeFrameUnit.Day)
    else:
        # fallback
        result = TimeFrame.Day
    
    return result


class AlpacaUtils:

    @staticmethod
    def get_stock_data(
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Optional[Union[str, datetime]] = None,
        timeframe: Union[str, TimeFrame] = "1Day",
        save_path: Optional[str] = None,
        feed: DataFeed = DataFeed.IEX
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a stock or crypto symbol.

        Args:
            symbol: The ticker symbol (e.g. "SPY" or "BTC/USD")
            start_date: 'YYYY-MM-DD' string or datetime
            end_date: optional 'YYYY-MM-DD' string or datetime
            timeframe: e.g. "1Min","5Min","15Min","1Hour","1Day" or a TimeFrame instance
            save_path: if provided, path to write a CSV
            feed: DataFeed enum (default IEX)

        Returns:
            pandas DataFrame with columns ['timestamp','open','high','low','close','volume']
        """
        # normalize dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date) + timedelta(days=1) if end_date else None

        tf = _parse_timeframe(timeframe)

        # choose client
        is_crypto = "/" in symbol
        client = get_alpaca_crypto_client() if is_crypto else get_alpaca_stock_client()

        # build request params; always use a list for symbol_or_symbols
        params = (
            CryptoBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=tf,
                start=start,
                end=end,
                feed=feed
            ) if is_crypto else
            StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=tf,
                start=start,
                end=end,
                feed=feed
            )
        )

        try:
            bars = client.get_crypto_bars(params) if is_crypto else client.get_stock_bars(params)
            # convert to DataFrame via the .df property
            df = bars.df.reset_index()  # multi-index ['symbol','timestamp']
            
            # filter for our symbol (in case of list) - only if symbol column exists
            if "symbol" in df.columns:
                df = df[df["symbol"] == symbol].drop(columns="symbol")
            else:
                # If no symbol column, assume all data is for the requested symbol
                pass
                
            if save_path:
                df.to_csv(save_path, index=False)
            return df

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_latest_quote(symbol: str) -> dict:
        """
        Get the latest bid/ask quote for a symbol.
        """
        is_crypto = "/" in symbol
        client = get_alpaca_crypto_client() if is_crypto else get_alpaca_stock_client()
        req = CryptoLatestQuoteRequest(symbol_or_symbols=[symbol]) if is_crypto else StockLatestQuoteRequest(symbol_or_symbols=[symbol])
        try:
            resp = client.get_crypto_latest_quote(req) if is_crypto else client.get_stock_latest_quote(req)
            quote = resp[symbol]
            return {
                "symbol": symbol,
                "bid_price": quote.bid_price,
                "bid_size": quote.bid_size,
                "ask_price": quote.ask_price,
                "ask_size": quote.ask_size,
                "timestamp": quote.timestamp
            }
        except Exception as e:
            print(f"Error fetching latest quote for {symbol}: {e}")
            return {}

    
    @staticmethod
    def get_stock_data_window(
        symbol: Annotated[str, "ticker symbol"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"] = None,
        look_back_days: Annotated[int, "Number of days to look back"] = 30,
        timeframe: Annotated[str, "Timeframe for data: 1Min, 5Min, 15Min, 1Hour, 1Day"] = "1Day",
    ) -> pd.DataFrame:
        """
        Fetches historical stock data from Alpaca for the specified symbol and a window of days.
        
        Args:
            symbol: The stock ticker symbol
            curr_date: Current date in yyyy-mm-dd format (optional - if not provided, will use today's date)
            look_back_days: Number of days to look back
            timeframe: Timeframe for data (1Min, 5Min, 15Min, 1Hour, 1Day)
            
        Returns:
            DataFrame containing the historical stock data
        """
        # Calculate start date based on look_back_days
        if curr_date:
            curr_dt = pd.to_datetime(curr_date)
        else:
            curr_dt = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
            
        start_dt = curr_dt - pd.Timedelta(days=look_back_days)
        
        # Don't pass end_date to avoid subscription limitations
        return AlpacaUtils.get_stock_data(
            symbol=symbol,
            start_date=start_dt.strftime("%Y-%m-%d"),
            timeframe=timeframe
        ) 

    @staticmethod
    def get_company_name(symbol: str) -> str:
        """
        Get company name for a ticker symbol using Alpaca API.
        
        Args:
            symbol: The ticker symbol (e.g. "AAPL")
            
        Returns:
            Company name as string or original symbol if not found
        """
        try:
            # Skip crypto or symbols with special characters
            if "/" in symbol:
                return symbol
                
            client = get_alpaca_trading_client()
            asset = client.get_asset(symbol)
            
            if asset and hasattr(asset, 'name') and asset.name:
                return asset.name
            else:
                # Use fallback if name is not available
                print(f"No company name found for {symbol} via API, using fallback.")
                return ticker_to_company_fallback.get(symbol, symbol)
                
        except Exception as e:
            print(f"Error fetching company name for {symbol}: {e}")
            print("This might be due to invalid API keys or insufficient permissions.")
            print("If you recently reset your paper trading account, you may need to generate new API keys.")
            return ticker_to_company_fallback.get(symbol, symbol) 

    @staticmethod
    def get_positions_data():
        """Get current positions from Alpaca account"""
        try:
            client = get_alpaca_trading_client()
            positions = client.get_all_positions()
            
            # Convert positions to a list of dictionaries
            positions_data = []
            for position in positions:
                current_price = float(position.current_price)
                avg_entry_price = float(position.avg_entry_price)
                qty = float(position.qty)
                market_value = float(position.market_value)
                cost_basis = avg_entry_price * qty
                
                # Calculate P/L values
                today_pl_dollars = float(position.unrealized_intraday_pl)
                total_pl_dollars = float(position.unrealized_pl)
                today_pl_percent = (today_pl_dollars / cost_basis) * 100 if cost_basis != 0 else 0
                total_pl_percent = (total_pl_dollars / cost_basis) * 100 if cost_basis != 0 else 0
                
                positions_data.append({
                    "Symbol": position.symbol,
                    "Qty": qty,
                    "Market Value": f"${market_value:.2f}",
                    "Avg Entry": f"${avg_entry_price:.2f}",
                    "Cost Basis": f"${cost_basis:.2f}",
                    "Today's P/L (%)": f"{today_pl_percent:.2f}%",
                    "Today's P/L ($)": f"${today_pl_dollars:.2f}",
                    "Total P/L (%)": f"{total_pl_percent:.2f}%",
                    "Total P/L ($)": f"${total_pl_dollars:.2f}"
                })
            
            return positions_data
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return []

    @staticmethod
    def get_recent_orders(page=1, page_size=7):
        """Get recent orders from Alpaca account, with simple pagination."""
        try:
            client = get_alpaca_trading_client()
            req = GetOrdersRequest(status="all", limit=page_size * page, nested=False)
            orders_page = client.get_orders(req)
            orders = list(orders_page)

            # Convert orders to a list of dictionaries
            orders_data = []
            for order in orders:
                qty = float(order.qty) if order.qty is not None else 0.0
                filled_qty = float(order.filled_qty) if order.filled_qty is not None else 0.0
                filled_avg_price = float(order.filled_avg_price) if order.filled_avg_price is not None else 0.0

                orders_data.append({
                    "Asset": order.symbol,
                    "Order Type": order.type,
                    "Side": order.side,
                    "Qty": qty,
                    "Filled Qty": filled_qty,
                    "Avg. Fill Price": f"${filled_avg_price:.2f}" if filled_avg_price > 0 else "-",
                    "Status": order.status,
                    "Source": order.client_order_id
                })

            # Now slice out the exact page we want (newest first)
            start = (page - 1) * page_size
            return orders_data[start : start + page_size]

        except Exception as e:
            print(f"Error fetching orders: {e}")
            return []

    @staticmethod
    def get_account_info():
        """Get account information from Alpaca"""
        try:
            client = get_alpaca_trading_client()
            account = client.get_account()
            
            # Extract the required values
            buying_power = float(account.buying_power)
            cash = float(account.cash)
            
            # Calculate daily change
            equity = float(account.equity)
            last_equity = float(account.last_equity)
            daily_change_dollars = equity - last_equity
            daily_change_percent = (daily_change_dollars / last_equity) * 100 if last_equity != 0 else 0
            
            return {
                "buying_power": buying_power,
                "cash": cash,
                "daily_change_dollars": daily_change_dollars,
                "daily_change_percent": daily_change_percent
            }
        except Exception as e:
            print(f"Error fetching account info: {e}")
            return {
                "buying_power": 0,
                "cash": 0,
                "daily_change_dollars": 0,
                "daily_change_percent": 0
            } 

    @staticmethod
    def get_current_position_state(symbol: str) -> str:
        """Return current position state for a symbol in the Alpaca account.

        Args:
            symbol: Ticker symbol (e.g. "AAPL" or "BTC/USD").  Crypto symbols will
                    be treated the same way as equities – a positive quantity is
                    considered a *LONG* position while a negative quantity (should
                    Alpaca ever allow it) is considered *SHORT*.

        Returns:
            One of "LONG", "SHORT", or "NEUTRAL" if no open position exists or we
            encounter an error.
        """
        try:
            # Skip if credentials are missing – the helper will raise inside but we
            # want to fail gracefully and just assume no position.
            client = get_alpaca_trading_client()

            # `get_all_positions()` is more broadly supported across Alpaca
            # versions than `get_position(symbol)` and avoids raising when the
            # asset is not found.
            positions = client.get_all_positions()

            # Normalise the requested symbol for comparisons – Alpaca symbols
            # for crypto may use different formats, so we normalize for position comparison only.
            requested_symbol_key = symbol.upper().replace("/", "")

            for pos in positions:
                if pos.symbol.upper() == requested_symbol_key:
                    try:
                        qty = float(pos.qty)
                    except (ValueError, AttributeError):
                        qty = 0.0

                    if qty > 0:
                        return "LONG"
                    elif qty < 0:
                        return "SHORT"
                    else:
                        # Zero quantity technically shouldn't appear but treat as
                        # neutral just in case.
                        return "NEUTRAL"
            # If we fall through the loop there is no open position for symbol.
            return "NEUTRAL"
        except Exception as e:
            # Log and default to neutral so agent prompts still work.
            print(f"Error determining current position for {symbol}: {e}")
            return "NEUTRAL"

    @staticmethod
    def place_market_order(symbol: str, side: str, notional: float = None, qty: float = None) -> dict:
        """
        Place a market order with Alpaca
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            side: "buy" or "sell"
            notional: Dollar amount to buy/sell (for fractional shares)
            qty: Number of shares (if not using notional)
            
        Returns:
            Dictionary with order result information
        """
        try:
            client = get_alpaca_trading_client()
            
            # Normalize symbol for Alpaca (remove "/" for crypto)
            alpaca_symbol = symbol.upper().replace("/", "")
            
            # Determine order side
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            
            # Determine proper time-in-force: crypto orders only allow GTC
            is_crypto = "/" in symbol.upper()
            tif = TimeInForce.GTC if is_crypto else TimeInForce.DAY

            # Create market order request
            if notional and notional > 0:
                # Use notional (dollar amount) for fractional shares
                order_request = MarketOrderRequest(
                    symbol=alpaca_symbol,
                    side=order_side,
                    time_in_force=tif,
                    notional=notional
                )
            elif qty and qty > 0:
                # Use quantity (number of shares)
                order_request = MarketOrderRequest(
                    symbol=alpaca_symbol,
                    side=order_side,
                    time_in_force=tif,
                    qty=qty
                )
            else:
                return {"success": False, "error": "Must specify either notional or qty"}
            
            # Submit the order
            order = client.submit_order(order_request)
            
            return {
                "success": True,
                "order_id": order.id,
                "symbol": order.symbol,
                "side": order.side,
                "qty": float(order.qty) if order.qty else None,
                "notional": float(order.notional) if order.notional else None,
                "status": order.status,
                "message": f"Successfully placed {side} order for {symbol}"
            }
            
        except Exception as e:
            error_msg = f"Error placing {side} order for {symbol}: {e}"
            print(error_msg)
            return {"success": False, "error": error_msg}

    @staticmethod
    def close_position(symbol: str, percentage: float = 100.0) -> dict:
        """
        Close a position (partially or completely)
        
        Args:
            symbol: Stock symbol
            percentage: Percentage of position to close (default 100% = full close)
            
        Returns:
            Dictionary with close result information
        """
        try:
            client = get_alpaca_trading_client()
            
            # Normalize symbol for Alpaca
            alpaca_symbol = symbol.upper().replace("/", "")
            
            # For full position close (100%), don't specify percentage - let Alpaca close entire position
            if percentage >= 100.0:
                # Close the entire position without specifying percentage
                order = client.close_position(alpaca_symbol)
            else:
                # Create close position request for partial close
                close_request = ClosePositionRequest(
                    percentage=str(percentage / 100.0)  # Convert percentage to decimal string
                )
                order = client.close_position(alpaca_symbol, close_request)
            
            return {
                "success": True,
                "order_id": order.id,
                "symbol": order.symbol,
                "side": order.side,
                "qty": float(order.qty) if order.qty else None,
                "status": order.status,
                "message": f"Successfully closed {percentage}% of {symbol} position"
            }
            
        except Exception as e:
            error_msg = f"Error closing position for {symbol}: {e}"
            print(error_msg)
            return {"success": False, "error": error_msg}

    @staticmethod
    def execute_trading_action(symbol: str, current_position: str, signal: str, 
                             dollar_amount: float, allow_shorts: bool = False) -> dict:
        """
        Execute trading action based on current position and signal
        
        Args:
            symbol: Stock symbol
            current_position: Current position state ("LONG", "SHORT", "NEUTRAL")
            signal: Trading signal from analysis
            dollar_amount: Dollar amount for trades
            allow_shorts: Whether short selling is allowed
            
        Returns:
            Dictionary with execution results
        """
        try:
            results = []
            
            # Helper to calculate integer quantity for any orders (used by both trading modes)
            def _calc_qty(sym: str, amount: float) -> int:
                """Return integer share qty based on latest quote price."""
                try:
                    quote = AlpacaUtils.get_latest_quote(sym)
                    price = quote.get("bid_price") or quote.get("ask_price")
                    if not price or price <= 0:
                        # Fallback: assume $1 to avoid div-by-zero; will raise later if Alpaca rejects
                        price = 1
                    qty = int(amount / price)
                    return max(qty, 1)
                except Exception:
                    # Fallback: at least 1 share
                    return 1
            
            if allow_shorts:
                # Trading mode: LONG/NEUTRAL/SHORT signals
                signal = signal.upper()
                
                if current_position == "LONG":
                    if signal == "LONG":
                        results.append({"action": "hold", "message": f"Keeping LONG position in {symbol}"})
                    elif signal == "NEUTRAL":
                        # Close LONG position
                        close_result = AlpacaUtils.close_position(symbol)
                        results.append({"action": "close_long", "result": close_result})
                    elif signal == "SHORT":
                        # Close LONG and open SHORT
                        close_result = AlpacaUtils.close_position(symbol)
                        results.append({"action": "close_long", "result": close_result})
                        if close_result.get("success"):
                            # Check if this is crypto - Alpaca doesn't support crypto short selling directly
                            is_crypto = "/" in symbol.upper()
                            if is_crypto:
                                error_msg = f"Direct short selling not supported for crypto assets like {symbol}. Position closed but short not opened."
                                results.append({"action": "open_short", "result": {"success": False, "error": error_msg}})
                            else:
                                # Calculate integer quantity for short (fractional shares cannot be shorted)
                                qty_int = _calc_qty(symbol, dollar_amount)
                                short_result = AlpacaUtils.place_market_order(symbol, "sell", qty=qty_int)
                                results.append({"action": "open_short", "result": short_result})
                
                elif current_position == "SHORT":
                    if signal == "SHORT":
                        results.append({"action": "hold", "message": f"Keeping SHORT position in {symbol}"})
                    elif signal == "NEUTRAL":
                        # Close SHORT position
                        close_result = AlpacaUtils.close_position(symbol)
                        results.append({"action": "close_short", "result": close_result})
                    elif signal == "LONG":
                        # Close SHORT and open LONG
                        close_result = AlpacaUtils.close_position(symbol)
                        results.append({"action": "close_short", "result": close_result})
                        if close_result.get("success"):
                            # Open LONG position - use notional amount for crypto, quantity for stocks
                            is_crypto = "/" in symbol.upper()
                            if is_crypto:
                                # For crypto, use exact dollar amount (notional)
                                long_result = AlpacaUtils.place_market_order(symbol, "buy", notional=dollar_amount)
                            else:
                                # For stocks, calculate quantity
                                qty_int = _calc_qty(symbol, dollar_amount)
                                long_result = AlpacaUtils.place_market_order(symbol, "buy", qty=qty_int)
                            results.append({"action": "open_long", "result": long_result})
                
                elif current_position == "NEUTRAL":
                    if signal == "LONG":
                        # Open LONG position - use notional amount for crypto, quantity for stocks
                        is_crypto = "/" in symbol.upper()
                        if is_crypto:
                            # For crypto, use exact dollar amount (notional)
                            long_result = AlpacaUtils.place_market_order(symbol, "buy", notional=dollar_amount)
                        else:
                            # For stocks, calculate quantity
                            qty_int = _calc_qty(symbol, dollar_amount)
                            long_result = AlpacaUtils.place_market_order(symbol, "buy", qty=qty_int)
                        results.append({"action": "open_long", "result": long_result})
                    elif signal == "SHORT":
                        # Check if this is crypto - Alpaca doesn't support crypto short selling directly
                        is_crypto = "/" in symbol.upper()
                        if is_crypto:
                            error_msg = f"Direct short selling not supported for crypto assets like {symbol}. Consider using derivatives or margin trading platforms."
                            results.append({"action": "open_short", "result": {"success": False, "error": error_msg}})
                        else:
                            # For stocks, attempt short selling
                            qty_int = _calc_qty(symbol, dollar_amount)
                            short_result = AlpacaUtils.place_market_order(symbol, "sell", qty=qty_int)
                            results.append({"action": "open_short", "result": short_result})
                    elif signal == "NEUTRAL":
                        results.append({"action": "hold", "message": f"No position needed for {symbol}"})
            
            else:
                # Investment mode: BUY/HOLD/SELL signals
                signal = signal.upper()
                has_position = current_position == "LONG"
                
                if signal == "BUY":
                    if has_position:
                        results.append({"action": "hold", "message": f"Already have position in {symbol}"})
                    else:
                        # Buy position - use notional amount for crypto, quantity for stocks
                        is_crypto = "/" in symbol.upper()
                        if is_crypto:
                            # For crypto, use exact dollar amount (notional)
                            buy_result = AlpacaUtils.place_market_order(symbol, "buy", notional=dollar_amount)
                        else:
                            # For stocks, calculate quantity
                            qty_int = _calc_qty(symbol, dollar_amount)
                            buy_result = AlpacaUtils.place_market_order(symbol, "buy", qty=qty_int)
                        results.append({"action": "buy", "result": buy_result})
                
                elif signal == "SELL":
                    if has_position:
                        # Sell position
                        sell_result = AlpacaUtils.close_position(symbol)
                        results.append({"action": "sell", "result": sell_result})
                    else:
                        results.append({"action": "hold", "message": f"No position to sell in {symbol}"})
                
                elif signal == "HOLD":
                    results.append({"action": "hold", "message": f"Holding current position in {symbol}"})
            
            # Check if any critical actions failed
            has_failures = False
            for action in results:
                if "result" in action and not action["result"].get("success", True):
                    has_failures = True
                    break
                    
            return {
                "success": not has_failures,
                "symbol": symbol,
                "current_position": current_position,
                "signal": signal,
                "actions": results
            }
            
        except Exception as e:
            error_msg = f"Error executing trading action for {symbol}: {e}"
            print(error_msg)
            return {"success": False, "error": error_msg} 