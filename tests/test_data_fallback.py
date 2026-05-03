import unittest
from unittest.mock import patch

import pandas as pd

from tradingagents.dataflows.alpaca_utils import AlpacaUtils
from tradingagents.dataflows.config import get_config, set_config


class DataFallbackTests(unittest.TestCase):
    def setUp(self):
        self.original_config = get_config()

    def tearDown(self):
        set_config(self.original_config)

    def test_yfinance_fallback_only_runs_when_enabled(self):
        set_config({**self.original_config, "data_fallback_enabled": False})
        with patch(
            "tradingagents.dataflows.alpaca_utils.get_alpaca_stock_client",
            side_effect=ValueError("Alpaca API key or secret not found."),
        ), patch("yfinance.download") as yf_download:
            result = AlpacaUtils.get_stock_data("AAPL", "2026-01-01", "2026-01-03")

        self.assertTrue(result.empty)
        yf_download.assert_not_called()

    def test_yfinance_fallback_normalizes_ohlcv_columns(self):
        fallback = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                "Open": [10.0, 11.0],
                "High": [12.0, 12.5],
                "Low": [9.5, 10.5],
                "Close": [11.5, 12.0],
                "Volume": [1000, 1100],
            }
        ).set_index("Date")

        set_config({**self.original_config, "data_fallback_enabled": True})
        with patch(
            "tradingagents.dataflows.alpaca_utils.get_alpaca_stock_client",
            side_effect=ValueError("Alpaca API key or secret not found."),
        ), patch("yfinance.download", return_value=fallback):
            result = AlpacaUtils.get_stock_data("AAPL", "2026-01-01", "2026-01-03")

        self.assertEqual(
            list(result.columns),
            ["timestamp", "open", "high", "low", "close", "volume"],
        )
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
