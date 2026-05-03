import unittest

from tradingagents.dataflows.utils import safe_ticker_component


class SafeTickerComponentTests(unittest.TestCase):
    def test_accepts_common_stock_and_crypto_symbols(self):
        cases = {
            "AAPL": "AAPL",
            "BRK.B": "BRK.B",
            "7203.T": "7203.T",
            "^GSPC": "^GSPC",
            "BTC/USD": "BTC_USD",
            "ETH-USD": "ETH-USD",
        }

        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(safe_ticker_component(raw), expected)

    def test_rejects_path_unsafe_values(self):
        bad_values = [
            "../AAPL",
            "AAPL ",
            "AA PL",
            "AAPL\x00",
            r"..\AAPL",
            "." * 2,
            "A" * 65,
        ]

        for value in bad_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    safe_ticker_component(value)


if __name__ == "__main__":
    unittest.main()
