import tempfile
import unittest
from pathlib import Path

from tradingagents.agents.utils.memory import TradingMemoryLog


FINAL_BUY = """Thesis body.

**Advisory Rating**: Overweight

FINAL TRANSACTION PROPOSAL: **BUY**"""


class TradingMemoryLogTests(unittest.TestCase):
    def test_store_dedupe_and_resolve_equity_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.md"
            log = TradingMemoryLog({"memory_log_path": str(path), "memory_log_max_entries": 3})

            log.store_decision("AAPL", "2026-01-02", FINAL_BUY)
            log.store_decision("AAPL", "2026-01-02", FINAL_BUY)

            entries = log.load_entries()
            self.assertEqual(len(entries), 1)
            self.assertTrue(entries[0]["pending"])
            self.assertEqual(entries[0]["action"], "BUY")
            self.assertEqual(entries[0]["rating"], "Overweight")

            log.update_with_outcome(
                ticker="AAPL",
                trade_date="2026-01-02",
                raw_return=0.04,
                alpha_return=0.01,
                holding_days=5,
                reflection="The setup worked.",
            )

            entries = log.load_entries()
            self.assertFalse(entries[0]["pending"])
            self.assertEqual(entries[0]["raw"], "+4.0%")
            self.assertEqual(entries[0]["alpha"], "+1.0%")
            self.assertIn("The setup worked.", log.get_past_context("AAPL"))

    def test_rotation_keeps_pending_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.md"
            log = TradingMemoryLog({"memory_log_path": str(path), "memory_log_max_entries": 1})

            log.store_decision("AAPL", "2026-01-01", FINAL_BUY)
            log.store_decision("MSFT", "2026-01-02", FINAL_BUY)
            log.update_with_outcome("AAPL", "2026-01-01", 0.02, None, 5, "First resolved.")

            entries = log.load_entries()
            self.assertEqual(len(entries), 2)
            self.assertEqual(len([entry for entry in entries if entry["pending"]]), 1)


if __name__ == "__main__":
    unittest.main()
