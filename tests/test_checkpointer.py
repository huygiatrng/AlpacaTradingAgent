import tempfile
import unittest
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from tradingagents.graph.checkpointer import (
    clear_checkpoint,
    get_checkpointer,
    has_checkpoint,
    thread_id,
)


class CounterState(TypedDict):
    value: int


class CheckpointerTests(unittest.TestCase):
    def test_checkpoint_is_ticker_date_isolated_and_clearable(self):
        workflow = StateGraph(CounterState)
        workflow.add_node("increment", lambda state: {"value": state["value"] + 1})
        workflow.add_edge(START, "increment")
        workflow.add_edge("increment", END)

        with tempfile.TemporaryDirectory() as tmp:
            config = {"configurable": {"thread_id": thread_id("BTC/USD", "2026-01-02")}}
            with get_checkpointer(tmp, "BTC/USD") as checkpointer:
                graph = workflow.compile(checkpointer=checkpointer)
                self.assertEqual(graph.invoke({"value": 1}, config=config)["value"], 2)

            self.assertTrue(has_checkpoint(tmp, "BTC/USD", "2026-01-02"))
            self.assertFalse(has_checkpoint(tmp, "ETH/USD", "2026-01-02"))

            clear_checkpoint(tmp, "BTC/USD", "2026-01-02")
            self.assertFalse(has_checkpoint(tmp, "BTC/USD", "2026-01-02"))


if __name__ == "__main__":
    unittest.main()
