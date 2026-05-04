# Prompt Templates

This folder contains the model-facing prompts used by the agent workflow.

Edit these Markdown files to tune agent behavior without hunting through the
Python graph code. Placeholders such as `{ticker}`, `{analysis_content}`, and
`{final_format}` are filled at runtime by the agent modules.

## Groups

- `shared/`: reusable collaboration and final recommendation scaffolding.
- `analysts/`: market, social, news, fundamentals, and macro analyst prompts.
- `researchers/`: bull and bear investment debate prompts.
- `managers/`: research manager and final risk manager prompts.
- `trader/`: trader system, plan, fallback, and final decision prompts.
- `risk/`: aggressive, conservative, and neutral risk debate prompts.
- `trading_modes/`: BUY/HOLD/SELL and LONG/NEUTRAL/SHORT mode instructions.
- `graph/`: signal extraction and reflection prompts.

To keep local prompt edits outside the repository, copy selected templates
elsewhere and set:

```bash
TRADINGAGENTS_PROMPT_DIR=/path/to/your/prompt/templates
```

The override directory only needs the files you want to override. Keep the same
group path, for example `analysts/market_system.md`. Missing templates fall
back to the bundled versions in this folder.
