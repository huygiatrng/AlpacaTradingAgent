"""
How to Run AlpacaTradingAgent with a Local LLM

This project can route the graph's quick/deep thinker LLM calls to an
OpenAI-compatible local endpoint such as LM Studio, Ollama, or vLLM.

What local mode covers:
- the main quick/deep thinker chat models used by the graph
- reflection-memory embeddings when your endpoint exposes an embeddings API

What local mode does not replace:
- OpenAI web-search tools in `tradingagents/dataflows/interface.py`

Those tools still rely on OpenAI cloud features. If you want a fully local run,
set `online_tools=False` so analysts avoid OpenAI web-search calls.

Environment-variable setup

Create a `.env` file with:

```env
OPENAI_USE_LOCAL=true
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_API_KEY=local-llm

# Optional when your endpoint exposes embeddings under a custom model name
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
```

Model configuration

You should also set `quick_think_llm` and `deep_think_llm` to model names your
local server actually exposes. Do not leave cloud-only names in place if your
server does not provide them.

Example:

```python
from tradingagents.dataflows.config import set_config

set_config(
    {
        "openai_use_local": True,
        "openai_base_url": "http://localhost:1234/v1",
        "quick_think_llm": "qwen2.5-7b-instruct",
        "deep_think_llm": "qwen2.5-14b-instruct",
        "online_tools": False,
    }
)
```

Supported endpoint patterns

1. LM Studio
   - `http://localhost:1234/v1`

2. Ollama (OpenAI-compatible mode)
   - `http://localhost:11434/v1`

3. vLLM OpenAI server
   - `http://localhost:8000/v1`

Embeddings behavior

Reflection memory uses embeddings. If your local endpoint does not expose a
compatible embeddings API, the app now degrades gracefully by skipping memory
lookups/additions for that run instead of crashing the graph.
"""
