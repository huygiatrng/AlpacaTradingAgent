# AlpacaTradingAgent: Enhanced Multi-Agent Alpaca Trading Framework

> 🚀 **AlpacaTradingAgent** - An independent enhanced version built upon the original TradingAgents framework, specifically designed for Alpaca users who want to test or use AI agents to trade on their Alpaca accounts.
>
> This project is an independent upgrade inspired by the original [TradingAgents](https://github.com/TauricResearch/TradingAgents) framework by Tauric Research, extending it with real-time Alpaca integration, crypto support, automated trading capabilities, and an enhanced web interface.
> 
> **Disclaimer**: This project is provided solely for educational and research purposes. It is not financial, investment, or trading advice. Trading involves risk, and users should conduct their own due diligence before making any trading decisions.

<div align="center">

🚀 [Enhanced Features](#enhanced-features) | ⚡ [Installation & Setup](#installation-and-setup) | 📦 [Package Usage](#alpacatradingagent-package) | 🌐 [Web Interface](#web-ui-usage) | 📖 [Complete Guide](#complete-guide) | 🤝 [Contributing](#contributing) | 📄 [Citation](#citation)

</div>

## Enhanced Features

AlpacaTradingAgent introduces powerful new capabilities specifically designed for Alpaca users:

### 🔄 **Real-Time Alpaca Integration**
- **Live Trading**: Direct integration with Alpaca API for real-time trading execution
- **Paper & Live Trading**: Support for both paper trading (testing) and live trading with real money
- **Margin Trading**: Full support for margin accounts, including short selling capabilities
- **Portfolio Management**: Real-time portfolio tracking, position monitoring, and order management

### 📈 **Dual Asset Support: Stocks & Crypto**
- **Multi-Asset Analysis**: Analyze both traditional stocks and cryptocurrencies in a single session
- **Crypto Format**: Use proper crypto format (e.g., `BTC/USD`, `ETH/USD`) for cryptocurrency analysis
- **Mixed Portfolios**: Support for mixed symbol inputs like `"NVDA, ETH/USD, AAPL"` for diversified analysis
- **Dedicated Data Sources**: CoinDesk/CryptoCompare-compatible crypto news and DeFi Llama for fundamental crypto data

### 🤖 **Enhanced Multi-Agent System (5 Agents)**
- **Market Analyst**: Evaluates overall market conditions and trends
- **Social Sentiment Analyst**: Analyzes social media sentiment and public opinion
- **News Analyst**: Monitors and interprets financial news and events
- **Fundamental Analyst**: Assesses company financials and intrinsic value
- **Macro Analyst**: Analyzes macroeconomic indicators and Federal Reserve data
- **Parallel Execution**: All 5 analysts run simultaneously for faster analysis with configurable delays to prevent API overload

### 🧠 **Multi-Provider LLM Runtime**
- **Default OpenAI GPT-5.4 Path**: Uses `gpt-5.4-nano` for quick agents and `gpt-5.4-mini` for deeper manager/trader agents
- **Provider Choice**: Supports OpenAI, local OpenAI-compatible endpoints, Google Gemini, Anthropic Claude, xAI, DeepSeek, Qwen, GLM, OpenRouter, Ollama, and Azure OpenAI
- **Provider-Specific Controls**: Preserves GPT reasoning controls, Gemini thinking level, Claude effort, custom model IDs, and Azure deployment names
- **Local Compatibility**: `OPENAI_USE_LOCAL` and `OPENAI_BASE_URL` continue to route core LLM calls to a local OpenAI-compatible backend

### 🧾 **Structured Decisions, Memory, and Resume**
- **Executable Final Action**: Final decisions preserve `BUY/HOLD/SELL` or `LONG/NEUTRAL/SHORT` for Alpaca execution
- **Advisory Ratings**: Upstream-style ratings are treated as metadata only and never directly trigger Alpaca orders
- **Structured Output Fallback**: Research Manager, Trader, and Risk Manager use structured schemas where supported and gracefully retry as free text otherwise
- **Persistent Decision Log**: Completed decisions are written to a markdown memory log and later resolved with realized returns and reflections
- **Checkpoint Resume**: Optional per-symbol SQLite checkpoints allow failed LangGraph runs to resume while successful runs clean up automatically
- **Safe Paths**: Report, cache, checkpoint, and log paths use safe ticker components, including crypto symbols like `BTC/USD -> BTC_USD`

### ⚡ **Automated Trading & Scheduling**
- **Market Hours Trading**: Automatic execution during market hours
- **Scheduled Analysis**: Configurable recurring analysis every N hours
- **Auto-Execution**: Optional automatic trade execution based on agent recommendations
- **Smart Scheduling**: Respects market hours for different asset classes

### 🌐 **Advanced Web Interface**
- **Multi-Symbol Dashboard**: Analyze and trade multiple symbols simultaneously
- **Progress Tracking**: Real-time progress table showing analysis status for each symbol
- **Interactive Charts**: Live Alpaca data integration with technical indicators
- **Tabbed Reports**: Organized analysis reports with easy navigation
- **Chat-Style Debates**: Visualize agent debates as conversation threads
- **Position Management**: View current positions, recent orders, and liquidate positions directly from UI
- **Model Configuration**: Choose provider, model, provider-specific parameters, output language, and checkpoint resume from the UI

## Complete Guide

For an in-depth, step-by-step walkthrough of using the AlpacaTradingAgent web UI for automated trading, check out the complete guide on Dev.to:

* **[Complete Guide: Using AlpacaTradingAgent Web UI for Automated Trading](https://dev.to/aarontrng/complete-guide-using-alpacatradingagent-web-ui-for-automated-trading-3k78)**

## AlpacaTradingAgent Framework

AlpacaTradingAgent is a multi-agent trading framework that mirrors the dynamics of real-world trading firms. By deploying specialized LLM-powered agents working collaboratively, the platform evaluates market conditions across multiple asset classes and executes informed trading decisions through the Alpaca API.

<p align="center">
  <img src="assets\schema.png" style="width: 100%; height: auto;">
</p>

> AlpacaTradingAgent framework is designed for research and educational purposes. Trading performance may vary based on many factors, including the chosen backbone language models, model temperature, trading periods, the quality of data, and other non-deterministic factors. [It is not intended as financial, investment, or trading advice.](https://tauric.ai/disclaimer/)

Our enhanced framework decomposes complex trading tasks into specialized roles while providing real-time market connectivity and execution capabilities.

### Enhanced Analyst Team (5 Agents)
- **Market Analyst**: Evaluates overall market conditions, sector trends, and market sentiment indicators
- **Social Sentiment Analyst**: Analyzes Reddit, OpenAI web-search sentiment, and public market narratives
- **News Analyst**: Monitors financial news, earnings announcements, and global events that impact markets
- **Fundamental Analyst**: Evaluates company financials, earnings reports, and intrinsic value calculations
- **Macro Analyst**: Analyzes Federal Reserve data, economic indicators, and macroeconomic trends using FRED API

### Researcher Team
- Comprises both bullish and bearish researchers who critically assess the insights provided by the Analyst Team. Through structured debates, they balance potential gains against inherent risks, now with enhanced support for both equity and crypto markets.

### Trader Agent
- Composes reports from analysts and researchers to make informed trading decisions. Determines timing, magnitude, and direction (long/short) of trades with direct execution through Alpaca API.

### Risk Management and Portfolio Manager
- Continuously evaluates portfolio risk across stocks and crypto assets. Monitors margin requirements, position sizes, and overall portfolio exposure. Provides real-time risk assessment and position management through the Alpaca integration.

## Installation and Setup

### Installation

Clone AlpacaTradingAgent:
```bash
git clone https://github.com/huygiatrng/AlpacaTradingAgent.git
cd AlpacaTradingAgent
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Required APIs Configuration

For full functionality including real-time trading, you'll need to set up the following API keys:

1. **Copy the sample environment file**:
   ```bash
   cp env.sample .env
   ```

2. **Edit the `.env` file** with your API keys:

#### Essential APIs
- **Alpaca API Keys** (Required for trading):
  - Sign up at [Alpaca Markets](https://app.alpaca.markets/signup)
  - Get your API key and secret from the dashboard
  - Set `ALPACA_USE_PAPER=True` for paper trading (recommended for testing)
  - Set `ALPACA_USE_PAPER=False` for live trading with real money

- **OpenAI API Key** (Default LLM provider and OpenAI web-search tools):
  - Sign up at [OpenAI Platform](https://platform.openai.com/api-keys)
  - Default models are `gpt-5.4-nano` and `gpt-5.4-mini`

#### LLM Provider APIs
Set `LLM_PROVIDER` in `.env`, the CLI, or the WebUI. Supported providers include:
- **OpenAI**: `OPENAI_API_KEY`
- **Local OpenAI-compatible**: `OPENAI_USE_LOCAL=true`, `OPENAI_BASE_URL`, optional `OPENAI_API_KEY`
- **Google Gemini**: `GOOGLE_API_KEY`
- **Anthropic Claude**: `ANTHROPIC_API_KEY`
- **xAI Grok**: `XAI_API_KEY`
- **DeepSeek**: `DEEPSEEK_API_KEY`
- **Qwen/DashScope**: `DASHSCOPE_API_KEY`
- **GLM/Zhipu**: `ZHIPU_API_KEY`
- **OpenRouter**: `OPENROUTER_API_KEY`
- **Ollama**: no API key by default; configure the backend URL
- **Azure OpenAI**: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME`, `AZURE_OPENAI_API_VERSION`

#### Financial Data APIs
- **Finnhub API Key** (Required for stock news and data):
  - Sign up at [Finnhub](https://finnhub.io/register)

- **FRED API Key** (Required for macro analysis):
  - Get your free key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html)

#### Crypto Data APIs
- **CoinDesk/CryptoCompare API Key** (Required for crypto news):
  - Sign up at [CryptoCompare](https://www.cryptocompare.com/cryptopian/api-keys)

#### Optional APIs
- **Alpha Vantage API Key** (Optional fallback market data):
  - Get from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
  - Fallback routing is optional and does not replace Alpaca as the primary market data path

#### Runtime Paths
`env.sample` also documents optional runtime paths:
- `TRADINGAGENTS_RESULTS_DIR` for report output
- `TRADINGAGENTS_CACHE_DIR` for cache and checkpoint files
- `TRADINGAGENTS_MEMORY_LOG_PATH` for persistent decision memory

3. **Restart the application** after setting up your API keys.

> **Note**: Without valid Alpaca API keys, the application will fall back to demo mode without trading capabilities.

### CLI Usage

You can try out the CLI by running:
```bash
python -m cli.main
```

The CLI now supports multiple symbols and crypto assets:
- Single stock: `NVDA`
- Single crypto: `BTC/USD`
- Multiple mixed assets: `NVDA, ETH/USD, AAPL, BTC/USD`
- Provider/model selection, custom model IDs, checkpoint resume, and provider-specific settings are available from the CLI prompts.

### Web UI Usage

Launch the enhanced Dash-based web interface:

```bash
python run_webui_dash.py
```

Common options:
- `--port PORT`: Specify a custom port (default: 7860)
- `--share`: Create a public link to share with others
- `--server-name`: Specify the server name/IP to bind to (default: 127.0.0.1)
- `--debug`: Run in debug mode with more logging
- `--max-threads N`: Set the maximum number of threads (default: 40)

or launch it with Docker:

```bash
cp env.sample .env
# Edit .env with your provider, market data, and Alpaca credentials first.
docker compose up -d --build
```

This starts a local web server at http://localhost:7860. To use a different
host port, set `HOST_PORT`, for example `HOST_PORT=7861 docker compose up -d --build`.

### Prompt Customization

Model-facing prompts live in `tradingagents/prompts/templates`. Edit those
Markdown templates to tune analyst, researcher, trader, risk, signal extraction,
and reflection behavior from one place. Templates are grouped by role:
`analysts/`, `researchers/`, `managers/`, `trader/`, `risk/`, `trading_modes/`,
`graph/`, and `shared/`.

To keep custom prompts outside the repo, copy selected templates to another
folder and set `TRADINGAGENTS_PROMPT_DIR` to that path. Keep the same group path
for overrides, for example `analysts/market_system.md`. Missing files fall back
to the bundled templates.

#### Enhanced Web UI Features

The web interface offers comprehensive trading and analysis capabilities:

**Multi-Asset Analysis Dashboard**
- Analyze multiple stocks and crypto assets simultaneously
- Real-time progress tracking for each symbol
- Support for mixed portfolios (e.g., `"NVDA, ETH/USD, AAPL"`)

<p align="center">
  <img src="assets\config_and_chart.png" style="width: 100%; height: auto;">
</p>

**Live Trading Integration**
- View current Alpaca positions and recent orders
- Execute trades directly from the interface
- Liquidate positions with one-click functionality
- Real-time portfolio value tracking

<p align="center">
  <img src="assets\portfolio.png" style="width: 100%; height: auto;">
</p>

**Interactive Charts & Data**
- Live price charts powered by Alpaca API
- Technical indicators and analysis overlays
- Support for both stock and crypto price data

**Enhanced Reporting Interface**
- Tabbed navigation for different analysis reports
- Chat-style conversation view for agent debates
- Progress table showing analysis status for each symbol
- Downloadable reports and trade recommendations

<p align="center">
  <img src="assets\reports.png" style="width: 100%; height: auto;">
</p>

**Automated Trading Controls**
- Schedule recurring analysis during market hours
- Configure auto-execution of trade recommendations
- Set custom analysis intervals (every N hours)
- Margin trading controls and risk management

**LLM and Runtime Controls**
- Select OpenAI, local OpenAI-compatible, Google, Anthropic, xAI, DeepSeek, Qwen, GLM, OpenRouter, Ollama, or Azure OpenAI
- Configure custom model IDs for compatible providers and Azure deployment names
- Tune GPT reasoning controls, Gemini thinking level, Claude effort, output language, and checkpoint resume

## AlpacaTradingAgent Package

### Implementation Details

Built with LangGraph for flexibility and modularity. The enhanced version integrates with multiple financial APIs and supports both paper and live trading through Alpaca. We recommend `gpt-5-nano` for the cheapest testing runs or `gpt-5.4-nano` for a newer low-cost default, as the framework makes numerous API calls across all 5 agents.

### Python Usage

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Initialize with default config
ta = TradingAgentsGraph(debug=True, config=DEFAULT_CONFIG.copy())

# Analyze a single stock
_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)

# Analyze multiple assets including crypto
symbols = ["NVDA", "ETH/USD", "AAPL"]
for symbol in symbols:
    _, decision = ta.propagate(symbol, "2024-05-10")
    print(f"{symbol}: {decision}")
```

### Custom Configuration

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Create custom config for enhanced features
config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = "gpt-5.4-mini"  # Balanced current default
config["quick_think_llm"] = "gpt-5.4-nano"  # New low-cost quick model
config["quick_llm_params"] = {
    "reasoning_effort": "low",
    "text_verbosity": "low",
    "reasoning_summary": "auto",
}
config["deep_llm_params"] = {
    "reasoning_effort": "medium",
    "text_verbosity": "medium",
    "reasoning_summary": "auto",
}
config["max_debate_rounds"] = 2  # Increase debate rounds
config["online_tools"] = True  # Use real-time data
config["allow_shorts"] = False  # Investment mode: BUY/HOLD/SELL
config["checkpoint_enabled"] = False  # Enable to resume failed graph runs
config["memory_log_path"] = "~/.tradingagents/memory/trading_memory.md"
config["news_global_openai_enabled"] = False  # Macro handles broad global context by default

# Parallel execution settings (to avoid API overload)
config["parallel_analysts"] = True  # Run analysts in parallel (default: True)
config["analyst_start_delay"] = 0.5  # Delay between starting each analyst (seconds)
config["analyst_call_delay"] = 0.1  # Delay before making analyst calls (seconds)
config["tool_result_delay"] = 0.2  # Delay between tool results and next call (seconds)

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# Analyze with crypto support
_, decision = ta.propagate("BTC/USD", "2024-05-10")
print(decision)
```

For non-OpenAI providers, switch the provider and model IDs:

```python
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "google"
config["quick_think_llm"] = "gemini-2.5-flash"
config["deep_think_llm"] = "gemini-3.1-pro-preview"
config["google_thinking_level"] = "high"

ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)
```

## Contributing

We welcome contributions from the community! AlpacaTradingAgent is an independent project that builds upon concepts from the original TradingAgents framework, continuously evolving with new features for Alpaca integration and multi-asset support.

## Acknowledgments

This project is inspired by and builds upon concepts from the original [TradingAgents](https://github.com/TauricResearch/TradingAgents) framework by Tauric Research. We extend our gratitude to the original authors for their pioneering work in multi-agent financial trading systems.

**AlpacaTradingAgent** is an independent project that focuses specifically on providing Alpaca users with a production-ready trading interface, real-time market connectivity, and expanded asset class support while implementing an enhanced multi-agent architecture.

## Citation

Please reference the original TradingAgents work that inspired this project:

```
@misc{xiao2025tradingagentsmultiagentsllmfinancial,
      title={TradingAgents: Multi-Agents LLM Financial Trading Framework}, 
      author={Yijia Xiao and Edward Sun and Di Luo and Wei Wang},
      year={2025},
      eprint={2412.20138},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2412.20138}, 
}
```
