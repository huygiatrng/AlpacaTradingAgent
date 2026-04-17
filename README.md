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
- **Dedicated Data Sources**: Coindesk API for crypto news and DeFi Llama for fundamental crypto data

### 🤖 **Enhanced Multi-Agent System (5 Agents)**
- **Market Analyst**: Evaluates overall market conditions and trends
- **Social Sentiment Analyst**: Analyzes social media sentiment and public opinion
- **News Analyst**: Monitors and interprets financial news and events
- **Fundamental Analyst**: Assesses company financials and intrinsic value
- **Macro Analyst**: Analyzes macroeconomic indicators and Federal Reserve data
- **Parallel Execution**: All 5 analysts run simultaneously for faster analysis with configurable delays to prevent API overload

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
- **Social Sentiment Analyst**: Analyzes Twitter, Reddit, and other social platforms to gauge market sentiment and momentum
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
git clone https://github.com/TauricResearch/TradingAgents.git
cd TradingAgents
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

- **OpenAI API Key** (Required for LLM agents):
  - Sign up at [OpenAI Platform](https://platform.openai.com/api-keys)

#### Financial Data APIs
- **Finnhub API Key** (Required for stock news and data):
  - Sign up at [Finnhub](https://finnhub.io/register)

- **FRED API Key** (Required for macro analysis):
  - Get your free key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html)

#### Crypto Data APIs
- **CoinDesk API Key** (Required for crypto news):
  - Sign up at [CryptoCompare](https://www.cryptocompare.com/cryptopian/api-keys)

#### Optional APIs
- **Twitter Bearer Token** (Optional for enhanced social sentiment):
  - Get from [Twitter Developer](https://developer.twitter.com/) for API-based sentiment analysis
  - The system can fallback to web scraping if not provided

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

or you can launch this with Docker via:
```bash
docker-compose up -d --build
```

These will start a local web server at http://localhost:7860.

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

## AlpacaTradingAgent Package

### Implementation Details

Built with LangGraph for flexibility and modularity. The enhanced version integrates with multiple financial APIs and supports both paper and live trading through Alpaca. We recommend using `gpt-5-mini` for testing to minimize API costs, as the framework makes numerous API calls across all 5 agents.

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
config["deep_think_llm"] = "gpt-5-mini"  # Cost-effective for testing
config["quick_think_llm"] = "gpt-5-mini"
config["max_debate_rounds"] = 2  # Increase debate rounds
config["online_tools"] = True  # Use real-time data
config["enable_margin_trading"] = True  # Allow short selling
config["auto_execute_trades"] = False  # Manual approval required

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
