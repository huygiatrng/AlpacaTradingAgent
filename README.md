# AlpacaTradingAgent

A multi-agent LLM trading framework that connects to your Alpaca account. Specialized AI agents analyze market conditions, debate trade ideas, and execute orders through a Dash web interface.

> **Disclaimer**: This project is for educational and research purposes only. It is not financial, investment, or trading advice. Trading involves significant risk of loss.

---

## Architecture

The system is a [LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph` that routes agent state through a fixed pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Analyst Team                             │
│  Market · Social · News · Fundamentals · Macro  (sequential    │
│                                          or parallel)           │
└────────────────────────────┬────────────────────────────────────┘
                             │  reports fed into →
┌────────────────────────────▼────────────────────────────────────┐
│                    Investment Debate                             │
│   Bull Researcher  ←──debate──→  Bear Researcher                │
│                        ↓ judge                                  │
│                    Invest Judge                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │  investment recommendation →
┌────────────────────────────▼────────────────────────────────────┐
│                      Trader Agent                               │
│   Synthesizes analyst reports + debate outcome into a           │
│   structured trade plan (direction, sizing, stops, targets)     │
└────────────────────────────┬────────────────────────────────────┘
                             │  trade plan →
┌────────────────────────────▼────────────────────────────────────┐
│                       Risk Debate                               │
│   Aggressive ←──debate──→ Neutral ←──debate──→ Conservative    │
│                        ↓ judge                                  │
│                     Risk Manager                                │
└────────────────────────────┬────────────────────────────────────┘
                             │  approved plan →
                    Alpaca order execution
```

### Agents

| Agent | Role |
|-------|------|
| **Market Analyst** | Technical analysis: price trends, OHLCV, indicators via stockstats |
| **Social Analyst** | Reddit/Twitter sentiment and social momentum |
| **News Analyst** | Financial news via Finnhub; crypto news via CryptoCompare |
| **Fundamentals Analyst** | Company financials, earnings; DeFi Llama for crypto |
| **Macro Analyst** | Federal Reserve data and economic indicators via FRED API |
| **Bull Researcher** | Argues the long/buy case in structured debate |
| **Bear Researcher** | Argues the short/sell case in structured debate |
| **Invest Judge** | Resolves the bull/bear debate; issues investment recommendation |
| **Trader** | Converts analyst + researcher output into a concrete trade plan |
| **Aggressive Debator** | Advocates for higher-risk position sizing in risk debate |
| **Neutral Debator** | Advocates for moderate sizing in risk debate |
| **Conservative Debator** | Advocates for capital preservation in risk debate |
| **Risk Manager** | Resolves risk debate; approves final position size and parameters |

---

## Features

### Order Execution
- **Market orders** — GTC time-in-force for both equities and crypto
- **Bracket orders** — atomic market entry with GTC stop-loss and take-profit legs
- **Standalone stop orders** — GTC stop-loss placed after entry fill
- **Standalone limit orders** — GTC take-profit (supports multiple scale-out targets)
- **Short selling** — supported for equities; crypto shorts not supported by Alpaca
- **Orphan cleanup** — open stop/target orders cancelled before closing or reversing a position

### Position Sizing
- AI-driven sizing: Risk Manager recommends dollar amount based on account equity and risk parameters
- Safety cap: AI-suggested size is validated against user-configured maximum
- Configurable limits: max % of buying power, max account risk per trade, minimum trade size

### Web Interface
- Single-page Dash/Flask app with dark theme
- Multi-symbol batch analysis with per-symbol status tracking
- Live Alpaca account panel: portfolio value, positions P&L, order history
- Interactive Plotly price charts
- Tabbed analyst reports, chat-style debate viewer, decision summary
- Debug panel with raw LLM prompt capture

### Watchdog Daemon
- Background daemon thread started at app launch
- Polls every 10 minutes; alerts to stdout if no heartbeat for 1 hour
- Thread-safe reference counting supports parallel batch mode (multiple tickers analysed concurrently)

### Smart Caching
- All data-fetching tools check a local disk cache before making API calls
- Cache directory: `tradingagents/dataflows/data_cache/`
- Per-source TTLs: 12 hours (news), 24 hours (price, macro, earnings)
- Reduces API costs and allows re-running analysis without re-fetching

### Memory & Reflection
- Each agent (bull, bear, trader, invest judge, risk manager) maintains a `FinancialSituationMemory` backed by ChromaDB
- After each trade, `reflect_and_remember()` updates agent memories with outcome feedback

### Asset Support
- **Equities**: any Alpaca-supported stock ticker (e.g. `NVDA`, `AAPL`)
- **Crypto**: `BTC/USD`, `ETH/USD` — use `/` separator format
- **Batch**: comma-separated mix, e.g. `"NVDA, ETH/USD, AAPL"`

---

## Stack

| Layer | Library |
|-------|---------|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM client | [LangChain OpenAI](https://github.com/langchain-ai/langchain), openai >= 1.30 |
| Web UI | [Dash](https://dash.plotly.com/) >= 3.0, [Flask](https://flask.palletsprojects.com/) >= 3.0, dash-bootstrap-components >= 2.0 |
| Charts | [Plotly](https://plotly.com/python/) |
| Broker / market data | [alpaca-py](https://github.com/alpacahq/alpaca-py) >= 0.8.2 |
| Agent memory | [ChromaDB](https://www.trychroma.com/) |
| Technical indicators | [stockstats](https://github.com/jealous/stockstats), yfinance |
| Retry / rate limits | [tenacity](https://github.com/jd/tenacity) >= 8.2 |
| Financial data | Finnhub, FRED (macro), CryptoCompare, DeFi Llama |
| Social data | PRAW (Reddit), feedparser |
| Python | 3.13 |

---

## Setup

### 1. Clone and create environment

```bash
git clone <repo-url>
cd AlpacaTradingAgent

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

Or with conda:

```bash
conda create -n alpacatradingagent python=3.13
conda activate alpacatradingagent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

```bash
cp env.sample .env
```

Edit `.env` with your keys:

```env
# Alpaca — required for trading and market data
# https://app.alpaca.markets/signup
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_USE_PAPER=True          # True = paper trading, False = live

# OpenAI — required for all LLM agents
# https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# Finnhub — required for stock news
# https://finnhub.io/register
FINNHUB_API_KEY=your_finnhub_api_key_here

# FRED — required for macro analysis
# https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY=your_fred_api_key_here

# CryptoCompare — required for crypto news
# https://www.cryptocompare.com/cryptopian/api-keys
COINDESK_API_KEY=your_cryptocompare_api_key_here
```

---

## Running

### Web UI (primary interface)

```bash
python run_webui_dash.py
```

Opens at **http://localhost:7860**

Options:

```
--port PORT         Port to listen on (default: 7860; auto-increments if taken)
--server-name HOST  Bind address (default: 127.0.0.1)
--debug             Enable Dash debug mode with more logging
--max-threads N     ThreadPoolExecutor size (default: 40)
--share             Expose a public URL (not recommended for live trading)
```

### CLI

```bash
python -m cli.main
```

Accepts single or multiple tickers interactively: `NVDA`, `BTC/USD`, `NVDA, ETH/USD, AAPL`.

### Installed entry points

```bash
pip install -e .

tradingagents        # CLI
tradingagents-web    # Web UI
```

---

## Configuration

Key settings in `tradingagents/default_config.py` (can be overridden per-session in the UI):

| Key | Default | Description |
|-----|---------|-------------|
| `deep_think_llm` | `"gpt-5.2-2025-12-11"` | Model for complex reasoning (researchers, judges) |
| `quick_think_llm` | `"gpt-5-mini-2025-08-07"` | Model for faster tasks (analysts) |
| `research_depth` | `"medium"` | `"shallow"` (1 round), `"medium"` (3 rounds), `"deep"` (5 rounds) |
| `allow_shorts` | `False` | `False` = investment mode (BUY/HOLD/SELL); `True` = trading mode (LONG/NEUTRAL/SHORT) |
| `parallel_analysts` | `False` | Run all 5 analysts concurrently via ThreadPoolExecutor |
| `ai_position_sizing` | `True` | Let Risk Manager determine position size |
| `max_position_pct_of_buying_power` | `30` | Hard cap on position size as % of buying power |
| `max_risk_pct_per_trade` | `3` | Maximum account risk per trade (%) |
| `use_stop_loss` | `True` | Place GTC stop order after entry |
| `use_take_profit` | `True` | Place GTC limit order(s) after entry |
| `scale_out_targets` | `True` | Split take-profit across multiple price targets |
| `alpaca_use_paper` | `"True"` | Paper vs live trading |

**Watchdog constants** in `webui/watchdog.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `WATCHDOG_INTERVAL_SECONDS` | `600` | How often the watchdog polls (10 min) |
| `STUCK_THRESHOLD_SECONDS` | `3600` | Flag-file age that triggers a stuck-analysis alert (1 hr) |

---

## Project Structure

```
AlpacaTradingAgent/
├── run_webui_dash.py              # Entry point: starts Dash server
├── env.sample                     # API key template
├── requirements.txt
│
├── tradingagents/
│   ├── default_config.py          # DEFAULT_CONFIG dict
│   ├── graph/
│   │   ├── trading_graph.py       # TradingAgentsGraph — top-level orchestrator
│   │   ├── setup.py               # StateGraph construction, parallel analyst coordinator
│   │   ├── conditional_logic.py   # Edge routing (debate round counters, etc.)
│   │   ├── propagation.py         # Initial state creation, graph invocation
│   │   ├── reflection.py          # Post-trade memory updates
│   │   └── signal_processing.py   # Final decision extraction
│   │
│   ├── agents/
│   │   ├── analysts/              # market, social_media, news, fundamentals, macro
│   │   ├── researchers/           # bull_researcher, bear_researcher
│   │   ├── risk_mgmt/             # aggressive, neutral, conservative debators
│   │   ├── trader/                # trader agent
│   │   ├── managers/              # risk_manager (judge), research_manager
│   │   └── utils/
│   │       ├── agent_states.py    # AgentState, InvestDebateState, RiskDebateState
│   │       ├── memory.py          # FinancialSituationMemory (ChromaDB)
│   │       ├── agent_utils.py     # Toolkit, LLM helpers
│   │       └── agent_trading_modes.py  # BUY/HOLD/SELL ↔ LONG/NEUTRAL/SHORT
│   │
│   └── dataflows/
│       ├── interface.py           # All LangChain tool definitions (3000+ lines)
│       ├── alpaca_utils.py        # AlpacaUtils: orders, positions, account info
│       ├── cache_utils.py         # @with_cache decorator, clear_cache()
│       ├── config.py              # API key resolution
│       ├── ticker_utils.py        # Crypto vs equity detection
│       ├── finnhub_utils.py       # Stock news
│       ├── coindesk_utils.py      # Crypto news (CryptoCompare)
│       ├── defillama_utils.py     # DeFi fundamentals
│       ├── macro_utils.py         # FRED macro indicators
│       ├── stockstats_utils.py    # Technical indicators
│       ├── earnings_utils.py      # Earnings calendar/surprises
│       └── data_cache/            # Auto-managed cache files
│
├── webui/
│   ├── app_dash.py                # Dash app factory, run_app()
│   ├── layout.py                  # Main page layout assembly
│   ├── watchdog.py                # Background stuck-analysis detector
│   ├── components/
│   │   ├── analysis.py            # Analysis lifecycle + trade execution
│   │   ├── alpaca_account.py      # Account/positions/orders panel
│   │   ├── config_panel.py        # Ticker input, model/depth controls
│   │   ├── chart_panel.py         # Plotly price chart
│   │   ├── reports_panel.py       # Tabbed analyst reports
│   │   ├── decision_panel.py      # Final trade decision display
│   │   ├── status_panel.py        # Per-agent progress indicators
│   │   └── batch_overview_panel.py  # Multi-symbol status table
│   ├── callbacks/                 # All Dash callback modules
│   └── utils/
│       ├── state.py               # app_state singleton
│       ├── charts.py              # Chart builder
│       └── market_hours.py        # Market hours checks
│
└── cli/
    └── main.py                    # Interactive CLI
```

---

## Python API

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = "gpt-5-mini-2025-08-07"   # cheaper for testing
config["quick_think_llm"] = "gpt-5-mini-2025-08-07"
config["research_depth"] = "shallow"
config["allow_shorts"] = False

ta = TradingAgentsGraph(debug=True, config=config)

# Returns (full_state, final_decision_string)
_, decision = ta.propagate("NVDA", "2025-01-15")
print(decision)
```

---

## Acknowledgments

Built upon concepts from the [TradingAgents](https://github.com/TauricResearch/TradingAgents) framework by Tauric Research.

```bibtex
@misc{xiao2025tradingagentsmultiagentsllmfinancial,
  title   = {TradingAgents: Multi-Agents LLM Financial Trading Framework},
  author  = {Yijia Xiao and Edward Sun and Di Luo and Wei Wang},
  year    = {2025},
  eprint  = {2412.20138},
  archivePrefix = {arXiv},
  primaryClass  = {q-fin.TR},
  url     = {https://arxiv.org/abs/2412.20138},
}
```
