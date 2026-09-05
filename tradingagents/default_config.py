import os

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "eval_results"),
    "memory_log_path": os.getenv(
        "TRADINGAGENTS_MEMORY_LOG_PATH",
        os.path.join(_TRADINGAGENTS_HOME, "memory", "trading_memory.md"),
    ),
    "memory_log_max_entries": None,
    # Self-learning memory: when set, the per-agent ChromaDB reflection
    # memories persist across restarts instead of resetting each session.
    "agent_memory_dir": os.getenv(
        "TRADINGAGENTS_AGENT_MEMORY_DIR",
        os.path.join(_TRADINGAGENTS_HOME, "memory", "agent_memory"),
    ),
    # Feed realized outcomes back into the per-agent memories (5 quick-LLM
    # reflection calls per resolved decision). Requires OpenAI embeddings.
    "reflection_on_outcome_enabled": True,
    # FinMem-style memory maintenance (arXiv:2311.13743): Ebbinghaus time
    # decay with importance-scaled stability, near-duplicate pruning, and a
    # per-collection size cap. Runs after outcome reflections.
    "memory_maintenance_enabled": True,
    "memory_max_entries_per_collection": 500,
    "memory_duplicate_similarity_threshold": 0.97,
    "memory_recency_purge_threshold": 0.05,
    "checkpoint_enabled": False,
    # "data_dir": "/Users/yluo/Documents/Code/ScAI/FR1-data",
    "data_dir": "data/ScAI/FR1-data",
    "data_cache_dir": os.getenv(
        "TRADINGAGENTS_CACHE_DIR",
        os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
            "dataflows/data_cache",
        ),
    ),
    # LLM settings
    "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
    "deep_think_llm": "gpt-5.4-mini",
    "quick_think_llm": "gpt-5.4-nano",
    "backend_url": None,
    "google_thinking_level": None,
    "openai_reasoning_effort": None,
    "anthropic_effort": None,
    "xai_reasoning_effort": None,
    "output_language": "English",
    "deep_llm_params": {
        "reasoning_effort": "medium",
        "text_verbosity": "medium",
        "reasoning_summary": "auto",
        "max_output_tokens": None,
        "store": False,
        "parallel_tool_calls": True,
    },
    "quick_llm_params": {
        "reasoning_effort": "low",
        "text_verbosity": "low",
        "reasoning_summary": "auto",
        "max_output_tokens": None,
        "store": False,
        "parallel_tool_calls": True,
    },
    # Debate and discussion settings
    "max_debate_rounds": 4,
    "max_risk_discuss_rounds": 3,
    "max_recur_limit": 200,
    # Trading settings
    "allow_shorts": False,  # False = Investment mode (BUY/HOLD/SELL), True = Trading mode (LONG/NEUTRAL/SHORT)
    # Deterministic risk sizing: when enabled, position size for opening trades
    # is recomputed by tradingagents.risk (fractional Kelly, ATR stops, exposure
    # caps) with the configured trade amount acting as a hard ceiling.
    "risk_sizing_enabled": False,
    "risk_sizing_params": {},  # optional RiskParameters overrides (see tradingagents/risk/position_sizing.py)
    "protective_bracket_orders_enabled": True,  # Submit stop-loss/take-profit as real bracket/OTO child orders on equity entries (crypto unsupported by Alpaca)
    # Production safety layer (deterministic, independent of agent logic)
    "safety_enabled": True,  # Master switch for pre-trade checks + circuit breakers + kill switch
    "max_trade_notional_usd": 25000.0,  # Per-order notional cap; 0 = uncapped
    "max_symbol_concentration_pct": 25.0,  # Max per-symbol exposure as % of equity; 0 = uncapped
    "daily_loss_halt_pct": 10.0,  # Circuit breaker: halt trading when equity falls this % vs yesterday
    "max_drawdown_halt_pct": 15.0,  # Circuit breaker: halt when equity falls this % below the high-water mark
    "max_consecutive_rejections": 5,  # Circuit breaker: halt after this many broker rejections in a row
    "daily_llm_token_budget": 0,  # Refuse new analyses after this many LLM tokens per day; 0 = unlimited
    # Operational alerts (Telegram / webhook; stdlib-only, failure-isolated)
    "alerts_enabled": True,  # Master switch; channels below must also be configured
    "alert_telegram_bot_token": None,  # Or env ALERT_TELEGRAM_BOT_TOKEN
    "alert_telegram_chat_id": None,  # Or env ALERT_TELEGRAM_CHAT_ID
    "alert_webhook_url": None,  # Generic JSON webhook; or env ALERT_WEBHOOK_URL
    "alert_cooldown_seconds": 900,  # Identical alerts suppressed within this window
    # Portfolio-level intelligence (deterministic sizing above per-symbol decisions)
    "portfolio_intelligence_enabled": True,  # Master switch for portfolio-aware sizing of new long exposure
    "portfolio_lookback_bars": 60,  # Daily bars used for correlation / volatility estimates
    "portfolio_high_correlation": 0.6,  # Positive correlation above this = duplicated risk
    "portfolio_correlated_size_factor": 0.5,  # Size multiplier applied on a correlation hit
    "portfolio_vol_sizing_enabled": True,  # Inverse-volatility (simplified risk parity) sizing
    "portfolio_target_daily_vol_pct": 2.0,  # Realized daily vol above this scales size by target/realized
    "portfolio_max_gross_exposure_pct": 100.0,  # Total book value cap as % of equity; 0 = uncapped
    "portfolio_min_size_factor": 0.25,  # Floor on combined penalties so trades never silently vanish
    # Market regime detection (deterministic, fit-free; filter not signal)
    "regime_detection_enabled": True,  # Inject regime block into the market report and scale sizing
    "regime_vol_window": 20,  # Bars for realized-volatility estimate
    "regime_turbulent_percentile": 75.0,  # Vol percentile above which the regime is turbulent
    "regime_turbulent_abs_annual_vol_pct": 40.0,  # Absolute annualized-vol floor for turbulence
    "regime_trend_window": 50,  # SMA window for the trend dimension
    "regime_thin_liquidity_ratio": 0.7,  # Recent/baseline dollar-volume ratio below this = thinning
    "regime_turbulent_size_factor": 0.5,  # Size multiplier in turbulent volatility
    "regime_downtrend_size_factor": 0.75,  # Size multiplier in a downtrend
    "regime_thin_liquidity_size_factor": 0.85,  # Size multiplier when liquidity is thinning
    "regime_min_risk_multiplier": 0.25,  # Floor for the combined regime multiplier
    # Execution settings
    "parallel_analysts": True,  # True = Run analysts in parallel for faster execution, False = Sequential execution
    "parallel_risk_first_round": True,  # Run Risky/Safe/Neutral in parallel only for round 1, then revert to linear flow
    "analyst_start_delay": 0.5,  # Delay in seconds between starting each analyst (to avoid API overload)
    "risk_analyst_start_delay": 0.35,  # Delay between starting first-round risk analysts in parallel mode
    "analyst_call_delay": 0.1,  # Delay in seconds before making analyst calls
    "tool_result_delay": 0.2,  # Delay in seconds between tool results and next analyst call
    # Context management settings (avoid prompt overflows in downstream agents)
    "report_context_budget_tokens": 5500,  # Max retrieved evidence budget per downstream agent call
    "report_context_max_chunks": 16,  # Max retrieved chunks injected into any single downstream prompt
    "report_context_min_chunks_per_report": 1,  # Ensure each non-empty analyst report is represented
    "report_context_chunk_chars": 900,  # Chunk size used to index analyst reports
    "report_context_chunk_overlap": 120,  # Overlap between report chunks
    "report_context_max_points_per_report": 8,  # Coverage bullets kept per report
    "report_context_point_chars": 220,  # Max chars per coverage bullet
    "report_context_excerpt_chars": 420,  # Max chars per retrieved excerpt injected into prompts
    "report_context_memory_chars": 12000,  # Max chars for memory embedding context
    "report_context_compact_points_per_report": 3,  # Claims per report for compact downstream packet
    "report_context_compact_point_chars": 180,  # Max chars per compact claim
    "report_context_compact_excerpt_chars": 240,  # Max chars per compact evidence excerpt
    "report_context_compact_max_excerpts": 8,  # Max compact evidence excerpts injected per prompt
    "report_context_max_claims_per_report": 8,  # Structured evidence claims scored per analyst report
    "report_context_claim_chars": 260,  # Max chars per structured evidence claim
    "report_context_scoreboard_claims_per_side": 3,  # Top scored claims shown for bull/bear sides
    "debate_digest_max_messages": 6,  # Max recent debate messages included in compact debate digest
    "debate_digest_message_chars": 520,  # Max chars per message in debate digest
    "debate_digest_total_chars": 2600,  # Total max chars for debate digest block
    "include_full_reports_in_prompts": False,  # If True, inject full raw analyst reports into downstream prompts (very slow)
    "max_tool_iterations_per_agent": 8,  # Max tool-call loop turns per analyst node
    "max_same_tool_call_repeats": 1,  # Max repeats for the same tool+args signature in a single analyst node
    # Tool settings
    "online_tools": True,
    "tool_semantic_retry_enabled": True,  # Retry web-search tool calls once on low-quality interactive/undersized output
    "tool_semantic_retry_max_retries": 1,
    "tool_semantic_retry_backoff_seconds": 0.8,
    "tool_semantic_retry_disabled_tools": [
        "get_global_news_openai",
        "get_macro_news_openai",
    ],
    "data_fallback_enabled": False,  # Optional yfinance backup for supported Alpaca data failures
    "web_search_timeout_extension_seconds": 45,  # Extra timeout buffer added by timing wrapper for web-search tools
    "news_global_openai_enabled": False,  # News analyst uses fast ticker sources by default; macro handles broad global context
    "global_news_fast_profile": True,  # Keep global-news tool lean even at medium/deep research depth
    "stock_news_fast_profile": True,  # Keep stock-news web-search tool lean at medium/deep depth
    "fundamentals_fast_profile": True,  # Keep fundamentals web-search tool lean at medium/deep depth
    "global_news_timeout_seconds": 150,  # Timeout for get_global_news_openai web-search calls
    "global_news_max_output_tokens": 1200,  # Applied to models that support explicit output-token caps
    "global_news_max_events": 8,  # Cap number of events requested from global-news tool
    "global_news_word_budget": 550,  # Target output length from global-news tool
    "stock_news_timeout_seconds": 120,  # Timeout for get_stock_news_openai web-search calls
    "stock_news_max_output_tokens": 900,  # Output-token cap for social/news web-search summary tool
    "fundamentals_timeout_seconds": 120,  # Timeout for get_fundamentals_openai web-search calls
    "fundamentals_max_output_tokens": 1000,  # Output-token cap for fundamentals web-search summary tool
    "google_news_max_pages": 3,  # Google News pages fetched before dedupe/limit
    "google_news_max_items": 18,  # Max deduped Google News items returned to analysts
    "openai_store_responses": False,  # Disable response storing by default to reduce latency/payload
    # API keys (these will be overridden by environment variables if present)
    "openai_api_key": None,
    "openai_use_local": False,  # Route core LLM calls to a local OpenAI-compatible endpoint
    "openai_base_url": None,  # Example: http://localhost:1234/v1
    "openai_embedding_model": "text-embedding-ada-002",
    "minimax_api_key": None,
    "finnhub_api_key": None,
    "alpaca_api_key": None,
    "alpaca_secret_key": None,
    "alpaca_use_paper": "True",  # Set to "True" to use paper trading, "False" for live trading
    "coindesk_api_key": None,
}
