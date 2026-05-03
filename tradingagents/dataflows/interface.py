from typing import Annotated, Dict, List, Tuple
from .reddit_utils import (
    fetch_top_from_category,
    fetch_top_from_category_online,
    get_search_terms,
)
from .stockstats_utils import *
from .googlenews_utils import *
from .finnhub_utils import (
    get_data_in_range,
    fetch_company_news_live,
    fetch_insider_sentiment_live,
    fetch_insider_transactions_live,
)
from .alpaca_utils import AlpacaUtils
from .coindesk_utils import get_news as get_coindesk_news_util
from .defillama_utils import get_fundamentals as get_defillama_fundamentals_util
from .earnings_utils import get_earnings_calendar_data, get_earnings_surprises_analysis
from .macro_utils import get_macro_economic_summary, get_economic_indicators_report, get_treasury_yield_curve
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
import pandas as pd
from .config import get_config, set_config, DATA_DIR, get_api_key
from .interface_utils import (
    _coerce_bool,
    extract_responses_text,
    _strip_trailing_interactive_followup,
    get_global_news_profile_for_depth,
    get_model_params,
    get_openai_client_with_timeout,
    get_search_context_for_depth,
)
from tradingagents.openai_model_registry import (
    apply_responses_model_params,
    is_responses_model,
    normalize_model_params,
)


def _cap_headline_sections(
    text: str,
    max_sections: int = 10,
    max_chars: int = 7000,
) -> str:
    """Keep at most N markdown `###` sections to avoid oversized fallback payloads."""
    body = str(text or "").strip()
    if not body:
        return ""

    lines = body.splitlines()
    kept: List[str] = []
    section_count = 0
    for line in lines:
        if line.startswith("### "):
            section_count += 1
            if section_count > max_sections:
                break
        kept.append(line)

    clipped = "\n".join(kept).strip()
    if len(clipped) > max_chars:
        clipped = clipped[:max_chars].rstrip() + "\n..."
    return clipped


def _build_empty_openai_global_fallback(curr_date: str, ticker_context: str | None = None) -> str:
    target = str(ticker_context or "global markets").strip()
    query = f"{target} macro economy central bank inflation"
    google = get_google_news(query=query, curr_date=curr_date, look_back_days=5)
    google = _cap_headline_sections(google, max_sections=8, max_chars=6000)
    if google:
        return (
            f"Fallback used because OpenAI web-search returned empty output.\n"
            f"## Global/Macro news proxy for {target} on {curr_date}\n\n{google}"
        )
    return (
        f"Fallback used because OpenAI web-search returned empty output.\n"
        f"No sufficiently relevant global-news items were found via fallback sources for {target} ({curr_date})."
    )


def _build_empty_openai_stock_news_fallback(ticker: str, curr_date: str) -> str:
    snippets: List[Tuple[str, str]] = []

    google = get_google_news(query=ticker, curr_date=curr_date, look_back_days=7)
    google = _cap_headline_sections(google, max_sections=8, max_chars=4500)
    if google:
        snippets.append(("Google News fallback", google))

    finnhub = get_finnhub_news(ticker=ticker, curr_date=curr_date, look_back_days=4)
    finnhub = _cap_headline_sections(finnhub, max_sections=8, max_chars=4500)
    if finnhub:
        snippets.append(("Finnhub fallback", finnhub))

    if not snippets:
        return (
            f"Fallback used because OpenAI web-search returned empty output.\n"
            f"No fallback stock-news items found for {ticker} as of {curr_date}."
        )

    merged = [f"Fallback used because OpenAI web-search returned empty output for {ticker}.", ""]
    for label, text in snippets:
        merged.append(f"## {label}")
        merged.append(text)
        merged.append("")
    return "\n".join(merged).strip()


def _build_empty_openai_fundamentals_fallback(ticker: str, curr_date: str) -> str:
    insider_sent = get_finnhub_company_insider_sentiment(ticker, curr_date, 30)
    insider_tx = get_finnhub_company_insider_transactions(ticker, curr_date, 30)
    finnhub_news = get_finnhub_news(ticker, curr_date, 5)

    sent_text = _cap_headline_sections(insider_sent, max_sections=8, max_chars=2500)
    tx_text = _cap_headline_sections(insider_tx, max_sections=10, max_chars=3500)
    news_text = _cap_headline_sections(finnhub_news, max_sections=6, max_chars=2800)

    return (
        f"Fallback used because OpenAI fundamentals web-search returned empty output for {ticker} ({curr_date}).\n\n"
        f"## Insider Sentiment Snapshot\n{sent_text}\n\n"
        f"## Insider Transactions Snapshot\n{tx_text}\n\n"
        f"## Recent Company News Snapshot\n{news_text}"
    ).strip()


def _uses_responses_for_web_search(model: str) -> bool:
    """Use Responses API for models that support hosted web search tools."""
    model_name = str(model or "")
    return is_responses_model(model_name) or model_name.startswith("gpt-4.1")


def _quick_model_params_for_tool(
    model: str,
    config: Dict,
    *,
    max_output_tokens: int,
    store_responses: bool,
) -> Dict:
    params = normalize_model_params(
        model,
        config.get("quick_llm_params"),
        role="quick",
    )
    params.setdefault("max_output_tokens", max_output_tokens)
    params.setdefault("store", store_responses)
    return params


def _build_web_search_response_params(
    *,
    model: str,
    developer_message: str,
    user_message: str,
    search_context: str,
    max_output_tokens: int,
    store_responses: bool,
    model_params: Dict,
    include_reasoning: bool = False,
) -> Dict:
    role = "developer" if is_responses_model(model) else "system"
    api_params = {
        "model": model,
        "input": [
            {
                "role": role,
                "content": [{"type": "input_text", "text": developer_message}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_message}],
            },
        ],
        "text": {"format": {"type": "text"}},
        "tools": [
            {
                "type": "web_search",
                "user_location": {"type": "approximate"},
                "search_context_size": search_context,
            }
        ],
        "include": ["web_search_call.action.sources"],
    }
    if include_reasoning:
        api_params["include"].insert(0, "reasoning.encrypted_content")

    tool_params = dict(model_params or {})
    tool_params.setdefault("max_output_tokens", max_output_tokens)
    tool_params.setdefault("store", store_responses)
    apply_responses_model_params(api_params, model, tool_params, role="quick")

    # Store is a Responses API-level field. Keep it available even for the
    # non-reasoning GPT-4.1 path, where it is not shown as a per-model UI knob.
    api_params.setdefault("store", store_responses)
    return api_params


def _create_response_with_output_cap_fallback(client, api_params: Dict):
    try:
        return client.responses.create(**api_params)
    except Exception as output_cap_error:
        if "max_output_tokens" in api_params and "max_output_tokens" in str(output_cap_error):
            fallback_params = dict(api_params)
            fallback_params.pop("max_output_tokens", None)
            return client.responses.create(**fallback_params)
        raise


def get_finnhub_news(
    ticker: Annotated[
        str,
        "Search query of a company's, e.g. 'AAPL, TSM, etc.",
    ],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
):
    """
    Retrieve news about a company within a time frame

    Args
        ticker (str): ticker for the company you are interested in
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns
        str: dataframe containing the news of the company in the time frame

    """

    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    online_tools_enabled = _coerce_bool(get_config().get("online_tools", True))
    combined_result = ""
    source_label = "cache"

    # Cache-first path
    try:
        result = get_data_in_range(ticker, before, curr_date, "news_data", DATA_DIR)
    except FileNotFoundError:
        result = {}

    for day, data in (result or {}).items():
        if not data:
            continue
        for entry in data:
            headline = entry.get("headline", "Untitled")
            summary = entry.get("summary", "")
            current_news = f"### {headline} ({day})\n{summary}"
            combined_result += current_news + "\n\n"

    # Live Finnhub API fallback when cache is missing/empty
    if not combined_result and online_tools_enabled:
        try:
            live_entries = fetch_company_news_live(ticker, before, curr_date)
            source_label = "finnhub_live_api"
        except Exception:
            live_entries = []

        for entry in live_entries:
            ts = entry.get("datetime", 0)
            if isinstance(ts, (int, float)) and ts > 0:
                day = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            else:
                day = entry.get("date", curr_date)
            headline = entry.get("headline", "Untitled")
            summary = entry.get("summary") or entry.get("url") or ""
            combined_result += f"### {headline} ({day})\n{summary}\n\n"

    if not combined_result:
        return f"## {ticker} News, from {before} to {curr_date}: No Finnhub news items found."

    return (
        f"## {ticker} News, from {before} to {curr_date} (source: {source_label}):\n"
        + str(combined_result)
    )


def get_finnhub_company_insider_sentiment(
    ticker: Annotated[str, "ticker symbol for the company"],
    curr_date: Annotated[
        str,
        "current date of you are trading at, yyyy-mm-dd",
    ],
    look_back_days: Annotated[int, "number of days to look back"],
):
    """
    Retrieve insider sentiment about a company (retrieved from public SEC information) for the past 15 days
    Args:
        ticker (str): ticker symbol of the company
        curr_date (str): current date you are trading on, yyyy-mm-dd
    Returns:
        str: a report of the sentiment in the past 15 days starting at curr_date
    """

    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    online_tools_enabled = _coerce_bool(get_config().get("online_tools", True))
    source_label = "cache"
    seen_records = set()
    result_lines = []

    try:
        data = get_data_in_range(ticker, before, curr_date, "insider_senti", DATA_DIR)
    except FileNotFoundError:
        data = {}

    for _, senti_list in (data or {}).items():
        for entry in senti_list:
            year = entry.get("year")
            month = entry.get("month")
            change = entry.get("change")
            mspr = entry.get("mspr")
            rec_key = (year, month, change, mspr)
            if rec_key in seen_records:
                continue
            seen_records.add(rec_key)
            result_lines.append(
                f"### {year}-{month}:\n"
                f"Change: {change}\n"
                f"Monthly Share Purchase Ratio: {mspr}\n"
            )

    if not result_lines and online_tools_enabled:
        try:
            live_entries = fetch_insider_sentiment_live(ticker, before, curr_date)
            source_label = "finnhub_live_api"
        except Exception:
            live_entries = []

        for entry in live_entries:
            year = entry.get("year")
            month = entry.get("month")
            change = entry.get("change")
            mspr = entry.get("mspr")
            rec_key = (year, month, change, mspr)
            if rec_key in seen_records:
                continue
            seen_records.add(rec_key)
            result_lines.append(
                f"### {year}-{month}:\n"
                f"Change: {change}\n"
                f"Monthly Share Purchase Ratio: {mspr}\n"
            )

    if not result_lines:
        return f"## {ticker} Insider Sentiment Data for {before} to {curr_date}: No records found."

    return (
        f"## {ticker} Insider Sentiment Data for {before} to {curr_date} (source: {source_label}):\n"
        + "\n".join(result_lines)
        + "\nThe change field refers to net insider buying/selling. "
        "The mspr field is the monthly share purchase ratio."
    )


def get_finnhub_company_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[
        str,
        "current date you are trading at, yyyy-mm-dd",
    ],
    look_back_days: Annotated[int, "how many days to look back"],
):
    """
    Retrieve insider transcaction information about a company (retrieved from public SEC information) for the past 15 days
    Args:
        ticker (str): ticker symbol of the company
        curr_date (str): current date you are trading at, yyyy-mm-dd
    Returns:
        str: a report of the company's insider transaction/trading informtaion in the past 15 days
    """

    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    online_tools_enabled = _coerce_bool(get_config().get("online_tools", True))
    source_label = "cache"
    seen_records = set()
    result_lines = []

    try:
        data = get_data_in_range(ticker, before, curr_date, "insider_trans", DATA_DIR)
    except FileNotFoundError:
        data = {}

    for _, senti_list in (data or {}).items():
        for entry in senti_list:
            filing_date = entry.get("filingDate", "N/A")
            name = entry.get("name", "Unknown Insider")
            change = entry.get("change", "N/A")
            shares = entry.get("share", "N/A")
            transaction_price = entry.get("transactionPrice", "N/A")
            transaction_code = entry.get("transactionCode", "N/A")
            rec_key = (
                filing_date,
                name,
                entry.get("transactionDate", ""),
                change,
                shares,
                transaction_price,
                transaction_code,
            )
            if rec_key in seen_records:
                continue
            seen_records.add(rec_key)
            result_lines.append(
                f"### Filing Date: {filing_date}, {name}:\n"
                f"Change: {change}\n"
                f"Shares: {shares}\n"
                f"Transaction Price: {transaction_price}\n"
                f"Transaction Code: {transaction_code}\n"
            )

    if not result_lines and online_tools_enabled:
        try:
            live_entries = fetch_insider_transactions_live(ticker, before, curr_date)
            source_label = "finnhub_live_api"
        except Exception:
            live_entries = []

        for entry in live_entries:
            filing_date = entry.get("filingDate", "N/A")
            name = entry.get("name", "Unknown Insider")
            change = entry.get("change", "N/A")
            shares = entry.get("share", "N/A")
            transaction_price = entry.get("transactionPrice", "N/A")
            transaction_code = entry.get("transactionCode", "N/A")
            rec_key = (
                filing_date,
                name,
                entry.get("transactionDate", ""),
                change,
                shares,
                transaction_price,
                transaction_code,
            )
            if rec_key in seen_records:
                continue
            seen_records.add(rec_key)
            result_lines.append(
                f"### Filing Date: {filing_date}, {name}:\n"
                f"Change: {change}\n"
                f"Shares: {shares}\n"
                f"Transaction Price: {transaction_price}\n"
                f"Transaction Code: {transaction_code}\n"
            )

    if not result_lines:
        return f"## {ticker} insider transactions from {before} to {curr_date}: No records found."

    return (
        f"## {ticker} insider transactions from {before} to {curr_date} (source: {source_label}):\n"
        + "\n".join(result_lines)
        + "\nThe change field reflects variation in insider holdings; share is volume; "
        "transactionPrice is execution price per share; transactionCode describes trade type."
    )


def get_coindesk_news(
    ticker: Annotated[str, "Ticker symbol, e.g. 'BTC/USD', 'ETH/USD', 'ETH', etc."],
    num_sentences: Annotated[int, "Number of sentences to include from news body."] = 5,
) -> str:
    """
    Retrieve news for a cryptocurrency.
    This function checks if the ticker is a crypto pair (like BTC/USD) and extracts the base currency.
    Then it fetches news for that cryptocurrency from CryptoCompare.

    Args:
        ticker (str): Ticker symbol for the cryptocurrency.
        num_sentences (int): Number of sentences to extract from the body of each news article.

    Returns:
        str: Formatted string containing news.
    """
    crypto_symbol = ticker.upper()
    if "/" in crypto_symbol:
        crypto_symbol = crypto_symbol.split('/')[0]
    else:
        crypto_symbol = crypto_symbol.replace("USDT", "").replace("USD", "")

    return get_coindesk_news_util(crypto_symbol, n=num_sentences)


def get_simfin_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
        str,
        "reporting frequency of the company's financial history: annual / quarterly",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
):
    data_path = os.path.join(
        DATA_DIR,
        "fundamental_data",
        "simfin_data_all",
        "balance_sheet",
        "companies",
        "us",
        f"us-balance-{freq}.csv",
    )

    if not os.path.exists(data_path):
        return (
            f"## {freq} balance sheet for {ticker}: unavailable "
            f"(SimFin file missing: {data_path})."
        )

    df = pd.read_csv(data_path, sep=";")
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    if filtered_df.empty:
        return (
            f"## {freq} balance sheet for {ticker}: no SimFin report available before {curr_date}."
        )

    latest_balance_sheet = filtered_df.loc[filtered_df["Publish Date"].idxmax()]
    if "SimFinId" in latest_balance_sheet.index:
        latest_balance_sheet = latest_balance_sheet.drop("SimFinId")

    return (
        f"## {freq} balance sheet for {ticker} released on {str(latest_balance_sheet['Publish Date'])[0:10]}: \n"
        + str(latest_balance_sheet)
        + "\n\nThis includes metadata like reporting dates and currency, share details, and a breakdown of assets, liabilities, and equity. Assets are grouped as current (liquid items like cash and receivables) and noncurrent (long-term investments and property). Liabilities are split between short-term obligations and long-term debts, while equity reflects shareholder funds such as paid-in capital and retained earnings. Together, these components ensure that total assets equal the sum of liabilities and equity."
    )


def get_simfin_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
        str,
        "reporting frequency of the company's financial history: annual / quarterly",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
):
    data_path = os.path.join(
        DATA_DIR,
        "fundamental_data",
        "simfin_data_all",
        "cash_flow",
        "companies",
        "us",
        f"us-cashflow-{freq}.csv",
    )

    if not os.path.exists(data_path):
        return (
            f"## {freq} cash flow statement for {ticker}: unavailable "
            f"(SimFin file missing: {data_path})."
        )

    df = pd.read_csv(data_path, sep=";")
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    if filtered_df.empty:
        return (
            f"## {freq} cash flow statement for {ticker}: no SimFin report available before {curr_date}."
        )

    latest_cash_flow = filtered_df.loc[filtered_df["Publish Date"].idxmax()]
    if "SimFinId" in latest_cash_flow.index:
        latest_cash_flow = latest_cash_flow.drop("SimFinId")

    return (
        f"## {freq} cash flow statement for {ticker} released on {str(latest_cash_flow['Publish Date'])[0:10]}: \n"
        + str(latest_cash_flow)
        + "\n\nThis includes metadata like reporting dates and currency, share details, and a breakdown of cash movements. Operating activities show cash generated from core business operations, including net income adjustments for non-cash items and working capital changes. Investing activities cover asset acquisitions/disposals and investments. Financing activities include debt transactions, equity issuances/repurchases, and dividend payments. The net change in cash represents the overall increase or decrease in the company's cash position during the reporting period."
    )


def get_simfin_income_statements(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
        str,
        "reporting frequency of the company's financial history: annual / quarterly",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
):
    data_path = os.path.join(
        DATA_DIR,
        "fundamental_data",
        "simfin_data_all",
        "income_statements",
        "companies",
        "us",
        f"us-income-{freq}.csv",
    )

    if not os.path.exists(data_path):
        return (
            f"## {freq} income statement for {ticker}: unavailable "
            f"(SimFin file missing: {data_path})."
        )

    df = pd.read_csv(data_path, sep=";")
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    if filtered_df.empty:
        return (
            f"## {freq} income statement for {ticker}: no SimFin report available before {curr_date}."
        )

    latest_income = filtered_df.loc[filtered_df["Publish Date"].idxmax()]
    if "SimFinId" in latest_income.index:
        latest_income = latest_income.drop("SimFinId")

    return (
        f"## {freq} income statement for {ticker} released on {str(latest_income['Publish Date'])[0:10]}: \n"
        + str(latest_income)
        + "\n\nThis includes metadata like reporting dates and currency, share details, and a comprehensive breakdown of the company's financial performance. Starting with Revenue, it shows Cost of Revenue and resulting Gross Profit. Operating Expenses are detailed, including SG&A, R&D, and Depreciation. The statement then shows Operating Income, followed by non-operating items and Interest Expense, leading to Pretax Income. After accounting for Income Tax and any Extraordinary items, it concludes with Net Income, representing the company's bottom-line profit or loss for the period."
    )


def _normalize_news_dedupe_key(title: str, link: str) -> str:
    title_key = " ".join(str(title or "").lower().split())
    link_key = str(link or "").strip().lower().rstrip("/")
    return f"{title_key}|{link_key}"


def _expand_google_news_query(query: str) -> str:
    base = str(query or "").strip()
    if not base:
        return base
    if "/" in base or base.upper().endswith("USD") or base.upper().endswith("USDT"):
        return base

    compact = base.upper().replace("/", "").replace("-", "")
    likely_ticker = compact.isalpha() and len(compact) <= 6 and " " not in base
    if not likely_ticker:
        return base

    try:
        company_name = AlpacaUtils.get_company_name(compact)
    except Exception:
        company_name = ""

    company_name = str(company_name or "").strip()
    if not company_name or company_name.upper() == compact:
        return base

    if " OR " in company_name:
        alias_terms = [part.strip() for part in company_name.split(" OR ") if part.strip()]
    else:
        alias_terms = [company_name]

    expanded_terms = [compact, f"${compact}"] + alias_terms
    expanded_terms = [term for term in dict.fromkeys(expanded_terms) if term]
    return " OR ".join(expanded_terms)


def get_google_news(
    query: Annotated[str, "Query to search with"],
    curr_date: Annotated[str, "Curr date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    query_for_search = _expand_google_news_query(query)
    query_encoded = query_for_search.replace(" ", "+")

    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    config = get_config()
    max_pages = int(config.get("google_news_max_pages", 3))
    max_items = int(config.get("google_news_max_items", 18))

    news_results = getNewsData(query_encoded, before, curr_date, max_pages=max_pages)
    if not news_results:
        return ""

    deduped: List[dict] = []
    seen = set()
    for news in news_results:
        key = _normalize_news_dedupe_key(news.get("title", ""), news.get("link", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(news)
        if len(deduped) >= max_items:
            break

    news_str = ""
    for news in deduped:
        source = news.get("source", "Unknown")
        date_text = news.get("date", "Unknown date")
        snippet = str(news.get("snippet", "")).strip()
        news_str += f"### {news.get('title', 'Untitled')} (source: {source}, date: {date_text})\n\n{snippet}\n\n"

    return (
        f"## {query_for_search} Google News, from {before} to {curr_date} "
        f"(items: {len(deduped)}, deduped from {len(news_results)}):\n\n{news_str}"
    )


def get_reddit_global_news(
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
    max_limit_per_day: Annotated[int, "Maximum number of news per day"],
) -> str:
    """
    Retrieve the latest top reddit news
    Args:
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the latest news articles posts on reddit and meta information in these columns: "created_utc", "id", "title", "selftext", "score", "num_comments", "url"
    """

    end_dt = datetime.strptime(start_date, "%Y-%m-%d")
    start_dt = end_dt - relativedelta(days=look_back_days)
    before = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    online_tools_enabled = _coerce_bool(get_config().get("online_tools", True))
    posts = []
    used_online_fallback = False
    local_data_path = os.path.join(DATA_DIR, "reddit_data")

    for offset in range((end_dt - start_dt).days + 1):
        day_str = (start_dt + relativedelta(days=offset)).strftime("%Y-%m-%d")
        try:
            fetch_result = fetch_top_from_category(
                "global_news",
                day_str,
                max_limit_per_day,
                data_path=local_data_path,
            )
            posts.extend(fetch_result)
        except (FileNotFoundError, NotADirectoryError, ValueError):
            used_online_fallback = True
            break

    if used_online_fallback and online_tools_enabled:
        posts = fetch_top_from_category_online(
            category="global_news",
            start_date=before,
            end_date=end_str,
            max_limit=max_limit_per_day * ((end_dt - start_dt).days + 1),
        )

    if len(posts) == 0:
        source = "reddit_live_api" if used_online_fallback else "reddit_local_dataset"
        return f"## Global News Reddit, from {before} to {end_str} (source: {source}): No posts found."

    news_str = ""
    for post in posts:
        content = post.get("content", "")
        subreddit = post.get("subreddit")
        source_tag = f" [r/{subreddit}]" if subreddit else ""
        if content == "":
            news_str += f"### {post.get('title', 'Untitled')}{source_tag}\n\n"
        else:
            news_str += f"### {post.get('title', 'Untitled')}{source_tag}\n\n{content}\n\n"

    source = "reddit_live_api" if used_online_fallback else "reddit_local_dataset"
    return f"## Global News Reddit, from {before} to {end_str} (source: {source}):\n{news_str}"


def get_reddit_company_news(
    ticker: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
    max_limit_per_day: Annotated[int, "Maximum number of news per day"],
) -> str:
    """
    Retrieve the latest top reddit news
    Args:
        ticker: ticker symbol of the company
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the latest news articles posts on reddit and meta information in these columns: "created_utc", "id", "title", "selftext", "score", "num_comments", "url"
    """

    end_dt = datetime.strptime(start_date, "%Y-%m-%d")
    start_dt = end_dt - relativedelta(days=look_back_days)
    before = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    online_tools_enabled = _coerce_bool(get_config().get("online_tools", True))
    posts = []
    used_online_fallback = False
    local_data_path = os.path.join(DATA_DIR, "reddit_data")
    attempted_terms = get_search_terms(ticker)

    for offset in range((end_dt - start_dt).days + 1):
        day_str = (start_dt + relativedelta(days=offset)).strftime("%Y-%m-%d")
        try:
            fetch_result = fetch_top_from_category(
                "company_news",
                day_str,
                max_limit_per_day,
                ticker,
                data_path=local_data_path,
            )
            posts.extend(fetch_result)
        except (FileNotFoundError, NotADirectoryError, ValueError):
            used_online_fallback = True
            break

    if used_online_fallback and online_tools_enabled:
        posts = fetch_top_from_category_online(
            category="company_news",
            start_date=before,
            end_date=end_str,
            max_limit=max_limit_per_day * ((end_dt - start_dt).days + 1),
            query=ticker,
        )

    if len(posts) == 0:
        source = "reddit_live_api" if used_online_fallback else "reddit_local_dataset"
        terms_preview = ", ".join(attempted_terms[:8]) if attempted_terms else ticker
        return (
            f"## {ticker} News Reddit, from {before} to {end_str} (source: {source}): No posts found.\n"
            f"Tried search terms: {terms_preview}"
        )

    news_str = ""
    for post in posts:
        content = post.get("content", "")
        subreddit = post.get("subreddit")
        source_tag = f" [r/{subreddit}]" if subreddit else ""
        if content == "":
            news_str += f"### {post.get('title', 'Untitled')}{source_tag}\n\n"
        else:
            news_str += f"### {post.get('title', 'Untitled')}{source_tag}\n\n{content}\n\n"

    source = "reddit_live_api" if used_online_fallback else "reddit_local_dataset"
    terms_preview = ", ".join(attempted_terms[:8]) if attempted_terms else ticker
    return (
        f"## {ticker} News Reddit, from {before} to {end_str} (source: {source}):\n"
        f"Searched terms: {terms_preview}\n\n{news_str}"
    )


def get_stock_stats_indicators_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    look_back_days: Annotated[int, "how many days to look back"],
    online: Annotated[bool, "to fetch data online or offline"],
) -> str:
    """
    Get a window of technical indicators for a stock
    Args:
        symbol: ticker symbol of the company
        indicator: technical indicator to get the analysis and report of
        curr_date: The current trading date you are trading on, YYYY-mm-dd
        look_back_days: how many days to look back
        online: to fetch data online or offline
    Returns:
        str: a report of the technical indicator for the stock
    """
    curr_date_dt = pd.to_datetime(curr_date)
    dates = []
    values = []

    # Generate dates
    for i in range(look_back_days, 0, -1):
        date = curr_date_dt - pd.DateOffset(days=i)
        dates.append(date.strftime("%Y-%m-%d"))

    # Add current date
    dates.append(curr_date)

    # Get indicator values for each date
    for date in dates:
        try:
            value = StockstatsUtils.get_stock_stats(
                symbol=symbol,
                indicator=indicator,
                curr_date=date,
                data_dir=DATA_DIR,
                online=online,
            )
            values.append(value)
        except Exception as e:
            values.append("N/A")

    # Format the result
    result = f"## {indicator} for {symbol} from {dates[0]} to {dates[-1]}:\n\n"
    for i in range(len(dates)):
        result += f"- {dates[i]}: {values[i]}\n"

    return result


def get_stockstats_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    online: Annotated[bool, "to fetch data online or offline"],
) -> str:
    """
    Get a technical indicator for a stock
    Args:
        symbol: ticker symbol of the company
        indicator: technical indicator to get the analysis and report of
        curr_date: The current trading date you are trading on, YYYY-mm-dd
        online: to fetch data online or offline
    Returns:
        str: a report of the technical indicator for the stock
    """
    try:
        value = StockstatsUtils.get_stock_stats(
            symbol=symbol,
            indicator=indicator,
            curr_date=curr_date,
            data_dir=DATA_DIR,
            online=online,
        )
        return f"## {indicator} for {symbol} on {curr_date}: {value}"
    except Exception as e:
        return f"Error getting {indicator} for {symbol}: {str(e)}"


_INDICATOR_ALIAS_MAP = {
    "close": "close",
    "open": "open",
    "high": "high",
    "low": "low",
    "volume": "volume",
    "vwap": "vwap",
    "ema_8": "ema_8",
    "close_8_ema": "ema_8",
    "ema_10": "ema_10",
    "close_10_ema": "ema_10",
    "ema_21": "ema_21",
    "close_21_ema": "ema_21",
    "sma_20": "sma_20",
    "close_20_sma": "sma_20",
    "sma_50": "sma_50",
    "close_50_sma": "sma_50",
    "rsi": "rsi_14",
    "rsi_14": "rsi_14",
    "adx": "adx_14",
    "adx_14": "adx_14",
    "macd": "macd",
    "macds": "macds",
    "macdh": "macdh",
    "atr": "atr_14",
    "atr_14": "atr_14",
    "boll_ub": "boll_ub",
    "boll_lb": "boll_lb",
    "stoch_k": "stoch_k",
    "stoch_d": "stoch_d",
    "obv": "obv",
    "volume_delta": "volume_delta",
}

_TIMEFRAME_TO_BRIEF_KEY = {
    "1h": "1h",
    "1hour": "1h",
    "4h": "4h",
    "4hour": "4h",
    "1d": "1d",
    "1day": "1d",
}


def _ensure_indicator_column(df: pd.DataFrame, indicator_col: str) -> pd.DataFrame:
    if indicator_col in df.columns:
        return df

    if indicator_col == "ema_10":
        df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    elif indicator_col == "sma_20":
        df["sma_20"] = df["close"].rolling(window=20).mean()
    elif indicator_col == "volume_delta":
        df["volume_delta"] = df["volume"].diff()
    return df


def _format_indicator_history_table(
    data: pd.DataFrame,
    indicator_columns: List[str],
    max_points: int,
) -> str:
    frame = data.tail(max(1, int(max_points))).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["date"] = frame["timestamp"].dt.strftime("%Y-%m-%d %H:%M")

    lines: List[str] = []
    header_columns = ["Date"] + [col.upper() for col in indicator_columns]
    lines.append("| " + " | ".join(header_columns) + " |")
    lines.append("|" + "|".join(["---"] * len(header_columns)) + "|")

    for _, row in frame.iterrows():
        values = [row["date"]]
        for col in indicator_columns:
            value = row.get(col)
            if pd.isna(value):
                values.append("N/A")
            elif col in ("volume", "obv", "volume_delta"):
                values.append(f"{float(value):,.0f}")
            elif col in ("macd", "macds", "macdh"):
                values.append(f"{float(value):.4f}")
            else:
                values.append(f"{float(value):.2f}")
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def get_stockstats_indicator_history(
    symbol: Annotated[str, "ticker symbol (stocks: AAPL, NVDA; crypto: ETH/USD, BTC/USD)"],
    indicator: Annotated[str, "indicator name (e.g. rsi_14, macd, close_8_ema, all)"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    timeframe: Annotated[str, "Chart timeframe: 1Hour, 4Hour, 1Day"] = "1Day",
    look_back_days: Annotated[int, "Calendar days to include in history window"] = 60,
    max_points: Annotated[int, "Maximum history points returned"] = 30,
) -> str:
    """
    Return indicator history on a chosen timeframe so analysts can inspect multiple
    indicators sequentially without requesting one giant report.
    """
    timeframe_key = _TIMEFRAME_TO_BRIEF_KEY.get(str(timeframe or "").lower().strip())
    if not timeframe_key:
        return "Error: timeframe must be one of 1Hour, 4Hour, 1Day."

    try:
        from .technical_brief import compute_indicators
    except Exception as exc:
        return f"Error loading technical indicator engine: {exc}"

    df = compute_indicators(symbol, curr_date, timeframe_key)
    if df is None or df.empty:
        return f"No indicator data available for {symbol} on timeframe {timeframe}."

    df = df.copy()
    if "timestamp" not in df.columns:
        return f"Error: indicator dataset for {symbol} lacks timestamp column."

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df.empty:
        return f"No indicator data available for {symbol} on timeframe {timeframe}."

    end_ts = pd.to_datetime(curr_date, utc=True) + pd.Timedelta(days=1)
    start_ts = end_ts - pd.Timedelta(days=max(1, int(look_back_days)))
    df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
    if df.empty:
        return (
            f"No indicator points found for {symbol} in lookback window {look_back_days} days "
            f"on timeframe {timeframe}."
        )

    requested = str(indicator or "").lower().strip()
    if not requested:
        requested = "all"

    if requested == "all":
        columns = [
            "close",
            "ema_8",
            "ema_21",
            "sma_50",
            "rsi_14",
            "macd",
            "macds",
            "macdh",
            "boll_ub",
            "boll_lb",
            "atr_14",
            "obv",
            "volume",
        ]
    else:
        canonical = _INDICATOR_ALIAS_MAP.get(requested)
        if not canonical:
            supported = ", ".join(sorted(_INDICATOR_ALIAS_MAP.keys()))
            return (
                f"Error: unsupported indicator '{indicator}'. "
                f"Supported indicators: {supported}, all."
            )
        columns = [canonical]

    for col in columns:
        df = _ensure_indicator_column(df, col)

    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        return f"Error: indicator data unavailable for columns: {', '.join(missing_cols)}."

    latest = df.iloc[-1]
    latest_lines = []
    for col in columns:
        value = latest.get(col)
        if pd.isna(value):
            latest_lines.append(f"- {col}: N/A")
        elif col in ("volume", "obv", "volume_delta"):
            latest_lines.append(f"- {col}: {float(value):,.0f}")
        elif col in ("macd", "macds", "macdh"):
            latest_lines.append(f"- {col}: {float(value):.4f}")
        else:
            latest_lines.append(f"- {col}: {float(value):.2f}")

    table = _format_indicator_history_table(df, columns, max_points=max_points)
    latest_ts = pd.to_datetime(latest["timestamp"]).strftime("%Y-%m-%d %H:%M")

    return (
        f"## Indicator history for {symbol}\n"
        f"- Timeframe: {timeframe}\n"
        f"- Lookback days: {look_back_days}\n"
        f"- Latest bar: {latest_ts}\n"
        f"- Requested indicator(s): {', '.join(columns)}\n\n"
        f"### Latest values\n"
        f"{chr(10).join(latest_lines)}\n\n"
        f"### History table\n"
        f"{table}"
    )


def get_stock_news_openai(ticker, curr_date):
    # Get API key from environment variables or config
    api_key = get_api_key("openai_api_key", "OPENAI_API_KEY")
    if not api_key:
        return f"Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
    
    try:
        # Standardize ticker format for consistent API calls
        from .ticker_utils import TickerUtils, normalize_ticker_for_logs
        ticker_info = TickerUtils.standardize_ticker(ticker)
        openai_ticker = ticker_info['openai_format']  # Use consistent format for OpenAI
        
        print(f"[SOCIAL] Using ticker format: {openai_ticker} (from input: {normalize_ticker_for_logs(ticker)})")
        
        config = get_config()
        timeout_seconds = float(config.get("stock_news_timeout_seconds", 120))
        max_output_tokens = int(config.get("stock_news_max_output_tokens", 900))
        store_responses = _coerce_bool(config.get("openai_store_responses", False))

        # Use client with timeout for web search operations
        client = get_openai_client_with_timeout(api_key, timeout_seconds=timeout_seconds)

        # Get the selected quick model from config
        model = config.get("quick_think_llm", "gpt-5.4-nano")  # fallback to default
        
        # Research depth controls prompt scope and search context
        research_depth = config.get("research_depth", "Medium")
        depth_key = research_depth.lower() if research_depth else "medium"
        fast_profile = _coerce_bool(config.get("stock_news_fast_profile", True))
        search_context = "low" if fast_profile else get_search_context_for_depth(research_depth)
        
        from datetime import datetime, timedelta
        lookback_days = 3 if depth_key == "shallow" else 7 if depth_key == "medium" else 14
        start_date = (datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        model_params = _quick_model_params_for_tool(
            model,
            config,
            max_output_tokens=max_output_tokens,
            store_responses=store_responses,
        )

        if _uses_responses_for_web_search(model):
            # Use responses.create() API with web search capabilities - use standardized ticker
            user_message = f"Search the web and analyze current social media sentiment and recent news for {ticker_info['display_format']} ({openai_ticker}) from {start_date} to {curr_date}. Include:\n" + \
                          f"1. Overall sentiment analysis from recent social media posts\n" + \
                          f"2. Key themes and discussions happening now\n" + \
                          f"3. Notable price-moving news or events from the past week\n" + \
                          f"4. Trading implications based on current sentiment\n" + \
                          f"5. Summary table with key metrics"
            api_params = _build_web_search_response_params(
                model=model,
                developer_message=(
                    "You are a financial research assistant with web search access. "
                    "Use real-time web search to provide focused social media sentiment "
                    "analysis and recent news about the specified ticker. Prioritize speed "
                    "and key insights."
                ),
                user_message=user_message,
                search_context=search_context,
                max_output_tokens=max_output_tokens,
                store_responses=store_responses,
                model_params=model_params,
            )
            response = _create_response_with_output_cap_fallback(client, api_params)
        else:
            # Use standard chat completions API for GPT-4 and other models
            chat_model_params = get_model_params(model, max_tokens_value=max_output_tokens)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial research assistant. Provide comprehensive social media sentiment analysis and recent news about the specified stock ticker. Focus on sentiment trends, key discussions, and any notable developments."
                    },
                    {
                        "role": "user",
                        "content": f"Analyze social media sentiment and recent news for {ticker_info['display_format']} ({openai_ticker}) from {start_date} to {curr_date}. Include:\n"
                                 f"1. Overall sentiment analysis\n"
                                 f"2. Key themes and discussions\n"
                                 f"3. Notable price-moving news or events\n"
                                 f"4. Trading implications based on sentiment\n"
                                 f"5. Summary table with key metrics"
                    }
                ],
                **chat_model_params
            )

        content = _strip_trailing_interactive_followup(extract_responses_text(response))
        
        # Check if content is empty
        if not content or content.strip() == "":
            return _build_empty_openai_stock_news_fallback(
                ticker=ticker_info["alpaca_format"],
                curr_date=curr_date,
            )
        
        return content
    except Exception as e:
        # Use standardized ticker in error message if available
        display_ticker = ticker
        try:
            display_ticker = normalize_ticker_for_logs(ticker)
        except:
            pass
        fallback = _build_empty_openai_stock_news_fallback(
            ticker=display_ticker,
            curr_date=curr_date,
        )
        return (
            f"OpenAI stock-news call failed for {display_ticker}: {str(e)}\n\n"
            f"{fallback}"
        )


def get_global_news_openai(curr_date, ticker_context=None):
    # Get API key from environment variables or config
    api_key = get_api_key("openai_api_key", "OPENAI_API_KEY")
    if not api_key:
        return f"Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
    
    try:
        config = get_config()
        timeout_seconds = float(config.get("global_news_timeout_seconds", 150))

        # Use client with timeout for web search operations
        client = get_openai_client_with_timeout(api_key, timeout_seconds=timeout_seconds)
        
        # Get the selected quick model from config
        model = config.get("quick_think_llm", "gpt-5.4-nano")  # fallback to default
        
        # Research depth controls prompt scope/search context; global news uses a tuned fast profile.
        research_depth = config.get("research_depth", "Medium")
        depth_key = research_depth.lower() if research_depth else "medium"
        fast_profile = _coerce_bool(config.get("global_news_fast_profile", True))
        profile = get_global_news_profile_for_depth(research_depth, fast_profile=fast_profile)
        search_context = profile["search_context"]
        max_events = int(config.get("global_news_max_events", 8))
        word_budget_default = 550 if depth_key in ("shallow", "medium") else 700
        word_budget = int(config.get("global_news_word_budget", word_budget_default))
        
        from datetime import datetime, timedelta
        lookback_days = int(profile["lookback_days"])
        start_date = (datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        max_output_tokens = int(config.get("global_news_max_output_tokens", 1800))
        store_responses = _coerce_bool(config.get("openai_store_responses", False))
        model_params = _quick_model_params_for_tool(
            model,
            config,
            max_output_tokens=max_output_tokens,
            store_responses=store_responses,
        )
        
        # Determine if this is crypto-related analysis
        is_crypto = ticker_context and ("/" in ticker_context or "USD" in ticker_context.upper() or "BTC" in ticker_context.upper() or "ETH" in ticker_context.upper())
        target = ticker_context if ticker_context else ("crypto markets" if is_crypto else "financial markets")
        if is_crypto:
            focus_points = [
                "Major crypto regulation, CBDC, or enforcement updates",
                "Institutional adoption/ETF flow developments tied to crypto demand",
                "Exchange, custody, security, or infrastructure incidents",
                "DeFi/protocol developments with market-wide relevance",
                "Macro policy or geopolitical events that shift crypto risk appetite",
            ]
            scope_label = "cryptocurrency markets and blockchain ecosystem"
        else:
            focus_points = [
                "Major economic data and growth/inflation surprises",
                "Central-bank policy signals and rates/liquidity implications",
                "Geopolitical events with cross-asset risk impact",
                f"Sector/industry developments relevant to {target}",
                "Cross-asset sentiment shifts with equity-market spillover",
            ]
            scope_label = "financial markets"

        focus_block = "\n".join(f"{i + 1}. {point}" for i, point in enumerate(focus_points))
        user_message = (
            f"Search the web for global news from {start_date} to {curr_date} most relevant to trading {target}.\n"
            "Prioritize catalysts with likely impact over the next 2-10 trading days.\n"
            f"Focus on:\n{focus_block}\n"
            f"Return at most {max_events} events ranked by impact.\n"
            "For each event include: date, what happened, impact level (minor/moderate/major), "
            "expected direction (bullish/bearish/mixed), and a short trading implication.\n"
            "Include a markdown table with columns: Event | Date | Impact | Direction | Trading Implication.\n"
            f"Keep the full response under about {word_budget} words.\n"
            "Do not ask follow-up questions. Do not offer extra files or downloads."
        )
        developer_message = (
            "You are a financial news analyst with web search access. Provide concise, source-grounded "
            f"analysis of global news that can impact {scope_label}. "
            "Write directly in final form and avoid interactive follow-up prompts."
        )
        
        if _uses_responses_for_web_search(model):
            api_params = _build_web_search_response_params(
                model=model,
                developer_message=developer_message,
                user_message=user_message,
                search_context=search_context,
                max_output_tokens=max_output_tokens,
                store_responses=store_responses,
                model_params=model_params,
            )
            response = _create_response_with_output_cap_fallback(client, api_params)
        else:
            # Use standard chat completions API for GPT-4 and other models
            chat_model_params = get_model_params(model, max_tokens_value=max_output_tokens)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": developer_message
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                **chat_model_params
            )

        content = extract_responses_text(response)

        content = _strip_trailing_interactive_followup(content)
        
        # Check if content is empty
        if not content or content.strip() == "":
            return _build_empty_openai_global_fallback(
                curr_date=curr_date,
                ticker_context=ticker_context,
            )
        
        return content
    except Exception as e:
        fallback = _build_empty_openai_global_fallback(
            curr_date=curr_date,
            ticker_context=ticker_context,
        )
        return f"OpenAI global-news call failed: {str(e)}\n\n{fallback}"


def get_fundamentals_openai(ticker, curr_date):
    # Get API key from environment variables or config
    api_key = get_api_key("openai_api_key", "OPENAI_API_KEY")
    if not api_key:
        return f"Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
    
    try:
        config = get_config()
        timeout_seconds = float(config.get("fundamentals_timeout_seconds", 120))
        max_output_tokens = int(config.get("fundamentals_max_output_tokens", 1000))
        store_responses = _coerce_bool(config.get("openai_store_responses", False))

        # Use client with timeout for web search operations
        client = get_openai_client_with_timeout(api_key, timeout_seconds=timeout_seconds)

        # Get the selected quick model from config
        model = config.get("quick_think_llm", "gpt-5.4-nano")  # fallback to default
        
        fundamentals_fast_profile = _coerce_bool(config.get("fundamentals_fast_profile", True))
        if fundamentals_fast_profile:
            search_context = "low"
        else:
            search_context = get_search_context_for_depth()
        
        from datetime import datetime, timedelta
        start_date = (datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")

        model_params = _quick_model_params_for_tool(
            model,
            config,
            max_output_tokens=max_output_tokens,
            store_responses=store_responses,
        )

        if _uses_responses_for_web_search(model):
            # Use responses.create() API with web search capabilities
            # Concise, swing-trading-focused prompt that avoids verbose multi-section output
            user_message = (
                f"Provide a concise fundamental analysis for {ticker} "
                f"covering {start_date} to {curr_date}. "
                f"Be brief and focus on what matters for a 2-10 day swing trade.\n\n"
                f"Cover these in SHORT paragraphs (not long essays):\n"
                f"1. Key valuation snapshot (P/E, EV/EBITDA, P/S — just the numbers)\n"
                f"2. Latest earnings/revenue vs estimates (beat or miss, magnitude)\n"
                f"3. Cash flow & balance sheet health (1-2 sentences)\n"
                f"4. Recent catalysts (earnings, leadership changes, M&A, etc.)\n"
                f"5. Key risk for the next 2-10 days\n\n"
                f"End with a compact summary table of key metrics.\n"
                f"Keep total response under 520 words."
            )
            
            system_text = (
                "You are a fundamental analyst providing concise, swing-trading-focused analysis. "
                "Use web search to find the latest financials but keep your output SHORT and actionable. "
                "Do not write long essays — be direct and data-driven."
            )
            api_params = _build_web_search_response_params(
                model=model,
                developer_message=system_text,
                user_message=user_message,
                search_context=search_context,
                max_output_tokens=max_output_tokens,
                store_responses=store_responses,
                model_params=model_params,
                include_reasoning=True,
            )
            response = _create_response_with_output_cap_fallback(client, api_params)
        else:
            # Use standard chat completions API for GPT-4 and other models
            chat_model_params = get_model_params(model, max_tokens_value=max_output_tokens)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a fundamental analyst specializing in financial analysis and valuation. Provide comprehensive fundamental analysis based on available financial metrics and recent company developments."
                    },
                    {
                        "role": "user",
                        "content": f"Provide a fundamental analysis for {ticker} covering the period from {start_date} to {curr_date}. Include:\n"
                                 f"1. Key financial metrics (P/E, P/S, P/B, EV/EBITDA, etc.)\n"
                                 f"2. Revenue and earnings trends\n"
                                 f"3. Cash flow analysis\n"
                                 f"4. Balance sheet strength\n"
                                 f"5. Competitive positioning\n"
                                 f"6. Recent business developments\n"
                                 f"7. Valuation assessment\n"
                                 f"8. Summary table with key fundamental metrics and ratios\n\n"
                                 f"Format the analysis professionally with clear sections and include a summary table at the end."
                    }
                ],
                **chat_model_params
            )

        content = _strip_trailing_interactive_followup(extract_responses_text(response))
        if not content or not content.strip():
            return _build_empty_openai_fundamentals_fallback(
                ticker=ticker,
                curr_date=curr_date,
            )

        return content
    except Exception as e:
        fallback = _build_empty_openai_fundamentals_fallback(
            ticker=ticker,
            curr_date=curr_date,
        )
        return f"OpenAI fundamentals call failed for {ticker}: {str(e)}\n\n{fallback}"


def get_defillama_fundamentals(
    ticker: Annotated[str, "Crypto ticker symbol (without USD/USDT suffix)"],
    lookback_days: Annotated[int, "Number of days to look back for data"] = 30,
) -> str:
    """
    Get fundamental data for a cryptocurrency from DeFi Llama
    
    Args:
        ticker: Crypto ticker symbol (e.g., BTC, ETH, UNI)
        lookback_days: Number of days to look back for data
        
    Returns:
        str: Markdown-formatted fundamentals report for the cryptocurrency
    """
    # Clean the ticker - remove any USD/USDT suffix if present
    clean_ticker = ticker.upper().replace("USD", "").replace("USDT", "")
    if "/" in clean_ticker:
        clean_ticker = clean_ticker.split("/")[0]
        
    try:
        return get_defillama_fundamentals_util(clean_ticker, lookback_days)
    except Exception as e:
        return f"Error fetching DeFi Llama data for {clean_ticker}: {str(e)}"


def get_alpaca_data_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"] = None,
    look_back_days: Annotated[int, "how many days to look back"] = 60,
    timeframe: Annotated[str, "Timeframe for data: 1Min, 5Min, 15Min, 1Hour, 1Day"] = "1Day",
) -> str:
    """
    Get a window of stock data from Alpaca
    Args:
        symbol: ticker symbol of the company
        curr_date: The current trading date you are trading on, YYYY-mm-dd (optional - if not provided, will use today's date)
        look_back_days: how many days to look back
        timeframe: Timeframe for data (1Min, 5Min, 15Min, 1Hour, 1Day)
    Returns:
        str: a report of the stock data
    """
    try:
        # Calculate start date based on look_back_days
        if curr_date:
            curr_dt = pd.to_datetime(curr_date)
        else:
            curr_dt = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
            
        start_dt = curr_dt - pd.Timedelta(days=look_back_days)
        start_date = start_dt.strftime("%Y-%m-%d")
        
        # Get data from Alpaca - don't pass end_date to avoid subscription limitations
        data = AlpacaUtils.get_stock_data(
            symbol=symbol,
            start_date=start_date,
            timeframe=timeframe
        )
        
        if data.empty:
            return f"No data found for {symbol} from {start_date} to present"
        
        # Format the result
        result = f"## Stock data for {symbol} from {start_date} to present:\n\n"
        result += data.to_string()
        
        # Add latest quote if available
        try:
            latest_quote = AlpacaUtils.get_latest_quote(symbol)
            if latest_quote:
                result += f"\n\n## Latest Quote for {symbol}:\n"
                result += f"Bid: {latest_quote['bid_price']} ({latest_quote['bid_size']}), "
                result += f"Ask: {latest_quote['ask_price']} ({latest_quote['ask_size']}), "
                result += f"Time: {latest_quote['timestamp']}"
        except Exception as quote_error:
            result += f"\n\nCould not fetch latest quote: {str(quote_error)}"
        
        return result
    except Exception as e:
        return f"Error getting stock data for {symbol}: {str(e)}"

def get_alpaca_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"] = None,
    timeframe: Annotated[str, "Timeframe for data: 1Min, 5Min, 15Min, 1Hour, 1Day"] = "1Day",
) -> str:
    """
    Get stock data from Alpaca
    Args:
        symbol: ticker symbol of the company
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format (optional - if not provided, will fetch up to latest available data)
        timeframe: Timeframe for data (1Min, 5Min, 15Min, 1Hour, 1Day)
    Returns:
        str: a report of the stock data
    """
    try:
        # Get data from Alpaca
        data = AlpacaUtils.get_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )
        
        if data.empty:
            date_range = f"from {start_date}" + (f" to {end_date}" if end_date else " to present")
            return f"No data found for {symbol} {date_range}"
        
        # Create a copy for formatting
        df_formatted = data.copy()
        
        # Format timestamp to be more readable (convert to date only for daily data)
        if timeframe == "1Day":
            df_formatted['date'] = pd.to_datetime(df_formatted['timestamp']).dt.strftime('%Y-%m-%d')
        else:
            df_formatted['date'] = pd.to_datetime(df_formatted['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Reorder columns for better readability
        columns_order = ['date', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
        available_columns = [col for col in columns_order if col in df_formatted.columns]
        df_display = df_formatted[available_columns].copy()
        
        # Round price columns for better readability
        price_columns = ['open', 'high', 'low', 'close', 'vwap']
        for col in price_columns:
            if col in df_display.columns:
                df_display[col] = df_display[col].round(2)
        
        # Format volume with thousands separators
        if 'volume' in df_display.columns:
            df_display['volume'] = df_display['volume'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")
        
        if 'trade_count' in df_display.columns:
            df_display['trade_count'] = df_display['trade_count'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")
        
        # Calculate some key metrics
        if len(df_formatted) > 1:
            current_close = df_formatted.iloc[-1]['close']
            previous_close = df_formatted.iloc[-2]['close']
            daily_change = current_close - previous_close
            daily_change_pct = (daily_change / previous_close) * 100
            
            current_volume = df_formatted.iloc[-1]['volume']
            avg_volume = df_formatted['volume'].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        else:
            daily_change = daily_change_pct = volume_ratio = 0
            current_close = df_formatted.iloc[0]['close'] if not df_formatted.empty else 0
        
        # Format the result
        date_range = f"from {start_date}" + (f" to {end_date}" if end_date else " to present")
        result = f"## Stock Data for {symbol} {date_range}:\n\n"
        result += df_display.to_string(index=False)
        
        # Add key metrics summary
        if len(df_formatted) > 1:
            result += f"\n\n## Key Swing Trading Metrics:\n"
            result += f"Current Close: ${current_close:.2f}\n"
            result += f"Daily Change: ${daily_change:.2f} ({daily_change_pct:+.2f}%)\n"
            result += f"Volume vs Avg: {volume_ratio:.2f}x ({int(current_volume):,} vs {int(avg_volume):,})\n"
            
            # Add daily range info
            latest_data = df_formatted.iloc[-1]
            daily_range = latest_data['high'] - latest_data['low']
            range_pct = (daily_range / latest_data['close']) * 100
            result += f"Daily Range: ${latest_data['low']:.2f} - ${latest_data['high']:.2f} ({range_pct:.2f}%)\n"
        
        # Add latest quote if available
        try:
            latest_quote = AlpacaUtils.get_latest_quote(symbol)
            if latest_quote:
                result += f"\n## Latest Real-Time Quote:\n"
                result += f"Bid: ${latest_quote['bid_price']:.2f} (Size: {int(latest_quote['bid_size']):,})\n"
                result += f"Ask: ${latest_quote['ask_price']:.2f} (Size: {int(latest_quote['ask_size']):,})\n"
                result += f"Spread: ${float(latest_quote['ask_price']) - float(latest_quote['bid_price']):.2f}\n"
                
                # Calculate quote vs close difference if we have close data
                if not data.empty:
                    mid_quote = (float(latest_quote['bid_price']) + float(latest_quote['ask_price'])) / 2
                    last_close = data.iloc[-1]['close']
                    after_hours_change = mid_quote - last_close
                    after_hours_pct = (after_hours_change / last_close) * 100
                    result += f"After-Hours Move: ${after_hours_change:+.2f} ({after_hours_pct:+.2f}%)\n"
                    
        except Exception as quote_error:
            result += f"\n\nNote: Real-time quote unavailable: {str(quote_error)}"
        
        return result
    except Exception as e:
        return f"Error getting stock data for {symbol}: {str(e)}"


def get_technical_brief(
    symbol: Annotated[str, "ticker symbol (stocks: AAPL; crypto: BTC/USD)"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
) -> str:
    """
    Build a standardized Technical Brief for LLM consumption.

    Computes indicators across 1h / 4h / 1d timeframes deterministically,
    then returns a compact JSON with trend, momentum, VWAP state, volatility,
    market structure, key levels, and an aggregate signal summary.

    Args:
        symbol: Ticker symbol (e.g. AAPL, BTC/USD)
        curr_date: The current trading date in YYYY-mm-dd format

    Returns:
        str: JSON string of the TechnicalBrief
    """
    from .technical_brief import build_technical_brief

    try:
        brief = build_technical_brief(symbol, curr_date)
        return brief.model_dump_json(indent=2)
    except Exception as e:
        import json as _json
        return _json.dumps({
            "error": f"Failed to build technical brief for {symbol}: {str(e)}",
            "symbol": symbol,
            "generated_at": curr_date,
        })


def get_earnings_calendar(
    ticker: Annotated[str, "Stock or crypto ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve earnings calendar data for stocks or major events for crypto.
    For stocks: Shows earnings dates, EPS estimates vs actuals, revenue estimates vs actuals, and surprise analysis.
    For crypto: Shows major protocol events, upgrades, and announcements that could impact price.
    
    Args:
        ticker (str): Stock ticker (e.g. AAPL, TSLA) or crypto ticker (e.g. BTC/USD, ETH/USD, SOL/USD)
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
        
    Returns:
        str: Formatted earnings calendar data with estimates, actuals, and surprise analysis
    """
    
    return get_earnings_calendar_data(ticker, start_date, end_date)


def get_earnings_surprise_analysis(
    ticker: Annotated[str, "Stock ticker symbol"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    lookback_quarters: Annotated[int, "Number of quarters to analyze"] = 8,
) -> str:
    """
    Analyze historical earnings surprises to identify patterns and trading implications.
    Shows consistency of beats/misses, magnitude of surprises, and seasonal patterns.
    
    Args:
        ticker (str): Stock ticker symbol, e.g. AAPL, TSLA
        curr_date (str): Current date in yyyy-mm-dd format
        lookback_quarters (int): Number of quarters to analyze (default 8 = ~2 years)
        
    Returns:
        str: Analysis of earnings surprise patterns with trading implications
    """
    
    return get_earnings_surprises_analysis(ticker, curr_date, lookback_quarters)


def get_macro_analysis(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    lookback_days: Annotated[int, "Number of days to look back for data"] = 90,
) -> str:
    """
    Retrieve comprehensive macro economic analysis including Fed funds, CPI, PPI, NFP, GDP, PMI, Treasury curve, VIX.
    Provides economic indicators, yield curve analysis, Fed policy updates, and trading implications.
    
    Args:
        curr_date (str): Current date in yyyy-mm-dd format
        lookback_days (int): Number of days to look back for data (default 90)
        
    Returns:
        str: Comprehensive macro economic analysis with trading implications
    """
    
    return get_macro_economic_summary(curr_date)


def get_economic_indicators(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    lookback_days: Annotated[int, "Number of days to look back for data"] = 90,
) -> str:
    """
    Retrieve key economic indicators report including Fed funds, CPI, PPI, unemployment, NFP, GDP, PMI, VIX.
    
    Args:
        curr_date (str): Current date in yyyy-mm-dd format
        lookback_days (int): Number of days to look back for data (default 90)
        
    Returns:
        str: Economic indicators report with analysis and interpretations
    """
    
    return get_economic_indicators_report(curr_date, lookback_days)


def get_yield_curve_analysis(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve Treasury yield curve analysis including inversion signals and recession indicators.
    
    Args:
        curr_date (str): Current date in yyyy-mm-dd format
        
    Returns:
        str: Treasury yield curve data with inversion analysis
    """
    
    return get_treasury_yield_curve(curr_date)
