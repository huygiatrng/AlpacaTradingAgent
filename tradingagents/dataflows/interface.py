from typing import Annotated, Dict
from .reddit_utils import fetch_top_from_category
from .stockstats_utils import *
from .googlenews_utils import *
from .finnhub_utils import get_data_in_range
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
from tqdm import tqdm
from openai import OpenAI
import httpx
from .config import get_config, set_config, DATA_DIR, get_api_key


def get_openai_client_with_timeout(api_key, timeout_seconds=300):
    """Create OpenAI client with configurable timeout for slow web search operations."""
    return OpenAI(
        api_key=api_key,
        timeout=httpx.Timeout(timeout_seconds, connect=10.0)
    )


def get_search_context_for_depth(research_depth=None):
    """Get the appropriate search_context_size based on research depth.
    
    Args:
        research_depth: "Shallow", "Medium", or "Deep" (or None for default)
    
    Returns:
        str: "low", "medium", or "high" for web search context size
    
    Research Depth Mapping:
        - Shallow: "low" - Faster, less comprehensive (5-10 sources)
        - Medium: "medium" - Balanced (10-20 sources)
        - Deep: "high" - Most comprehensive, slowest (20+ sources)
    """
    if research_depth is None:
        config = get_config()
        research_depth = config.get("research_depth", "Medium")
    
    depth_mapping = {
        "shallow": "low",
        "medium": "medium", 
        "deep": "high"
    }
    
    return depth_mapping.get(research_depth.lower() if research_depth else "medium", "medium")


def get_llm_params_for_depth(research_depth=None):
    """Get reasoning effort and verbosity matching the research depth.

    Returns:
        dict with keys ``effort`` and ``verbosity`` (both str).
    """
    if research_depth is None:
        config = get_config()
        research_depth = config.get("research_depth", "Medium")

    depth = research_depth.lower() if research_depth else "medium"
    mapping = {
        "shallow": {"effort": "low",    "verbosity": "low"},
        "medium":  {"effort": "medium",  "verbosity": "medium"},
        "deep":    {"effort": "high",    "verbosity": "high"},
    }
    return mapping.get(depth, mapping["medium"])


def get_model_params(model_name, max_tokens_value=3000):
    """Get appropriate parameters for different model types."""
    params = {}
    
    # GPT-5 and GPT-4.1 models use the responses.create() API 
    # Older models use the standard chat.completions.create() API
    gpt5_models = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]
    gpt52_models = ["gpt-5.2", "gpt-5.2-pro"]
    gpt41_models = ["gpt-4.1"]
    
    if any(model_prefix in model_name for model_prefix in gpt52_models):
        # GPT-5.2 models: use responses.create() API with specific parameters
        params["text"] = {"format": "text"}
        params["summary"] = "auto"
        
        if "gpt-5.2-pro" in model_name:
            # GPT-5.2-pro specific: store responses for fine-tuning
            params["store"] = True
        else:
            # GPT-5.2 specific: effort and verbosity controls
            params["reasoning"] = {"effort": "medium"}
            params["verbosity"] = "medium"
    elif any(model_prefix in model_name for model_prefix in gpt5_models):
        # GPT-5 models: use responses.create() API with no token parameters
        # Token limits are handled by the model automatically
        pass  # No additional parameters needed for GPT-5
    elif any(model_prefix in model_name for model_prefix in gpt41_models):
        # GPT-4.1 models: use responses.create() API with specific parameters
        params["temperature"] = 0.2
        params["max_output_tokens"] = max_tokens_value
        params["top_p"] = 1
    else:
        # Standard models (GPT-4, etc.)
        params["temperature"] = 0.2
        params["max_tokens"] = max_tokens_value
    
    return params


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

    result = get_data_in_range(ticker, before, curr_date, "news_data", DATA_DIR)

    if len(result) == 0:
        return ""

    combined_result = ""
    for day, data in result.items():
        if len(data) == 0:
            continue
        for entry in data:
            current_news = (
                "### " + entry["headline"] + f" ({day})" + "\n" + entry["summary"]
            )
            combined_result += current_news + "\n\n"

    return f"## {ticker} News, from {before} to {curr_date}:\n" + str(combined_result)


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

    data = get_data_in_range(ticker, before, curr_date, "insider_senti", DATA_DIR)

    if len(data) == 0:
        return ""

    result_str = ""
    seen_dicts = []
    for date, senti_list in data.items():
        for entry in senti_list:
            if entry not in seen_dicts:
                result_str += f"### {entry['year']}-{entry['month']}:\nChange: {entry['change']}\nMonthly Share Purchase Ratio: {entry['mspr']}\n\n"
                seen_dicts.append(entry)

    return (
        f"## {ticker} Insider Sentiment Data for {before} to {curr_date}:\n"
        + result_str
        + "The change field refers to the net buying/selling from all insiders' transactions. The mspr field refers to monthly share purchase ratio."
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

    data = get_data_in_range(ticker, before, curr_date, "insider_trans", DATA_DIR)

    if len(data) == 0:
        return ""

    result_str = ""

    seen_dicts = []
    for date, senti_list in data.items():
        for entry in senti_list:
            if entry not in seen_dicts:
                result_str += f"### Filing Date: {entry['filingDate']}, {entry['name']}:\nChange:{entry['change']}\nShares: {entry['share']}\nTransaction Price: {entry['transactionPrice']}\nTransaction Code: {entry['transactionCode']}\n\n"
                seen_dicts.append(entry)

    return (
        f"## {ticker} insider transactions from {before} to {curr_date}:\n"
        + result_str
        + "The change field reflects the variation in share count—here a negative number indicates a reduction in holdings—while share specifies the total number of shares involved. The transactionPrice denotes the per-share price at which the trade was executed, and transactionDate marks when the transaction occurred. The name field identifies the insider making the trade, and transactionCode (e.g., S for sale) clarifies the nature of the transaction. FilingDate records when the transaction was officially reported, and the unique id links to the specific SEC filing, as indicated by the source. Additionally, the symbol ties the transaction to a particular company, isDerivative flags whether the trade involves derivative securities, and currency notes the currency context of the transaction."
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
    df = pd.read_csv(data_path, sep=";")

    # Convert date strings to datetime objects and remove any time components
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()

    # Convert the current date to datetime and normalize
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()

    # Filter the DataFrame for the given ticker and for reports that were published on or before the current date
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    # Check if there are any available reports; if not, return a notification
    if filtered_df.empty:
        print("No balance sheet available before the given current date.")
        return ""

    # Get the most recent balance sheet by selecting the row with the latest Publish Date
    latest_balance_sheet = filtered_df.loc[filtered_df["Publish Date"].idxmax()]

    # drop the SimFinID column
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
    df = pd.read_csv(data_path, sep=";")

    # Convert date strings to datetime objects and remove any time components
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()

    # Convert the current date to datetime and normalize
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()

    # Filter the DataFrame for the given ticker and for reports that were published on or before the current date
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    # Check if there are any available reports; if not, return a notification
    if filtered_df.empty:
        print("No cash flow statement available before the given current date.")
        return ""

    # Get the most recent cash flow statement by selecting the row with the latest Publish Date
    latest_cash_flow = filtered_df.loc[filtered_df["Publish Date"].idxmax()]

    # drop the SimFinID column
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
    df = pd.read_csv(data_path, sep=";")

    # Convert date strings to datetime objects and remove any time components
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()

    # Convert the current date to datetime and normalize
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()

    # Filter the DataFrame for the given ticker and for reports that were published on or before the current date
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    # Check if there are any available reports; if not, return a notification
    if filtered_df.empty:
        print("No income statement available before the given current date.")
        return ""

    # Get the most recent income statement by selecting the row with the latest Publish Date
    latest_income = filtered_df.loc[filtered_df["Publish Date"].idxmax()]

    # drop the SimFinID column
    latest_income = latest_income.drop("SimFinId")

    return (
        f"## {freq} income statement for {ticker} released on {str(latest_income['Publish Date'])[0:10]}: \n"
        + str(latest_income)
        + "\n\nThis includes metadata like reporting dates and currency, share details, and a comprehensive breakdown of the company's financial performance. Starting with Revenue, it shows Cost of Revenue and resulting Gross Profit. Operating Expenses are detailed, including SG&A, R&D, and Depreciation. The statement then shows Operating Income, followed by non-operating items and Interest Expense, leading to Pretax Income. After accounting for Income Tax and any Extraordinary items, it concludes with Net Income, representing the company's bottom-line profit or loss for the period."
    )


def get_google_news(
    query: Annotated[str, "Query to search with"],
    curr_date: Annotated[str, "Curr date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    query = query.replace(" ", "+")

    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    # Limit to 2 pages for better performance (about 20 articles max)
    news_results = getNewsData(query, before, curr_date, max_pages=2)

    news_str = ""

    for news in news_results:
        news_str += (
            f"### {news['title']} (source: {news['source']}) \n\n{news['snippet']}\n\n"
        )

    if len(news_results) == 0:
        return ""

    return f"## {query} Google News, from {before} to {curr_date}:\n\n{news_str}"


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

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    posts = []
    # iterate from start_date to end_date
    curr_date = datetime.strptime(before, "%Y-%m-%d")

    total_iterations = (start_date - curr_date).days + 1
    pbar = tqdm(desc=f"Getting Global News on {start_date}", total=total_iterations)

    while curr_date <= start_date:
        curr_date_str = curr_date.strftime("%Y-%m-%d")
        fetch_result = fetch_top_from_category(
            "global_news",
            curr_date_str,
            max_limit_per_day,
            data_path=os.path.join(DATA_DIR, "reddit_data"),
        )
        posts.extend(fetch_result)
        curr_date += relativedelta(days=1)
        pbar.update(1)

    pbar.close()

    if len(posts) == 0:
        return ""

    news_str = ""
    for post in posts:
        if post["content"] == "":
            news_str += f"### {post['title']}\n\n"
        else:
            news_str += f"### {post['title']}\n\n{post['content']}\n\n"

    return f"## Global News Reddit, from {before} to {curr_date}:\n{news_str}"


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

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    posts = []
    # iterate from start_date to end_date
    curr_date = datetime.strptime(before, "%Y-%m-%d")

    total_iterations = (start_date - curr_date).days + 1
    pbar = tqdm(
        desc=f"Getting Company News for {ticker} on {start_date}",
        total=total_iterations,
    )

    while curr_date <= start_date:
        curr_date_str = curr_date.strftime("%Y-%m-%d")
        fetch_result = fetch_top_from_category(
            "company_news",
            curr_date_str,
            max_limit_per_day,
            ticker,
            data_path=os.path.join(DATA_DIR, "reddit_data"),
        )
        posts.extend(fetch_result)
        curr_date += relativedelta(days=1)

        pbar.update(1)

    pbar.close()

    if len(posts) == 0:
        return ""

    news_str = ""
    for post in posts:
        if post["content"] == "":
            news_str += f"### {post['title']}\n\n"
        else:
            news_str += f"### {post['title']}\n\n{post['content']}\n\n"

    return f"##{ticker} News Reddit, from {before} to {curr_date}:\n\n{news_str}"


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
        
        # Use client with timeout for web search operations
        client = get_openai_client_with_timeout(api_key)
        
        # Get the selected quick model from config
        config = get_config()
        model = config.get("quick_think_llm", "gpt-4o-mini")  # fallback to default
        
        # Research depth controls prompt scope and search context
        research_depth = config.get("research_depth", "Medium")
        depth_key = research_depth.lower() if research_depth else "medium"
        search_context = get_search_context_for_depth(research_depth)
        
        from datetime import datetime, timedelta
        lookback_days = 3 if depth_key == "shallow" else 7 if depth_key == "medium" else 14
        start_date = (datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        # Get model-specific parameters
        model_params = get_model_params(model)
        
        # Check if this is a GPT-5/GPT-5.2 or GPT-4.1 model (both use responses.create())
        gpt5_models = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]
        gpt52_models = ["gpt-5.2", "gpt-5.2-pro"]
        gpt41_models = ["gpt-4.1"]
        is_gpt5 = any(model_prefix in model for model_prefix in gpt5_models)
        is_gpt52 = any(model_prefix in model for model_prefix in gpt52_models)
        is_gpt41 = any(model_prefix in model for model_prefix in gpt41_models)
        
        if is_gpt5 or is_gpt52 or is_gpt41:
            # Use responses.create() API with web search capabilities - use standardized ticker
            user_message = f"Search the web and analyze current social media sentiment and recent news for {ticker_info['display_format']} ({openai_ticker}) from {start_date} to {curr_date}. Include:\n" + \
                          f"1. Overall sentiment analysis from recent social media posts\n" + \
                          f"2. Key themes and discussions happening now\n" + \
                          f"3. Notable price-moving news or events from the past week\n" + \
                          f"4. Trading implications based on current sentiment\n" + \
                          f"5. Summary table with key metrics"
            
            # Base parameters for responses.create()
            if is_gpt52:
                # GPT-5.2 uses "developer" role with specific parameters
                api_params = {
                    "model": model,
                    "input": [
                        {
                            "role": "developer",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "You are a financial research assistant with web search access. Use real-time web search to provide focused social media sentiment analysis and recent news about the specified ticker. Prioritize speed and key insights."
                                }
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": user_message
                                }
                            ]
                        }
                    ],
                    "text": {"format": {"type": "text"}},
                    "tools": [{
                        "type": "web_search",
                        "user_location": {"type": "approximate"},
                        "search_context_size": search_context
                    }],
                    "include": ["web_search_call.action.sources"]
                }
                # Apply GPT-5.2 specific parameters
                api_params["summary"] = "auto"
                if "gpt-5.2-pro" in model:
                    api_params["store"] = True
                else:
                    effort_map = {"shallow": "low", "medium": "medium", "deep": "high"}
                    verbosity_map = {"shallow": "low", "medium": "medium", "deep": "high"}
                    api_params["reasoning"] = {"effort": effort_map.get(depth_key, "medium")}
                    api_params["verbosity"] = verbosity_map.get(depth_key, "medium")
            elif is_gpt5:
                # GPT-5 uses "developer" role - optimized for speed
                api_params = {
                    "model": model,
                    "input": [
                        {
                            "role": "developer",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "You are a financial research assistant with web search access. Use real-time web search to provide focused social media sentiment analysis and recent news about the specified ticker. Prioritize speed and key insights."
                                }
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": user_message
                                }
                            ]
                        }
                    ],
                    "text": {"format": {"type": "text"}, "verbosity": "low"},
                    "reasoning": {"effort": "low", "summary": "auto"},
                    "tools": [{
                        "type": "web_search",
                        "user_location": {"type": "approximate"},
                        "search_context_size": "low"
                    }],
                    "store": True,
                    "include": ["web_search_call.action.sources"]
                }
            elif is_gpt41:
                # GPT-4.1 uses "system" role  
                api_params = {
                    "model": model,
                    "input": [
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "You are a financial research assistant with web search access. Use real-time web search to provide comprehensive social media sentiment analysis and recent news about the specified stock ticker. Focus on sentiment trends, key discussions, and any notable developments."
                                }
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": user_message
                                }
                            ]
                        }
                    ],
                    "text": {"format": {"type": "text"}},
                    "reasoning": {},
                    "tools": [{
                        "type": "web_search",
                        "user_location": {"type": "approximate"},
                        "search_context_size": search_context
                    }],
                    "store": True,
                    "include": ["web_search_call.action.sources"]
                }
                api_params.update(model_params)  # Add temperature, max_output_tokens, top_p
            
            response = client.responses.create(**api_params)
        else:
            # Use standard chat completions API for GPT-4 and other models
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
                **model_params
            )

        # Parse response based on API type
        if is_gpt5 or is_gpt52 or is_gpt41:
            # Extract content from GPT-5/GPT-5.2 responses.create() structure
            content = None
            if hasattr(response, 'output_text') and response.output_text:
                content = response.output_text
            elif hasattr(response, 'output') and response.output:
                # Navigate through output array to find text content
                for item in response.output:
                    if hasattr(item, 'content') and item.content:
                        for content_item in item.content:
                            if hasattr(content_item, 'text'):
                                content = content_item.text
                                break
                        if content:
                            break
                if not content:
                    content = str(response.output)
            else:
                content = str(response)
        else:
            content = response.choices[0].message.content  # Standard chat.completions.create() structure
        
        # Check if content is empty
        if not content or content.strip() == "":
            return f"Error: Empty response from model {model}. This may indicate the model used all tokens for reasoning."
        
        return content
    except Exception as e:
        # Use standardized ticker in error message if available
        display_ticker = ticker
        try:
            display_ticker = normalize_ticker_for_logs(ticker)
        except:
            pass
        return f"Error fetching social media analysis for {display_ticker}: {str(e)}"


def get_global_news_openai(curr_date, ticker_context=None):
    # Get API key from environment variables or config
    api_key = get_api_key("openai_api_key", "OPENAI_API_KEY")
    if not api_key:
        return f"Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
    
    try:
        # Use client with timeout for web search operations
        client = get_openai_client_with_timeout(api_key)
        
        # Get the selected quick model from config
        config = get_config()
        model = config.get("quick_think_llm", "gpt-4o-mini")  # fallback to default
        
        # Research depth controls prompt scope and search context
        research_depth = config.get("research_depth", "Medium")
        depth_key = research_depth.lower() if research_depth else "medium"
        search_context = get_search_context_for_depth(research_depth)
        
        from datetime import datetime, timedelta
        lookback_days = 3 if depth_key == "shallow" else 7 if depth_key == "medium" else 14
        start_date = (datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        # Get model-specific parameters
        model_params = get_model_params(model)
        
        # Check if this is a GPT-5/GPT-5.2 or GPT-4.1 model (both use responses.create())
        gpt5_models = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]
        gpt52_models = ["gpt-5.2", "gpt-5.2-pro"]
        gpt41_models = ["gpt-4.1"]
        is_gpt5 = any(model_prefix in model for model_prefix in gpt5_models)
        is_gpt52 = any(model_prefix in model for model_prefix in gpt52_models)
        is_gpt41 = any(model_prefix in model for model_prefix in gpt41_models)
        
        # Determine if this is crypto-related analysis
        is_crypto = ticker_context and ("/" in ticker_context or "USD" in ticker_context.upper() or "BTC" in ticker_context.upper() or "ETH" in ticker_context.upper())
        
        if is_gpt5 or is_gpt52 or is_gpt41:
            # Use responses.create() API with web search capabilities
            if is_crypto:
                if depth_key == "shallow":
                    user_message = (
                        f"Search the web for key crypto market news from {start_date} to {curr_date} that could impact "
                        f"{ticker_context if ticker_context else 'crypto'} trading. Focus on:\n"
                        f"1. Major regulatory headlines\n"
                        f"2. Major exchange or security events\n"
                        f"3. Macro events affecting crypto sentiment\n"
                        f"4. Summary table with key events and impact levels"
                    )
                else:
                    user_message = f"Search the web for current global news and developments from {start_date} to {curr_date} that would impact cryptocurrency markets and {ticker_context if ticker_context else 'crypto'} trading. Include:\n" + \
                                  f"1. Major cryptocurrency and blockchain regulatory developments\n" + \
                                  f"2. Central bank digital currency (CBDC) announcements and crypto policy updates\n" + \
                                  f"3. Institutional crypto adoption, ETF developments, and major investment flows\n" + \
                                  f"4. Major DeFi, smart contract, and blockchain protocol developments\n" + \
                                  f"5. Crypto exchange developments, security issues, and market infrastructure news\n" + \
                                  f"6. Macro events affecting crypto (Fed policy, inflation data, geopolitical developments)\n" + \
                                  f"7. Trading implications and crypto market sentiment\n" + \
                                  f"8. Summary table with key events and impact levels on crypto markets"
            else:
                if depth_key == "shallow":
                    user_message = (
                        f"Search the web for key global and macro news from {start_date} to {curr_date} that could impact "
                        f"{ticker_context if ticker_context else 'financial markets'}. Focus on:\n"
                        f"1. Major economic events and announcements\n"
                        f"2. Central bank policy updates\n"
                        f"3. Geopolitical developments affecting markets\n"
                        f"4. Summary table with key events and impact levels"
                    )
                else:
                    user_message = f"Search the web for current global and macroeconomic news from {start_date} to {curr_date} that would be informative for trading {ticker_context if ticker_context else 'financial markets'}. Include:\n" + \
                                  f"1. Major economic events and announcements\n" + \
                                  f"2. Central bank policy updates\n" + \
                                  f"3. Geopolitical developments affecting markets\n" + \
                                  f"4. Economic data releases and their implications\n" + \
                                  f"5. Sector-specific developments relevant to {ticker_context if ticker_context else 'the market'}\n" + \
                                  f"6. Trading implications and market sentiment\n" + \
                                  f"7. Summary table with key events and impact levels"
            
            # Base parameters for responses.create()
            if is_gpt52:
                # GPT-5.2 uses "developer" role with specific parameters
                api_params = {
                    "model": model,
                    "input": [
                        {
                            "role": "developer",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": f"You are a financial news analyst with web search access. Use real-time web search to provide comprehensive analysis of global news that could impact {'cryptocurrency markets and blockchain ecosystem' if is_crypto else 'financial markets'} and trading decisions."
                                }
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": user_message
                                }
                            ]
                        }
                    ],
                    "text": {"format": {"type": "text"}},
                    "tools": [{
                        "type": "web_search",
                        "user_location": {"type": "approximate"},
                        "search_context_size": search_context
                    }],
                    "include": ["web_search_call.action.sources"]
                }
                # Apply GPT-5.2 specific parameters
                api_params["summary"] = "auto"
                if "gpt-5.2-pro" in model:
                    api_params["store"] = True
                else:
                    effort_map = {"shallow": "low", "medium": "medium", "deep": "high"}
                    verbosity_map = {"shallow": "low", "medium": "medium", "deep": "high"}
                    api_params["reasoning"] = {"effort": effort_map.get(depth_key, "medium")}
                    api_params["verbosity"] = verbosity_map.get(depth_key, "medium")
            elif is_gpt5:
                # GPT-5 uses "developer" role
                api_params = {
                    "model": model,
                    "input": [
                        {
                            "role": "developer",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": f"You are a financial news analyst with web search access. Use real-time web search to provide comprehensive analysis of global news that could impact {'cryptocurrency markets and blockchain ecosystem' if is_crypto else 'financial markets'} and trading decisions."
                                }
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": user_message
                                }
                            ]
                        }
                    ],
                    "text": {"format": {"type": "text"}, "verbosity": "medium"},
                    "reasoning": {"effort": "medium", "summary": "auto"},
                    "tools": [{
                        "type": "web_search",
                        "user_location": {"type": "approximate"},
                        "search_context_size": search_context
                    }],
                    "store": True,
                    "include": ["reasoning.encrypted_content", "web_search_call.action.sources"]
                }
                effort_map = {"shallow": "low", "medium": "medium", "deep": "high"}
                verbosity_map = {"shallow": "low", "medium": "medium", "deep": "high"}
                api_params["reasoning"]["effort"] = effort_map.get(depth_key, "medium")
                api_params["text"]["verbosity"] = verbosity_map.get(depth_key, "medium")
            elif is_gpt41:
                # GPT-4.1 uses "system" role  
                api_params = {
                    "model": model,
                    "input": [
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": f"You are a financial news analyst with web search access. Use real-time web search to provide comprehensive analysis of global news that could impact {'cryptocurrency markets and blockchain ecosystem' if is_crypto else 'financial markets'} and trading decisions."
                                }
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": user_message
                                }
                            ]
                        }
                    ],
                    "text": {"format": {"type": "text"}},
                    "reasoning": {},
                    "tools": [{
                        "type": "web_search",
                        "user_location": {"type": "approximate"},
                        "search_context_size": search_context
                    }],
                    "store": True,
                    "include": ["web_search_call.action.sources"]
                }
                api_params.update(model_params)  # Add temperature, max_output_tokens, top_p
            
            response = client.responses.create(**api_params)
        else:
            # Use standard chat completions API for GPT-4 and other models
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a financial news analyst. Provide comprehensive analysis of global news that could impact {'cryptocurrency markets and blockchain ecosystem' if is_crypto else 'financial markets'} and trading decisions."
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                **model_params
            )

        # Parse response based on API type
        if is_gpt5 or is_gpt52 or is_gpt41:
            # Extract content from GPT-5/GPT-5.2 responses.create() structure
            content = None
            if hasattr(response, 'output_text') and response.output_text:
                content = response.output_text
            elif hasattr(response, 'output') and response.output:
                # Navigate through output array to find text content
                for item in response.output:
                    if hasattr(item, 'content') and item.content:
                        for content_item in item.content:
                            if hasattr(content_item, 'text'):
                                content = content_item.text
                                break
                        if content:
                            break
                if not content:
                    content = str(response.output)
            else:
                content = str(response)
        else:
            content = response.choices[0].message.content  # Standard chat.completions.create() structure
        
        # Check if content is empty
        if not content or content.strip() == "":
            return f"Error: Empty response from model {model}. This may indicate the model used all tokens for reasoning."
        
        return content
    except Exception as e:
        return f"Error fetching global news analysis: {str(e)}"


def get_fundamentals_openai(ticker, curr_date):
    # Get API key from environment variables or config
    api_key = get_api_key("openai_api_key", "OPENAI_API_KEY")
    if not api_key:
        return f"Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
    
    try:
        # Use client with timeout for web search operations
        client = get_openai_client_with_timeout(api_key)
        
        # Get the selected quick model from config
        config = get_config()
        model = config.get("quick_think_llm", "gpt-4o-mini")  # fallback to default
        
        # Get search context size and LLM params based on research depth
        search_context = get_search_context_for_depth()
        depth_params = get_llm_params_for_depth()
        
        from datetime import datetime, timedelta
        start_date = (datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")

        # Get model-specific parameters
        model_params = get_model_params(model)
        
        # Check if this is a GPT-5/GPT-5.2 or GPT-4.1 model (both use responses.create())
        gpt5_models = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]
        gpt52_models = ["gpt-5.2", "gpt-5.2-pro"]
        gpt41_models = ["gpt-4.1"]
        is_gpt5 = any(model_prefix in model for model_prefix in gpt5_models)
        is_gpt52 = any(model_prefix in model for model_prefix in gpt52_models)
        is_gpt41 = any(model_prefix in model for model_prefix in gpt41_models)
        
        if is_gpt5 or is_gpt52 or is_gpt41:
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
                f"Keep total response under 800 words."
            )
            
            system_text = (
                "You are a fundamental analyst providing concise, swing-trading-focused analysis. "
                "Use web search to find the latest financials but keep your output SHORT and actionable. "
                "Do not write long essays — be direct and data-driven."
            )
            
            # Base parameters for responses.create()
            if is_gpt52:
                # GPT-5.2 uses "developer" role with specific parameters
                api_params = {
                    "model": model,
                    "input": [
                        {
                            "role": "developer",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": system_text
                                }
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": user_message
                                }
                            ]
                        }
                    ],
                    "text": {"format": {"type": "text"}},
                    "tools": [{
                        "type": "web_search",
                        "user_location": {"type": "approximate"},
                        "search_context_size": search_context
                    }],
                    "include": ["web_search_call.action.sources"]
                }
                # Apply depth-aware parameters
                api_params["summary"] = "auto"
                if "gpt-5.2-pro" in model:
                    api_params["store"] = True
                else:
                    api_params["reasoning"] = {"effort": depth_params["effort"]}
                    api_params["verbosity"] = depth_params["verbosity"]
            elif is_gpt5:
                # GPT-5 uses "developer" role
                api_params = {
                    "model": model,
                    "input": [
                        {
                            "role": "developer",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": system_text
                                }
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": user_message
                                }
                            ]
                        }
                    ],
                    "text": {"format": {"type": "text"}, "verbosity": depth_params["verbosity"]},
                    "reasoning": {"effort": depth_params["effort"], "summary": "auto"},
                    "tools": [{
                        "type": "web_search",
                        "user_location": {"type": "approximate"},
                        "search_context_size": search_context
                    }],
                    "store": True,
                    "include": ["reasoning.encrypted_content", "web_search_call.action.sources"]
                }
            elif is_gpt41:
                # GPT-4.1 uses "system" role  
                api_params = {
                    "model": model,
                    "input": [
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": system_text
                                }
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": user_message
                                }
                            ]
                        }
                    ],
                    "text": {"format": {"type": "text"}},
                    "reasoning": {},
                    "tools": [{
                        "type": "web_search",
                        "user_location": {"type": "approximate"},
                        "search_context_size": search_context
                    }],
                    "store": True,
                    "include": ["web_search_call.action.sources"]
                }
                api_params.update(model_params)  # Add temperature, max_output_tokens, top_p
            
            response = client.responses.create(**api_params)
        else:
            # Use standard chat completions API for GPT-4 and other models
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
                **model_params
            )

        # Parse response based on API type
        if is_gpt5 or is_gpt52 or is_gpt41:
            # Extract content from GPT-5/GPT-5.2 responses.create() structure
            content = None
            if hasattr(response, 'output_text') and response.output_text:
                content = response.output_text
            elif hasattr(response, 'output') and response.output:
                # Navigate through output array to find text content
                for item in response.output:
                    if hasattr(item, 'content') and item.content:
                        for content_item in item.content:
                            if hasattr(content_item, 'text'):
                                content = content_item.text
                                break
                        if content:
                            break
                if not content:
                    content = str(response.output)
            else:
                content = str(response)
        else:
            content = response.choices[0].message.content  # Standard chat.completions.create() structure
        
        return content
    except Exception as e:
        return f"Error fetching fundamental analysis for {ticker}: {str(e)}"


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