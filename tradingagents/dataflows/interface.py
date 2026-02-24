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
from .config import get_config, set_config, DATA_DIR, get_api_key
from .cache_utils import with_cache, load_from_cache, save_to_cache, generate_cache_key
from .openai_client import get_openai_client


def get_model_params(model_name, max_tokens_value=3000):
    """Get appropriate parameters for different model types."""
    params = {}
    
    # GPT-5 and GPT-4.1 models use the responses.create() API 
    # Older models use the standard chat.completions.create() API
    gpt5_models = ["gpt-5", "gpt-5.2", "gpt-5-mini", "gpt-5-nano"]
    gpt41_models = ["gpt-4.1"]
    
    if any(model_prefix in model_name for model_prefix in gpt5_models):
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

    try:
        # Try to get data from cache
        result = get_data_in_range(ticker, before, curr_date, "news_data", DATA_DIR)
    except FileNotFoundError:
        # Fallback to Finnhub API if cache unavailable
        print(f"[FINNHUB] Cache unavailable for {ticker}, using real-time API...")
        try:
            from .finnhub_utils import get_finnhub_client
            client = get_finnhub_client()
            # Finnhub API uses from/to parameters
            news_data = client.company_news(ticker, _from=before, to=curr_date)

            if not news_data or len(news_data) == 0:
                return f"## {ticker} News, from {before} to {curr_date}:\nNo news available."

            combined_result = ""
            for entry in news_data[:20]:  # Limit to 20 articles
                headline = entry.get('headline', 'No headline')
                summary = entry.get('summary', 'No summary')
                date_ts = entry.get('datetime', 0)
                date_str = datetime.fromtimestamp(date_ts).strftime("%Y-%m-%d") if date_ts else "Unknown date"
                current_news = f"### {headline} ({date_str})\n{summary}"
                combined_result += current_news + "\n\n"

            return f"## {ticker} News, from {before} to {curr_date}:\n" + str(combined_result)
        except Exception as api_err:
            print(f"[FINNHUB] API error: {api_err}")
            return f"## {ticker} News, from {before} to {curr_date}:\nError fetching news: {str(api_err)}"

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

    try:
        data = get_data_in_range(ticker, before, curr_date, "insider_senti", DATA_DIR)
    except FileNotFoundError:
        # Fallback to Finnhub API if cache unavailable
        print(f"[FINNHUB] Cache unavailable for {ticker} insider sentiment, using real-time API...")
        try:
            from .finnhub_utils import get_finnhub_client
            client = get_finnhub_client()

            # Call Finnhub API
            api_data = client.stock_insider_sentiment(ticker, _from=before, to=curr_date)

            if not api_data or 'data' not in api_data or len(api_data['data']) == 0:
                return f"## {ticker} Insider Sentiment Data for {before} to {curr_date}:\nNo data available from Finnhub API."

            # Format the API response
            result_str = ""
            seen_dicts = []
            for entry in api_data['data']:
                if entry not in seen_dicts:
                    result_str += f"### {entry['year']}-{entry['month']}:\nChange: {entry['change']}\nMonthly Share Purchase Ratio: {entry['mspr']}\n\n"
                    seen_dicts.append(entry)

            return (
                f"## {ticker} Insider Sentiment Data for {before} to {curr_date}:\n"
                + result_str
                + "The change field refers to the net buying/selling from all insiders' transactions. The mspr field refers to monthly share purchase ratio."
            )
        except Exception as api_err:
            print(f"[FINNHUB] API error for insider sentiment: {api_err}")
            return f"## {ticker} Insider Sentiment Data:\nError fetching from Finnhub API: {str(api_err)}"

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

    try:
        data = get_data_in_range(ticker, before, curr_date, "insider_trans", DATA_DIR)
    except FileNotFoundError:
        # Fallback to Finnhub API if cache unavailable
        print(f"[FINNHUB] Cache unavailable for {ticker} insider transactions, using real-time API...")
        try:
            from .finnhub_utils import get_finnhub_client
            client = get_finnhub_client()

            # Call Finnhub API
            api_data = client.stock_insider_transactions(ticker, _from=before, to=curr_date)

            if not api_data or 'data' not in api_data or len(api_data['data']) == 0:
                return f"## {ticker} Insider Transactions from {before} to {curr_date}:\nNo data available from Finnhub API."

            # Format the API response
            result_str = ""
            seen_dicts = []
            for entry in api_data['data']:
                if entry not in seen_dicts:
                    filing_date = entry.get('filingDate', 'Unknown')
                    name = entry.get('name', 'Unknown')
                    change = entry.get('change', 0)
                    shares = entry.get('share', 0)
                    price = entry.get('transactionPrice', 0)
                    code = entry.get('transactionCode', 'N/A')

                    result_str += f"### Filing Date: {filing_date}, {name}:\nChange: {change}\nShares: {shares}\nTransaction Price: {price}\nTransaction Code: {code}\n\n"
                    seen_dicts.append(entry)

            return (
                f"## {ticker} insider transactions from {before} to {curr_date}:\n"
                + result_str
                + "The change field reflects the variation in share count—here a negative number indicates a reduction in holdings—while share specifies the total number of shares involved. The transactionPrice denotes the per-share price at which the trade was executed, and transactionDate marks when the transaction occurred. The name field identifies the insider making the trade, and transactionCode (e.g., S for sale) clarifies the nature of the transaction. FilingDate records when the transaction was officially reported, and the unique id links to the specific SEC filing, as indicated by the source. Additionally, the symbol ties the transaction to a particular company, isDerivative flags whether the trade involves derivative securities, and currency notes the currency context of the transaction."
            )
        except Exception as api_err:
            print(f"[FINNHUB] API error for insider transactions: {api_err}")
            return f"## {ticker} Insider Transactions:\nError fetching from Finnhub API: {str(api_err)}"

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


@with_cache(cache_category="crypto_news", max_age_hours=12, extension="json")
def get_coindesk_news(
    ticker: Annotated[str, "Ticker symbol, e.g. 'BTC/USD', 'ETH/USD', 'ETH', etc."],
    num_sentences: Annotated[int, "Number of sentences to include from news body."] = 5,
) -> str:
    """
    Retrieve news for a cryptocurrency (with automatic caching - 12 hour cache lifetime)
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


@with_cache(cache_category="google_news", max_age_hours=12, extension="json")
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

    try:
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
    except FileNotFoundError as e:
        print(f"[REDDIT] Cache unavailable: {e}")
        return f"## Reddit News: Cache not available. Reddit news requires pre-cached data. Continuing analysis with other sources..."
    except Exception as e:
        print(f"[REDDIT] Error fetching Reddit news: {e}")
        return f"## Reddit News: Error - {str(e)}"

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

    try:
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
    except FileNotFoundError as e:
        print(f"[REDDIT] Cache unavailable for {ticker}: {e}")
        return f"## {ticker} Reddit News: Cache not available. Reddit news requires pre-cached data. Continuing analysis with other sources..."
    except Exception as e:
        print(f"[REDDIT] Error fetching Reddit news for {ticker}: {e}")
        return f"## {ticker} Reddit News: Error - {str(e)}"

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
            print(f"[STOCKSTATS] Error calculating {indicator} for {symbol} on {date}: {e}")
            import traceback
            traceback.print_exc()
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

        client = get_openai_client()
        
        # Get the selected quick model from config
        config = get_config()
        model = config.get("quick_think_llm", "gpt-4o-mini")  # fallback to default
        
        from datetime import datetime, timedelta
        start_date = (datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")

        # Get model-specific parameters
        model_params = get_model_params(model)
        
        # Check if this is a GPT-5 or GPT-4.1 model (both use responses.create())
        gpt5_models = ["gpt-5", "gpt-5.2", "gpt-5-mini", "gpt-5-nano"]
        gpt41_models = ["gpt-4.1"]
        is_gpt5 = any(model_prefix in model for model_prefix in gpt5_models)
        is_gpt41 = any(model_prefix in model for model_prefix in gpt41_models)
        
        if is_gpt5 or is_gpt41:
            # Use responses.create() API with web search capabilities - use standardized ticker
            user_message = f"Search the web and analyze current social media sentiment and recent news for {ticker_info['display_format']} ({openai_ticker}) from {start_date} to {curr_date}. Include:\n" + \
                          f"1. Overall sentiment analysis from recent social media posts\n" + \
                          f"2. Key themes and discussions happening now\n" + \
                          f"3. Notable price-moving news or events from the past week\n" + \
                          f"4. Trading implications based on current sentiment\n" + \
                          f"5. Summary table with key metrics"
            
            # Base parameters for responses.create()
            if is_gpt5:
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
                        "search_context_size": "medium"
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
        if is_gpt5 or is_gpt41:
            # Extract content from GPT-5 responses.create() structure
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


async def get_global_news_openai_parallel(curr_date, ticker_context=None):
    """
    Parallelized version of get_global_news_openai - 83% faster via concurrent search areas.
    Splits 6-8 search topics into concurrent API calls instead of sequential searches.
    """
    from openai import AsyncOpenAI
    import asyncio

    # Get API key from environment variables or config
    api_key = get_api_key("openai_api_key", "OPENAI_API_KEY")
    if not api_key:
        return f"Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable."

    try:
        # Use AsyncOpenAI with increased connection limits for true parallelism
        import httpx

        # Create custom httpx client with higher limits for parallel searches
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=20,      # Allow up to 20 concurrent connections
                max_keepalive_connections=10
            ),
            timeout=180.0  # 3 min timeout per search (increased from 120s)
        )

        client = AsyncOpenAI(api_key=api_key, http_client=http_client)
        config = get_config()

        # Use configured model for parallel searches
        model = config.get("quick_think_llm", "gpt-4o-mini")

        # Check if this model supports web search
        gpt5_models = ["gpt-5", "gpt-5.2", "gpt-5-mini", "gpt-5-nano"]
        gpt41_models = ["gpt-4.1"]
        is_gpt5 = any(m in model for m in gpt5_models)
        is_gpt41 = any(m in model for m in gpt41_models)

        if not (is_gpt5 or is_gpt41):
            # Model doesn't support web search, fall back to sequential
            print(f"[NEWS] ⚠️ Model {model} doesn't support web search in parallel mode")
            print(f"[NEWS] 🔄 Falling back to sequential implementation")
            raise ValueError(f"Model {model} not supported for parallel web search")

        print(f"[NEWS] Using model: {model} (web_search: True)")

        from datetime import datetime, timedelta
        start_date = (datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")

        # Determine if this is crypto-related analysis
        is_crypto = ticker_context and ("/" in ticker_context or "USD" in ticker_context.upper() or "BTC" in ticker_context.upper() or "ETH" in ticker_context.upper())

        # Define focused search areas based on asset type
        if is_crypto:
            search_areas = [
                ("Regulatory & Policy", f"Search for cryptocurrency regulatory developments, CBDC announcements, and crypto policy updates from {start_date} to {curr_date} that could impact {ticker_context if ticker_context else 'crypto markets'}. Include impact assessment on trading and adoption."),
                ("Institutional Adoption", f"Search for institutional crypto adoption news, ETF developments, and major investment flows related to {ticker_context if ticker_context else 'cryptocurrencies'} from {start_date} to {curr_date}. Include major announcements and fund flows."),
                ("DeFi & Protocol", f"Search for DeFi developments, smart contract updates, and blockchain protocol news affecting {ticker_context if ticker_context else 'crypto ecosystem'} from {start_date} to {curr_date}. Include major upgrades and vulnerabilities."),
                ("Exchange & Infrastructure", f"Search for crypto exchange developments, security issues, market infrastructure changes, and custody solutions affecting {ticker_context if ticker_context else 'crypto markets'} from {start_date} to {curr_date}."),
                ("Macro Impact", f"Search for Federal Reserve policy, inflation data, interest rate decisions, and geopolitical events affecting cryptocurrency markets and {ticker_context if ticker_context else 'digital assets'} from {start_date} to {curr_date}."),
                ("Market Sentiment", f"Search for crypto market sentiment, social media trends, whale movements, and community sentiment for {ticker_context if ticker_context else 'cryptocurrencies'} from {start_date} to {curr_date}. Assess bullish vs bearish sentiment."),
                ("Trading Catalysts", f"Search for upcoming crypto catalysts, protocol upgrades, token unlocks, major announcements, and price-moving events for {ticker_context if ticker_context else 'crypto markets'} from {start_date} to {curr_date}."),
            ]
        else:
            search_areas = [
                ("Economic Events", f"Search for major economic events and announcements affecting {ticker_context if ticker_context else 'stock markets'} from {start_date} to {curr_date}. Include major policy announcements and economic surprises."),
                ("Central Bank Policy", f"Search for Federal Reserve decisions, central bank policy updates, interest rate changes, and monetary policy affecting {ticker_context if ticker_context else 'equity markets'} from {start_date} to {curr_date}."),
                ("Geopolitical Developments", f"Search for geopolitical events, trade policy changes, international relations, and regulatory developments affecting {ticker_context if ticker_context else 'markets'} and its sector from {start_date} to {curr_date}."),
                ("Economic Data Releases", f"Search for economic data releases (GDP, inflation, employment, consumer spending) and their implications for {ticker_context if ticker_context else 'stock markets'} from {start_date} to {curr_date}."),
                ("Sector & Industry News", f"Search for sector-specific developments, competitive dynamics, technological breakthroughs, and industry trends relevant to {ticker_context if ticker_context else 'the market'} from {start_date} to {curr_date}."),
                ("Market Sentiment", f"Search for market sentiment, analyst ratings, institutional positioning, and trading implications for {ticker_context if ticker_context else 'stocks'} from {start_date} to {curr_date}."),
                ("Key Catalysts", f"Search for upcoming catalysts, earnings events, product launches, and major announcements that could impact {ticker_context if ticker_context else 'stock prices'} from {start_date} to {curr_date}."),
            ]

        async def search_single_area(area_name, area_prompt):
            """Execute a single focused search with web_search tool"""
            try:
                if is_gpt5:
                    # GPT-5 API format with web search
                    response = await client.responses.create(
                        model=model,
                        input=[
                            {
                                "role": "developer",
                                "content": [{
                                    "type": "input_text",
                                    "text": f"You are a financial news analyst focusing on {area_name} analysis. Use web search to find current, real-time information."
                                }]
                            },
                            {
                                "role": "user",
                                "content": [{
                                    "type": "input_text",
                                    "text": area_prompt
                                }]
                            }
                        ],
                        text={"format": {"type": "text"}, "verbosity": "medium"},
                        reasoning={"effort": "medium", "summary": "auto"},
                        tools=[{
                            "type": "web_search",  # ← ENABLES WEB SEARCH
                            "user_location": {"type": "approximate"},
                            "search_context_size": "medium"
                        }],
                        store=True,
                        include=["reasoning.encrypted_content", "web_search_call.action.sources"]
                    )

                    # Extract content from GPT-5 response
                    content = None
                    if hasattr(response, 'output_text') and response.output_text:
                        content = response.output_text
                    elif hasattr(response, 'output') and response.output:
                        for item in response.output:
                            if hasattr(item, 'content') and item.content:
                                for content_item in item.content:
                                    if hasattr(content_item, 'text'):
                                        content = content_item.text
                                        break
                                if content:
                                    break
                    if not content:
                        content = str(response.output) if hasattr(response, 'output') else str(response)

                elif is_gpt41:
                    # GPT-4.1 API format with web search
                    response = await client.responses.create(
                        model=model,
                        input=[
                            {
                                "role": "system",
                                "content": [{
                                    "type": "input_text",
                                    "text": f"You are a financial news analyst focusing on {area_name} analysis. Use web search to find current, real-time information."
                                }]
                            },
                            {
                                "role": "user",
                                "content": [{
                                    "type": "input_text",
                                    "text": area_prompt
                                }]
                            }
                        ],
                        text={"format": {"type": "text"}},
                        reasoning={},
                        tools=[{
                            "type": "web_search",  # ← ENABLES WEB SEARCH
                            "user_location": {"type": "approximate"},
                            "search_context_size": "medium"
                        }],
                        store=True,
                        include=["web_search_call.action.sources"]
                    )

                    # Extract content from GPT-4.1 response
                    content = None
                    if hasattr(response, 'output_text') and response.output_text:
                        content = response.output_text
                    elif hasattr(response, 'output') and response.output:
                        for item in response.output:
                            if hasattr(item, 'content') and item.content:
                                for content_item in item.content:
                                    if hasattr(content_item, 'text'):
                                        content = content_item.text
                                        break
                                if content:
                                    break
                    if not content:
                        content = str(response.output) if hasattr(response, 'output') else str(response)

                if not content or content.strip() == "":
                    content = f"Error: Empty response from {area_name} search"

                return {
                    "area": area_name,
                    "content": content,
                    "success": True
                }

            except Exception as e:
                print(f"[NEWS] ⚠️ Search area '{area_name}' failed: {str(e)[:100]}")
                return {
                    "area": area_name,
                    "content": f"Search failed: {str(e)[:200]}",
                    "success": False
                }

        # Execute all searches in parallel
        print(f"[NEWS] 🚀 Starting {len(search_areas)} parallel searches for {ticker_context or 'markets'}...")
        start_time = asyncio.get_event_loop().time()

        tasks = [search_single_area(name, prompt) for name, prompt in search_areas]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = asyncio.get_event_loop().time() - start_time

        # Aggregate results
        final_report = f"# Global News Analysis for {ticker_context or 'Markets'}\n"
        final_report += f"**Analysis Period:** {start_date} to {curr_date}\n"
        final_report += f"**Asset Type:** {'Cryptocurrency' if is_crypto else 'Stock/Equity'}\n\n"

        successful_searches = 0
        failed_areas = []

        for result in results:
            if isinstance(result, Exception):
                print(f"[NEWS] ⚠️ Search task raised exception: {str(result)[:100]}")
                continue

            if result.get("success"):
                final_report += f"## {result['area']}\n{result['content']}\n\n"
                successful_searches += 1
            else:
                failed_areas.append(result['area'])

        # Add summary footer
        final_report += f"\n---\n"
        final_report += f"**Search Summary:** {successful_searches}/{len(search_areas)} areas completed successfully in {elapsed:.1f}s\n"
        if failed_areas:
            final_report += f"**Failed Areas:** {', '.join(failed_areas)}\n"

        print(f"[NEWS] ✅ Parallel search completed: {successful_searches}/{len(search_areas)} successful in {elapsed:.1f}s")

        if successful_searches == 0:
            return f"Error: All parallel searches failed. This may indicate API issues or network problems."

        return final_report

    except Exception as e:
        return f"Error in parallel global news search: {str(e)}"


def get_global_news_openai_sync_wrapper(curr_date, ticker_context=None):
    """Synchronous wrapper for async parallel function with fallback"""
    import asyncio

    # Get or create event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Try parallel execution first
    try:
        return loop.run_until_complete(get_global_news_openai_parallel(curr_date, ticker_context))
    except ValueError as e:
        # Model doesn't support web search, fall back to sequential
        print(f"[NEWS] 📊 Falling back to sequential implementation...")
        return get_global_news_openai_sequential(curr_date, ticker_context)
    except Exception as e:
        print(f"[NEWS] ⚠️ Parallel execution failed: {str(e)[:100]}")
        raise


def get_global_news_openai_sequential(curr_date, ticker_context=None):
    """
    Original sequential implementation - kept as fallback.
    This is the original function that makes 1 API call with multiple search prompts.
    """
    # Get API key from environment variables or config
    api_key = get_api_key("openai_api_key", "OPENAI_API_KEY")
    if not api_key:
        return f"Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable."

    try:
        # Initialize OpenAI client with 300s timeout for SDK-level enforcement
        client = get_openai_client(timeout=300.0)

        # Get the selected quick model from config
        config = get_config()
        model = config.get("quick_think_llm", "gpt-4o-mini")  # fallback to default

        from datetime import datetime, timedelta
        start_date = (datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")

        # Get model-specific parameters
        model_params = get_model_params(model)
        
        # Check if this is a GPT-5 or GPT-4.1 model (both use responses.create())
        gpt5_models = ["gpt-5", "gpt-5.2", "gpt-5-mini", "gpt-5-nano"]
        gpt41_models = ["gpt-4.1"]
        is_gpt5 = any(model_prefix in model for model_prefix in gpt5_models)
        is_gpt41 = any(model_prefix in model for model_prefix in gpt41_models)
        
        # Determine if this is crypto-related analysis
        is_crypto = ticker_context and ("/" in ticker_context or "USD" in ticker_context.upper() or "BTC" in ticker_context.upper() or "ETH" in ticker_context.upper())
        
        if is_gpt5 or is_gpt41:
            # Use responses.create() API with web search capabilities
            if is_crypto:
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
                user_message = f"Search the web for current global and macroeconomic news from {start_date} to {curr_date} that would be informative for trading {ticker_context if ticker_context else 'financial markets'}. Include:\n" + \
                              f"1. Major economic events and announcements\n" + \
                              f"2. Central bank policy updates\n" + \
                              f"3. Geopolitical developments affecting markets\n" + \
                              f"4. Economic data releases and their implications\n" + \
                              f"5. Sector-specific developments relevant to {ticker_context if ticker_context else 'the market'}\n" + \
                              f"6. Trading implications and market sentiment\n" + \
                              f"7. Summary table with key events and impact levels"
            
            # Base parameters for responses.create()
            if is_gpt5:
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
                        "search_context_size": "medium"
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
                        "search_context_size": "medium"
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
        if is_gpt5 or is_gpt41:
            # Extract content from GPT-5 responses.create() structure
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


def get_global_news_openai(curr_date, ticker_context=None):
    """
    Main entry point for global news analysis.
    Uses parallelized async version for 83% speed improvement (30-40s vs 180-240s).
    Falls back to sequential version if parallel fails.
    """
    try:
        # Attempt parallel version (6-8 concurrent API calls, much faster)
        return get_global_news_openai_sync_wrapper(curr_date, ticker_context)
    except Exception as e:
        print(f"[NEWS] ⚠️ Parallel search failed: {str(e)[:100]}")
        print(f"[NEWS] 🔄 Falling back to sequential search...")
        # Fall back to original sequential implementation
        return get_global_news_openai_sequential(curr_date, ticker_context)


def get_fundamentals_openai(ticker, curr_date):
    # Get API key from environment variables or config
    api_key = get_api_key("openai_api_key", "OPENAI_API_KEY")
    if not api_key:
        return f"Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
    
    try:
        # Initialize OpenAI client with 300s timeout for SDK-level enforcement
        client = get_openai_client(timeout=300.0)

        # Get the selected quick model from config
        config = get_config()
        model = config.get("quick_think_llm", "gpt-4o-mini")  # fallback to default

        from datetime import datetime, timedelta
        start_date = (datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")

        # Get model-specific parameters
        model_params = get_model_params(model)

        # Check if this is a GPT-5 or GPT-4.1 model (both use responses.create())
        gpt5_models = ["gpt-5", "gpt-5.2", "gpt-5-mini", "gpt-5-nano"]
        gpt41_models = ["gpt-4.1"]
        is_gpt5 = any(model_prefix in model for model_prefix in gpt5_models)
        is_gpt41 = any(model_prefix in model for model_prefix in gpt41_models)

        if is_gpt5 or is_gpt41:
            # Use responses.create() API with web search capabilities
            user_message = f"Search the web and provide a current fundamental analysis for {ticker} covering the period from {start_date} to {curr_date}. Include:\n" + \
                          f"1. Key financial metrics (P/E, P/S, P/B, EV/EBITDA, etc.)\n" + \
                          f"2. Revenue and earnings trends\n" + \
                          f"3. Cash flow analysis\n" + \
                          f"4. Balance sheet strength\n" + \
                          f"5. Competitive positioning\n" + \
                          f"6. Recent business developments\n" + \
                          f"7. Valuation assessment\n" + \
                          f"8. Summary table with key fundamental metrics and ratios\n\n" + \
                          f"Format the analysis professionally with clear sections and include a summary table at the end."
            
            # Base parameters for responses.create()
            if is_gpt5:
                # GPT-5 uses "developer" role
                api_params = {
                    "model": model,
                    "input": [
                        {
                            "role": "developer",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "You are a fundamental analyst with web search access specializing in financial analysis and valuation. Use real-time web search to provide comprehensive fundamental analysis based on available financial metrics and recent company developments."
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
                        "search_context_size": "medium"
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
                                    "text": "You are a fundamental analyst with web search access specializing in financial analysis and valuation. Use real-time web search to provide comprehensive fundamental analysis based on available financial metrics and recent company developments."
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
                        "search_context_size": "medium"
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
        if is_gpt5 or is_gpt41:
            # Extract content from GPT-5 responses.create() structure
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


@with_cache(cache_category="crypto_fundamentals", max_age_hours=24, extension="json")
def get_defillama_fundamentals(
    ticker: Annotated[str, "Crypto ticker symbol (without USD/USDT suffix)"],
    lookback_days: Annotated[int, "Number of days to look back for data"] = 30,
) -> str:
    """
    Get fundamental data for a cryptocurrency from DeFi Llama (with automatic caching - 24 hour cache lifetime)

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

@with_cache(cache_category="alpaca_data", max_age_hours=24, extension="json")
def get_alpaca_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"] = None,
    timeframe: Annotated[str, "Timeframe for data: 1Min, 5Min, 15Min, 1Hour, 1Day"] = "1Day",
) -> str:
    """
    Get stock data from Alpaca (with automatic caching - 24 hour cache lifetime)
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
            result += f"\n\n## Key EOD Trading Metrics:\n"
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


@with_cache(cache_category="earnings", max_age_hours=24, extension="json")
def get_earnings_calendar(
    ticker: Annotated[str, "Stock or crypto ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve earnings calendar data for stocks or major events for crypto (with automatic caching - 24 hour cache lifetime)
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


@with_cache(cache_category="earnings_analysis", max_age_hours=24, extension="json")
def get_earnings_surprise_analysis(
    ticker: Annotated[str, "Stock ticker symbol"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    lookback_quarters: Annotated[int, "Number of quarters to analyze"] = 8,
) -> str:
    """
    Analyze historical earnings surprises to identify patterns and trading implications (with automatic caching - 24 hour cache lifetime)
    Shows consistency of beats/misses, magnitude of surprises, and seasonal patterns.

    Args:
        ticker (str): Stock ticker symbol, e.g. AAPL, TSLA
        curr_date (str): Current date in yyyy-mm-dd format
        lookback_quarters (int): Number of quarters to analyze (default 8 = ~2 years)

    Returns:
        str: Analysis of earnings surprise patterns with trading implications
    """

    return get_earnings_surprises_analysis(ticker, curr_date, lookback_quarters)


@with_cache(cache_category="macro_analysis", max_age_hours=24, extension="json")
def get_macro_analysis(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    lookback_days: Annotated[int, "Number of days to look back for data"] = 90,
) -> str:
    """
    Retrieve comprehensive macro economic analysis including Fed funds, CPI, PPI, NFP, GDP, PMI, Treasury curve, VIX (with automatic caching - 24 hour cache lifetime)
    Provides economic indicators, yield curve analysis, Fed policy updates, and trading implications.

    Args:
        curr_date (str): Current date in yyyy-mm-dd format
        lookback_days (int): Number of days to look back for data (default 90)

    Returns:
        str: Comprehensive macro economic analysis with trading implications
    """

    return get_macro_economic_summary(curr_date)


@with_cache(cache_category="economic_indicators", max_age_hours=24, extension="json")
def get_economic_indicators(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    lookback_days: Annotated[int, "Number of days to look back for data"] = 90,
) -> str:
    """
    Retrieve key economic indicators report including Fed funds, CPI, PPI, unemployment, NFP, GDP, PMI, VIX (with automatic caching - 24 hour cache lifetime)

    Args:
        curr_date (str): Current date in yyyy-mm-dd format
        lookback_days (int): Number of days to look back for data (default 90)

    Returns:
        str: Economic indicators report with analysis and interpretations
    """

    return get_economic_indicators_report(curr_date, lookback_days)


@with_cache(cache_category="yield_curve", max_age_hours=24, extension="json")
def get_yield_curve_analysis(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve Treasury yield curve analysis including inversion signals and recession indicators (with automatic caching - 24 hour cache lifetime)

    Args:
        curr_date (str): Current date in yyyy-mm-dd format

    Returns:
        str: Treasury yield curve data with inversion analysis
    """

    return get_treasury_yield_curve(curr_date)