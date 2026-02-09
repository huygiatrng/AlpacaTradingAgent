import requests
import time
import json
from datetime import datetime, timedelta
from contextlib import contextmanager
from typing import Annotated, List
import os
import re
from .alpaca_utils import AlpacaUtils

REDDIT_USER_AGENT = "TradingAgents/1.0"
REDDIT_CATEGORY_SUBREDDITS = {
    "global_news": [
        "news",
        "worldnews",
        "economics",
        "stocks",
        "investing",
    ],
    "company_news": [
        "stocks",
        "investing",
        "wallstreetbets",
        "SecurityAnalysis",
        "stockmarket",
        "dividends",
        "ValueInvesting",
    ],
}

_SEARCH_TERMS_CACHE = {}


def get_company_name(ticker: str) -> str:
    """
    Get company name from ticker symbol using Alpaca API.
    The fallback logic is handled in AlpacaUtils.
    
    Args:
        ticker: Ticker symbol
    
    Returns:
        Company name or the original ticker if not found
    """

    normalized = (ticker or "").strip().upper()
    if not normalized:
        return ticker

    company_name = AlpacaUtils.get_company_name(normalized)
    if company_name and company_name.strip().upper() != normalized:
        return company_name

    # Secondary fallback: yfinance name lookup for better social search recall.
    try:
        import yfinance as yf

        info = yf.Ticker(normalized).info or {}
        candidate = (
            info.get("shortName")
            or info.get("longName")
            or info.get("displayName")
            or ""
        )
        candidate = str(candidate).strip()
        if candidate and candidate.upper() != normalized:
            return candidate
    except Exception:
        pass

    return company_name or ticker


def get_search_terms(ticker: str) -> List[str]:
    """
    Generate a list of search terms for a company based on ticker symbol
    
    Args:
        ticker: Ticker symbol
    
    Returns:
        List of search terms including company name, ticker, and common variations
    """
    normalized = (ticker or "").strip().upper()
    if not normalized:
        return []

    cached = _SEARCH_TERMS_CACHE.get(normalized)
    if cached:
        return list(cached)

    search_terms = [normalized, f"${normalized}"]  # Include ticker + common cashtag format
    
    # Get company name from Alpaca
    company_name = get_company_name(normalized)
    
    if company_name == normalized:
        # If we couldn't get a company name, just return ticker/cashtag
        _SEARCH_TERMS_CACHE[normalized] = tuple(dict.fromkeys([term for term in search_terms if term]))
        return list(_SEARCH_TERMS_CACHE[normalized])
    
    # Handle company names with "Common Stock", "Class A", etc.
    if isinstance(company_name, str):
        # Add the full company name
        search_terms.append(company_name)
        
        # Split by "Common Stock", "Class A", etc.
        name_parts = re.split(r'\s+(?:Common Stock|Class [A-Z]|Inc\.?|Corp\.?|Corporation|Ltd\.?|Limited|LLC)', company_name)
        if name_parts and name_parts[0].strip():
            search_terms.append(name_parts[0].strip())
        
        # If company name has OR, split into separate terms
        if " OR " in company_name:
            or_terms = company_name.split(" OR ")
            search_terms.extend([term.strip() for term in or_terms])
    
    deduped_terms = [term for term in dict.fromkeys([t.strip() for t in search_terms if isinstance(t, str) and t.strip()])]
    _SEARCH_TERMS_CACHE[normalized] = tuple(deduped_terms)
    return list(_SEARCH_TERMS_CACHE[normalized])


def fetch_top_from_category(
    category: Annotated[
        str, "Category to fetch top post from. Collection of subreddits."
    ],
    date: Annotated[str, "Date to fetch top posts from."],
    max_limit: Annotated[int, "Maximum number of posts to fetch."],
    query: Annotated[str, "Optional query to search for in the subreddit."] = None,
    data_path: Annotated[
        str,
        "Path to the data folder. Default is 'reddit_data'.",
    ] = "reddit_data",
):
    base_path = data_path

    all_content = []

    if max_limit < len(os.listdir(os.path.join(base_path, category))):
        raise ValueError(
            "REDDIT FETCHING ERROR: max limit is less than the number of files in the category. Will not be able to fetch any posts"
        )

    limit_per_subreddit = max_limit // len(
        os.listdir(os.path.join(base_path, category))
    )

    search_terms = None
    if "company" in category and query:
        search_terms = get_search_terms(query)

    for data_file in os.listdir(os.path.join(base_path, category)):
        # check if data_file is a .jsonl file
        if not data_file.endswith(".jsonl"):
            continue

        all_content_curr_subreddit = []

        with open(os.path.join(base_path, category, data_file), "rb") as f:
            for i, line in enumerate(f):
                # skip empty lines
                if not line.strip():
                    continue

                parsed_line = json.loads(line)

                # select only lines that are from the date
                post_date = datetime.utcfromtimestamp(
                    parsed_line["created_utc"]
                ).strftime("%Y-%m-%d")
                if post_date != date:
                    continue

                # if is company_news, check that the title or the content has the company's name (query) mentioned
                if "company" in category and query:
                    found = False
                    for term in (search_terms or []):
                        # Only search if we have a valid term
                        if term and isinstance(term, str):
                            if re.search(
                                re.escape(term), parsed_line["title"], re.IGNORECASE
                            ) or re.search(
                                re.escape(term), parsed_line["selftext"], re.IGNORECASE
                            ):
                                found = True
                                break

                    if not found:
                        continue

                post = {
                    "title": parsed_line["title"],
                    "content": parsed_line["selftext"],
                    "url": parsed_line["url"],
                    "upvotes": parsed_line["ups"],
                    "posted_date": post_date,
                }

                all_content_curr_subreddit.append(post)

        # sort all_content_curr_subreddit by upvote_ratio in descending order
        all_content_curr_subreddit.sort(key=lambda x: x["upvotes"], reverse=True)

        all_content.extend(all_content_curr_subreddit[:limit_per_subreddit])

    return all_content


def fetch_top_from_category_online(
    category: str,
    start_date: str,
    end_date: str,
    max_limit: int,
    query: str = None,
) -> List[dict]:
    """
    Fetch Reddit posts directly from Reddit JSON endpoints as a realtime fallback
    when local reddit_data files are missing.
    """
    if max_limit <= 0:
        return []

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    if end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt

    subreddits = REDDIT_CATEGORY_SUBREDDITS.get(
        category,
        REDDIT_CATEGORY_SUBREDDITS["company_news" if "company" in category else "global_news"],
    )
    if not subreddits:
        return []

    search_terms = None
    if "company" in category and query:
        search_terms = [term for term in get_search_terms(query) if term]
        search_query = " OR ".join(search_terms[:12]) if search_terms else query
    else:
        search_query = "market OR economy OR inflation OR central bank"

    headers = {"User-Agent": REDDIT_USER_AGENT}
    per_subreddit_limit = max(20, min(100, max_limit * 2))

    posts = []
    seen = set()

    for subreddit in subreddits:
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {
            "q": search_query,
            "restrict_sr": "1",
            "sort": "new",
            "t": "year",
            "limit": per_subreddit_limit,
        }
        try:
            response = requests.get(url, headers=headers, params=params, timeout=12)
            if response.status_code != 200:
                continue
            payload = response.json() or {}
            children = ((payload.get("data") or {}).get("children") or [])
        except Exception:
            continue

        for child in children:
            data = child.get("data") if isinstance(child, dict) else None
            if not data:
                continue

            created_utc = data.get("created_utc")
            if not created_utc:
                continue
            post_date = datetime.utcfromtimestamp(created_utc)
            if post_date < start_dt or post_date > (end_dt + timedelta(days=1)):
                continue

            title = data.get("title", "")
            content = data.get("selftext", "")
            # For online path, rely on Reddit search API filtering; avoid over-filtering
            # by local regex that can drop valid posts with sparse selftext.

            url_value = data.get("url") or ""
            if not url_value and data.get("permalink"):
                url_value = f"https://www.reddit.com{data.get('permalink')}"

            dedupe_key = (title.strip().lower(), url_value.strip().lower())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            posts.append(
                {
                    "title": title,
                    "content": content,
                    "url": url_value,
                    "upvotes": int(data.get("ups", 0) or 0),
                    "posted_date": post_date.strftime("%Y-%m-%d"),
                    "subreddit": subreddit,
                }
            )

        time.sleep(0.2)

    posts.sort(key=lambda item: (item.get("upvotes", 0), item.get("posted_date", "")), reverse=True)
    return posts[:max_limit]
