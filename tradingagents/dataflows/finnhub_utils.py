import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests

from .config import get_finnhub_api_key

_FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
_FINNHUB_TIMEOUT_SECONDS = 20.0


def get_finnhub_client():
    """
    Get a finnhub client using the API key from environment variables or config.
    """
    api_key = get_finnhub_api_key()
    if not api_key:
        raise ValueError("Finnhub API key not found. Please set FINNHUB_API_KEY environment variable or in .env file.")
    try:
        import finnhub  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "finnhub package is not installed. Install finnhub-python or use HTTP helpers."
        ) from exc
    return finnhub.Client(api_key=api_key)


def _request_finnhub_json(path: str, params: Dict[str, Any]) -> Any:
    api_key = get_finnhub_api_key()
    if not api_key:
        raise ValueError(
            "Finnhub API key not found. Please set FINNHUB_API_KEY environment variable or in .env file."
        )

    request_params = dict(params or {})
    request_params["token"] = api_key
    url = f"{_FINNHUB_BASE_URL}{path}"
    response = requests.get(url, params=request_params, timeout=_FINNHUB_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"Finnhub API error: {data.get('error')}")
    return data


def _timestamp_to_date_str(value: Any) -> str:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%d")
    return ""


def fetch_company_news_live(ticker: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    data = _request_finnhub_json(
        "/company-news",
        {"symbol": ticker.upper(), "from": start_date, "to": end_date},
    )
    if not isinstance(data, list):
        return []
    return sorted(
        [entry for entry in data if isinstance(entry, dict)],
        key=lambda entry: entry.get("datetime", 0) or 0,
        reverse=True,
    )


def fetch_insider_sentiment_live(ticker: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    data = _request_finnhub_json(
        "/stock/insider-sentiment",
        {"symbol": ticker.upper(), "from": start_date, "to": end_date},
    )
    if isinstance(data, dict):
        payload = data.get("data", [])
        if isinstance(payload, list):
            return [entry for entry in payload if isinstance(entry, dict)]
    return []


def fetch_insider_transactions_live(ticker: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    data = _request_finnhub_json(
        "/stock/insider-transactions",
        {"symbol": ticker.upper(), "from": start_date, "to": end_date},
    )
    if isinstance(data, dict):
        payload = data.get("data", [])
        if isinstance(payload, list):
            parsed = [entry for entry in payload if isinstance(entry, dict)]
            return sorted(
                parsed,
                key=lambda entry: entry.get("filingDate", "") or _timestamp_to_date_str(entry.get("transactionDate")),
                reverse=True,
            )
    return []




def get_data_in_range(ticker, start_date, end_date, data_type, data_dir, period=None):
    """
    Gets finnhub data saved and processed on disk.
    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        data_type (str): Type of data from finnhub to fetch. Can be insider_trans, SEC_filings, news_data, insider_senti, or fin_as_reported.
        data_dir (str): Directory where the data is saved.
        period (str): Default to none, if there is a period specified, should be annual or quarterly.
    """

    if period:
        data_path = os.path.join(
            data_dir,
            "finnhub_data",
            data_type,
            f"{ticker}_{period}_data_formatted.json",
        )
    else:
        data_path = os.path.join(
            data_dir, "finnhub_data", data_type, f"{ticker}_data_formatted.json"
        )

    with open(data_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    # filter keys (date, str in format YYYY-MM-DD) by the date range (str, str in format YYYY-MM-DD)
    filtered_data = {}
    for key, value in data.items():
        if start_date <= key <= end_date and len(value) > 0:
            filtered_data[key] = value
    return filtered_data
