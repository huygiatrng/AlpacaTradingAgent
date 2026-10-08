"""FXMacroData helpers for the macro analyst.

FRED only covers the US. These helpers add headline releases for other
economies (policy rate, CPI, unemployment, GDP, government bond yields), the
scheduled economic release calendar, and FX reference rates.

API docs: https://fxmacrodata.com/documentation/reference

USD announcements and the USD calendar are readable without a key (recent
90 days, delayed 15 minutes). Other currencies and FX rates need
FXMACRODATA_API_KEY.
"""

from datetime import datetime, timedelta, timezone
import json
import math
from typing import Dict, List, Optional
from urllib.parse import quote

import requests

from .config import get_config, get_fxmacrodata_api_key

BASE_URL = "https://api.fxmacrodata.com/v1"
REQUEST_TIMEOUT = 15
PAGE_LIMIT = 100  # API maximum per page
MAX_PAGES = 5

# Headline series shown per currency. Slugs follow /v1/data_catalogue/{currency};
# a currency that does not publish one of these is simply skipped.
HEADLINE_INDICATORS = {
    "policy_rate": "Policy Rate",
    "inflation": "Inflation",
    "core_inflation": "Core Inflation",
    "unemployment": "Unemployment Rate",
    "gdp": "GDP",
    "gov_bond_2y": "2Y Government Bond",
    "gov_bond_10y": "10Y Government Bond",
}

DEFAULT_CURRENCIES = ["USD", "EUR", "GBP", "JPY"]
DEFAULT_FX_PAIRS = ["EUR/USD", "USD/JPY", "GBP/USD"]


def _redact_error_detail(detail, api_key: str) -> str:
    """Preserve useful provider diagnostics while removing echoed credentials."""
    text = detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False)
    if api_key:
        for secret in {api_key, json.dumps(api_key)[1:-1], quote(api_key, safe="")}:
            text = text.replace(secret, "[REDACTED]")
    return text


def _fxmacrodata_get(path: str, params: Optional[Dict] = None) -> Dict:
    """GET an FXMacroData endpoint. Returns the JSON body or {"error": ...}."""
    headers = {"Accept": "application/json"}
    api_key = get_fxmacrodata_api_key()
    if api_key is not None and not isinstance(api_key, str):
        return {"error": "FXMACRODATA_API_KEY must be a string.", "key_required": True}
    api_key = (api_key or "").strip()
    if api_key:
        # requests echoes an invalid header value in its exception message, and
        # that message is returned to the model, so never let the key reach it.
        if any(char.isspace() or not char.isprintable() for char in api_key):
            return {
                "error": "FXMACRODATA_API_KEY contains whitespace or control characters.",
                "key_required": True,
            }
        try:
            api_key.encode("latin-1")  # Encoding used by the HTTP transport.
        except UnicodeEncodeError:
            return {
                "error": "FXMACRODATA_API_KEY cannot be encoded as an HTTP header.",
                "key_required": True,
            }
        headers["X-API-Key"] = api_key

    try:
        # requests only strips Authorization on a cross-host redirect, so a
        # followed redirect would forward X-API-Key to the new host.
        response = requests.get(
            f"{BASE_URL}{path}",
            params=params or {},
            headers=headers,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=False,
        )
    except (requests.exceptions.RequestException, UnicodeError) as e:
        # Even a well-formed credential may be echoed by a transport adapter.
        return {"error": f"Failed to fetch FXMacroData {path}: {type(e).__name__}"}

    if 300 <= response.status_code < 400:
        return {
            "error": f"FXMacroData {path} returned a redirect (HTTP {response.status_code}), which is not followed.",
            "status_code": response.status_code,
        }

    invalid_json = False
    try:
        body = response.json()
    except ValueError:
        body = {}
        invalid_json = True

    if response.status_code in (401, 403):
        return {
            "error": f"FXMacroData {path} returned HTTP {response.status_code}: a valid "
            "FXMACRODATA_API_KEY is required (USD works without a key).",
            "key_required": True,
            "status_code": response.status_code,
        }
    if response.status_code >= 400:
        detail = body.get("detail") if isinstance(body, dict) else None
        return {
            "error": f"FXMacroData {path} returned HTTP {response.status_code}: "
                     f"{_redact_error_detail(detail or response.reason, api_key)}",
            "status_code": response.status_code,
        }
    if invalid_json or not _valid_payload(body, path):
        detail = body.get("detail") if isinstance(body, dict) else None
        suffix = f": {_redact_error_detail(detail, api_key)}" if isinstance(detail, str) else "."
        return {"error": f"FXMacroData {path} returned an invalid JSON data response{suffix}"}
    return body


def _valid_payload(body, path: str) -> bool:
    """Validate fields used by the reports before treating HTTP 200 as success."""
    if not isinstance(body, dict) or not isinstance(body.get("data"), list):
        return False
    for key in ("pagination", "value_metadata", "replay", "data_quality"):
        if body.get(key) is not None and not isinstance(body[key], dict):
            return False
    unit = (body.get("value_metadata") or {}).get("source_unit")
    if unit is not None and not isinstance(unit, str):
        return False
    pagination = body.get("pagination") or {}
    if "has_more" in pagination and not isinstance(pagination["has_more"], bool):
        return False
    next_offset = pagination.get("next_offset")
    if next_offset is not None and (type(next_offset) is not int or next_offset < 0):
        return False
    for row in body["data"]:
        if not isinstance(row, dict):
            return False
        value = row.get("val")
        if value is not None:
            try:
                if isinstance(value, bool) or not math.isfinite(float(value)):
                    return False
            except (ValueError, TypeError, OverflowError):
                return False
            if path.startswith(("/announcements/", "/forex/")):
                try:
                    datetime.strptime(row["date"], "%Y-%m-%d")
                except (KeyError, ValueError, TypeError):
                    return False
        announced = row.get("announcement_datetime")
        if announced is not None and (
            type(announced) not in (int, float) or not 0 <= announced < 253402300800
        ):
            return False
        when = row.get("announcement_datetime_utc")
        if when is not None and not isinstance(when, str):
            return False
    return True


def _fetch_rows(path: str, params: Dict) -> Dict:
    """Fetch a complete window within the request budget, or return an error."""
    rows: List[Dict] = []
    params = dict(params)
    params.setdefault("limit", PAGE_LIMIT)
    offset = 0
    body: Dict = {}

    for _ in range(MAX_PAGES):
        params["offset"] = offset
        body = _fxmacrodata_get(path, params)
        if "error" in body:
            return body
        page = body.get("data") or []
        rows.extend(page)
        pagination = body.get("pagination") or {}
        if not pagination.get("has_more"):
            break
        next_offset = pagination.get("next_offset", offset + len(page))
        if not page or next_offset is None or next_offset <= offset:
            return {"error": f"FXMacroData {path} returned incomplete or stalled pagination."}
        offset = next_offset
        if body.get("dataset_version"):
            params.setdefault("dataset_version", body["dataset_version"])
    else:
        return {
            "error": f"FXMacroData {path} window is incomplete after {MAX_PAGES} pages; "
            "use a shorter lookback. No full-window statistics were calculated."
        }

    body = dict(body)
    body["data"] = rows
    return body


def _parse_currencies(currencies) -> List[str]:
    if not currencies:
        currencies = get_config().get("fxmacrodata_currencies") or DEFAULT_CURRENCIES
    if isinstance(currencies, str):
        currencies = currencies.split(",")
    return [c.strip().upper() for c in currencies if c and c.strip()]


def _split_by_access(currencies: List[str]):
    """Without a key only USD is served, so don't spend requests on the rest."""
    if get_fxmacrodata_api_key():
        return currencies, []
    return [c for c in currencies if c == "USD"], [c for c in currencies if c != "USD"]


def _end_of_day_epoch(curr_date: str) -> int:
    day = datetime.strptime(curr_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int((day + timedelta(days=1)).timestamp())


def _is_historical_date(curr_date: str) -> bool:
    return datetime.strptime(curr_date, "%Y-%m-%d").date() < datetime.now(timezone.utc).date()


def _released_rows(rows: List[Dict], curr_date: str, *, historical: bool = False) -> List[Dict]:
    """Keep rows that have a value and were public by the end of curr_date.

    `val` can be null for a scheduled-but-unpublished period; those rows are
    dropped rather than read as zero. Historical callers also require a
    provider replay pinned to the cutoff and confirmed publication times;
    an old period or an assumed timestamp is not proof of availability.
    """
    cutoff = _end_of_day_epoch(curr_date)
    released = []
    for row in rows:
        if row.get("val") is None:
            continue
        announced = row.get("announcement_datetime")
        if historical and (
            row.get("publication_time_status") != "confirmed"
            or row.get("release_time_assumed")
            or announced is None
        ):
            continue
        if announced is not None and announced >= cutoff:
            continue
        released.append(row)
    return released


def _format_value(value: float, unit: Optional[str]) -> str:
    if unit and unit.strip() == "%":
        return f"{value:.2f}%"
    if unit:
        return f"{value:,.2f} {unit}"
    return f"{value:,.2f}"


def get_global_macro_indicators(curr_date: str, currencies=None) -> str:
    """
    Latest headline macro releases per currency, as known on curr_date.

    Args:
        curr_date: Current date in YYYY-MM-DD format
        currencies: List or comma-separated string of currency codes
            (defaults to config "fxmacrodata_currencies")

    Returns:
        Markdown report with one table per currency
    """
    currency_list, skipped_for_key = _split_by_access(_parse_currencies(currencies))
    try:
        historical = _is_historical_date(curr_date)
        as_of = datetime.fromtimestamp(
            _end_of_day_epoch(curr_date) - 1, timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError):
        return "Error: curr_date must be a valid YYYY-MM-DD date."
    result = f"## Global Macro Indicators as of {curr_date} (FXMacroData)\n\n"
    if historical:
        result += f"Only verified publication vintages known by {as_of} are included.\n\n"
    else:
        result += "Latest stored values; these are not a verified historical replay.\n\n"
    errors = []
    fetched_any = False

    for currency in currency_list:
        table_rows = []
        key_required = False

        for slug, label in HEADLINE_INDICATORS.items():
            params = {"end_date": curr_date, "limit": 6}
            if historical:
                params["as_of"] = as_of
            data = _fxmacrodata_get(
                f"/announcements/{currency.lower()}/{slug}",
                params,
            )
            if "error" in data:
                # A currency may not publish every headline series.
                if data.get("status_code") == 404:
                    continue
                errors.append(f"{currency}/{slug}: {data['error']}")
                if data.get("key_required"):
                    key_required = True
                    break
                continue

            if historical and (data.get("replay") or {}).get("as_of") != as_of:
                errors.append(
                    f"{currency}/{slug}: provider did not verify the requested historical cutoff; values omitted."
                )
                continue
            fetched_any = True
            rows = _released_rows(data.get("data") or [], curr_date, historical=historical)
            if not rows:
                continue

            unit = (data.get("value_metadata") or {}).get("source_unit")
            latest = rows[0]
            latest_value = float(latest["val"])
            previous_str = "-"
            change_str = "-"
            if len(rows) >= 2:
                previous_value = float(rows[1]["val"])
                previous_str = f"{_format_value(previous_value, unit)} ({rows[1]['date']})"
                change_str = f"{latest_value - previous_value:+.2f}"

            table_rows.append(
                f"| {data.get('name') or label} | {_format_value(latest_value, unit)} "
                f"| {latest['date']} | {previous_str} | {change_str} |\n"
            )

        if key_required:
            skipped_for_key.append(currency)
            continue

        result += f"### {currency}\n"
        if not table_rows:
            result += (
                "No verified point-in-time releases available for this date.\n\n"
                if historical else "No recent releases available.\n\n"
            )
            continue
        result += "| Indicator | Latest | Period | Previous | Change |\n"
        result += "|-----------|--------|--------|----------|--------|\n"
        result += "".join(table_rows) + "\n"

    if skipped_for_key:
        result += (
            f"**Not loaded**: {', '.join(skipped_for_key)} need FXMACRODATA_API_KEY "
            "(only USD is available without a key).\n"
        )

    for error in errors:
        result += f"\n**Error**: {error}\n"
    if errors and not fetched_any:
        result = "Error: FXMacroData macro indicators could not be loaded.\n\n" + result
    return result


def get_economic_release_calendar(curr_date: str, currencies=None, days_ahead: int = 10) -> str:
    """
    Scheduled economic releases from curr_date through curr_date + days_ahead.

    Args:
        curr_date: Current date in YYYY-MM-DD format
        currencies: List or comma-separated string of currency codes
        days_ahead: Number of days ahead to include (default 10)

    Returns:
        Markdown table of upcoming releases sorted by release time
    """
    currency_list, skipped_for_key = _split_by_access(_parse_currencies(currencies))
    end_date = (datetime.strptime(curr_date, "%Y-%m-%d") + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    result = f"## Economic Release Calendar ({curr_date} to {end_date}, FXMacroData)\n\n"
    result += "Current release schedule; historical schedule vintages are not available for strict backtests.\n\n"

    events = []
    errors = []
    for currency in currency_list:
        data = _fxmacrodata_get(
            f"/calendar/{currency.lower()}",
            {"start_date": curr_date, "end_date": end_date},
        )
        if "error" in data:
            if data.get("key_required"):
                skipped_for_key.append(currency)
            else:
                errors.append(f"{currency}: {data['error']}")
            continue
        for row in data.get("data") or []:
            events.append((currency, row))

    events.sort(key=lambda item: item[1].get("announcement_datetime") or 0)

    if events:
        result += "| Date/Time (UTC) | Currency | Release | Importance | Reference Period |\n"
        result += "|-----------------|----------|---------|------------|------------------|\n"
        for currency, row in events:
            when = row.get("announcement_datetime_utc")
            if when:
                when = when.replace("T", " ")[:16]
            elif row.get("announcement_datetime"):
                when = datetime.fromtimestamp(row["announcement_datetime"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            else:
                when = row.get("date", "-")
            result += (
                f"| {when} | {currency} | {row.get('name') or row.get('release')} "
                f"| {row.get('event_importance') or '-'} | {row.get('reference_period') or '-'} |\n"
            )
    else:
        result += "No scheduled releases found for this window.\n"

    if skipped_for_key:
        result += (
            f"\n**Not loaded**: {', '.join(skipped_for_key)} need FXMACRODATA_API_KEY "
            "(only USD is available without a key).\n"
        )
    for error in errors:
        result += f"\n**Error**: {error}\n"

    return result


def get_fx_rates_report(curr_date: str, pairs=None, lookback_days: int = 30) -> str:
    """
    FX reference rates for a few pairs with change over the lookback window.
    Requires FXMACRODATA_API_KEY.

    Args:
        curr_date: Current date in YYYY-MM-DD format
        pairs: List or comma-separated string of pairs like "EUR/USD"
        lookback_days: Number of days to look back (default 30)

    Returns:
        Markdown table of FX rates
    """
    if not get_fxmacrodata_api_key():
        return "Error: FXMacroData FX rates require FXMACRODATA_API_KEY."

    if not pairs:
        pairs = get_config().get("fxmacrodata_fx_pairs") or DEFAULT_FX_PAIRS
    if isinstance(pairs, str):
        pairs = pairs.split(",")

    start_date = (datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    result = f"## FX Rates ({start_date} to {curr_date}, FXMacroData)\n\n"
    result += "Reference-rate history; publication-time availability is not verified for strict backtests.\n\n"
    result += "| Pair | Latest | Date | Change | Range (Low - High) |\n"
    result += "|------|--------|------|--------|--------------------|\n"

    errors = []
    for pair in pairs:
        pair = pair.strip().upper().replace("-", "/")
        if "/" not in pair:
            continue
        base, quote = pair.split("/", 1)
        data = _fetch_rows(
            f"/forex/{base.lower()}/{quote.lower()}",
            {"start_date": start_date, "end_date": curr_date},
        )
        if "error" in data:
            errors.append(f"{pair}: {data['error']}")
            continue

        # Rows come back most recent first.
        rows = [row for row in data.get("data") or [] if row.get("val") is not None]
        if not rows:
            errors.append(f"{pair}: no data in window")
            continue

        latest = float(rows[0]["val"])
        oldest = float(rows[-1]["val"])
        values = [float(row["val"]) for row in rows]
        change_pct = (latest - oldest) / oldest * 100 if oldest else 0
        result += (
            f"| {pair} | {latest:.4f} | {rows[0]['date']} | {change_pct:+.2f}% "
            f"| {min(values):.4f} - {max(values):.4f} |\n"
        )

    for error in errors:
        result += f"\n**Error**: {error}\n"

    return result
