"""Macro data must preserve historical evidence and surface provider failures."""

import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest
import requests

from tradingagents.dataflows import fxmacrodata_utils as fx


@pytest.fixture(autouse=True)
def historical_dates():
    # Keep these cases independent of the day on which CI runs.
    with patch.object(fx, "_is_historical_date", return_value=True):
        yield


def epoch(day):
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp())


class Response:
    status_code = 200
    reason = "OK"

    def __init__(self, body=None, malformed=False):
        self.body = body
        self.malformed = malformed

    def json(self):
        if self.malformed:
            raise ValueError("not JSON")
        return self.body


def payload(rows):
    return {
        "name": "GDP",
        "value_metadata": {"source_unit": "%QoQ"},
        "pagination": {"has_more": False},
        "data": rows,
    }


def indicators(body, day="2026-08-01"):
    with patch.object(fx, "get_fxmacrodata_api_key", return_value="test-key"), patch.object(
        fx, "HEADLINE_INDICATORS", {"gdp": "GDP"}
    ), patch.object(fx, "_fxmacrodata_get", return_value=body):
        return fx.get_global_macro_indicators(day, "USD")


def test_historical_request_pins_known_at_time():
    with patch.object(fx, "get_fxmacrodata_api_key", return_value="test-key"), patch.object(
        fx, "HEADLINE_INDICATORS", {"gdp": "GDP"}
    ), patch.object(fx, "_fxmacrodata_get", return_value=payload([])) as get:
        fx.get_global_macro_indicators("2026-08-01", "USD")
    assert "as_of" in get.call_args.args[1], "end_date filters periods, not revision vintages"


def test_future_revision_not_used_as_historical_value():
    report = indicators(payload([{
        "date": "2026-06-30",
        "announcement_datetime": epoch("2026-07-30"),
        "val": 3.0,
        "revisions": [
            {"epoch": epoch("2026-07-30"), "val": 2.0},
            {"epoch": epoch("2026-09-30"), "val": 3.0},
        ],
    }]))
    assert "| GDP | 3.00" not in report, "September revision leaked into August analysis"


def test_missing_release_time_and_later_capture_are_not_historical_evidence():
    report = indicators(payload([{
        "date": "2026-07-31",
        "val": 4.83,
        "publication_time_status": "unknown",
        "vintage_status": "captured_snapshot",
        "observed_at_ns": epoch("2026-08-03") * 1_000_000_000,
        "collected_at_iso": "2026-08-03T05:18:43Z",
    }]))
    assert "4.83" not in report, "Value only captured after cutoff must be excluded or disclaimed"


@pytest.mark.parametrize("failure", ["timeout", "HTTP 429: quota exceeded", "HTTP 503: unavailable"])
def test_macro_report_exposes_provider_failure(failure):
    if failure == "timeout":
        with patch.object(fx, "get_fxmacrodata_api_key", return_value="test-key"), patch.object(
            fx, "HEADLINE_INDICATORS", {"gdp": "GDP"}
        ), patch.object(fx.requests, "get", side_effect=requests.exceptions.Timeout("provider timeout")):
            report = fx.get_global_macro_indicators("2026-08-01", "USD")
    else:
        report = indicators({"error": failure})
    assert "error" in report.lower(), "Outage must not look like an empty successful report"
    assert report.startswith("Error:"), "A complete provider outage must be a tool failure"


@pytest.mark.parametrize("body,malformed", [(None, True), (None, False), ([], False), ("unexpected", False)])
def test_invalid_success_payload_is_reported_as_error(body, malformed):
    with patch.object(fx, "get_fxmacrodata_api_key", return_value=None), patch.object(
        fx.requests, "get", return_value=Response(body, malformed)
    ):
        result = fx._fxmacrodata_get("/calendar/usd")
    assert isinstance(result, dict) and result.get("error"), "Malformed HTTP 200 cannot be accepted"


def test_fx_report_does_not_claim_full_window_after_page_cap():
    end = datetime(2026, 10, 3)

    def page(_path, params):
        offset = params["offset"]
        rows = [
            {"date": (end - timedelta(days=i)).strftime("%Y-%m-%d"), "val": 1.0 + i / 1000}
            for i in range(offset, offset + 100)
        ]
        return {"data": rows, "pagination": {"has_more": True, "next_offset": offset + 100}}

    with patch.object(fx, "get_fxmacrodata_api_key", return_value="test-key"), patch.object(
        fx, "_fxmacrodata_get", side_effect=page
    ):
        report = fx.get_fx_rates_report("2026-10-03", "EUR/USD", lookback_days=1000)
    assert any(word in report.lower() for word in ("incomplete", "truncated", "error")), report
    assert "| EUR/USD |" not in report, "Incomplete history must not produce window statistics"


def test_zero_is_valid_and_null_is_skipped():
    rows = [
        {"date": "2026-07-31", "val": 0, "announcement_datetime": epoch("2026-08-01")},
        {"date": "2026-07-31", "val": None, "announcement_datetime": epoch("2026-08-01")},
    ]
    assert fx._released_rows(rows, "2026-08-01") == [rows[0]]


def verified_payload(value=2.0):
    body = payload([{
        "date": "2026-06-30",
        "val": value,
        "announcement_datetime": epoch("2026-07-30"),
        "publication_time_status": "confirmed",
    }])
    body["replay"] = {"as_of": "2026-08-01T23:59:59Z"}
    return body


def test_verified_replay_uses_selected_vintage_and_not_later_revision():
    body = verified_payload()
    body["data"][0]["revisions"] = [
        {"epoch": epoch("2026-07-30"), "val": 2.0},
        {"epoch": epoch("2026-09-30"), "val": 3.0},
    ]
    report = indicators(body)
    assert "| GDP | 2.00 %QoQ |" in report
    assert "3.00" not in report


@pytest.mark.parametrize("status", [None, "unknown", "unverified", "assumed_historical"])
def test_replay_without_confirmed_publication_omits_value(status):
    body = verified_payload()
    body["data"][0]["publication_time_status"] = status
    report = indicators(body)
    assert "2.00" not in report
    assert "No verified point-in-time" in report


@pytest.mark.parametrize("as_of", [None, "2026-09-01T23:59:59Z"])
def test_wrong_replay_cutoff_is_a_failure(as_of):
    body = verified_payload()
    body["replay"]["as_of"] = as_of
    report = indicators(body)
    assert report.startswith("Error:")
    assert "2.00" not in report


def test_today_keeps_latest_snapshot_without_requesting_historical_replay():
    body = payload([{"date": "2026-08-01", "val": 4.83}])
    with patch.object(fx, "_is_historical_date", return_value=False), patch.object(
        fx, "get_fxmacrodata_api_key", return_value="test-key"
    ), patch.object(fx, "HEADLINE_INDICATORS", {"gdp": "GDP"}), patch.object(
        fx, "_fxmacrodata_get", return_value=body
    ) as get:
        report = fx.get_global_macro_indicators("2026-08-01", "USD")
    assert "4.83" in report
    assert "as_of" not in get.call_args.args[1]
    assert "Latest stored values" in report


def test_partial_failure_keeps_valid_evidence_and_reports_missing_source():
    with patch.object(fx, "get_fxmacrodata_api_key", return_value="test-key"), patch.object(
        fx, "HEADLINE_INDICATORS", {"gdp": "GDP", "policy_rate": "Policy Rate"}
    ), patch.object(fx, "_fxmacrodata_get", side_effect=[verified_payload(), {"error": "HTTP 429"}]):
        report = fx.get_global_macro_indicators("2026-08-01", "USD")
    assert "| GDP | 2.00 %QoQ |" in report
    assert "USD/policy_rate: HTTP 429" in report
    assert not report.startswith("Error:")


@pytest.mark.parametrize("body", [
    {"data": None}, {"data": [None]}, {"data": [], "pagination": []},
    {"data": [], "pagination": {"has_more": "yes"}},
    {"data": [], "value_metadata": {"source_unit": 1}},
    {"data": [{"date": "2026-07-30", "val": "NaN"}]},
    {"data": [{"date": "2026-07-30", "val": True}]},
    {"data": [{"date": "not-a-date", "val": 1}]},
    {"data": [{"date": "2026-07-30", "val": 1, "announcement_datetime": "yesterday"}]},
])
def test_unusable_data_fields_return_error(body):
    with patch.object(fx, "get_fxmacrodata_api_key", return_value=None), patch.object(
        fx.requests, "get", return_value=Response(body)
    ):
        result = fx._fxmacrodata_get("/announcements/usd/gdp")
    assert result.get("error")


def test_empty_valid_response_is_not_an_operational_error():
    with patch.object(fx, "get_fxmacrodata_api_key", return_value=None), patch.object(
        fx.requests, "get", return_value=Response({"data": []})
    ):
        result = fx._fxmacrodata_get("/calendar/usd")
    assert result == {"data": []}


@pytest.mark.parametrize("rows,next_offset", [([], 100), ([{"val": 1}], 0)])
def test_stalled_pagination_returns_error(rows, next_offset):
    page = {"data": rows, "pagination": {"has_more": True, "next_offset": next_offset}}
    with patch.object(fx, "_fxmacrodata_get", return_value=page):
        result = fx._fetch_rows("/forex/eur/usd", {})
    assert "error" in result


def test_final_page_at_request_limit_is_complete_and_pins_dataset():
    first = {"data": [{"val": 2}], "pagination": {"has_more": True, "next_offset": 1}, "dataset_version": "v1"}
    last = {"data": [{"val": 1}], "pagination": {"has_more": False}, "dataset_version": "v1"}
    seen = []

    def get(_path, params):
        seen.append(dict(params))
        return first if params["offset"] == 0 else last

    with patch.object(fx, "MAX_PAGES", 2), patch.object(fx, "_fxmacrodata_get", side_effect=get):
        result = fx._fetch_rows("/forex/eur/usd", {})
    assert result["data"] == [{"val": 2}, {"val": 1}]
    assert seen[1]["dataset_version"] == "v1"


class RedirectHandler(BaseHTTPRequestHandler):
    requested = []
    redirect_status = 302

    def do_GET(self):
        self.requested.append(self.path)
        self.send_response(self.redirect_status)
        self.send_header("Location", "/elsewhere")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirect_is_not_followed_so_key_is_not_forwarded(status):
    RedirectHandler.requested = []
    RedirectHandler.redirect_status = status
    server = HTTPServer(("127.0.0.1", 0), RedirectHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with patch.object(fx, "BASE_URL", f"http://127.0.0.1:{server.server_port}/v1"), patch.object(
            fx, "get_fxmacrodata_api_key", return_value="test-key"
        ):
            result = fx._fxmacrodata_get("/calendar/usd")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert RedirectHandler.requested == ["/v1/calendar/usd"], "Redirect target must not be requested"
    assert "redirect" in result["error"]
    assert "test-key" not in str(result)


@pytest.mark.parametrize("api_key", ["test-key\r\nX-Injected: 1", "test key", "test-key\x00"])
def test_malformed_key_is_rejected_without_echoing_it(api_key):
    with patch.object(fx, "get_fxmacrodata_api_key", return_value=api_key), patch.object(
        fx.requests, "get", side_effect=AssertionError("no request expected")
    ):
        result = fx._fxmacrodata_get("/calendar/gbp")
        report = fx.get_economic_release_calendar("2026-10-01", "GBP")
    assert result.get("error") and "whitespace or control" in result["error"]
    assert "test" not in str(result), "Key must never reach text returned to the model"
    assert api_key not in report and "test-key" not in report


def test_surrounding_whitespace_is_stripped_from_key():
    with patch.object(fx, "get_fxmacrodata_api_key", return_value="  test-key\n"), patch.object(
        fx.requests, "get", return_value=Response({"data": []})
    ) as get:
        fx._fxmacrodata_get("/calendar/usd")
    assert get.call_args.kwargs["headers"]["X-API-Key"] == "test-key"
    assert get.call_args.kwargs["allow_redirects"] is False


def test_error_body_on_success_status_includes_api_detail():
    with patch.object(fx, "get_fxmacrodata_api_key", return_value=None), patch.object(
        fx.requests, "get", return_value=Response({"detail": "Unknown indicator"})
    ):
        result = fx._fxmacrodata_get("/announcements/usd/gdp")
    assert "Unknown indicator" in result["error"]


@pytest.mark.parametrize("status", [200, 400])
def test_provider_error_detail_redacts_the_api_key_in_reports(status):
    response = Response({"detail": "Rejected credential DUMMY_SECRET_52"})
    response.status_code = status
    with patch.object(fx, "get_fxmacrodata_api_key", return_value="DUMMY_SECRET_52"), patch.object(
        fx.requests, "get", return_value=response
    ):
        result = fx._fxmacrodata_get("/calendar/gbp")
        report = fx.get_economic_release_calendar("2026-10-07", "GBP")
    assert result.get("error")
    assert "Rejected credential" in result["error"]
    assert "DUMMY_SECRET_52" not in str(result)
    assert "DUMMY_SECRET_52" not in report


def test_http_reason_redacts_the_api_key():
    response = Response({})
    response.status_code = 400
    response.reason = "Rejected DUMMY_SECRET_52"
    with patch.object(fx, "get_fxmacrodata_api_key", return_value="DUMMY_SECRET_52"), patch.object(
        fx.requests, "get", return_value=response
    ):
        result = fx._fxmacrodata_get("/calendar/gbp")
    assert result.get("error") and "DUMMY_SECRET_52" not in str(result)


@pytest.mark.parametrize("error", [
    requests.exceptions.InvalidHeader("Invalid credential DUMMY_SECRET_52"),
    requests.exceptions.Timeout("Request using DUMMY_SECRET_52 timed out"),
    UnicodeEncodeError("latin-1", "DUMMY_SECRET_52\u2603", 15, 16, "cannot encode"),
])
def test_transport_failures_never_echo_credentials(error):
    with patch.object(fx, "get_fxmacrodata_api_key", return_value="DUMMY_SECRET_52"), patch.object(
        fx.requests, "get", side_effect=error
    ):
        result = fx._fxmacrodata_get("/calendar/gbp")
    assert result.get("error")
    assert type(error).__name__ in result["error"]
    assert "DUMMY_SECRET_52" not in str(result)


@pytest.mark.parametrize("key", ["DUMMY_SECRET_52\u2603", 123, True])
def test_invalid_header_key_is_rejected_before_transport(key):
    with patch.object(fx, "get_fxmacrodata_api_key", return_value=key), patch.object(fx.requests, "get") as get:
        result = fx._fxmacrodata_get("/calendar/gbp")
    get.assert_not_called()
    assert result.get("error") and result.get("key_required")
    assert "DUMMY_SECRET_52" not in str(result)
