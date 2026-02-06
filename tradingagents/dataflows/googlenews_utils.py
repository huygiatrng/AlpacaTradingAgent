import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import random
import urllib.parse
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_result,
)

try:
    import feedparser
except ImportError:
    feedparser = None


def is_rate_limited(response):
    """Check if the response indicates rate limiting (status code 429)"""
    return response.status_code == 429


@retry(
    retry=(retry_if_result(is_rate_limited)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
)
def make_request(url, headers):
    """Make a request with retry logic for rate limiting"""
    # Reduced delay for better performance while still avoiding detection
    time.sleep(random.uniform(1, 3))
    response = requests.get(url, headers=headers, timeout=15)
    return response


def _getNewsDataRSS(query, start_date, end_date, max_results=20):
    """
    Fetch Google News results via the public RSS feed (much more reliable than
    HTML scraping because the feed format is stable).

    query:       str - search query (can contain spaces)
    start_date:  str - start date in ``yyyy-mm-dd`` **or** ``mm/dd/yyyy``
    end_date:    str - end date in ``yyyy-mm-dd`` **or** ``mm/dd/yyyy``
    max_results: int - maximum number of articles to return (default 20)

    Returns a list of dicts with keys: link, title, snippet, date, source
    """
    if feedparser is None:
        print("[GOOGLE-NEWS] feedparser not installed, falling back to HTML scraping")
        return []

    # The caller may have pre-encoded spaces as '+'; undo before URL encoding
    query = query.replace("+", " ")

    # Normalise dates to yyyy-mm-dd for the ``after:`` / ``before:`` operators
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            start_dt = datetime.strptime(start_date, fmt)
            break
        except ValueError:
            pass
    else:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")

    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            end_dt = datetime.strptime(end_date, fmt)
            break
        except ValueError:
            pass
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    after_str = start_dt.strftime("%Y-%m-%d")
    before_str = end_dt.strftime("%Y-%m-%d")

    encoded_q = urllib.parse.quote_plus(f"{query} after:{after_str} before:{before_str}")
    rss_url = (
        f"https://news.google.com/rss/search?q={encoded_q}"
        f"&hl=en-US&gl=US&ceid=US:en"
    )

    try:
        feed = feedparser.parse(rss_url)
    except Exception as exc:
        print(f"[GOOGLE-NEWS] RSS parse error: {exc}")
        return []

    news_results = []
    for entry in feed.entries[:max_results]:
        title = getattr(entry, "title", "")
        link = getattr(entry, "link", "")
        published = getattr(entry, "published", "Unknown")
        # Google News RSS puts the source at the end of the title after " - "
        source = "Unknown"
        if " - " in title:
            parts = title.rsplit(" - ", 1)
            title = parts[0]
            source = parts[1] if len(parts) > 1 else "Unknown"
        # RSS entries often lack a separate snippet; use title as a fallback
        snippet = getattr(entry, "summary", title)
        # Strip HTML tags from snippet (Google wraps it in <a> tags)
        if "<" in snippet:
            try:
                snippet = BeautifulSoup(snippet, "html.parser").get_text()
            except Exception:
                pass

        news_results.append(
            {
                "link": link,
                "title": title,
                "snippet": snippet,
                "date": published,
                "source": source,
            }
        )

    return news_results


def _getNewsDataScrape(query, start_date, end_date, max_pages=3):
    """
    Legacy HTML-scraping approach (fallback if RSS fails).
    Scrape Google News search results for a given query and date range.
    """
    if "-" in start_date:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        start_date = start_date.strftime("%m/%d/%Y")
    if "-" in end_date:
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        end_date = end_date.strftime("%m/%d/%Y")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    news_results = []
    page = 0
    while page < max_pages:
        offset = page * 10
        url = (
            f"https://www.google.com/search?q={query}"
            f"&tbs=cdr:1,cd_min:{start_date},cd_max:{end_date}"
            f"&tbm=nws&start={offset}"
        )

        try:
            response = make_request(url, headers)
            soup = BeautifulSoup(response.content, "html.parser")

            # Try multiple CSS selector strategies (Google changes these often)
            results_on_page = soup.select("div.SoaBEf")
            if not results_on_page:
                results_on_page = soup.select("div.xuvV6b")
            if not results_on_page:
                results_on_page = soup.select("div.dbsr")

            if not results_on_page:
                break  # No more results found

            for el in results_on_page:
                try:
                    link = el.find("a")["href"] if el.find("a") else ""
                    # Try several known selector patterns for title/snippet
                    title_el = (
                        el.select_one("div.MBeuO")
                        or el.select_one("div.n0jPhd")
                        or el.select_one("div.JheGif")
                        or el.select_one("a div[role='heading']")
                    )
                    snippet_el = (
                        el.select_one(".GI74Re")
                        or el.select_one(".Y3v8qd")
                        or el.select_one(".dG2XIf")
                    )
                    date_el = el.select_one(".LfVVr") or el.select_one(".WG9SHc span")
                    source_el = el.select_one(".NUnG9d span") or el.select_one(".CEMjEf span")

                    if not title_el:
                        continue

                    title = title_el.get_text()
                    snippet = snippet_el.get_text() if snippet_el else title
                    date = date_el.get_text() if date_el else "Unknown"
                    source = source_el.get_text() if source_el else "Unknown"
                    news_results.append(
                        {
                            "link": link,
                            "title": title,
                            "snippet": snippet,
                            "date": date,
                            "source": source,
                        }
                    )
                except Exception as e:
                    print(f"Error processing result: {e}")
                    continue

            next_link = soup.find("a", id="pnnext")
            if not next_link:
                break

            page += 1

        except Exception as e:
            print(f"Failed after multiple retries: {e}")
            break

    return news_results


def getNewsData(query, start_date, end_date, max_pages=3):
    """
    Fetch Google News results for a given query and date range.

    Uses the RSS feed first (fast and reliable). Falls back to HTML scraping
    if RSS returns nothing.

    query:      str - search query
    start_date: str - start date (yyyy-mm-dd or mm/dd/yyyy)
    end_date:   str - end date   (yyyy-mm-dd or mm/dd/yyyy)
    max_pages:  int - max pages for the HTML-scraping fallback (default 3)
    """
    # --- Primary: RSS feed (fast, stable) ---
    results = _getNewsDataRSS(query, start_date, end_date, max_results=max_pages * 10)
    if results:
        return results

    # --- Fallback: HTML scraping (slow, fragile) ---
    print(f"[GOOGLE-NEWS] RSS returned 0 results for '{query}', trying HTML scrape fallback...")
    return _getNewsDataScrape(query, start_date, end_date, max_pages=max_pages)
