import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import random
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_result,
)


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
    response = requests.get(url, headers=headers)
    return response


def getNewsData(query, start_date, end_date, max_pages=3):
    """
    Scrape Google News search results for a given query and date range.
    query: str - search query
    start_date: str - start date in the format yyyy-mm-dd or mm/dd/yyyy
    end_date: str - end date in the format yyyy-mm-dd or mm/dd/yyyy
    max_pages: int - maximum number of pages to scrape (default 3, each page has ~10 results)
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
            "Chrome/131.0.0.0 Safari/537.36"
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
            results_on_page = soup.select("div.SoaBEf")

            if not results_on_page:
                print(f"[GOOGLE NEWS] No results found on page {page}")
                print(f"[GOOGLE NEWS] Response status: {response.status_code}")
                break  # No more results found

            for el in results_on_page:
                try:
                    link = el.find("a")["href"]
                    title_el = el.select_one("div.MBeuO")

                    # Try multiple selectors for snippet (Google changes these frequently)
                    snippet_el = (
                        el.select_one(".VwiC3b") or
                        el.select_one(".GI74Re") or
                        el.select_one("div[data-hveid] > div")
                    )
                    date_el = el.select_one(".LfVVr")
                    source_el = el.select_one(".NUnG9d span")

                    # Skip this result if mandatory fields are missing
                    if not title_el or not snippet_el:
                        continue

                    title = title_el.get_text()
                    snippet = snippet_el.get_text().strip()
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
                    # If one of the fields is not found, skip this result
                    continue

            # Update the progress bar with the current count of results scraped

            # Check for the "Next" link (pagination)
            next_link = soup.find("a", id="pnnext")
            if not next_link:
                break

            page += 1

        except Exception as e:
            print(f"Failed after multiple retries: {e}")
            break

    return news_results
