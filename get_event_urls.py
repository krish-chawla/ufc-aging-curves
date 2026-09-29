"""
Companion script: collect every UFC event-details URL from UFCStats.com's
completed-events list, from a given start year through today, and save
them to a plain text file (one URL per line) that scrape_card.py can
consume via --event-file.

Usage
-----
    python3 get_event_urls.py --since 2010 --out events.txt

Then feed that file into the main scraper:
    python3 scrape_card.py --event-file events.txt --out ufc_full.csv
"""

import argparse
import hashlib
import re
import time
from datetime import datetime
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

COMPLETED_EVENTS_URL = "http://ufcstats.com/statistics/events/completed?page=all"


def _looks_like_challenge(text: str) -> bool:
    return "Checking your browser" in text and "nonce=" in text


def _solve_challenge(resp: requests.Response, url: str) -> None:
    text = resp.text
    nonce_match = re.search(r'nonce\s*=\s*"([a-f0-9]+)"', text)
    target_match = re.search(r"new Array\((\d+)\+1\)", text)
    if not nonce_match or not target_match:
        raise RuntimeError("Couldn't parse the bot-check challenge — page format may have changed.")

    nonce = nonce_match.group(1)
    zeros = int(target_match.group(1))
    target = "0" * zeros

    n = 0
    while not hashlib.sha256(f"{nonce}:{n}".encode()).hexdigest().startswith(target):
        n += 1

    parsed = urlparse(url)
    challenge_url = f"{parsed.scheme}://{parsed.netloc}/__c"
    SESSION.post(
        challenge_url,
        data={"nonce": nonce, "n": n},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=20,
    )


def get_soup(url: str) -> BeautifulSoup:
    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()

    if _looks_like_challenge(resp.text):
        _solve_challenge(resp, url)
        resp = SESSION.get(url, timeout=30)
        resp.raise_for_status()
        if _looks_like_challenge(resp.text):
            raise RuntimeError("Still hitting the bot-check page after solving the challenge.")

    return BeautifulSoup(resp.text, "lxml")


def get_all_event_urls(since_year: int) -> list[dict]:
    """Return [{'url': ..., 'name': ..., 'date': datetime}] for every completed
    event on or after Jan 1 of since_year, most recent first."""
    soup = get_soup(COMPLETED_EVENTS_URL)

    rows = soup.select("tr.b-statistics__table-row")
    events = []
    for row in rows:
        link = row.select_one("a.b-link")
        date_span = row.select_one("span.b-statistics__date")
        if not link or not date_span:
            continue  # header row or an upcoming/unlinked event

        url = link.get("href")
        name = link.get_text(strip=True)
        date_text = date_span.get_text(strip=True)
        try:
            date = datetime.strptime(date_text, "%B %d, %Y")
        except ValueError:
            continue

        if date.year < since_year:
            continue  # events are listed newest-first; could `break` instead,
                       # but `continue` is safer against any out-of-order rows

        events.append({"url": url, "name": name, "date": date})

    events.sort(key=lambda e: e["date"])
    return events


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", type=int, default=2010, help="Earliest year to include (default 2010)")
    parser.add_argument("--out", default="events.txt", help="Output text file, one event URL per line")
    args = parser.parse_args()

    print(f"Fetching completed-events list (since {args.since}) ...")
    events = get_all_event_urls(args.since)

    if not events:
        raise SystemExit(
            "No events found — UFCStats may have changed the page layout, "
            "or ?page=all isn't returning the full list. Run with a browser "
            "dev-tools inspection of http://ufcstats.com/statistics/events/completed "
            "and send me the row HTML."
        )

    with open(args.out, "w") as f:
        for e in events:
            f.write(e["url"] + "\n")

    print(f"Found {len(events)} events from {events[0]['date'].date()} to {events[-1]['date'].date()}.")
    print(f"Saved event URLs to {args.out}")
    print("\nFirst few:")
    for e in events[:5]:
        print(f"  {e['date'].date()}  {e['name']}")
    print("...")
    for e in events[-5:]:
        print(f"  {e['date'].date()}  {e['name']}")