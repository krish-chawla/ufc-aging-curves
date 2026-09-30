"""
Phase 1 scraper for UFCStats.com — now supports multiple event cards at once,
bout dates, and fighter age-at-fight (per Dr. Johnson's follow-up questions).

What this does
---------------
1. Given one or more event-details URLs, pulls every fight on each card
   (fighters + their fighter-detail links, weight class, method, round,
   time, bout date, and the link to each fight's detail page).
2. For each fight, visits the fight-details page and pulls the full
   "totals" stat line for both fighters (knockdowns, sig. strikes,
   takedowns, control time, etc.).
3. For each unique fighter encountered, visits their fighter-details page
   once (cached, since the same fighter shows up across many cards) and
   pulls their date of birth.
4. Computes age_at_fight = bout_date - DOB, adds a derived `gender` column
   (from the "Women's ..." weight-class prefix), and writes one combined
   CSV across every card given.

Usage
-----
    pip install requests beautifulsoup4 pandas lxml

    # single card:
    python scrape_card.py --event http://ufcstats.com/event-details/638cfec7ec559d6e

    # multiple cards in one run:
    python scrape_card.py \\
        --event http://ufcstats.com/event-details/638cfec7ec559d6e \\
        --event http://ufcstats.com/event-details/<another-hash> \\
        --out combined.csv

    # or a text file with one event URL per line:
    python scrape_card.py --event-file events.txt --out combined.csv
"""

import argparse
import hashlib
import re
import time
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

TOTALS_COLUMNS = [
    "fighter", "kd", "sig_str", "sig_str_pct", "total_str",
    "td", "td_pct", "sub_att", "rev", "ctrl",
]

# One shared session so the "you're not a bot" cookie earned on the first
# request carries over to every later request instead of re-solving each time.
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# Cache fighter DOBs across the whole run so a fighter who appears on
# multiple cards only gets their fighter-details page fetched once.
_DOB_CACHE: dict[str, "datetime | None"] = {}


def _looks_like_challenge(text: str) -> bool:
    return "Checking your browser" in text and "nonce=" in text


def _solve_challenge(resp: requests.Response, url: str) -> None:
    """UFCStats fronts requests with a small proof-of-work check: find a
    number `n` such that sha256(f"{nonce}:{n}") starts with N zeros, then
    POST it to /__c. Solving it is cheap (milliseconds) since it's just math,
    no real browser/JS engine required."""
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
    resp = SESSION.get(url, timeout=20)
    resp.raise_for_status()

    if _looks_like_challenge(resp.text):
        _solve_challenge(resp, url)
        resp = SESSION.get(url, timeout=20)
        resp.raise_for_status()
        if _looks_like_challenge(resp.text):
            raise RuntimeError(
                "Still hitting the bot-check page after solving the challenge — "
                "the cookie may not be sticking. Check that SESSION is reused everywhere."
            )

    return BeautifulSoup(resp.text, "lxml")


def parse_event_date(soup: BeautifulSoup) -> "datetime | None":
    """Event pages show 'Date: September 13, 2025' inside the info box list."""
    for li in soup.select("li.b-list__box-list-item"):
        label = li.select_one("i.b-list__box-item-title")
        if label and "date" in label.get_text(strip=True).lower():
            text = li.get_text(" ", strip=True)
            text = re.sub(r"(?i)^date:\s*", "", text).strip()
            try:
                return datetime.strptime(text, "%B %d, %Y")
            except ValueError:
                return None
    return None


def parse_result_flag(col) -> "str | None":
    """Read the W/L column of an event-page row.
    UFCStats lists the WINNER as fighter_1 and shows one 'win' flag;
    draws show 'draw' and no contests show 'nc'.
    Returns 'win' (fighter_1 won), 'draw', 'nc', or None if unreadable."""
    flags = [t.get_text(strip=True).lower() for t in col.select(".b-flag__text")]
    text = " ".join(flags) if flags else col.get_text(" ", strip=True).lower()
    for key in ("win", "draw", "nc"):
        if key in text.split():
            return key
    return None


def add_outcome(row: dict) -> dict:
    """Per-fighter outcome from bout_result: W / L / D (draw) / NC."""
    res = row.get("bout_result")
    me = (row.get("fighter") or "").strip().lower()
    f1 = (row.get("fighter_1") or "").strip().lower()
    if res == "win":
        row["outcome"] = "W" if me == f1 else "L"
    elif res == "draw":
        row["outcome"] = "D"
    elif res == "nc":
        row["outcome"] = "NC"
    else:
        row["outcome"] = None
    return row


def get_fights_on_card(event_url: str) -> list[dict]:
    """Return one dict per fight on the card, including fighter-detail links."""
    soup = get_soup(event_url)

    event_name_tag = soup.select_one("h2.b-content__title span") or soup.select_one("h2.b-content__title")
    event_name = event_name_tag.get_text(strip=True) if event_name_tag else None
    event_date = parse_event_date(soup)

    fights = []
    rows = soup.select("tr.b-fight-details__table-row")
    for row in rows:
        link = row.get("data-link") or ""
        if not link:
            onclick = row.get("onclick", "")
            match = re.search(r"'(http[^']+)'", onclick)
            link = match.group(1) if match else None

        cols = row.select("td.b-fight-details__table-col")
        if len(cols) < 7 or not link:
            continue  # header row or malformed row

        fighter_anchors = cols[1].select("a")
        fighter_names = [a.get_text(strip=True) for a in fighter_anchors]
        fighter_urls = [a.get("href") for a in fighter_anchors]
        if not fighter_names:  # fallback if names aren't wrapped in <a>
            fighter_names = [p.get_text(strip=True) for p in cols[1].select("p")]
            fighter_urls = [None, None]

        weight_class = cols[6].get_text(" ", strip=True) if len(cols) > 6 else None
        method = cols[7].get_text(" ", strip=True) if len(cols) > 7 else None
        round_num = cols[8].get_text(" ", strip=True) if len(cols) > 8 else None
        fight_time = cols[9].get_text(" ", strip=True) if len(cols) > 9 else None

        gender = "Women" if weight_class and weight_class.strip().startswith("Women's") else "Men"
        bout_result = parse_result_flag(cols[0])

        fights.append({
            "event": event_name,
            "event_date": event_date,
            "fighter_1": fighter_names[0] if len(fighter_names) > 0 else None,
            "fighter_1_url": fighter_urls[0] if len(fighter_urls) > 0 else None,
            "fighter_2": fighter_names[1] if len(fighter_names) > 1 else None,
            "fighter_2_url": fighter_urls[1] if len(fighter_urls) > 1 else None,
            "weight_class": weight_class,
            "gender": gender,
            "method": method,
            "round": round_num,
            "time": fight_time,
            "fight_url": link,
            "bout_result": bout_result,
        })
    return fights


def get_fighter_dob(fighter_url: str) -> "datetime | None":
    """Fetch (and cache) a fighter's date of birth from their fighter-details page."""
    if not fighter_url:
        return None
    if fighter_url in _DOB_CACHE:
        return _DOB_CACHE[fighter_url]

    soup = get_soup(fighter_url)
    dob = None
    for li in soup.select("li.b-list__box-list-item"):
        label = li.select_one("i.b-list__box-item-title")
        if label and "dob" in label.get_text(strip=True).lower():
            text = li.get_text(" ", strip=True)
            text = re.sub(r"(?i)^dob:\s*", "", text).strip()
            if text and text != "--":
                try:
                    dob = datetime.strptime(text, "%b %d, %Y")
                except ValueError:
                    dob = None
            break

    _DOB_CACHE[fighter_url] = dob
    time.sleep(0.5)  # be polite — this hits a new page per unique fighter
    return dob


def parse_totals_table(table) -> pd.DataFrame:
    """Parse a 'totals' style table (fight totals) into a 2-row frame, one row per fighter."""
    body_row = table.select_one("tbody tr")
    if body_row is None:
        return pd.DataFrame(columns=TOTALS_COLUMNS)

    cols = body_row.select("td")
    field_order = ["fighter", "kd", "sig_str", "sig_str_pct", "total_str",
                   "td", "td_pct", "sub_att", "rev", "ctrl"]

    data = {"__row0": {}, "__row1": {}}
    for field, col in zip(field_order, cols):
        values = [p.get_text(strip=True) for p in col.select("p")]
        while len(values) < 2:
            values.append(None)
        for i in range(2):
            data[f"__row{i}"][field] = values[i]

    return pd.DataFrame([data["__row0"], data["__row1"]], columns=TOTALS_COLUMNS)


_ROUND_LABEL = re.compile(r"\bRound\s+\d+\b", re.IGNORECASE)


def _is_per_round_table(table) -> bool:
    """Per-round tables have the `js-fight-table` class and 'Round 1', 'Round 2', ...
    header rows inside them. The fight TOTALS table has neither."""
    classes = table.get("class") or []
    if "js-fight-table" in classes:
        return True
    return bool(_ROUND_LABEL.search(table.get_text(" ", strip=True)))


def find_totals_table(soup: BeautifulSoup):
    """Return the fight-TOTALS table (the one with KD / Sig. str. / Td / Ctrl
    covering the whole fight), or None.

    BUG FIX (Sep 2026): the old code searched only `table.b-fight-details__table`.
    On UFCStats the totals table does NOT have that class, but the per-round
    breakdown table does, so the old code silently returned ROUND 1 stats for
    every fight that went past round 1 (confirmed against Kaggle round.csv:
    462 of 462 comparable rows matched round 1 exactly)."""
    for t in soup.find_all("table"):
        headers = [th.get_text(strip=True) for th in t.select("thead th")]
        if "KD" not in headers or "Ctrl" not in headers:
            continue  # the second totals table (Head/Body/Leg) has no KD/Ctrl columns
        if _is_per_round_table(t):
            continue
        return t
    return None


def parse_fight(fight_url: str) -> pd.DataFrame:
    """Return a 2-row DataFrame (one row per fighter) of fight totals for this fight."""
    soup = get_soup(fight_url)

    totals_table = find_totals_table(soup)
    if totals_table is None:
        # Raise instead of falling back to some other table: a loud failure gets
        # logged and retried; a silent fallback gets saved as wrong data.
        raise RuntimeError(f"No fight-totals table found on {fight_url}")

    df = parse_totals_table(totals_table)
    df["fight_url"] = fight_url
    return df


def compute_age(dob: "datetime | None", event_date: "datetime | None") -> "float | None":
    if dob is None or event_date is None:
        return None
    days = (event_date - dob).days
    return round(days / 365.25, 2)


def build_card_dataset(event_url: str, pause: float = 1.0, strict: bool = False) -> pd.DataFrame:
    """strict=True: raise if ANY fight on the card fails, so a resumable driver
    (run_historical.py) never checkpoints a card with fights missing."""
    fights = get_fights_on_card(event_url)
    if not fights:
        raise RuntimeError(
            f"No fights found on {event_url} — UFCStats likely changed a "
            "class name, or this event page has a different layout."
        )

    all_rows = []
    for fight in fights:
        try:
            totals = parse_fight(fight["fight_url"])
        except Exception as exc:  # noqa: BLE001 - keep going on one bad fight
            if strict:
                raise
            print(f"  ! failed to parse {fight['fight_url']}: {exc}")
            continue

        # IMPORTANT: the totals table's row order does NOT reliably match the
        # fighter_1/fighter_2 order from the event page (UFCStats sometimes
        # lists the winner first in the stats table regardless of how the
        # event page ordered them). So match DOB to a stats row by fighter
        # NAME, not by row position — matching by position silently pairs
        # the wrong fighter's age with the wrong fighter's stats.
        name_to_url = {
            (fight["fighter_1"] or "").strip().lower(): fight["fighter_1_url"],
            (fight["fighter_2"] or "").strip().lower(): fight["fighter_2_url"],
        }

        for i, (_, stat_row) in enumerate(totals.iterrows()):
            merged = {**fight, **stat_row.to_dict()}

            stat_name = (stat_row.get("fighter") or "").strip().lower()
            fighter_url = name_to_url.get(stat_name)
            if fighter_url is None:
                # Name didn't match exactly (e.g. nickname/formatting
                # difference) — fall back to position and flag it so it's
                # visible rather than silently wrong.
                fallback_urls = [fight["fighter_1_url"], fight["fighter_2_url"]]
                fighter_url = fallback_urls[i] if i < len(fallback_urls) else None
                print(f"    ! name match failed for '{stat_row.get('fighter')}' "
                      f"in {fight['fighter_1']} vs {fight['fighter_2']} — used position fallback")

            dob = get_fighter_dob(fighter_url)
            merged["dob"] = dob.strftime("%Y-%m-%d") if dob else None
            merged["age_at_fight"] = compute_age(dob, fight["event_date"])
            all_rows.append(add_outcome(merged))

        print(f"  parsed: {fight['fighter_1']} vs {fight['fighter_2']}")
        time.sleep(pause)  # be polite to the server

    df = pd.DataFrame(all_rows)
    if "event_date" in df.columns:
        df["event_date"] = df["event_date"].apply(lambda d: d.strftime("%Y-%m-%d") if pd.notnull(d) else None)
    return df


def build_multi_card_dataset(event_urls: list[str], pause: float = 1.0) -> pd.DataFrame:
    all_dfs = []
    for i, url in enumerate(event_urls, 1):
        print(f"\n[{i}/{len(event_urls)}] Fetching {url} ...")
        try:
            df = build_card_dataset(url, pause=pause)
        except Exception as exc:  # noqa: BLE001 - one bad card shouldn't kill the whole run
            print(f"  ! failed to process event {url}: {exc}")
            continue
        all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("No cards were successfully scraped.")

    return pd.concat(all_dfs, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", action="append", default=[],
                         help="UFCStats event-details URL (repeatable: --event url1 --event url2)")
    parser.add_argument("--event-file", default=None,
                         help="Path to a text file with one event-details URL per line")
    parser.add_argument("--out", default="ufc_data.csv", help="Output CSV path")
    args = parser.parse_args()

    event_urls = list(args.event)
    if args.event_file:
        with open(args.event_file) as f:
            event_urls += [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not event_urls:
        parser.error("Provide at least one --event URL or an --event-file")

    df = build_multi_card_dataset(event_urls)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} rows ({len(df) // 2} fights) across {len(event_urls)} card(s) to {args.out}")
    print(df.head(10))