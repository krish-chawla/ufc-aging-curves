"""
add_results.py — add fight OUTCOMES (W / L / D / NC) to data/ufc_2010_present.csv
without re-running the full scrape.

It fetches only each event page (649 requests, not ~8,000), reads the W/L
column, and joins it onto your existing rows by fight_url. It then checks the
result against Kaggle's winner_id for every bout the two share.

Resumable: event results are cached in data/event_results.json, so if it stops,
run it again and it skips events already fetched.

Usage (run from ~/ufc_scraper):
  caffeinate -ims python3 add_results.py
Output:
  data/ufc_2010_present_with_results.csv
"""

import json
import os
import time
from pathlib import Path

import pandas as pd

import scrape_card

DATA = Path("data")
IN_CSV = DATA / "ufc_2010_present.csv"
OUT_CSV = DATA / "ufc_2010_present_with_results.csv"
URLS = DATA / "event_urls.json"
CACHE = DATA / "event_results.json"
KAGGLE = Path("kaggle_long_2010plus.csv")


def load_cache() -> dict:
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def save_cache(cache: dict) -> None:
    tmp = CACHE.with_name(CACHE.name + ".tmp")
    tmp.write_text(json.dumps(cache))
    os.replace(tmp, CACHE)


def fetch_results(event_urls: list[str]) -> dict:
    """cache: {event_url: {fight_url: bout_result}}"""
    cache = load_cache()
    todo = [u for u in event_urls if u not in cache]
    print(f"{len(event_urls)} events | {len(event_urls) - len(todo)} cached | {len(todo)} to fetch")
    for i, url in enumerate(todo, 1):
        for attempt in range(1, 5):
            try:
                fights = scrape_card.get_fights_on_card(url)
                break
            except Exception as e:  # noqa: BLE001
                print(f"  attempt {attempt} failed for {url}: {e}")
                scrape_card.SESSION.cookies.clear()
                time.sleep(5 * attempt)
        else:
            print(f"  ! giving up on {url} for now; rerun to retry")
            continue
        cache[url] = {f["fight_url"]: f["bout_result"] for f in fights}
        save_cache(cache)
        if i % 25 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] fetched")
        time.sleep(1.0)
    return cache


def main() -> None:
    df = pd.read_csv(IN_CSV, low_memory=False)
    urls = json.loads(URLS.read_text())
    cache = fetch_results(urls)

    result_by_fight = {f: r for ev in cache.values() for f, r in ev.items()}
    df["bout_result"] = df["fight_url"].map(result_by_fight)
    df = df.apply(scrape_card.add_outcome, axis=1)

    print("\nOutcome counts:")
    print(df["outcome"].value_counts(dropna=False).to_string())
    missing = df["outcome"].isna().sum()
    print(f"Rows with no outcome: {missing}  (should be 0; rerun if not)")

    # Each decided bout should have exactly one W and one L
    per_bout = df.groupby("fight_url")["outcome"].apply(lambda s: "".join(sorted(s.fillna("?"))))
    bad = per_bout[~per_bout.isin(["LW", "DD", "NCNC"])]
    print(f"Bouts without a clean W/L, D/D or NC/NC pair: {len(bad)}")
    if len(bad):
        print(bad.head(20).to_string())

    # Cross-check against Kaggle winner_id
    if KAGGLE.exists():
        k = pd.read_csv(KAGGLE, low_memory=False, usecols=["fight_id", "fighter_id", "outcome"])
        d = df.copy()
        d["fight_id"] = d["fight_url"].str.rstrip("/").str.split("/").str[-1]
        is_f1 = d["fighter"].str.strip() == d["fighter_1"].str.strip()
        d["fighter_id"] = d["fighter_1_url"].where(is_f1, d["fighter_2_url"]).str.rstrip("/").str.split("/").str[-1]
        k["outcome_k"] = k["outcome"].replace({"DRAW": "D", "NO_CONTEST": "NC"})
        m = d.merge(k[["fight_id", "fighter_id", "outcome_k"]], on=["fight_id", "fighter_id"])
        diff = m[m["outcome"] != m["outcome_k"]]
        print(f"\nKaggle cross-check: {len(m):,} rows compared, {len(diff)} disagree")
        if len(diff):
            print(diff[["event_date", "event", "fighter", "outcome", "outcome_k"]].head(20).to_string(index=False))

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV} ({len(df):,} rows)")


if __name__ == "__main__":
    main()