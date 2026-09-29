"""
run_historical.py — resumable bulk scrape of UFCStats events (2010–present).

How checkpointing works:
  * Each event is saved to its own file: data/raw/events/<event_id>.csv
  * A file is written atomically (temp file + rename), so a crash mid-write
    never leaves a half-written checkpoint.
  * "Done" = that event's CSV exists. Rerunning the script skips done events
    and retries everything else (including earlier failures) automatically.
  * The event URL list is cached in data/event_urls.json so a resume doesn't
    re-crawl the events index.
  * Failures are logged to data/failed_events.jsonl for inspection.

Usage:
  python run_historical.py --limit 3          # small test run
  caffeinate -i python run_historical.py      # full run (Mac: prevents sleep)
  python run_historical.py --merge-only       # combine checkpoints into one CSV
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# ADAPT THESE IMPORTS to match your existing scripts.
#   get_event_urls(start_year, session) -> list[str]         (event-details URLs)
#   scrape_card(event_url, session)     -> DataFrame or list[dict] (one row per fighter per bout)
#   make_session()                      -> requests.Session with the PoW check solved
# ---------------------------------------------------------------------------
from get_event_urls import get_event_urls
from scrape_card import scrape_card, make_session

DATA_DIR = Path("data")
EVENTS_DIR = DATA_DIR / "raw" / "events"
URLS_FILE = DATA_DIR / "event_urls.json"
FAILED_FILE = DATA_DIR / "failed_events.jsonl"
LOG_FILE = DATA_DIR / "scrape.log"
FINAL_CSV = DATA_DIR / "ufc_2010_present.csv"


# ----------------------------- helpers -------------------------------------
def event_id(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def out_path(url: str) -> Path:
    return EVENTS_DIR / f"{event_id(url)}.csv"


def is_done(url: str) -> bool:
    return out_path(url).exists()


def atomic_write_text(text: str, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def record_failure(url: str, err: Exception) -> None:
    with FAILED_FILE.open("a") as f:
        f.write(json.dumps({
            "time": datetime.now().isoformat(timespec="seconds"),
            "event_url": url,
            "error": f"{type(err).__name__}: {err}",
        }) + "\n")


def setup_logging() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
    )


# ----------------------------- core ----------------------------------------
def load_or_fetch_urls(start_year: int, session, refresh: bool) -> list[str]:
    if URLS_FILE.exists() and not refresh:
        urls = json.loads(URLS_FILE.read_text())
        logging.info(f"Loaded {len(urls)} cached event URLs from {URLS_FILE}")
        return urls
    logging.info(f"Fetching event URLs from {start_year} to present...")
    urls = list(dict.fromkeys(get_event_urls(start_year, session)))  # dedupe, keep order
    atomic_write_text(json.dumps(urls, indent=1), URLS_FILE)
    logging.info(f"Cached {len(urls)} event URLs to {URLS_FILE}")
    return urls


def scrape_with_retries(url: str, holder: dict, max_retries: int, base_wait: float) -> pd.DataFrame:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            result = scrape_card(url, holder["session"])
            return result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = e
            logging.warning(f"  attempt {attempt}/{max_retries} failed: {type(e).__name__}: {e}")
            if attempt >= 2:
                # Repeated failure: often an expired PoW cookie. Start a fresh session.
                logging.info("  refreshing session (re-solving bot check)")
                try:
                    holder["session"] = make_session()
                except Exception as se:
                    logging.warning(f"  session refresh failed: {se}")
            if attempt < max_retries:
                time.sleep(base_wait * 2 ** (attempt - 1) + random.uniform(0, 2))
    raise last_err


def run(args) -> None:
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    holder = {"session": make_session()}

    urls = load_or_fetch_urls(args.start_year, holder["session"], args.refresh_urls)
    todo = [u for u in urls if not is_done(u)]
    if args.limit:
        todo = todo[: args.limit]
    logging.info(f"{len(urls)} events total | {len(urls) - len([u for u in urls if not is_done(u)])} "
                 f"already done | {len(todo)} to scrape this run")

    ok = failed = empty = 0
    t0 = time.time()
    try:
        for i, url in enumerate(todo, 1):
            logging.info(f"[{i}/{len(todo)}] {url}")
            try:
                df = scrape_with_retries(url, holder, args.max_retries, args.backoff)
            except Exception as e:
                failed += 1
                record_failure(url, e)
                logging.error(f"  FAILED after {args.max_retries} attempts; logged, moving on")
                continue

            if df.empty:
                # Not checkpointed, so it gets retried next run (covers upcoming/cancelled events).
                empty += 1
                logging.warning("  no rows returned; not checkpointed")
            else:
                if "event_url" not in df.columns:
                    df.insert(0, "event_url", url)
                atomic_write_csv(df, out_path(url))
                ok += 1
                elapsed = time.time() - t0
                eta_min = elapsed / i * (len(todo) - i) / 60
                logging.info(f"  saved {len(df)} rows | ETA ~{eta_min:.0f} min")

            time.sleep(args.delay + random.uniform(0, args.delay / 2))
    except KeyboardInterrupt:
        logging.warning("Interrupted. All completed events are saved; rerun to resume.")

    logging.info(f"Run summary: {ok} saved, {empty} empty, {failed} failed. "
                 f"Remaining: {sum(not is_done(u) for u in urls)} events.")
    if failed:
        logging.info(f"See {FAILED_FILE} for errors. Rerunning retries them automatically.")


def merge() -> None:
    files = sorted(EVENTS_DIR.glob("*.csv"))
    if not files:
        logging.error("No checkpoint files to merge.")
        return
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    before = len(df)
    df = df.drop_duplicates()
    atomic_write_csv(df, FINAL_CSV)
    logging.info(f"Merged {len(files)} events -> {FINAL_CSV} "
                 f"({len(df)} rows, {before - len(df)} exact duplicates dropped)")

    date_cols = [c for c in df.columns if "date" in c.lower()]
    if date_cols:
        years = pd.to_datetime(df[date_cols[0]], errors="coerce").dt.year
        logging.info("Rows per year:\n" + years.value_counts().sort_index().to_string())
    for col in df.columns:
        if "age" in col.lower() or "dob" in col.lower():
            logging.info(f"Missing values in '{col}': {df[col].isna().sum()}")


def main() -> None:
    p = argparse.ArgumentParser(description="Resumable UFCStats historical scrape")
    p.add_argument("--start-year", type=int, default=2010)
    p.add_argument("--delay", type=float, default=2.0, help="base seconds between events")
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--backoff", type=float, default=5.0, help="base seconds for retry backoff")
    p.add_argument("--limit", type=int, default=0, help="scrape at most N events (testing)")
    p.add_argument("--refresh-urls", action="store_true", help="re-crawl the event list")
    p.add_argument("--merge-only", action="store_true", help="skip scraping; just merge")
    args = p.parse_args()

    setup_logging()
    if not args.merge_only:
        run(args)
    merge()


if __name__ == "__main__":
    main()