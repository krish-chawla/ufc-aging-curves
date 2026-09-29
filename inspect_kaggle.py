"""
inspect_kaggle.py — profile every CSV in a folder (e.g. the Kaggle UFC dataset)
and write a plain-text report you can paste back to Claude.

Usage:
  python inspect_kaggle.py path/to/kaggle_folder
  python inspect_kaggle.py path/to/kaggle_folder --compare ufc_2026.csv

Output: kaggle_profile.txt (also printed to the screen)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

KEYWORDS = ["date", "dob", "birth", "age", "sig", "str", "td", "takedown",
            "ctrl", "control", "kd", "sub", "weight", "gender", "method", "winner", "result"]


def profile_csv(path: Path, lines: list[str]) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        lines.append(f"!! Could not read {path.name}: {e}")
        return None

    lines.append("=" * 80)
    lines.append(f"FILE: {path.name}   rows={len(df):,}   cols={df.shape[1]}")
    lines.append("=" * 80)

    lines.append("Columns (dtype, % missing, example value):")
    for c in df.columns:
        miss = df[c].isna().mean() * 100
        example = df[c].dropna().iloc[0] if df[c].notna().any() else ""
        example = str(example)[:40]
        flag = "  <--" if any(k in c.lower() for k in KEYWORDS) else ""
        lines.append(f"  {c:<35} {str(df[c].dtype):<10} {miss:5.1f}%  {example}{flag}")

    # Date ranges for anything that looks like a date column
    for c in df.columns:
        if any(k in c.lower() for k in ("date", "dob", "birth")):
            d = pd.to_datetime(df[c], errors="coerce")
            if d.notna().any():
                lines.append(f"  date range [{c}]: {d.min().date()} -> {d.max().date()} "
                             f"({d.isna().sum():,} unparseable/missing)")
                if "dob" not in c.lower() and "birth" not in c.lower():
                    counts = d.dt.year.value_counts().sort_index()
                    recent = counts[counts.index >= 2010]
                    lines.append("  rows per year (2010+): " +
                                 ", ".join(f"{int(y)}:{n}" for y, n in recent.items()))

    lines.append("")
    lines.append("First 3 rows:")
    with pd.option_context("display.max_columns", 50, "display.width", 200):
        lines.append(df.head(3).to_string())
    lines.append("")
    return df


def compare_2026(kaggle_dfs: dict, scraped_path: Path, lines: list[str]) -> None:
    lines.append("=" * 80)
    lines.append(f"COMPARISON vs {scraped_path.name}")
    lines.append("=" * 80)
    scraped = pd.read_csv(scraped_path, low_memory=False)
    s_date = next((c for c in scraped.columns if "date" in c.lower()), None)
    lines.append(f"Your scrape: {len(scraped):,} rows; date column = {s_date}")
    if s_date:
        sd = pd.to_datetime(scraped[s_date], errors="coerce")
        lines.append(f"  your date range: {sd.min().date()} -> {sd.max().date()}")

    for name, df in kaggle_dfs.items():
        d_col = next((c for c in df.columns
                      if "date" in c.lower() and "dob" not in c.lower() and "birth" not in c.lower()), None)
        if not d_col:
            continue
        d = pd.to_datetime(df[d_col], errors="coerce")
        n2026 = int((d.dt.year == 2026).sum())
        lines.append(f"  {name}: {n2026:,} rows dated 2026 (latest date {d.max().date()})")
    lines.append("")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("folder", help="folder containing the Kaggle CSVs")
    p.add_argument("--compare", help="your scraped CSV, e.g. ufc_2026.csv")
    args = p.parse_args()

    folder = Path(args.folder)
    files = sorted(folder.rglob("*.csv"))
    if not files:
        sys.exit(f"No CSV files found under {folder.resolve()}")

    lines: list[str] = [f"Found {len(files)} CSV file(s) under {folder.resolve()}", ""]
    dfs = {}
    for f in files:
        df = profile_csv(f, lines)
        if df is not None:
            dfs[f.name] = df

    if args.compare:
        compare_2026(dfs, Path(args.compare), lines)

    report = "\n".join(lines)
    Path("kaggle_profile.txt").write_text(report)
    print(report)
    print("\nSaved report to kaggle_profile.txt")


if __name__ == "__main__":
    main()