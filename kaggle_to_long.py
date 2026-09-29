"""
kaggle_to_long.py — reshape the Kaggle UFC master.csv (one row per bout, red/blue
columns) into ONE ROW PER FIGHTER PER BOUT, matching the shape of your scraper,
with age-at-fight, gender, outcome, fight length and per-minute rates.

Usage (run from ~/ufc_scraper):
  python3 kaggle_to_long.py Whole_Dataset/master.csv

Output:
  kaggle_long_2010plus.csv
  kaggle_long_report.txt   (summary + sanity checks; paste this back to Claude)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

START_DATE = "2010-01-01"

SHARED = ["fight_id", "event_id", "event_name", "event_date", "event_location",
          "weight_class", "title_fight", "winner_id", "result_status", "method",
          "finish_round", "finish_time", "time_format", "referee", "details",
          "bonuses", "rounds_fought"]

# Fighter profile fields (fixed per fighter, safe to use)
PROFILE = ["fighter_id", "fighter_name", "height", "reach_inches", "stance", "dob"]

# CAREER AVERAGES copied from each fighter's CURRENT profile page.
# They are identical across every fight a fighter has, so they are not
# at-the-time performance and must NOT be used for aging curves. Dropped.
CAREER = ["slpm", "str_acc", "sapm", "str_def", "td_avg", "td_acc", "td_def", "sub_avg"]


def one_side(df: pd.DataFrame, me: str, opp: str) -> pd.DataFrame:
    out = df[SHARED].copy()
    out["corner"] = "red" if me == "r" else "blue"
    for c in PROFILE:
        out[c] = df[f"{me}_{c}"]
    out["opponent_id"] = df[f"{opp}_fighter_id"]
    out["opponent_name"] = df[f"{opp}_fighter_name"]
    out["opp_dob"] = df[f"{opp}_dob"]

    # Per-fight totals, e.g. r_total_sig_landed -> sig_landed, b_total_... -> opp_sig_landed
    my_prefix, opp_prefix = f"{me}_total_", f"{opp}_total_"
    for c in df.columns:
        if c.startswith(my_prefix):
            out[c[len(my_prefix):]] = df[c]
        elif c.startswith(opp_prefix):
            out["opp_" + c[len(opp_prefix):]] = df[c]
    return out


def classify_promotion(name) -> str:
    """UFC = official UFC cards (incl. Noche UFC, TUF finales).
    DWCS and Road to UFC are UFC-run tryout/qualifier series, kept separate.
    Strikeforce, WEC, DREAM etc. are other promotions."""
    n = str(name).strip().lower()
    if n.startswith(("ufc", "the ultimate fighter", "noche ufc")):
        return "UFC"
    if n.startswith(("dwcs", "dana white")):
        return "DWCS"
    if n.startswith("road to ufc"):
        return "Road to UFC"
    if n.startswith("strikeforce"):
        return "Strikeforce"
    if n.startswith("wec"):
        return "WEC"
    return "Other"


def mmss_to_seconds(s: pd.Series) -> pd.Series:
    parts = s.astype(str).str.split(":", expand=True)
    return pd.to_numeric(parts[0], errors="coerce") * 60 + pd.to_numeric(parts[1], errors="coerce")


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return np.where(den > 0, num / den.where(den > 0), np.nan)


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "Whole_Dataset/master.csv")
    df = pd.read_csv(src, low_memory=False)
    report = []

    dup = df["fight_id"].duplicated().sum()
    report.append(f"Loaded {len(df):,} bouts from {src} ({dup} duplicate fight_ids)")

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df[df["event_date"] >= START_DATE].copy()
    report.append(f"Bouts on/after {START_DATE}: {len(df):,}")

    # ---- flag events that don't look like UFC events -----------------------
    df["promotion"] = df["event_name"].map(classify_promotion)
    df["is_ufc_event"] = df["promotion"] == "UFC"
    non_ufc = df.loc[~df["is_ufc_event"], "event_name"].value_counts()
    report.append("\nBouts per promotion (2010+):\n" + df["promotion"].value_counts().to_string())
    report.append(f"\nEvents per year (all):\n"
                  f"{df.groupby(df.event_date.dt.year)['event_id'].nunique().to_string()}")
    report.append(f"\nBouts at events NOT starting with 'UFC'/'The Ultimate Fighter': "
                  f"{(~df['is_ufc_event']).sum():,} across {len(non_ufc)} events")
    if len(non_ufc):
        report.append("Top 25 of those event names (check these by eye):\n" + non_ufc.head(25).to_string())

    # ---- reshape to one row per fighter per bout ---------------------------
    long = pd.concat([one_side(df, "r", "b"), one_side(df, "b", "r")], ignore_index=True)
    long["promotion"] = long["fight_id"].map(df.set_index("fight_id")["promotion"])
    long["is_ufc_event"] = long["promotion"] == "UFC"

    # ---- derived columns ---------------------------------------------------
    long["dob"] = pd.to_datetime(long["dob"], errors="coerce")
    long["opp_dob"] = pd.to_datetime(long["opp_dob"], errors="coerce")
    long["age_at_fight"] = (long["event_date"] - long["dob"]).dt.days / 365.25
    long["opp_age_at_fight"] = (long["event_date"] - long["opp_dob"]).dt.days / 365.25
    long["age_diff"] = long["age_at_fight"] - long["opp_age_at_fight"]

    long["gender"] = np.where(long["weight_class"].str.contains("Women", case=False, na=False), "F", "M")

    long["outcome"] = np.select(
        [long["winner_id"] == long["fighter_id"], long["winner_id"] == long["opponent_id"]],
        ["W", "L"],
        default=long["result_status"].astype(str).str.upper(),
    )
    long["win"] = (long["outcome"] == "W").astype(int)
    long["finish"] = long["method"].str.contains("KO|TKO|Submission", case=False, na=False).astype(int)

    # Fight length (UFC rounds are 5 minutes)
    long["fight_seconds"] = (long["finish_round"] - 1) * 300 + mmss_to_seconds(long["finish_time"])
    minutes = long["fight_seconds"] / 60

    long["sig_acc"] = safe_ratio(long["sig_landed"], long["sig_atmp"])
    long["td_acc"] = safe_ratio(long["td_success"], long["td_atmp"])
    long["sig_landed_per_min"] = safe_ratio(long["sig_landed"], minutes)
    long["sig_absorbed_per_min"] = safe_ratio(long["opp_sig_landed"], minutes)
    long["td_per_15min"] = safe_ratio(long["td_success"] * 15, minutes)
    long["ctrl_share"] = safe_ratio(long["ctrl_seconds"], long["fight_seconds"])

    long = long.sort_values(["event_date", "fight_id", "corner"]).reset_index(drop=True)
    out = Path("kaggle_long_2010plus.csv")
    long.to_csv(out, index=False)

    # ---- sanity checks -----------------------------------------------------
    report.append(f"\nLong format: {len(long):,} rows (should be 2 x bouts = {2 * len(df):,})")
    report.append(f"Unique fighters: {long['fighter_id'].nunique():,}")
    report.append(f"Rows missing DOB/age: {long['age_at_fight'].isna().sum():,} "
                  f"({long['age_at_fight'].isna().mean() * 100:.1f}%)")
    odd = long[(long["age_at_fight"] < 18) | (long["age_at_fight"] > 50)]
    report.append(f"Rows with age < 18 or > 50 (likely bad DOB): {len(odd)}")
    if len(odd):
        report.append(odd[["event_date", "fighter_name", "dob", "age_at_fight"]].head(20).to_string())
    report.append(f"\nAge summary:\n{long['age_at_fight'].describe().round(2).to_string()}")
    report.append(f"\nGender counts (rows):\n{long['gender'].value_counts().to_string()}")
    report.append(f"\nOutcome counts:\n{long['outcome'].value_counts().to_string()}")
    report.append(f"\nMethod counts:\n{long['method'].value_counts().to_string()}")
    report.append(f"\nFight length: {long['fight_seconds'].isna().sum()} unparseable; "
                  f"max {long['fight_seconds'].max() / 60:.1f} min")
    report.append(f"Rows with sig_atmp == 0 (possible missing stats): {(long['sig_atmp'] == 0).sum():,}")
    both0 = (long["ctrl_seconds"] == 0) & (long["opp_ctrl_seconds"] == 0)
    report.append(f"Rows with ctrl_seconds == 0: {(long['ctrl_seconds'] == 0).mean() * 100:.1f}%; "
                  f"BOTH fighters 0 control: {both0.mean() * 100:.1f}% "
                  f"(of those, {(both0 & (long['fight_seconds'] > 600)).sum() // 2} bouts lasted > 10 min: suspicious)")
    report.append(f"\nRows per year:\n{long.groupby(long.event_date.dt.year).size().to_string()}")
    report.append(f"\nColumns ({long.shape[1]}): {', '.join(long.columns)}")

    text = "\n".join(report)
    Path("kaggle_long_report.txt").write_text(text)
    print(text)
    print(f"\nSaved {out} and kaggle_long_report.txt")


if __name__ == "__main__":
    main()