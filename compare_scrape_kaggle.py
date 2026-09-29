"""
compare_scrape_kaggle.py — check your scraper against the Kaggle dataset on the
dates both cover (2026-01-24 through 2026-08-08 for your current files).

Matching is by UFCStats IDs, not names or row order:
  fight_id   = last part of your fight_url
  fighter_id = last part of fighter_1_url / fighter_2_url (whichever is `fighter`)

Usage (run from ~/ufc_scraper):
  python3 compare_scrape_kaggle.py ufc_2026.csv kaggle_long_2010plus.csv

Output:
  compare_report.txt     summary (paste this back to Claude)
  compare_mismatches.csv every stat that disagrees, one row per disagreement
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def url_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.rstrip("/").str.split("/").str[-1]


def split_of(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'27 of 45' -> (27, 45)"""
    parts = s.astype(str).str.extract(r"(\d+)\s*of\s*(\d+)")
    return pd.to_numeric(parts[0], errors="coerce"), pd.to_numeric(parts[1], errors="coerce")


def mmss(s: pd.Series) -> pd.Series:
    parts = s.astype(str).str.extract(r"(\d+):(\d+)")
    return pd.to_numeric(parts[0], errors="coerce") * 60 + pd.to_numeric(parts[1], errors="coerce")


def prep_scrape(path: Path) -> pd.DataFrame:
    s = pd.read_csv(path, low_memory=False)
    s["event_date"] = pd.to_datetime(s["event_date"], errors="coerce")
    s["fight_id"] = url_id(s["fight_url"])
    is_f1 = s["fighter"].astype(str).str.strip() == s["fighter_1"].astype(str).str.strip()
    s["fighter_id"] = np.where(is_f1, url_id(s["fighter_1_url"]), url_id(s["fighter_2_url"]))
    s["sig_landed"], s["sig_atmp"] = split_of(s["sig_str"])
    s["total_str_landed"], s["total_str_atmp"] = split_of(s["total_str"])
    s["td_success"], s["td_atmp"] = split_of(s["td"])
    s["ctrl_seconds"] = mmss(s["ctrl"])
    s["finish_round"] = pd.to_numeric(s["round"], errors="coerce")
    s["finish_seconds"] = mmss(s["time"])
    s["dob"] = pd.to_datetime(s["dob"], errors="coerce")
    for c in ["kd", "sub_att", "rev", "age_at_fight"]:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s["name_matched_side"] = is_f1 | (s["fighter"].astype(str).str.strip() == s["fighter_2"].astype(str).str.strip())
    return s


def classify_promotion(name) -> str:
    n = str(name).strip().lower()
    if n.startswith(("ufc", "the ultimate fighter", "noche ufc")):
        return "UFC"
    if n.startswith(("dwcs", "dana white")):
        return "DWCS"
    if n.startswith("road to ufc"):
        return "Road to UFC"
    return "Other"


def prep_kaggle(path: Path) -> pd.DataFrame:
    k = pd.read_csv(path, low_memory=False)
    k["promotion"] = k["event_name"].map(classify_promotion)  # recomputed so older files work too
    k["event_date"] = pd.to_datetime(k["event_date"], errors="coerce")
    k["dob"] = pd.to_datetime(k["dob"], errors="coerce")
    k["finish_seconds"] = mmss(k["finish_time"])
    return k


NUMERIC = ["kd", "sig_landed", "sig_atmp", "total_str_landed", "total_str_atmp",
           "td_success", "td_atmp", "sub_att", "rev", "ctrl_seconds",
           "finish_round", "finish_seconds"]


def main() -> None:
    scrape_path = Path(sys.argv[1] if len(sys.argv) > 1 else "ufc_2026.csv")
    kaggle_path = Path(sys.argv[2] if len(sys.argv) > 2 else "kaggle_long_2010plus.csv")
    s, k = prep_scrape(scrape_path), prep_kaggle(kaggle_path)
    rep = []

    # Only compare the dates both files cover
    lo = max(s["event_date"].min(), k["event_date"].min())
    hi = min(s["event_date"].max(), k["event_date"].max())
    s_o = s[s["event_date"].between(lo, hi)].copy()
    k_o = k[k["event_date"].between(lo, hi)].copy()
    rep.append(f"Kaggle rows in window by promotion: {k_o['promotion'].value_counts().to_dict()} "
               f"(only UFC cards are compared, since your scraper pulls UFC events)")
    k_o = k_o[k_o["promotion"] == "UFC"]
    rep.append(f"Overlap window: {lo.date()} -> {hi.date()}")
    rep.append(f"Scrape rows in window: {len(s_o):,} ({s_o['fight_id'].nunique()} bouts, "
               f"{s_o['event'].nunique()} events)")
    rep.append(f"Kaggle rows in window: {len(k_o):,} ({k_o['fight_id'].nunique()} bouts, "
               f"{k_o['event_id'].nunique()} events)")
    rep.append(f"Scrape rows where `fighter` matched neither fighter_1 nor fighter_2 by name: "
               f"{(~s_o['name_matched_side']).sum()}  (these get a wrong fighter_id; should be 0)")
    dup = s_o.duplicated(["fight_id", "fighter_id"]).sum()
    rep.append(f"Duplicate (fight, fighter) rows in scrape: {dup}")

    # ---- coverage: bouts on one side only --------------------------------
    s_f, k_f = set(s_o["fight_id"]), set(k_o["fight_id"])
    only_s, only_k = s_f - k_f, k_f - s_f
    rep.append(f"\nBouts in BOTH: {len(s_f & k_f)} | only in scrape: {len(only_s)} | only in Kaggle: {len(only_k)}")
    if only_s:
        rep.append("Only in your scrape:\n" + s_o[s_o.fight_id.isin(only_s)]
                   .drop_duplicates("fight_id")[["event_date", "event", "fighter_1", "fighter_2", "method"]]
                   .to_string(index=False))
    if only_k:
        rep.append("Only in Kaggle:\n" + k_o[k_o.fight_id.isin(only_k)]
                   .drop_duplicates("fight_id")[["event_date", "event_name", "promotion", "fighter_name",
                                                  "opponent_name", "method"]]
                   .to_string(index=False))

    # ---- stat-by-stat agreement ------------------------------------------
    m = s_o.merge(k_o, on=["fight_id", "fighter_id"], how="inner", suffixes=("_s", "_k"))
    rep.append(f"\nMatched fighter-bout rows: {len(m):,} of {len(s_o):,} scrape rows")

    name_diff = (m["fighter"].str.strip().str.lower() != m["fighter_name"].str.strip().str.lower()).sum()
    rep.append(f"Same fighter_id but different name spelling: {name_diff}")

    mism = []
    rep.append("\nAgreement by stat (exact match):")
    for c in NUMERIC:
        a, b = m[f"{c}_s"], m[f"{c}_k"]
        both = a.notna() & b.notna()
        bad = both & (a != b)
        rep.append(f"  {c:<18} {both.sum():>5} compared  {bad.sum():>4} differ  "
                   f"{(a.isna() & b.notna()).sum():>3} missing in scrape  "
                   f"{(a.notna() & b.isna()).sum():>3} missing in Kaggle")
        for _, r in m[bad].iterrows():
            mism.append({"event_date": r["event_date_s"].date(), "event": r["event"],
                         "fighter": r["fighter"], "fight_url": r["fight_url"], "stat": c,
                         "scrape": r[f"{c}_s"], "kaggle": r[f"{c}_k"]})

    d = m["dob_s"].notna() & m["dob_k"].notna()
    dob_bad = d & (m["dob_s"] != m["dob_k"])
    rep.append(f"  {'dob':<18} {d.sum():>5} compared  {dob_bad.sum():>4} differ  "
               f"{(m['dob_s'].isna() & m['dob_k'].notna()).sum():>3} missing in scrape  "
               f"{(m['dob_s'].notna() & m['dob_k'].isna()).sum():>3} missing in Kaggle")
    for _, r in m[dob_bad].iterrows():
        mism.append({"event_date": r["event_date_s"].date(), "event": r["event"], "fighter": r["fighter"],
                     "fight_url": r["fight_url"], "stat": "dob",
                     "scrape": r["dob_s"].date(), "kaggle": r["dob_k"].date()})

    age = (m["age_at_fight_s"] - m["age_at_fight_k"]).abs()
    rep.append(f"  age_at_fight       max abs difference {age.max():.3f} years "
               f"(rows > 0.02 yr apart: {(age > 0.02).sum()})")

    date_bad = (m["event_date_s"] != m["event_date_k"]).sum()
    rep.append(f"  event_date         {date_bad} differ")

    rep.append("\nMethod labels, scrape vs Kaggle (how to map one onto the other):")
    rep.append(m.groupby(["method_s", "method_k"]).size().rename("rows").reset_index().to_string(index=False))

    rep.append("\nYour gender label vs Kaggle weight class:")
    rep.append(pd.crosstab(m["weight_class_k"], m["gender_s"]).to_string())

    pd.DataFrame(mism).to_csv("compare_mismatches.csv", index=False)
    text = "\n".join(rep)
    Path("compare_report.txt").write_text(text)
    print(text)
    print(f"\nSaved compare_report.txt and compare_mismatches.csv ({len(mism)} disagreements)")


if __name__ == "__main__":
    main()