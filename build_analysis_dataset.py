"""
build_analysis_dataset.py — turn the raw scrape into the analysis-ready dataset.

Input : data/ufc_2010_present_with_results.csv   (one row per fighter per bout)
Output: data/ufc_analysis.csv                     (same rows, cleaned + derived columns)
        data/build_report.txt                     (sanity checks; paste back to Claude)

Optional: if Whole_Dataset/master.csv (Kaggle) is present, it is used ONLY to count
each fighter's UFC fights BEFORE 2010, so career-stage variables are correct for
fighters who debuted before the data window. All stats still come from your scrape.

Usage (run from ~/ufc_scraper):
  python3 build_analysis_dataset.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

IN_CSV = Path("data/ufc_2010_present_with_results.csv")
OUT_CSV = Path("data/ufc_analysis.csv")
REPORT = Path("data/build_report.txt")
KAGGLE_MASTER = Path("Whole_Dataset/master.csv")
WINDOW_START = pd.Timestamp("2010-01-01")

BASE_CLASSES = [  # order matters: "Light Heavyweight" before "Heavyweight"
    "Strawweight", "Flyweight", "Bantamweight", "Featherweight", "Lightweight",
    "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight",
]
CLASS_ORDER = {c: i for i, c in enumerate(BASE_CLASSES)}


# ------------------------------------------------------------------ parsers
def url_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.rstrip("/").str.split("/").str[-1]


def split_of(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'27 of 45' -> (27, 45)"""
    p = s.astype(str).str.extract(r"(\d+)\s*of\s*(\d+)")
    return pd.to_numeric(p[0], errors="coerce"), pd.to_numeric(p[1], errors="coerce")


def mmss(s: pd.Series) -> pd.Series:
    """'4:12' -> 252 ; '--' or blank -> NaN"""
    p = s.astype(str).str.extract(r"^\s*(\d+):(\d{2})\s*$")
    return pd.to_numeric(p[0], errors="coerce") * 60 + pd.to_numeric(p[1], errors="coerce")


def ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.where(den > 0)).astype(float)


def base_weight_class(wc: str) -> str:
    """'Ultimate Fighter 18 Women's Bantamweight Tournament' -> "Women's Bantamweight"
    'UFC Lightweight Title' -> 'Lightweight' ; 'Catch Weight' -> 'Catch Weight'."""
    t = str(wc)
    women = "women" in t.lower()
    for c in sorted(BASE_CLASSES, key=len, reverse=True):
        if re.search(rf"\b{c}\b", t, re.IGNORECASE):
            return f"Women's {c}" if women else c
    if re.search(r"catch", t, re.IGNORECASE):
        return "Catch Weight"
    if re.search(r"open", t, re.IGNORECASE):
        return "Open Weight"
    return "Unknown"


def method_group(m: str) -> str:
    t = str(m).strip().upper()
    if t.startswith(("U-DEC", "S-DEC", "M-DEC")) or "DECISION" in t:
        return "Decision"
    if t.startswith("KO/TKO") or t.startswith("TKO"):
        return "KO/TKO"
    if t.startswith("SUB"):
        return "Submission"
    if t.startswith("DQ"):
        return "DQ"
    if t.startswith("OVERTURNED"):
        return "Overturned"
    if t.startswith("CNC") or "COULD NOT CONTINUE" in t:
        return "Could Not Continue"
    return "Other"


def decision_type(m: str) -> str | float:
    t = str(m).strip().upper()
    return {"U-DEC": "Unanimous", "S-DEC": "Split", "M-DEC": "Majority"}.get(t[:5], np.nan)


# ------------------------------------------------------------ pre-2010 counts
def prior_ufc_fights_before_window(fighter_ids: pd.Series) -> pd.Series | None:
    """Count each fighter's official-UFC fights dated before 2010 (from Kaggle master.csv)."""
    if not KAGGLE_MASTER.exists():
        return None
    k = pd.read_csv(KAGGLE_MASTER, low_memory=False,
                    usecols=["event_name", "event_date", "r_fighter_id", "b_fighter_id"])
    k["event_date"] = pd.to_datetime(k["event_date"], errors="coerce")
    n = k["event_name"].astype(str).str.strip().str.lower()
    ufc = n.str.startswith(("ufc", "the ultimate fighter", "noche ufc"))
    k = k[ufc & (k["event_date"] < WINDOW_START)]
    counts = pd.concat([k["r_fighter_id"], k["b_fighter_id"]]).value_counts()
    return fighter_ids.map(counts).fillna(0).astype(int)


# --------------------------------------------------------------------- main
def main() -> None:
    rep: list[str] = []
    df = pd.read_csv(IN_CSV, low_memory=False)
    rep.append(f"Loaded {len(df):,} rows from {IN_CSV}")

    # ---- identifiers ------------------------------------------------------
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["fight_id"] = url_id(df["fight_url"])
    is_f1 = df["fighter"].astype(str).str.strip() == df["fighter_1"].astype(str).str.strip()
    df["fighter_id"] = np.where(is_f1, url_id(df["fighter_1_url"]), url_id(df["fighter_2_url"]))
    df["opponent_id"] = np.where(is_f1, url_id(df["fighter_2_url"]), url_id(df["fighter_1_url"]))
    df["opponent"] = np.where(is_f1, df["fighter_2"], df["fighter_1"])
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")

    # ---- parse raw stat strings ------------------------------------------
    df["sig_landed"], df["sig_atmp"] = split_of(df["sig_str"])
    df["total_landed"], df["total_atmp"] = split_of(df["total_str"])
    df["td_landed"], df["td_atmp"] = split_of(df["td"])
    df["ctrl_seconds"] = mmss(df["ctrl"])
    for c in ["kd", "sub_att", "rev", "age_at_fight"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- fight length ------------------------------------------------------
    df["finish_round"] = pd.to_numeric(df["round"], errors="coerce")
    df["fight_seconds"] = (df["finish_round"] - 1) * 300 + mmss(df["time"])
    df["fight_minutes"] = df["fight_seconds"] / 60

    # ---- categories --------------------------------------------------------
    df["weight_class_raw"] = df["weight_class"]
    df["weight_class"] = df["weight_class_raw"].map(base_weight_class)
    df["gender"] = np.where(df["weight_class"].str.startswith("Women's"), "F", "M")
    df["weight_class_order"] = df["weight_class"].str.replace("Women's ", "", regex=False).map(CLASS_ORDER)
    df["title_or_tournament"] = df["weight_class_raw"].astype(str).str.contains(
        "Title|Tournament", case=False, regex=True)
    df["method_detail"] = df["method"]
    df["method_group"] = df["method"].map(method_group)
    df["decision_type"] = df["method"].map(decision_type)

    # ---- outcomes ----------------------------------------------------------
    df["win"] = df["outcome"].map({"W": 1, "L": 0})            # NaN for draws / NCs
    df["finish_win"] = np.where(df["outcome"] == "W",
                                df["method_group"].isin(["KO/TKO", "Submission"]).astype(int), 0)
    df["finished"] = np.where(df["outcome"] == "L",               # was finished (lost by KO/SUB)
                              df["method_group"].isin(["KO/TKO", "Submission"]).astype(int), 0)
    df["ko_loss"] = ((df["outcome"] == "L") & (df["method_group"] == "KO/TKO")).astype(int)

    # ---- opponent's line (self-join on fight_id) -------------------------
    opp_cols = ["kd", "sig_landed", "sig_atmp", "total_landed", "total_atmp",
                "td_landed", "td_atmp", "sub_att", "ctrl_seconds", "age_at_fight"]
    opp = df[["fight_id", "fighter_id"] + opp_cols].rename(
        columns={"fighter_id": "opponent_id", **{c: f"opp_{c}" for c in opp_cols}})
    df = df.merge(opp, on=["fight_id", "opponent_id"], how="left")
    df["age_diff"] = df["age_at_fight"] - df["opp_age_at_fight"]

    # ---- per-fight performance rates -------------------------------------
    m = df["fight_minutes"].where(df["fight_minutes"] > 0)
    df["sig_acc"] = ratio(df["sig_landed"], df["sig_atmp"])
    df["sig_def"] = 1 - ratio(df["opp_sig_landed"], df["opp_sig_atmp"])
    df["sig_landed_pm"] = df["sig_landed"] / m
    df["sig_absorbed_pm"] = df["opp_sig_landed"] / m
    df["sig_diff_pm"] = df["sig_landed_pm"] - df["sig_absorbed_pm"]
    df["td_acc"] = ratio(df["td_landed"], df["td_atmp"])
    df["td_def"] = 1 - ratio(df["opp_td_landed"], df["opp_td_atmp"])
    df["td_landed_p15"] = df["td_landed"] / m * 15
    df["kd_p15"] = df["kd"] / m * 15
    df["sub_att_p15"] = df["sub_att"] / m * 15
    df["ctrl_share"] = ratio(df["ctrl_seconds"], df["fight_seconds"])

    # ---- career stage (sorted by date within fighter) ---------------------
    df = df.sort_values(["fighter_id", "event_date", "fight_id"]).reset_index(drop=True)
    prior = prior_ufc_fights_before_window(df["fighter_id"])
    df["ufc_fights_before_2010"] = prior if prior is not None else np.nan
    df["fight_num_in_data"] = df.groupby("fighter_id").cumcount() + 1
    df["ufc_fight_number"] = df["fight_num_in_data"] + df["ufc_fights_before_2010"].fillna(0).astype(int)
    df["debuted_in_window"] = df["ufc_fights_before_2010"].eq(0) if prior is not None else np.nan
    df["days_since_last_fight"] = df.groupby("fighter_id")["event_date"].diff().dt.days
    df["prior_wins_in_data"] = df.groupby("fighter_id")["win"].transform(lambda s: s.fillna(0).shift().cumsum()).fillna(0)
    df["prior_losses_in_data"] = df.groupby("fighter_id")["win"].transform(
        lambda s: (s == 0).astype(int).shift().cumsum()).fillna(0)
    df["prior_ko_losses_in_data"] = df.groupby("fighter_id")["ko_loss"].transform(lambda s: s.shift().cumsum()).fillna(0)
    df["age_at_debut_in_data"] = df.groupby("fighter_id")["age_at_fight"].transform("first")
    df["n_fights_in_data"] = df.groupby("fighter_id")["fight_id"].transform("size")
    df["is_last_fight_in_data"] = df["fight_num_in_data"] == df["n_fights_in_data"]

    # ---- data-quality flags -----------------------------------------------
    df["flag_no_stats"] = (df["sig_atmp"].fillna(0) == 0) & (df["opp_sig_atmp"].fillna(0) == 0) & \
                          (df["total_atmp"].fillna(0) == 0)
    df["flag_ctrl_missing"] = df["ctrl_seconds"].isna()
    df["flag_missing_age"] = df["age_at_fight"].isna()
    df["flag_nc_or_draw"] = df["outcome"].isin(["NC", "D"])
    df["flag_short_fight"] = df["fight_seconds"] < 60   # rates are unstable under 1 min
    df["include_main"] = ~(df["flag_no_stats"] | df["flag_missing_age"] | df["flag_nc_or_draw"])

    # ---- column order ------------------------------------------------------
    front = ["fight_id", "event", "event_date", "fighter_id", "fighter", "opponent_id", "opponent",
             "gender", "weight_class", "weight_class_order", "weight_class_raw", "title_or_tournament",
             "dob", "age_at_fight", "opp_age_at_fight", "age_diff",
             "outcome", "win", "finish_win", "finished", "ko_loss",
             "method_group", "method_detail", "decision_type", "finish_round", "fight_seconds", "fight_minutes",
             "kd", "sig_landed", "sig_atmp", "total_landed", "total_atmp", "td_landed", "td_atmp",
             "sub_att", "rev", "ctrl_seconds",
             "sig_acc", "sig_def", "sig_landed_pm", "sig_absorbed_pm", "sig_diff_pm",
             "td_acc", "td_def", "td_landed_p15", "kd_p15", "sub_att_p15", "ctrl_share",
             "ufc_fights_before_2010", "debuted_in_window", "fight_num_in_data", "ufc_fight_number",
             "n_fights_in_data", "is_last_fight_in_data", "days_since_last_fight",
             "prior_wins_in_data", "prior_losses_in_data", "prior_ko_losses_in_data", "age_at_debut_in_data"]
    front = [c for c in front if c in df.columns]
    rest = [c for c in df.columns if c.startswith(("opp_", "flag_", "include_"))]
    df = df[front + [c for c in rest if c not in front]].sort_values(
        ["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)

    # ---- report -------------------------------------------------------------
    rep.append(f"Saved {OUT_CSV}: {len(df):,} rows, {df.shape[1]} columns")
    rep.append(f"Bouts: {df['fight_id'].nunique():,} | fighters: {df['fighter_id'].nunique():,} | "
               f"events: {df['event'].nunique()}")
    rep.append(f"Every fighter-row has an opponent row: {df['opp_sig_atmp'].notna().all()}")
    rep.append(f"\nRows in main analysis sample (include_main): {df['include_main'].sum():,}")
    for f in [c for c in df.columns if c.startswith("flag_")]:
        rep.append(f"  {f:<20} {df[f].sum():>6,}")
    rep.append("\nWeight classes (base) x gender:")
    rep.append(pd.crosstab(df["weight_class"], df["gender"]).to_string())
    unk = df.loc[df["weight_class"] == "Unknown", "weight_class_raw"].value_counts()
    rep.append(f"\nUnmapped weight classes: {len(unk)}" + (("\n" + unk.to_string()) if len(unk) else ""))
    rep.append("\nMethod groups:\n" + df["method_group"].value_counts().to_string())
    rep.append("\nOutcome:\n" + df["outcome"].value_counts().to_string())
    rep.append("\nAge at fight:\n" + df["age_at_fight"].describe().round(2).to_string())
    rep.append("\nRows by age bracket (main sample):")
    br = pd.cut(df.loc[df.include_main, "age_at_fight"], [17, 23, 26, 29, 32, 35, 38, 41, 50])
    rep.append(br.value_counts().sort_index().to_string())
    rep.append("\nFights per fighter in data:\n" +
               df.drop_duplicates("fighter_id")["n_fights_in_data"].describe().round(2).to_string())
    if prior is not None:
        rep.append(f"\nFighters who debuted before 2010 (left-censored careers): "
                   f"{(~df.drop_duplicates('fighter_id')['debuted_in_window'].astype(bool)).sum():,}")
    else:
        rep.append("\nWhole_Dataset/master.csv not found: pre-2010 fight counts not added.")
    rep.append("\nSanity: rates")
    rep.append(df[["sig_acc", "sig_def", "sig_landed_pm", "td_acc", "ctrl_share"]].describe().round(3).to_string())
    bad = df[(df["ctrl_share"] > 1.01) | (df["sig_acc"] > 1) | (df["fight_seconds"] > 25 * 60)]
    rep.append(f"Impossible values (ctrl_share > 1, accuracy > 1, fight > 25 min): {len(bad)}")
    if len(bad):
        rep.append(bad[["event_date", "fighter", "fight_seconds", "ctrl_seconds", "sig_landed", "sig_atmp"]]
                   .head(15).to_string(index=False))

    text = "\n".join(rep)
    REPORT.write_text(text)
    print(text)
    print(f"\nSaved {REPORT}")


if __name__ == "__main__":
    main()