"""
build_prefight_features.py — one row per bout with ONLY information known before
the fight, for win/loss prediction (logistic regression vs. random forest).

Input : data/ufc_analysis.csv          (from build_analysis_dataset.py)
Output: data/bouts_prefight.csv         one row per decided bout (draws / NCs dropped)
        data/prefight_report.txt        sanity checks (paste back to Claude)

Why "pre-fight only": strikes landed, takedowns, control time etc. in THIS fight
largely decide who wins, so using them would make prediction trivially easy and
meaningless. Every feature here comes from the fighter's age/division or from
fights that happened BEFORE this one.

Layout: each bout's two fighters are randomly assigned to side A and side B
(fixed seed), and the target is `a_win` (1 if A won). Random sides make ~50% the
no-skill baseline. Features are given for A, for B, and as A - B differences.

Usage (from ~/ufc_scraper):
  python3 build_prefight_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

IN_CSV = Path("data/ufc_analysis.csv")
OUT_CSV = Path("data/bouts_prefight.csv")
REPORT = Path("data/prefight_report.txt")
SEED = 2026

# Per-fight performance stats whose PRIOR career averages become features
PERF = ["sig_landed_pm", "sig_absorbed_pm", "sig_acc", "sig_def",
        "td_landed_p15", "td_acc", "td_def", "ctrl_share", "kd_p15", "sub_att_p15"]


def fighter_history(d: pd.DataFrame) -> pd.DataFrame:
    """Per fighter-bout features computed from that fighter's EARLIER fights only."""
    d = d.sort_values(["fighter_id", "event_date", "fight_id"]).copy()
    g = d.groupby("fighter_id", sort=False)

    d["won"] = (d["outcome"] == "W").astype(float)
    d["lost"] = (d["outcome"] == "L").astype(float)
    d["prior_fights"] = g.cumcount()                                  # fights in data before this one
    d["prior_w"] = g["won"].transform(lambda s: s.shift().cumsum()).fillna(0)
    d["prior_l"] = g["lost"].transform(lambda s: s.shift().cumsum()).fillna(0)
    d["prior_win_pct"] = (d["prior_w"] + 1) / (d["prior_w"] + d["prior_l"] + 2)   # smoothed; debut = 0.5
    d["last3_win_pct"] = g["won"].transform(lambda s: s.shift().rolling(3, min_periods=1).mean())
    d["prior_ko_losses"] = g["ko_loss"].transform(lambda s: s.shift().cumsum()).fillna(0)
    d["prior_finish_wins"] = g["finish_win"].transform(lambda s: s.shift().cumsum()).fillna(0)

    # current streak before this fight: +n = n straight wins, -n = n straight losses
    def streak(res: pd.Series) -> pd.Series:
        out, cur = [], 0
        for r in res:
            out.append(cur)
            cur = (cur + 1 if cur >= 0 else 1) if r == "W" else \
                  (cur - 1 if cur <= 0 else -1) if r == "L" else 0
        return pd.Series(out, index=res.index)
    d["streak"] = g["outcome"].transform(streak)

    d["layoff_days"] = g["event_date"].diff().dt.days
    d["debut_in_data"] = (d["prior_fights"] == 0).astype(int)
    d["ufc_experience"] = d["ufc_fight_number"] - 1                    # UFC fights before this one (incl. pre-2010)

    # Prior career averages of performance stats (expanding mean of earlier fights)
    for c in PERF:
        d[f"prior_{c}"] = g[c].transform(lambda s: s.shift().expanding().mean())
    return d


def main() -> None:
    d = pd.read_csv(IN_CSV, low_memory=False, parse_dates=["event_date"])
    rep = [f"Loaded {len(d):,} fighter-rows"]

    d = fighter_history(d)

    # Debut / missing prior stats: fill with the division-and-year mean, keep a debut flag
    d["year"] = d["event_date"].dt.year
    for c in [f"prior_{p}" for p in PERF] + ["last3_win_pct"]:
        fill = d.groupby(["weight_class", "year"])[c].transform("mean")
        d[c] = d[c].fillna(fill).fillna(d[c].mean())
    d["layoff_days"] = d["layoff_days"].fillna(d["layoff_days"].median())

    feats = ["age_at_fight", "ufc_experience", "prior_fights", "prior_win_pct", "last3_win_pct",
             "streak", "prior_ko_losses", "prior_finish_wins", "layoff_days", "debut_in_data"] + \
            [f"prior_{p}" for p in PERF]

    # ---- one row per decided bout, sides randomized -------------------------
    dec = d[d["outcome"].isin(["W", "L"]) & d["age_at_fight"].notna()].copy()
    complete = dec.groupby("fight_id")["fighter_id"].transform("size") == 2
    dec = dec[complete]
    rng = np.random.default_rng(SEED)
    dec = dec.sort_values(["fight_id", "fighter_id"])
    fights = dec["fight_id"].unique()
    flip = pd.Series(rng.random(len(fights)) < 0.5, index=fights)
    first = dec.groupby("fight_id").cumcount() == 0
    dec["side"] = np.where(first ^ dec["fight_id"].map(flip).values, "A", "B")

    a = dec[dec["side"] == "A"].set_index("fight_id")
    b = dec[dec["side"] == "B"].set_index("fight_id")
    bouts = pd.DataFrame(index=a.index)
    bouts["event_date"] = a["event_date"]
    bouts["year"] = a["year"]
    bouts["weight_class"] = a["weight_class"]
    bouts["gender"] = a["gender"]
    bouts["title_or_tournament"] = a["title_or_tournament"].astype(int)
    bouts["a_fighter"], bouts["b_fighter"] = a["fighter"], b.loc[a.index, "fighter"]
    bouts["a_win"] = (a["outcome"] == "W").astype(int)
    for f in feats:
        bouts[f"a_{f}"] = a[f]
        bouts[f"b_{f}"] = b.loc[a.index, f]
        bouts[f"diff_{f}"] = bouts[f"a_{f}"] - bouts[f"b_{f}"]
    bouts = bouts.rename(columns={"diff_age_at_fight": "age_diff", "a_age_at_fight": "a_age",
                                  "b_age_at_fight": "b_age"})
    bouts = bouts.reset_index().sort_values("event_date").reset_index(drop=True)
    bouts.to_csv(OUT_CSV, index=False)

    # ---- report -----------------------------------------------------------------
    rep.append(f"Saved {OUT_CSV}: {len(bouts):,} decided bouts, {bouts.shape[1]} columns")
    rep.append(f"Side A win rate (should be ~0.50 because sides are random): {bouts['a_win'].mean():.3f}")
    young = np.sign(-bouts["age_diff"])           # +1 if A is younger
    yw = ((young > 0) & (bouts.a_win == 1)) | ((young < 0) & (bouts.a_win == 0))
    rep.append(f"Younger fighter won: {yw[young != 0].mean():.3f} of bouts")
    more_exp = np.sign(bouts["diff_ufc_experience"])
    ew = ((more_exp > 0) & (bouts.a_win == 1)) | ((more_exp < 0) & (bouts.a_win == 0))
    rep.append(f"More UFC-experienced fighter won: {ew[more_exp != 0].mean():.3f} of bouts")
    better = np.sign(bouts["diff_prior_win_pct"])
    bw = ((better > 0) & (bouts.a_win == 1)) | ((better < 0) & (bouts.a_win == 0))
    rep.append(f"Fighter with better prior record won: {bw[better != 0].mean():.3f} of bouts")
    rep.append(f"\nBouts per year:\n{bouts.groupby('year').size().to_string()}")
    rep.append(f"\nMissing values after filling: {int(bouts.isna().sum().sum())}")
    rep.append("\nCorrelation of each A-B difference with A winning:")
    diffs = [c for c in bouts.columns if c.startswith("diff_") or c == "age_diff"]
    cor = bouts[diffs].corrwith(bouts["a_win"]).sort_values(key=abs, ascending=False).round(3)
    rep.append(cor.to_string())
    text = "\n".join(rep)
    REPORT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
