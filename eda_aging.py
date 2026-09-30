"""
eda_aging.py — first exploratory pass at UFC aging curves.

Input : data/ufc_analysis.csv  (from build_analysis_dataset.py; uses rows with include_main)
Output: eda/  folder with
  fig1_sample_by_age.png     how many fighter-rows exist at each age (men / women)
  fig2_raw_by_age.png        raw mean of each metric by age, 95% CI (men / women)
  fig3_raw_vs_delta.png      raw curve vs within-fighter (delta method, shrunk) curve, men
  fig4_attrition_by_age.png  share of fights that were a fighter's last, by age
  raw_by_age.csv, delta_curves.csv, attrition_by_age.csv
  eda_summary.txt            peak ages and key numbers (paste back to Claude)

Usage (run from ~/ufc_scraper):
  pip3 install matplotlib   # once, if needed
  python3 eda_aging.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

IN_CSV = Path("data/ufc_analysis.csv")
OUT = Path("eda")
AGE_MIN, AGE_MAX = 21, 40         # plotted range
MIN_N_RAW = 40                    # min fighter-rows per age/gender to plot a raw point
MIN_PAIRS_DELTA = 25              # min fighter pairs per age step for a delta point
ACTIVE_DAYS = 548                 # fought within ~18 months of data end = still active (censored)

# metric -> (label, weight column for per-fighter-age averages, higher_is_better)
METRICS = {
    "sig_acc":         ("Sig. strike accuracy", "sig_atmp", True),
    "sig_def":         ("Sig. strike defense", "opp_sig_atmp", True),
    "sig_landed_pm":   ("Sig. strikes landed / min", "fight_minutes", True),
    "sig_absorbed_pm": ("Sig. strikes absorbed / min", "fight_minutes", False),
    "td_acc":          ("Takedown accuracy", "td_atmp", True),
    "td_def":          ("Takedown defense", "opp_td_atmp", True),
    "ctrl_share":      ("Control time share", "fight_minutes", True),
    "win":             ("Win rate", None, True),
}
PCT = {"sig_acc", "sig_def", "td_acc", "td_def", "ctrl_share", "win"}

# Reference data-viz palette (categorical slots 1-2), light mode
C = {"M": "#2a78d6", "F": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABEL = {"M": "Men", "F": "Women"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.titlesize": 10, "axes.titlecolor": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def wmean(x: pd.Series, w: pd.Series | None) -> float:
    ok = x.notna() if w is None else x.notna() & w.notna() & (w > 0)
    if not ok.any():
        return np.nan
    return x[ok].mean() if w is None else np.average(x[ok], weights=w[ok])


# ------------------------------------------------------------------ raw curves
def raw_by_age(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (g, a), grp in d.groupby(["gender", "age"]):
        for m in METRICS:
            x = grp[m].dropna()
            n = len(x)
            if n < 2:
                continue
            se = x.std(ddof=1) / np.sqrt(n)
            rows.append({"gender": g, "age": a, "metric": m, "n": n, "mean": x.mean(),
                         "lo": x.mean() - 1.96 * se, "hi": x.mean() + 1.96 * se})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- delta method
def per_fighter_age(d: pd.DataFrame, m: str, wcol: str | None) -> pd.DataFrame:
    """One row per (gender, fighter, age): weighted mean of metric m and its total weight."""
    x = d[["gender", "fighter_id", "age", m]].copy()
    x["w"] = 1.0 if wcol is None else d[wcol]
    x = x[x[m].notna() & (x["w"] > 0)]
    x["vw"] = x[m] * x["w"]
    x["v2w"] = x[m] ** 2 * x["w"]
    g = x.groupby(["gender", "fighter_id", "age"]).agg(vw=("vw", "sum"), v2w=("v2w", "sum"),
                                                     w=("w", "sum"), n=(m, "size")).reset_index()
    g["v"] = g["vw"] / g["w"]
    return g


def shrink(g: pd.DataFrame) -> pd.Series:
    """Empirical-Bayes shrinkage of each fighter-age value toward the age mean.
    k = (noise variance per unit weight) / (true between-fighter variance).
    A fighter-age with weight w keeps w / (w + k) of its own deviation."""
    multi = g[g["n"] >= 2]
    # pooled within-fighter-age variance per unit weight: E[w_i (x_i - mean)^2]
    s2 = ((multi["v2w"] - multi["w"] * multi["v"] ** 2).sum() /
          max(multi["n"].sum() - len(multi), 1))
    # weighted (DerSimonian-Laird style) estimate of the true between-fighter variance,
    # so fighter-ages with tiny weights (1-2 strike attempts) don't dominate it
    cell = [g["gender"], g["age"]]
    mu = (g["v"] * g["w"]).groupby(cell).transform("sum") / g["w"].groupby(cell).transform("sum")
    q = (g["w"] * (g["v"] - mu) ** 2).sum()
    dof = len(g) - g.groupby(["gender", "age"]).ngroups
    sw = g["w"].groupby(cell).transform("sum")
    denom = (g["w"] - g["w"] ** 2 / sw).sum()
    tau2 = max((q - dof * s2) / denom, 1e-12)
    k = s2 / tau2
    return mu + (g["w"] / (g["w"] + k)) * (g["v"] - mu), k


def delta_curves(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Delta method: compare the SAME fighter at age a and age a+1, average the change
    across fighters (weight = harmonic mean of the two ages' weights), and add up the
    changes to get the curve.

    Correction for survivor bias: fighters who continue to age a+1 were, on average,
    luckier at age a than fighters who stopped (they had a good year and kept their
    contract). That luck regresses, so raw deltas drift downward at every step. Each
    fighter-age value is therefore shrunk toward the age mean by its reliability
    (empirical Bayes) BEFORE differencing, which removes most of the luck component.

    Win rate is excluded: matchmaking pairs winners with tougher opponents, so a
    fighter's win rate falls over a career even without aging. Age effects on
    winning are modeled with logistic regression (controlling for opponent age)."""
    out, ks = [], {}
    for m, (_, wcol, better) in METRICS.items():
        if m == "win":
            continue
        g = per_fighter_age(d, m, wcol)
        g["v"], ks[m] = shrink(g)
        nxt = g[["gender", "fighter_id", "age", "v", "w"]].copy()
        nxt["age"] -= 1
        pairs = g.merge(nxt, on=["gender", "fighter_id", "age"], suffixes=("", "_next"))
        pairs["delta"] = pairs["v_next"] - pairs["v"]
        pairs["hw"] = 2 / (1 / pairs["w"] + 1 / pairs["w_next"])
        for gen, gp in pairs.groupby("gender"):
            steps = (gp.groupby("age")
                       .apply(lambda s: pd.Series({"delta": np.average(s["delta"], weights=s["hw"]),
                                                   "pairs": len(s)}), include_groups=False)
                       .reset_index())
            steps = steps[(steps["pairs"] >= MIN_PAIRS_DELTA) &
                          steps["age"].between(AGE_MIN, AGE_MAX - 1)].sort_values("age")
            if steps.empty:
                continue
            run = (steps["age"].diff() != 1).cumsum()      # longest run of consecutive ages
            steps = steps[run == run.value_counts().idxmax()]
            ages = np.r_[steps["age"].iloc[0], steps["age"].values + 1]
            curve = np.r_[0, steps["delta"].cumsum().values]
            best = np.nanargmax(curve) if better else np.nanargmin(curve)
            curve = curve - curve[best]                    # 0 at peak
            pairs_into = dict(zip(steps["age"] + 1, steps["pairs"]))
            for a, v in zip(ages, curve):
                out.append({"metric": m, "gender": gen, "age": int(a), "rel_to_peak": v,
                            "peak_age": int(ages[best]), "pairs_into_age": pairs_into.get(a, np.nan)})
    return pd.DataFrame(out), ks


# ------------------------------------------------------------------- attrition
def attrition(d_all: pd.DataFrame) -> pd.DataFrame:
    end = d_all["event_date"].max()
    last = d_all.groupby("fighter_id")["event_date"].transform("max")
    d = d_all[(end - last).dt.days > ACTIVE_DAYS]           # drop still-active fighters
    t = (d.groupby(["gender", "age"])
           .agg(fights=("fight_id", "size"), last_fights=("is_last_fight_in_data", "sum"))
           .reset_index())
    t["exit_rate"] = t["last_fights"] / t["fights"]
    return t


# --------------------------------------------------------------------- figures
def fmt_axis(ax, metric, change=False):
    """change=True: y is a difference from peak, so proportions are shown in percentage points."""
    if metric in PCT:
        if change:
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: f"{round(v * 100, 1):+g} pp" if abs(v) > 1e-9 else "0"))
        else:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(AGE_MIN - 0.7, AGE_MAX + 0.7)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    d0 = pd.read_csv(IN_CSV, low_memory=False, parse_dates=["event_date"])
    d0["age"] = np.floor(d0["age_at_fight"])
    d = d0[d0["include_main"] & d0["age"].between(AGE_MIN, AGE_MAX)].copy()
    rep = [f"Main sample, ages {AGE_MIN}-{AGE_MAX}: {len(d):,} fighter-rows, "
           f"{d['fighter_id'].nunique():,} fighters "
           f"({(d.gender == 'M').sum():,} men's rows / {(d.gender == 'F').sum():,} women's rows)"]

    # ---- fig 1: sample size by age -------------------------------------
    cnt = d.groupby(["age", "gender"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    x = cnt.index.values
    for i, g in enumerate(["M", "F"]):
        if g in cnt:
            ax.bar(x + (i - 0.5) * 0.4, cnt[g], width=0.38, color=C[g], label=LABEL[g], linewidth=0)
    ax.set_title("Fighter-bouts by age at fight", loc="left")
    ax.set_xlabel("Age at fight (years)")
    ax.set_ylabel("Fighter-bouts")
    fmt_axis(ax, "n")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.savefig(OUT / "fig1_sample_by_age.png")
    plt.close(fig)

    # ---- fig 2: raw means by age -----------------------------------------
    raw = raw_by_age(d)
    raw.to_csv(OUT / "raw_by_age.csv", index=False)
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6), sharex=True)
    for ax, (m, (label, _, _)) in zip(axes.flat, METRICS.items()):
        for g in ["M", "F"]:
            r = raw[(raw.metric == m) & (raw.gender == g) & (raw.n >= MIN_N_RAW)]
            if r.empty:
                continue
            ax.fill_between(r.age, r.lo, r.hi, color=C[g], alpha=0.15, linewidth=0)
            ax.plot(r.age, r["mean"], color=C[g], linewidth=2, marker="o", markersize=3, label=LABEL[g])
        ax.set_title(label, loc="left")
        fmt_axis(ax, m)
    for ax in axes[1]:
        ax.set_xlabel("Age at fight")
    axes[0, 0].legend(loc="lower left")
    fig.suptitle(f"Raw averages by age (95% CI; ages with ≥{MIN_N_RAW} fighter-bouts)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_raw_by_age.png")
    plt.close(fig)

    # ---- fig 3: raw vs delta (men) ----------------------------------------
    dc, ks = delta_curves(d)
    dc.to_csv(OUT / "delta_curves.csv", index=False)
    show = ["sig_acc", "sig_landed_pm", "sig_def", "ctrl_share"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharex=True)
    for ax, m in zip(axes, show):
        r = raw[(raw.metric == m) & (raw.gender == "M") & (raw.n >= MIN_N_RAW)].sort_values("age")
        c = dc[(dc.metric == m) & (dc.gender == "M")].sort_values("age")
        if not r.empty:
            peak = r["mean"].max() if METRICS[m][2] else r["mean"].min()
            ax.plot(r.age, r["mean"] - peak, color=INK2, linewidth=2, linestyle="--", label="Raw average")
        if not c.empty:
            ax.plot(c.age, c.rel_to_peak, color=C["M"], linewidth=2, label="Delta method")
            pk = c["peak_age"].iloc[0]
            ax.axvline(pk, color=C["M"], linewidth=0.8, alpha=0.5)
            ax.annotate(f"peak {pk}", (pk, 0), xytext=(4, -12), textcoords="offset points",
                        color=INK2, fontsize=8)
        ax.axhline(0, color=INK2, linewidth=0.6)
        ax.set_title(METRICS[m][0], loc="left")
        ax.set_xlabel("Age at fight")
        fmt_axis(ax, m, change=True)
    axes[0].set_ylabel("Change vs. peak")
    axes[0].legend(loc="lower center")
    fig.suptitle("Men: raw average vs. within-fighter (delta method, shrinkage-corrected) aging curve",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_raw_vs_delta.png")
    plt.close(fig)

    # ---- fig 4: attrition ---------------------------------------------------
    at = attrition(d0[d0["include_main"] & d0["age"].between(AGE_MIN, AGE_MAX)])
    at.to_csv(OUT / "attrition_by_age.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for g in ["M", "F"]:
        a = at[(at.gender == g) & (at.fights >= MIN_N_RAW)]
        ax.plot(a.age, a.exit_rate, color=C[g], linewidth=2, marker="o", markersize=3, label=LABEL[g])
    ax.set_title("Share of fights that were the fighter's last UFC fight, by age "
                 f"(fighters inactive > {ACTIVE_DAYS // 30} months)", loc="left")
    ax.set_xlabel("Age at fight")
    fmt_axis(ax, "win")
    ax.legend(loc="upper left")
    fig.savefig(OUT / "fig4_attrition_by_age.png")
    plt.close(fig)

    # ---- summary -------------------------------------------------------------
    rep.append("\nDelta-method peak age and decline from peak (men | women):")
    for m, (label, _, better) in METRICS.items():
        if m == "win":
            continue
        parts = []
        for g in ["M", "F"]:
            c = dc[(dc.metric == m) & (dc.gender == g)]
            if c.empty:
                parts.append("n/a")
                continue
            pk = int(c["peak_age"].iloc[0])
            at35 = c.loc[c.age == 35, "rel_to_peak"]
            span = f"ages {c.age.min()}-{c.age.max()}"
            chg = (f", at 35: {at35.iloc[0]:+.1%}" if m in PCT else f", at 35: {at35.iloc[0]:+.2f}") \
                if len(at35) else ""
            parts.append(f"peak {pk}{chg} ({span})")
        rep.append(f"  {label:<28} {parts[0]:<38} | {parts[1]}")
    rep.append("\nShrinkage constant k (weight units at which a fighter-age keeps half its deviation):")
    rep.append("  " + ", ".join(f"{m}: {k:.1f}" for m, k in ks.items()))
    rep.append("\nAttrition (exit rate) by age bracket, men:")
    am = at[at.gender == "M"].copy()
    am["bracket"] = pd.cut(am.age, [20, 25, 29, 33, 36, 40])
    b = am.groupby("bracket", observed=True)[["fights", "last_fights"]].sum()
    b["exit_rate"] = (b.last_fights / b.fights).round(3)
    rep.append(b.to_string())
    text = "\n".join(rep)
    (OUT / "eda_summary.txt").write_text(text)
    print(text)
    print(f"\nSaved figures and tables to {OUT}/")


if __name__ == "__main__":
    main()
