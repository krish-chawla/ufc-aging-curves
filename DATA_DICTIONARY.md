# Data dictionary — `data/ufc_analysis.csv`

One row per fighter per bout. Official UFC cards only (incl. TUF finales and Noche UFC),
January 2010 onward. Source: UFCStats.com, collected with `scrape_card.py`.
Validated against the Kaggle dataset *UFC DATASETS 1994-2026* (neelagiriaditya): exact
agreement on every per-fight statistic and outcome across 7,555 bouts (2010-01-02 to 2026-08-08).

## Identifiers
| Column | Meaning |
|---|---|
| `fight_id` | UFCStats fight ID (from the fight-details URL) |
| `fighter_id`, `opponent_id` | UFCStats fighter IDs (from fighter-details URLs) |
| `event`, `event_date` | Event name and date |
| `fighter`, `opponent` | Names as listed on UFCStats |

## Fighter and bout context
| Column | Meaning |
|---|---|
| `gender` | `F` if the weight class is a women's division, else `M` |
| `weight_class` | Base division. Tournament/title labels are mapped to it, e.g. "Ultimate Fighter 19 Middleweight Tournament" → "Middleweight" |
| `weight_class_order` | 0 = Strawweight … 8 = Heavyweight (same scale for men and women) |
| `weight_class_raw` | Weight class exactly as scraped |
| `title_or_tournament` | Raw label mentions Title or Tournament |
| `dob`, `age_at_fight` | Date of birth; age in years on the event date |
| `opp_age_at_fight`, `age_diff` | Opponent's age; fighter age minus opponent age |

## Outcomes
| Column | Meaning |
|---|---|
| `outcome` | `W`, `L`, `D` (draw), `NC` (no contest) |
| `win` | 1 / 0; blank for draws and no contests |
| `finish_win` | 1 if won by KO/TKO or submission |
| `finished` | 1 if lost by KO/TKO or submission |
| `ko_loss` | 1 if lost by KO/TKO |
| `method_group` | Decision, KO/TKO, Submission, DQ, Overturned, Could Not Continue |
| `method_detail` | Method as scraped, e.g. "SUB Rear Naked Choke", "KO/TKO Punch" |
| `decision_type` | Unanimous / Split / Majority (decisions only) |
| `finish_round`, `fight_seconds`, `fight_minutes` | When the fight ended; total time fought (5-minute rounds) |

## Per-fight counts (whole fight, not per round)
`kd`, `sig_landed`, `sig_atmp`, `total_landed`, `total_atmp`, `td_landed`, `td_atmp`,
`sub_att`, `rev`, `ctrl_seconds`. The same stats for the opponent carry an `opp_` prefix.

## Per-fight rates (use these for aging curves: they don't depend on fight length)
| Column | Formula |
|---|---|
| `sig_acc` | sig_landed / sig_atmp |
| `sig_def` | 1 − opp_sig_landed / opp_sig_atmp |
| `sig_landed_pm`, `sig_absorbed_pm` | Significant strikes landed / absorbed per minute |
| `sig_diff_pm` | sig_landed_pm − sig_absorbed_pm |
| `td_acc` | td_landed / td_atmp (blank if no attempts) |
| `td_def` | 1 − opp_td_landed / opp_td_atmp (blank if opponent made no attempts) |
| `td_landed_p15`, `kd_p15`, `sub_att_p15` | Per 15 minutes |
| `ctrl_share` | ctrl_seconds / fight_seconds |

## Career stage
| Column | Meaning |
|---|---|
| `ufc_fights_before_2010` | Official UFC fights before the data window (counted from Kaggle `master.csv`) |
| `debuted_in_window` | True if the fighter's UFC debut is on or after 2010-01-01. False = career is left-censored |
| `fight_num_in_data`, `ufc_fight_number` | Fight count within the data / including pre-2010 UFC fights |
| `n_fights_in_data`, `is_last_fight_in_data` | Total fights in data; whether this is the last one observed (useful for survival analysis) |
| `days_since_last_fight` | Layoff length; blank for a fighter's first fight in the data |
| `prior_wins_in_data`, `prior_losses_in_data`, `prior_ko_losses_in_data` | Counts before this fight, 2010 onward only |
| `age_at_debut_in_data` | Age at first fight in the data |

## Quality flags
| Column | Meaning |
|---|---|
| `flag_no_stats` | Both fighters show zero strike attempts (stats not recorded) |
| `flag_ctrl_missing` | Control time blank on UFCStats |
| `flag_missing_age` | No DOB on UFCStats |
| `flag_nc_or_draw` | Draw or no contest |
| `flag_short_fight` | Under 60 seconds; per-minute rates are unstable |
| `include_main` | Not no-stats, not missing age, not draw/NC. Use this as the main sample |
