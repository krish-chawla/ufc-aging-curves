# models_aging.R — mixed-effects aging curves for UFC fighters
#
# Input : data/ufc_analysis.csv   (from build_analysis_dataset.py; rows with include_main)
# Output: models/                 (or models_experience/ for the sensitivity run)
#   curves.csv               predicted curve by age, men & women, with 95% intervals
#   peak_ages.csv            peak age per metric with a 95% interval
#   fig_model_curves.png     all curves, small multiples
#   model_summaries.txt      peak table + full model output (paste this back to Claude)
#
# Each metric is modeled on its natural scale with lme4, fighter as a random effect:
#   accuracy / defense  -> binomial GLMM (successes out of attempts) + observation-level
#                          random effect for extra-binomial variation (opponent, fight style)
#   per-minute output   -> Poisson GLMM with log(minutes) offset + observation-level
#                          random effect for overdispersion (Poisson-lognormal)
#   control time        -> binomial GLMM: control seconds out of fight seconds
#   (glmer.nb was used first but can converge to a degenerate fit with the fighter
#    variance at 0; the Poisson-lognormal form is stable and gives the same curves)
#   win                 -> head-to-head (paired) logistic model, one row per bout:
#                          logit P(win) = f(own age) - f(opponent age), f a spline per gender
#   KO/TKO loss         -> logistic model on own age + opponent age + division + era
# Fixed effects in the performance models: a natural spline of age for men and for
# women, opponent age (spline), weight class, event year (spline, for era). The
# fighter random intercept separates "the older fighters in the data are different
# people" (between-fighter) from "fighters change as they age" (within-fighter).
#
# Why win and KO loss have NO fighter random effect: UFC careers end on losses (a
# finished career's last fight is a win only ~8% of the time, vs ~51% otherwise),
# so within-fighter models of outcomes show a decline even without any aging. The
# paired model compares the two fighters in each bout instead.
#
# Usage (from ~/ufc_scraper, in Terminal):
#   Rscript models_aging.R              # main models (about 1-2 minutes)
#   Rscript models_aging.R experience   # sensitivity: also control for UFC fight number
#   Rscript models_aging.R drop_last    # sensitivity: drop each finished career's last fight
#
# One-time setup (type R in Terminal, paste this line, then q() to quit):
#   install.packages(c("lme4", "dplyr", "readr", "tidyr", "ggplot2"))

suppressPackageStartupMessages({
  library(lme4); library(splines); library(dplyr); library(readr); library(tidyr); library(ggplot2)
})
set.seed(2026)

args <- commandArgs(trailingOnly = TRUE)
WITH_EXPERIENCE <- "experience" %in% args
DROP_LAST <- "drop_last" %in% args
OUT <- paste(c("models", if (WITH_EXPERIENCE) "experience", if (DROP_LAST) "droplast"), collapse = "_")
ACTIVE_DAYS <- 548                   # fought within ~18 months of data end = career not finished
dir.create(OUT, showWarnings = FALSE)
AGE_MIN <- 21; AGE_MAX <- 40
AGE_DF <- 4                          # spline flexibility for age (4 = smooth, allows one peak + decline)
N_SIM <- 2000                        # simulation draws for intervals and peak ages
CTRL <- glmerControl(optimizer = "bobyqa", calc.derivs = FALSE)

# ------------------------------------------------------------------ data
d <- read_csv("data/ufc_analysis.csv", show_col_types = FALSE) |>
  filter(include_main, age_at_fight >= AGE_MIN, age_at_fight < AGE_MAX + 1) |>
  mutate(
    age         = age_at_fight,
    opp_age     = opp_age_at_fight,
    gender      = factor(gender, levels = c("M", "F")),
    fighter_id  = factor(fighter_id),
    wc          = factor(weight_class),
    year        = as.numeric(format(event_date, "%Y")) + (as.numeric(format(event_date, "%j")) - 1) / 365.25,
    log_fightno = log(ufc_fight_number),
    minutes     = fight_minutes,
    obs         = factor(row_number())
  ) |>
  filter(!is.na(opp_age), minutes > 0)

if (DROP_LAST) {
  end_date <- max(d$event_date)
  d <- d |> group_by(fighter_id) |>
    mutate(finished = as.numeric(end_date - max(event_date)) > ACTIVE_DAYS) |> ungroup() |>
    filter(!(is_last_fight_in_data & finished)) |> select(-finished)
}

# Spline bases are fixed on the full data so every model and the prediction grid share them
age_knots <- list(knots = quantile(d$age, seq(0, 1, length.out = AGE_DF + 1))[2:AGE_DF],
                  Boundary.knots = range(d$age))
AGE <- function(x) ns(x, knots = age_knots$knots, Boundary.knots = age_knots$Boundary.knots)
OPP <- function(x) ns(x, knots = quantile(d$opp_age, c(1, 2) / 3), Boundary.knots = range(d$opp_age))
YR  <- function(x) ns(x, knots = quantile(d$year, c(1, 2) / 3), Boundary.knots = range(d$year))
EXP <- function(x) ns(x, knots = quantile(d$log_fightno, c(1, 2) / 3), Boundary.knots = range(d$log_fightno))

# Weight class already identifies gender (women's divisions), so gender enters only
# through its own age curve: gender:AGE(age). Levels come from weight class.
fixed_rhs <- "wc + gender:AGE(age) + OPP(opp_age) + YR(year)"
if (WITH_EXPERIENCE) fixed_rhs <- paste(fixed_rhs, "+ EXP(log_fightno)")

cat(sprintf("Rows: %d | fighters: %d | men rows: %d | women rows: %d\n",
            nrow(d), n_distinct(d$fighter_id), sum(d$gender == "M"), sum(d$gender == "F")))

# One row per bout for the win model (both rows of a bout carry the same information)
bout_one <- d |> filter(!is.na(win)) |> group_by(fight_id) |> slice_sample(n = 1) |> ungroup()

# ------------------------------------------------------------- model list
specs <- list(
  sig_acc         = list(label = "Sig. strike accuracy", type = "binom",
                         lhs = "cbind(sig_landed, sig_atmp - sig_landed)", data = \() filter(d, sig_atmp > 0)),
  sig_def         = list(label = "Sig. strike defense", type = "binom",
                         lhs = "cbind(opp_sig_atmp - opp_sig_landed, opp_sig_landed)",
                         data = \() filter(d, opp_sig_atmp > 0)),
  sig_landed_pm   = list(label = "Sig. strikes landed / min", type = "pois",
                         lhs = "sig_landed", offset = "offset(log(minutes))", data = \() d),
  sig_absorbed_pm = list(label = "Sig. strikes absorbed / min", type = "pois", worse_high = TRUE,
                         lhs = "opp_sig_landed", offset = "offset(log(minutes))", data = \() d),
  td_acc          = list(label = "Takedown accuracy", type = "binom",
                         lhs = "cbind(td_landed, td_atmp - td_landed)", data = \() filter(d, td_atmp > 0)),
  td_def          = list(label = "Takedown defense", type = "binom",
                         lhs = "cbind(opp_td_atmp - opp_td_landed, opp_td_landed)",
                         data = \() filter(d, opp_td_atmp > 0)),
  ctrl_share      = list(label = "Control time share", type = "binom",
                         lhs = "cbind(ctrl_s, fight_s - ctrl_s)",
                         data = \() filter(d, !is.na(ctrl_seconds)) |>
                           mutate(fight_s = round(minutes * 60), ctrl_s = pmin(round(ctrl_seconds), fight_s))),
  win             = list(label = "Win probability vs. a 30-year-old", type = "paired",
                         lhs = "win", data = \() bout_one),
  ko_loss         = list(label = "KO/TKO loss probability", type = "logit", worse_high = TRUE,
                         lhs = "ko_loss", data = \() d)
)

# Paired design matrix for the win model: own age basis minus opponent age basis,
# separately for men and women (no intercept: sides are assigned at random)
paired_X <- function(dat) {
  D <- AGE(dat$age) - AGE(dat$opp_age)
  X <- cbind(D * (dat$gender == "M"), D * (dat$gender == "F"))
  colnames(X) <- c(paste0("M_age", seq_len(ncol(D))), paste0("F_age", seq_len(ncol(D))))
  X
}

fit_one <- function(s, dat) {
  if (s$type == "paired") {
    Xw <- paired_X(dat); y <- dat$win
    return(glm(y ~ 0 + Xw, family = binomial))
  }
  if (s$type == "logit") {
    return(glm(as.formula(paste(s$lhs, "~", fixed_rhs)), data = dat, family = binomial))
  }
  rhs <- paste(fixed_rhs, if (!is.null(s$offset)) paste("+", s$offset), "+ (1 | fighter_id)")
  rhs <- paste(rhs, "+ (1 | obs)")   # observation-level effect: extra-binomial / extra-Poisson variation
  f <- as.formula(paste(s$lhs, "~", rhs))
  glmer(f, data = dat, family = if (s$type == "pois") poisson else binomial, control = CTRL, nAGQ = 0)
}

is_mixed <- function(fit) inherits(fit, "merMod")
fixed_coef <- function(fit) if (is_mixed(fit)) fixef(fit) else coef(fit)

# ------------------------------------------------------- prediction grid
ref_wc <- d |> count(gender, wc) |> group_by(gender) |> slice_max(n, n = 1, with_ties = FALSE) |>
  ungroup() |> select(gender, wc)
f_range <- quantile(d$age[d$gender == "F"], c(0.02, 0.98))
grid <- expand_grid(gender = factor(c("M", "F"), levels = c("M", "F")),
                    age = seq(AGE_MIN, AGE_MAX, by = 0.25)) |>
  filter(gender == "M" | (age >= f_range[1] & age <= f_range[2])) |>
  left_join(ref_wc, by = "gender") |>
  mutate(opp_age = median(d$opp_age), year = median(d$year), log_fightno = median(d$log_fightno),
         minutes = 1, win = 0, ko_loss = 0, sig_landed = 0, opp_sig_landed = 0, ctrl_seconds = 0,
         ctrl_s = 0, fight_s = 1)

# --------------------------------------------------------------- fit all
curves <- list(); peaks <- list(); summaries <- character()
for (m in names(specs)) {
  s <- specs[[m]]
  t0 <- Sys.time()
  dat <- s$data()
  fit <- suppressMessages(fit_one(s, dat))
  cat(sprintf("%-16s n=%6d  fit %4.1fs%s\n", m, nobs(fit), as.numeric(Sys.time() - t0, units = "secs"),
              if (is_mixed(fit) && isSingular(fit))
                "  ** WARNING: singular fit (a random-effect variance is ~0) - check before using **" else ""))

  # Population curve (random effects = 0) with intervals from the fixed-effect covariance
  if (s$type == "paired") {
    g2 <- grid |> mutate(opp_age = 30)              # win probability against a 30-year-old
    X <- paired_X(g2)
    colnames(X) <- paste0("Xw", colnames(X))
  } else {
    X <- model.matrix(delete.response(terms(fit)), data = grid)
  }
  b <- fixed_coef(fit); b <- b[!is.na(b)]
  X <- X[, names(b), drop = FALSE]
  V <- as.matrix(vcov(fit))[names(b), names(b)]
  B <- MASS::mvrnorm(N_SIM, b, V)
  # offsets are left out of X, so the rate models give per-minute rates
  eta_hat <- as.numeric(X %*% b)
  eta_sim <- X %*% t(B)
  # Poisson-lognormal: mean rate for a typical fighter is exp(eta + sigma_obs^2 / 2)
  if (s$type == "pois") {
    s2 <- attr(VarCorr(fit)$obs, "stddev")^2
    inv <- function(eta) exp(eta + s2 / 2)
  } else inv <- plogis
  est <- inv(eta_hat); sim <- inv(eta_sim)
  curves[[m]] <- grid |> select(gender, age) |>
    mutate(metric = m, label = s$label, est = est,
           lo = apply(sim, 1, quantile, 0.025), hi = apply(sim, 1, quantile, 0.975))

  worse_high <- isTRUE(s$worse_high)
  for (g in c("M", "F")) {
    idx <- which(grid$gender == g)
    a <- grid$age[idx]; e <- est[idx]
    pick <- if (worse_high) which.min else which.max
    pk <- apply(sim[idx, , drop = FALSE], 2, \(v) a[pick(v)])
    best <- pick(e)
    peaks[[paste(m, g)]] <- tibble(
      metric = m, label = s$label, gender = g,
      peak_age = a[best], peak_lo = unname(quantile(pk, 0.025)), peak_hi = unname(quantile(pk, 0.975)),
      value_at_peak = e[best],
      change_at_35 = if (any(a == 35)) e[a == 35] - e[best] else NA_real_,
      edge_peak = a[best] %in% range(a))
  }
  # likelihood-ratio test: does the age curve matter at all? (compare with age removed)
  if (s$type == "paired") {                         # vs. no age effect at all (eta = 0)
    chi <- fit$null.deviance - fit$deviance; ldf <- length(coef(fit))
  } else {
    fit0 <- suppressMessages(update(fit, . ~ . - gender:AGE(age), data = dat))
    lrt <- anova(fit0, fit, test = "Chisq")
    chi <- if (is_mixed(fit)) lrt$Chisq[2] else lrt$Deviance[2]
    ldf <- abs(lrt$Df[2])
  }
  pv <- pchisq(chi, ldf, lower.tail = FALSE)
  summaries <- c(summaries,
    sprintf("\n==================== %s (%s) ====================", m, s$label),
    sprintf("n = %d%s | LRT for age curve: chi2 = %.1f, df = %d, p = %.2g",
            nobs(fit), if (is_mixed(fit)) sprintf(" | fighters = %d", ngrps(fit)[["fighter_id"]]) else "",
            chi, ldf, pv),
    if (is_mixed(fit)) capture.output(print(VarCorr(fit))),
    capture.output(printCoefmat(coef(summary(fit)), digits = 3)))
}

curves <- bind_rows(curves)
peaks  <- bind_rows(peaks)
write_csv(curves, file.path(OUT, "curves.csv"))
write_csv(peaks,  file.path(OUT, "peak_ages.csv"))

# ---------------------------------------------------------------- figure
pct_metrics <- c("sig_acc", "sig_def", "td_acc", "td_def", "ctrl_share", "win", "ko_loss")
pal <- c(Men = "#2a78d6", Women = "#eb6834")
plot_df <- curves |> mutate(
  gender = factor(ifelse(gender == "M", "Men", "Women"), levels = c("Men", "Women")),
  label  = factor(label, levels = sapply(specs, `[[`, "label")),
  across(c(est, lo, hi), \(v) ifelse(metric %in% pct_metrics, 100 * v, v)),
  label  = factor(ifelse(metric %in% pct_metrics, paste0(label, " (%)"), as.character(label)),
                  levels = unique(ifelse(names(specs) %in% pct_metrics,
                                         paste0(sapply(specs, `[[`, "label"), " (%)"),
                                         sapply(specs, `[[`, "label")))))
p <- ggplot(plot_df, aes(age, est, colour = gender, fill = gender)) +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.15, colour = NA) +
  geom_line(linewidth = 0.9) +
  facet_wrap(~label, scales = "free_y", ncol = 3) +
  scale_colour_manual(values = pal, name = NULL) +
  scale_fill_manual(values = pal, name = NULL) +
  scale_x_continuous(breaks = seq(20, 40, 5)) +
  labs(x = "Age at fight", y = NULL,
       title = "UFC aging curves: mixed-effects models (performance) and head-to-head model (wins)",
       subtitle = paste0("Adjusted for opponent age, weight class and era; 95% intervals; ",
                         "curves for a typical fighter in the most common division",
                         if (WITH_EXPERIENCE) "; + UFC fight number" else "",
                         if (DROP_LAST) "; finished careers' last fights dropped" else "")) +
  theme_minimal(base_size = 10) +
  theme(panel.grid.minor = element_blank(), panel.grid.major = element_line(colour = "#e4e3df"),
        legend.position = "top", legend.justification = "left",
        plot.title.position = "plot", strip.text = element_text(hjust = 0, face = "bold"),
        text = element_text(colour = "#0b0b0b"), axis.text = element_text(colour = "#52514e"))
ggsave(file.path(OUT, "fig_model_curves.png"), p, width = 10, height = 8.5, dpi = 150, bg = "white")

# ---------------------------------------------------------------- report
tab <- peaks |> mutate(
  peak = sprintf("%.1f [%.1f-%.1f]%s", peak_age, peak_lo, peak_hi, ifelse(edge_peak, " (edge)", "")),
  change_to_35 = case_when(
    is.na(change_at_35) ~ "",
    metric %in% c("sig_landed_pm", "sig_absorbed_pm") ~ sprintf("%+.2f /min", change_at_35),
    TRUE ~ sprintf("%+.1f pp", 100 * change_at_35))) |>
  select(metric, gender, peak, change_to_35)
report <- c(
  sprintf("Rows: %d | fighters: %d%s%s", nrow(d), n_distinct(d$fighter_id),
          if (WITH_EXPERIENCE) " | sensitivity: + UFC fight number" else "",
          if (DROP_LAST) " | sensitivity: finished careers' last fights dropped" else ""),
  sprintf("Curves: opponent age %.1f, year %.0f, reference divisions: %s",
          median(d$opp_age), median(d$year), paste(ref_wc$wc, collapse = " / ")),
  "", "Best age [95% interval] and change from best age to 35:",
  capture.output(print(as.data.frame(tab), row.names = FALSE)),
  "", "(edge) = best value is at the edge of the age range: no interior peak, the metric only",
  "declines (or only improves) across 21-40. For 'absorbed' and 'KO/TKO loss', best = lowest.",
  summaries)
writeLines(report, file.path(OUT, "model_summaries.txt"))
cat("\n", paste(report[1:(6 + nrow(tab) + 3)], collapse = "\n"), "\n", sep = "")
cat(sprintf("\nSaved %s/curves.csv, peak_ages.csv, fig_model_curves.png, model_summaries.txt\n", OUT))
