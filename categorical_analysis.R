# categorical_analysis.R - categorical data analysis of age and fight outcomes,
#                          following Dr. Johnson's "Lecture Notes on Categorical Data Analysis"
#
# Input : data/bouts_prefight.csv   (from build_prefight_features.py)
#         data/ufc_analysis.csv     (for how each fight ended: method_group)
# Output: categorical/
#   categorical_report.txt          every test below, in the order of the lecture notes
#   table_2x2.csv                   younger/older x won/lost
#   breslow_day_by_class.csv        odds ratio (younger vs older) in each weight class
#   table_ageband_result_M.csv / _F.csv   observed counts, expected counts, standardized residuals
#   drop_in_deviance.csv            model comparison matrix (notes Example 8.6.3)
#   multinomial_coefficients.csv    baseline-category logit estimates (notes Ch. 9)
#   fig_or_by_class.png             forest plot: younger fighter's odds ratio by weight class
#   fig_residuals_ageband.png       standardized residuals, age band x result
#   fig_multinomial_probs.png       estimated probability of each result vs fighter age
#
# Sections (lecture-note chapter in brackets)
#   0. Check: Breslow-Day code reproduces notes Examples 6.5.2 and 6.5.3
#   1. One-sample test: does the younger fighter win more than half the time?   [Ch. 4]
#   2. 2x2 table: difference in proportions, relative risk, odds ratio, X^2, G^2 [Ch. 5, 6.1-6.2]
#   3. Breslow-Day: is the younger fighter's odds ratio the same in every
#      weight class? for men and women?                                          [Ch. 6.5]
#   4. Age band x result (win / decision loss / KO-TKO loss / submission loss):
#      test of independence + standardized residuals                             [Ch. 6.1-6.3]
#   5. Logistic regression on age difference: Wald test, G statistic,
#      confidence intervals for beta, the odds ratio, and pi(x)                  [Ch. 8.4]
#   6. Comparing nested models by drop in deviance (M1 -> M4)                    [Ch. 8.6.2]
#   7. Multinomial (baseline-category) logistic regression: how a fight ends
#      as a function of each fighter's age                                       [Ch. 9]
#
# Each bout is used ONCE, from the point of view of fighter A (sides were assigned at random),
# so all observations are independent, as the chi-squared tests assume. Uses all decided bouts.
#
# Usage (from ~/ufc_scraper):
#   Rscript categorical_analysis.R

suppressPackageStartupMessages({
  library(dplyr); library(readr); library(tidyr); library(ggplot2); library(nnet)
})
OUT <- "categorical"; dir.create(OUT, showWarnings = FALSE)
MIN_STRATUM <- 5                        # Breslow-Day: every cell in a weight class needs >= 5 bouts

fmt_p <- function(p) ifelse(p < 0.001, "< 0.001", sprintf("= %.3f", p))
rep <- character(0); say <- function(...) rep <<- c(rep, sprintf(...))
ilogit <- plogis

# ------------------------------------------------------------------ Breslow-Day (notes 6.5)
mh_or <- function(tab) {                # tab: 2 x 2 x K array
  n <- apply(tab, 3, sum)
  sum(tab[1, 1, ] * tab[2, 2, ] / n) / sum(tab[1, 2, ] * tab[2, 1, ] / n)
}
breslow_day <- function(tab) {
  th <- mh_or(tab); K <- dim(tab)[3]; bd <- 0
  for (k in 1:K) {
    t <- tab[, , k]; r1 <- sum(t[1, ]); r2 <- sum(t[2, ]); c1 <- sum(t[, 1])
    # x (r2 - c1 + x) = th (r1 - x)(c1 - x)   ->   (1 - th) x^2 + (r2 - c1 + th (r1 + c1)) x - th r1 c1 = 0
    A <- 1 - th; B <- r2 - c1 + th * (r1 + c1); C <- -th * r1 * c1
    roots <- if (abs(A) < 1e-12) -C / B else (-B + c(-1, 1) * sqrt(B^2 - 4 * A * C)) / (2 * A)
    lo <- max(0, c1 - r2); hi <- min(r1, c1)
    x <- roots[roots > lo & roots < hi][1]          # the root that makes every expected count positive
    e <- c(x, r1 - x, c1 - x, r2 - c1 + x)
    bd <- bd + (t[1, 1] - x)^2 * sum(1 / e)
  }
  list(theta_mh = th, BD = bd, df = K - 1, p = pchisq(bd, K - 1, lower.tail = FALSE))
}
g2 <- function(obs) {                   # likelihood-ratio chi-squared statistic G^2
  e <- outer(rowSums(obs), colSums(obs)) / sum(obs)
  2 * sum(ifelse(obs > 0, obs * log(obs / e), 0))
}

# ------------------------------------------------------------------ 0. check against the notes
ex652 <- array(c(10, 5, 8, 8, 9, 6, 5, 10), dim = c(2, 2, 2))
ex653 <- array(c(56, 42, 44, 58, 54, 45, 46, 55, 58, 43, 42, 57), dim = c(2, 2, 3))
b652 <- breslow_day(ex652); b653 <- breslow_day(ex653)
say("CATEGORICAL DATA ANALYSIS - age and UFC fight outcomes")
say("Methods follow Dr. Johnson's Lecture Notes on Categorical Data Analysis (chapter numbers in brackets).")
say("")
say("0. CHECK: Breslow-Day code vs. the lecture notes")
say("   Example 6.5.2: theta_MH = %.4f (notes 2.4366), BD = %.4f (notes 0.1459), p = %.2f (notes ~0.7)",
    b652$theta_mh, b652$BD, b652$p)
say("   Example 6.5.3: BD = %.2f (notes ~0.42), p = %.2f (notes ~0.81)", b653$BD, b653$p)
say("")

# ------------------------------------------------------------------ data
b <- read_csv("data/bouts_prefight.csv", show_col_types = FALSE) |>
  filter(!is.na(age_diff), !is.na(a_win), age_diff != 0)
meth <- read_csv("data/ufc_analysis.csv", show_col_types = FALSE,
                 col_select = c("fight_id", "method_group")) |>
  distinct(fight_id, .keep_all = TRUE)
b <- b |> left_join(meth, by = "fight_id") |>
  mutate(younger = factor(ifelse(age_diff < 0, "Younger", "Older"), levels = c("Younger", "Older")),
         result  = factor(ifelse(a_win == 1, "Won", "Lost"), levels = c("Won", "Lost")),
         weight_class = factor(weight_class))
say("Data: %d decided bouts with an age difference (%s to %s). Fighter A = randomly chosen side.",
    nrow(b), min(b$event_date), max(b$event_date))
say("")

# ------------------------------------------------------------------ 1. one-sample [Ch. 4]
yw <- with(b, (age_diff < 0 & a_win == 1) | (age_diff > 0 & a_win == 0))
n <- length(yw); x <- sum(yw); ph <- x / n; p0 <- 0.5
z_score <- (ph - p0) / sqrt(p0 * (1 - p0) / n)
z_wald  <- (ph - p0) / sqrt(ph * (1 - ph) / n)
G_one   <- 2 * (x * log(ph / p0) + (n - x) * log((1 - ph) / (1 - p0)))
wald_ci <- ph + c(-1, 1) * 1.96 * sqrt(ph * (1 - ph) / n)
score_ci <- prop.test(x, n, correct = FALSE)$conf.int
say("1. ONE-SAMPLE TEST: proportion of bouts won by the younger fighter [Ch. 4]")
say("   Younger fighter won %d of %d bouts: p-hat = %.4f", x, n, ph)
say("   95%% Wald CI (%.4f, %.4f);  95%% score CI (%.4f, %.4f)", wald_ci[1], wald_ci[2], score_ci[1], score_ci[2])
say("   H0: p = 0.5   score z = %.2f, p %s | Wald z = %.2f, p %s | likelihood-ratio G = %.2f, p %s",
    z_score, fmt_p(2 * pnorm(-abs(z_score))), z_wald, fmt_p(2 * pnorm(-abs(z_wald))),
    G_one, fmt_p(pchisq(G_one, 1, lower.tail = FALSE)))
say("")

# ------------------------------------------------------------------ 2. 2x2 table [Ch. 5, 6]
tab <- table(b$younger, b$result)
write_csv(as.data.frame.matrix(tab) |> tibble::rownames_to_column("fighter_A"), file.path(OUT, "table_2x2.csv"))
n11 <- tab[1, 1]; n12 <- tab[1, 2]; n21 <- tab[2, 1]; n22 <- tab[2, 2]
n1 <- n11 + n12; n2 <- n21 + n22
p1 <- n11 / n1; p2 <- n21 / n2
se_d <- sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
pp <- (n11 + n21) / (n1 + n2); z_pool <- (p1 - p2) / sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
rr <- p1 / p2; se_lrr <- sqrt((1 - p1) / n11 + (1 - p2) / n21)
or <- (n11 * n22) / (n12 * n21); se_lor <- sqrt(1 / n11 + 1 / n12 + 1 / n21 + 1 / n22)
X2 <- suppressWarnings(chisq.test(tab, correct = FALSE)); G2 <- g2(tab)
say("2. TWO-WAY TABLE: is fighter A younger or older x did A win? [Ch. 5, 6.1-6.2]")
say("                 Won    Lost   Total")
say("   Younger   %6d  %6d  %6d", n11, n12, n1)
say("   Older     %6d  %6d  %6d", n21, n22, n2)
say("   P(win | younger) = %.4f,  P(win | older) = %.4f", p1, p2)
say("   Difference in proportions = %.4f, 95%% Wald CI (%.4f, %.4f); pooled z = %.2f, p %s",
    p1 - p2, p1 - p2 - 1.96 * se_d, p1 - p2 + 1.96 * se_d, z_pool, fmt_p(2 * pnorm(-abs(z_pool))))
say("   Relative risk (of winning) = %.3f, 95%% CI (%.3f, %.3f)",
    rr, exp(log(rr) - 1.96 * se_lrr), exp(log(rr) + 1.96 * se_lrr))
say("   Odds ratio = %.3f, 95%% CI (%.3f, %.3f): the younger fighter's odds of winning are %.2f times the older fighter's",
    or, exp(log(or) - 1.96 * se_lor), exp(log(or) + 1.96 * se_lor), or)
say("   Test of independence: Pearson X^2 = %.2f, G^2 = %.2f, df = 1, p %s",
    X2$statistic, G2, fmt_p(X2$p.value))
say("")

# ------------------------------------------------------------------ 3. Breslow-Day [Ch. 6.5]
stratum_or <- function(t) {
  o <- (t[1, 1] * t[2, 2]) / (t[1, 2] * t[2, 1]); s <- sqrt(sum(1 / t))
  c(or = o, lo = exp(log(o) - 1.96 * s), hi = exp(log(o) + 1.96 * s))
}
by_class <- b |> count(weight_class, gender, younger, result) |>
  complete(nesting(weight_class, gender), younger, result, fill = list(n = 0)) |>
  group_by(weight_class, gender) |> filter(min(n) >= MIN_STRATUM) |> ungroup()
classes <- as.character(unique(by_class$weight_class))
dropped <- setdiff(levels(b$weight_class), as.character(classes))
arr <- array(0, dim = c(2, 2, length(classes)), dimnames = list(c("Younger", "Older"), c("Won", "Lost"), classes))
for (cl in classes) {
  s <- filter(by_class, weight_class == cl)
  for (i in 1:2) for (j in 1:2)
    arr[i, j, cl] <- s$n[s$younger == c("Younger", "Older")[i] & s$result == c("Won", "Lost")[j]]
}
bd_cls <- breslow_day(arr); mh_cls <- mantelhaen.test(arr, correct = FALSE)
cls_tab <- bind_rows(lapply(classes, \(cl) {
  t <- arr[, , cl]
  tibble(weight_class = cl, gender = by_class$gender[by_class$weight_class == cl][1], bouts = sum(t),
         !!!as.list(stratum_or(t)))
})) |> arrange(gender, desc(bouts))
write_csv(cls_tab, file.path(OUT, "breslow_day_by_class.csv"))

arr_g <- table(b$younger, b$result, b$gender); storage.mode(arr_g) <- "double"
bd_g <- breslow_day(unclass(arr_g)); mh_g <- mantelhaen.test(arr_g, correct = FALSE)
say("3. BRESLOW-DAY TEST OF HOMOGENEOUS ASSOCIATION [Ch. 6.5]")
say("   Question: is the younger fighter's odds ratio the same in every weight class?")
say("   (Same design as notes Example 6.5.3, home-field advantage across MLB / NFL / NHL.)")
say("   Odds ratio (younger vs older winning) in each weight class:")
for (i in seq_len(nrow(cls_tab)))
  say("     %-24s %s  bouts %5d   OR %.2f (95%% CI %.2f-%.2f)", cls_tab$weight_class[i], cls_tab$gender[i],
      cls_tab$bouts[i], cls_tab$or[i], cls_tab$lo[i], cls_tab$hi[i])
if (length(dropped)) say("   Left out (a cell had < %d bouts): %s", MIN_STRATUM, paste(dropped, collapse = ", "))
say("   Mantel-Haenszel common OR = %.3f (95%% CI %.3f-%.3f); CMH test of no association X^2 = %.1f, p %s",
    bd_cls$theta_mh, mh_cls$conf.int[1], mh_cls$conf.int[2], mh_cls$statistic, fmt_p(mh_cls$p.value))
say("   Breslow-Day BD = %.2f, df = %d, p %s", bd_cls$BD, bd_cls$df, fmt_p(bd_cls$p))
say("   -> %s", ifelse(bd_cls$p > 0.05,
    "fail to reject homogeneity: the younger fighter's advantage is statistically the same in every division.",
    "reject homogeneity: the size of the younger fighter's advantage differs between divisions."))
say("   Men vs women: OR men %.2f, OR women %.2f; common OR %.3f; Breslow-Day BD = %.2f, df = 1, p %s",
    stratum_or(arr_g[, , "M"])[1], stratum_or(arr_g[, , "F"])[1], bd_g$theta_mh, bd_g$BD, fmt_p(bd_g$p))
say("")

# ------------------------------------------------------------------ 4. age band x result [Ch. 6.1-6.3]
b4 <- b |> filter(a_win == 1 | method_group %in% c("Decision", "KO/TKO", "Submission")) |>
  mutate(res4 = factor(case_when(a_win == 1 ~ "Won",
                                 method_group == "Decision" ~ "Lost: decision",
                                 method_group == "KO/TKO" ~ "Lost: KO/TKO",
                                 TRUE ~ "Lost: submission"),
                       levels = c("Won", "Lost: decision", "Lost: KO/TKO", "Lost: submission")),
         band = cut(a_age, c(-Inf, 25.999, 29.999, 33.999, 37.999, Inf),
                    labels = c("25 or under", "26-29", "30-33", "34-37", "38+")))
say("4. AGE BAND x FIGHT RESULT: test of independence and standardized residuals [Ch. 6.1-6.3]")
say("   %d bouts left out (losses by DQ, overturned, could not continue).", nrow(b) - nrow(b4))
res_long <- list()
for (g in c("M", "F")) {
  ct <- table(filter(b4, gender == g)$band, filter(b4, gender == g)$res4)
  ct <- ct[rowSums(ct) > 0, , drop = FALSE]
  cs <- suppressWarnings(chisq.test(ct, correct = FALSE))
  out <- cbind(band = rownames(ct), as.data.frame.matrix(ct),
               setNames(as.data.frame.matrix(round(cs$expected, 1)), paste0("expected_", colnames(ct))),
               setNames(as.data.frame.matrix(round(cs$stdres, 2)), paste0("stdres_", colnames(ct))))
  write_csv(out, file.path(OUT, sprintf("table_ageband_result_%s.csv", g)))
  say("   %s (n = %d):  X^2 = %.1f, G^2 = %.1f, df = %d, p %s%s", ifelse(g == "M", "MEN", "WOMEN"), sum(ct),
      cs$statistic, g2(ct), cs$parameter, fmt_p(cs$p.value),
      ifelse(any(cs$expected < 5), sprintf("  [%d expected counts < 5: treat with care]", sum(cs$expected < 5)), ""))
  say("     Row %% (observed)          Won   Dec.L   KO.L  Sub.L |  Std. residuals:  Won  Dec.L   KO.L  Sub.L")
  rp <- prop.table(ct, 1) * 100
  for (i in seq_len(nrow(ct)))
    say("     %-12s (n=%4d)  %5.1f  %5.1f  %5.1f  %5.1f |                %6.2f %6.2f %6.2f %6.2f",
        rownames(ct)[i], sum(ct[i, ]), rp[i, 1], rp[i, 2], rp[i, 3], rp[i, 4],
        cs$stdres[i, 1], cs$stdres[i, 2], cs$stdres[i, 3], cs$stdres[i, 4])
  res_long[[g]] <- as.data.frame(as.table(cs$stdres)) |> setNames(c("band", "result", "stdres")) |>
    mutate(gender = ifelse(g == "M", "Men", "Women"))
}
say("   |standardized residual| > 2 marks a cell with clearly more (+) or fewer (-) bouts than independence predicts.")
say("")

# ------------------------------------------------------------------ 5. logistic on age_diff [Ch. 8.4]
m1 <- glm(a_win ~ age_diff, data = b, family = binomial)
cf <- coef(m1); V <- vcov(m1); se <- sqrt(diag(V)); zb <- cf[2] / se[2]
G_m1 <- m1$null.deviance - m1$deviance
say("5. LOGISTIC REGRESSION: logit(pi(x)) = alpha + beta x, x = age difference in years [Ch. 8.4]")
say("   alpha-hat = %.5f (SE %.5f), beta-hat = %.5f (SE %.5f), Cov(alpha-hat, beta-hat) = %.3g",
    cf[1], se[1], cf[2], se[2], V[1, 2])
say("   95%% CI for beta: (%.5f, %.5f);  Wald z = %.2f, p %s", cf[2] - 1.96 * se[2], cf[2] + 1.96 * se[2],
    zb, fmt_p(2 * pnorm(-abs(zb))))
say("   Odds ratio per year e^beta = %.4f, 95%% CI (%.4f, %.4f): each extra year older multiplies the odds of winning by %.3f",
    exp(cf[2]), exp(cf[2] - 1.96 * se[2]), exp(cf[2] + 1.96 * se[2]), exp(cf[2]))
say("   G = null deviance - residual deviance = %.2f - %.2f = %.2f, df = 1, p %s",
    m1$null.deviance, m1$deviance, G_m1, fmt_p(pchisq(G_m1, 1, lower.tail = FALSE)))
say("   Confidence intervals for pi(x), computed as in notes 8.4.2 (SE from vcov, then inverse logit):")
say("        x (years)   pi-hat(x)   95%% CI")
for (xx in c(-10, -5, -2, 0, 2, 5, 10)) {
  lg <- cf[1] + cf[2] * xx; s <- sqrt(V[1, 1] + xx^2 * V[2, 2] + 2 * xx * V[1, 2])
  say("        %+5d        %.3f      (%.3f, %.3f)", xx, ilogit(lg), ilogit(lg - 1.96 * s), ilogit(lg + 1.96 * s))
}
say("")

# ------------------------------------------------------------------ 6. drop in deviance [Ch. 8.6.2]
diff_vars <- c("age_diff", "diff_prior_win_pct", "diff_ufc_experience", "diff_prior_fights",
               "diff_last3_win_pct", "diff_streak", "diff_prior_ko_losses", "diff_prior_finish_wins",
               "diff_layoff_days", "diff_debut_in_data",
               "diff_prior_sig_landed_pm", "diff_prior_sig_absorbed_pm", "diff_prior_sig_acc",
               "diff_prior_sig_def", "diff_prior_td_landed_p15", "diff_prior_td_acc",
               "diff_prior_td_def", "diff_prior_ctrl_share", "diff_prior_kd_p15", "diff_prior_sub_att_p15")
forms <- list(
  `Model 1` = a_win ~ age_diff,
  `Model 2` = a_win ~ age_diff + age_diff:weight_class,
  `Model 3` = a_win ~ age_diff + age_diff:weight_class + diff_prior_win_pct,
  `Model 4` = as.formula(paste("a_win ~ age_diff:weight_class +", paste(diff_vars, collapse = " + "))))
bb <- b[complete.cases(b[, c(diff_vars, "weight_class")]), ]
fits <- lapply(forms, \(f) glm(f, data = bb, family = binomial))
rd <- sapply(fits, deviance); k <- sapply(fits, \(f) sum(!is.na(coef(f))))
dd_rows <- list()
say("6. COMPARING MODELS BY DROP IN DEVIANCE [Ch. 8.6.2, same layout as Example 8.6.3]")
say("   Model 1: age difference | Model 2: + age difference x weight class | Model 3: + prior win %% difference")
say("   Model 4: + all other pre-fight differences (experience, streak, layoff, prior performance averages)")
say("   (%d bouts with every variable available)", nrow(bb))
for (i in seq_along(fits)) say("   %s: residual deviance = %.2f, parameters = %d", names(fits)[i], rd[i], k[i])
say("   Drop in deviance DD (df, p-value), row = smaller model, column = larger model:")
for (i in 1:(length(fits) - 1)) for (j in (i + 1):length(fits)) {
  DD <- rd[i] - rd[j]; df <- k[j] - k[i]; p <- pchisq(DD, df, lower.tail = FALSE)
  dd_rows[[length(dd_rows) + 1]] <- tibble(smaller = names(fits)[i], larger = names(fits)[j], DD = DD, df = df, p = p)
  say("     %s -> %s: DD = %7.2f, df = %2d, p %s  %s", names(fits)[i], names(fits)[j], DD, df, fmt_p(p),
      ifelse(p <= 0.1, "(larger model preferred, notes' rule of thumb p <= 0.1)", "(extra parameters not worth it)"))
}
write_csv(bind_rows(dd_rows), file.path(OUT, "drop_in_deviance.csv"))
say("")

# ------------------------------------------------------------------ 7. multinomial [Ch. 9]
say("7. MULTINOMIAL (BASELINE-CATEGORY) LOGISTIC REGRESSION [Ch. 9]")
say("   Response Y = fighter A's result: Won (baseline), Lost: decision, Lost: KO/TKO, Lost: submission")
say("   log(pi_i / pi_Won) = beta_0i + beta_1i * (A's age) + beta_2i * (opponent's age), fitted separately for men and women")
coef_rows <- list(); prob_rows <- list()
for (g in c("M", "F")) {
  d7 <- filter(b4, gender == g)
  mfull <- multinom(res4 ~ a_age + b_age, data = d7, trace = FALSE, maxit = 500)
  mnoA  <- multinom(res4 ~ b_age, data = d7, trace = FALSE, maxit = 500)
  mnull <- multinom(res4 ~ 1, data = d7, trace = FALSE, maxit = 500)
  sm <- summary(mfull); est <- sm$coefficients; ses <- sm$standard.errors
  DD_a <- deviance(mnoA) - deviance(mfull); DD_all <- deviance(mnull) - deviance(mfull)
  say("   %s (n = %d)", ifelse(g == "M", "MEN", "WOMEN"), nrow(d7))
  say("     Likelihood-ratio test, H0: own age has no effect on any logit: DD = %.2f, df = 3, p %s",
      DD_a, fmt_p(pchisq(DD_a, 3, lower.tail = FALSE)))
  say("     Likelihood-ratio test, H0: neither age matters: DD = %.2f, df = 6, p %s",
      DD_all, fmt_p(pchisq(DD_all, 6, lower.tail = FALSE)))
  say("     %-18s %-10s %9s %8s %7s  %-9s %s", "logit vs Won", "term", "estimate", "SE", "z", "p", "odds ratio per year (95% CI)")
  for (r in rownames(est)) for (term in c("a_age", "b_age")) {
    e <- est[r, term]; s <- ses[r, term]; z <- e / s
    say("     %-18s %-10s %9.4f %8.4f %7.2f  %-9s %.3f (%.3f-%.3f)", r,
        ifelse(term == "a_age", "own age", "opp. age"), e, s, z, fmt_p(2 * pnorm(-abs(z))),
        exp(e), exp(e - 1.96 * s), exp(e + 1.96 * s))
    coef_rows[[length(coef_rows) + 1]] <- tibble(gender = g, logit = paste(r, "vs Won"), term = term,
      estimate = e, se = s, z = z, p = 2 * pnorm(-abs(z)), odds_ratio = exp(e))
  }
  ages <- if (g == "M") seq(21, 41, 0.5) else seq(21, 37, 0.5)
  pr <- predict(mfull, newdata = data.frame(a_age = ages, b_age = 30), type = "probs")
  prob_rows[[g]] <- data.frame(age = ages, pr, check.names = FALSE) |>
    pivot_longer(-age, names_to = "result", values_to = "prob") |>
    mutate(gender = ifelse(g == "M", "Men", "Women"))
  at <- function(a) predict(mfull, newdata = data.frame(a_age = a, b_age = 30), type = "probs")
  p25 <- at(25); p35 <- at(35)
  say("     Estimated probabilities vs a 30-year-old opponent (notes 9.2):")
  say("        age 25: won %.1f%%, decision loss %.1f%%, KO/TKO loss %.1f%%, submission loss %.1f%%",
      100 * p25[1], 100 * p25[2], 100 * p25[3], 100 * p25[4])
  say("        age 35: won %.1f%%, decision loss %.1f%%, KO/TKO loss %.1f%%, submission loss %.1f%%",
      100 * p35[1], 100 * p35[2], 100 * p35[3], 100 * p35[4])
}
write_csv(bind_rows(coef_rows), file.path(OUT, "multinomial_coefficients.csv"))
say("   Reading the odds ratios: e.g. 'Lost: KO/TKO vs Won, own age, OR 1.08' = each extra year of the fighter's")
say("   age multiplies the odds of being knocked out (rather than winning) by 1.08, opponent's age held fixed.")

writeLines(rep, file.path(OUT, "categorical_report.txt"))
cat(rep, sep = "\n")

# ------------------------------------------------------------------ figures
theme_set(theme_minimal(base_size = 12))
p1 <- cls_tab |> mutate(label = sprintf("%s (n=%d)", weight_class, bouts),
                        label = reorder(label, or)) |>
  ggplot(aes(or, label, colour = ifelse(gender == "F", "Women", "Men"))) +
  geom_vline(xintercept = 1, colour = "grey60", linetype = "dashed") +
  geom_vline(xintercept = bd_cls$theta_mh, colour = "#1f5aa6", alpha = 0.5) +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0) + geom_point(size = 3) +
  scale_colour_manual(values = c(Men = "#1f5aa6", Women = "#d1495b"), name = NULL) +
  scale_x_log10() +
  labs(x = "Odds ratio: younger fighter winning vs older fighter (log scale, 95% CI)", y = NULL,
       title = "Is the younger fighter's advantage the same in every division?",
       subtitle = sprintf("Blue line = common OR %.2f (Mantel-Haenszel)\nBreslow-Day BD = %.1f, df = %d, p %s",
                          bd_cls$theta_mh, bd_cls$BD, bd_cls$df, fmt_p(bd_cls$p))) +
  theme(legend.position = "bottom")
ggsave(file.path(OUT, "fig_or_by_class.png"), p1, width = 9, height = 6, dpi = 200, bg = "white")

p2 <- bind_rows(res_long) |>
  ggplot(aes(result, band, fill = stdres)) + geom_tile(colour = "white") +
  geom_text(aes(label = sprintf("%.1f", stdres)), size = 3.6) +
  scale_fill_gradient2(low = "#1f5aa6", mid = "white", high = "#d1495b", midpoint = 0, name = "Std.\nresidual") +
  facet_wrap(~gender) +
  labs(x = NULL, y = "Fighter's age", title = "Age band x fight result: where does the association come from?",
       subtitle = "Standardized residuals (notes 6.3). Red = more bouts than expected, blue = fewer.\n|value| > 2 is notable.") +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))
ggsave(file.path(OUT, "fig_residuals_ageband.png"), p2, width = 10, height = 5, dpi = 200, bg = "white")

p3 <- bind_rows(prob_rows) |>
  mutate(result = factor(result, levels = c("Won", "Lost: decision", "Lost: KO/TKO", "Lost: submission"))) |>
  ggplot(aes(age, prob, colour = result)) + geom_line(linewidth = 1.1) +
  facet_wrap(~gender, scales = "free_x") +
  scale_colour_manual(values = c("Won" = "#2a9d8f", "Lost: decision" = "grey50",
                                 "Lost: KO/TKO" = "#d1495b", "Lost: submission" = "#e9a23b"), name = NULL) +
  scale_y_continuous(labels = scales::percent) +
  labs(x = "Fighter's age (opponent fixed at 30)", y = "Estimated probability",
       title = "How fights end as fighters age: multinomial logistic regression",
       subtitle = "Baseline-category logits (notes Ch. 9), baseline = Won") +
  theme(legend.position = "bottom")
ggsave(file.path(OUT, "fig_multinomial_probs.png"), p3, width = 10, height = 5.5, dpi = 200, bg = "white")
cat("\nSaved categorical_report.txt, 6 CSV tables and 3 figures to", OUT, "/\n")
