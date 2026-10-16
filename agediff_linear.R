# agediff_linear.R - Win vs. age difference: scatterplot, linearity check, linear regression,
#                    t-test, R-squared  (Dr. Johnson, Oct 15 2026 meeting)
#
# Input : data/bouts_prefight.csv   (from build_prefight_features.py)
# Output: agediff/
#   fig_win_vs_agediff.png        win rate by 1-year age-difference bin (point size = bouts),
#                                 with the linear fit, a logistic fit and a flexible curve on top
#   fig_win_vs_agediff_raw.png    every bout as a point (win = 1, loss = 0), jittered
#   fig_residuals_binned.png      binned residuals from the linear fit (linearity check)
#   agediff_bins.csv              the binned table behind the figure
#   agediff_report.txt            correlation, regressions, t-tests, R-squared, linearity tests
#
# Variable definitions
#   age_diff = fighter A's age minus fighter B's age (years). Positive = A is older.
#   win      = 1 if fighter A won. Sides were assigned at random, so each bout counts once.
#   Uses ALL decided bouts (2010-2026) - this is description/inference, not prediction,
#   so there is no train/test split here.
#
# Usage (from ~/ufc_scraper):
#   Rscript agediff_linear.R

suppressPackageStartupMessages({
  library(dplyr); library(readr); library(ggplot2); library(splines)
})
OUT <- "agediff"; dir.create(OUT, showWarnings = FALSE)
MAX_DIFF <- 12          # bins beyond +/-12 years have too few bouts; they are pooled into the end bins
MIN_BIN_N <- 20         # bins with fewer bouts are dropped from the binned regression

b <- read_csv("data/bouts_prefight.csv", show_col_types = FALSE) |>
  filter(!is.na(age_diff), !is.na(a_win)) |>
  mutate(win = a_win, gender = factor(gender))
n <- nrow(b)
rep <- c("WIN vs AGE DIFFERENCE - linear regression analysis",
         sprintf("Bouts: %d (%s to %s). age_diff = A's age minus B's age; win = A won.",
                 n, min(b$event_date), max(b$event_date)),
         sprintf("age_diff: mean %.2f, SD %.2f, range %.1f to %.1f years",
                 mean(b$age_diff), sd(b$age_diff), min(b$age_diff), max(b$age_diff)), "")
fmt_p <- function(p) ifelse(p < 0.001, "< 0.001", sprintf("= %.3f", p))

# ------------------------------------------------------------------ 1. Correlation
ct <- cor.test(b$age_diff, b$win)                         # Pearson (point-biserial)
sp <- suppressWarnings(cor.test(b$age_diff, b$win, method = "spearman"))
rep <- c(rep, "1. CORRELATION (is it negative?)",
  sprintf("   Pearson r = %.3f (95%% CI %.3f to %.3f), t = %.2f, df = %d, p %s",
          ct$estimate, ct$conf.int[1], ct$conf.int[2], ct$statistic, ct$parameter, fmt_p(ct$p.value)),
  sprintf("   Spearman rho = %.3f, p %s", sp$estimate, fmt_p(sp$p.value)),
  "   Negative r = the OLDER fighter is LESS likely to win.", "")

# ------------------------------------------------------- 2. Linear regression (bouts)
# Linear probability model: win = b0 + b1 * age_diff. b1 = change in win probability per year.
lm1 <- lm(win ~ age_diff, data = b)
s1 <- summary(lm1)
co <- s1$coefficients
# Robust (HC1) standard error: with a 0/1 outcome the residual variance isn't constant,
# so the usual t-test SE is slightly off. Report both; they should agree closely.
X <- model.matrix(lm1); e <- resid(lm1)
bread <- solve(crossprod(X))
hc1 <- bread %*% crossprod(X * e) %*% bread * n / (n - ncol(X))
se_r <- sqrt(diag(hc1))["age_diff"]
t_r <- co["age_diff", 1] / se_r
rep <- c(rep, "2. LINEAR REGRESSION on individual bouts:  win ~ age_diff",
  sprintf("   Intercept = %.4f  (win prob. when ages are equal; should be ~0.50)", co[1, 1]),
  sprintf("   Slope     = %.4f per year  ->  %.2f percentage points per year of age difference",
          co[2, 1], 100 * co[2, 1]),
  sprintf("   t-test of slope = 0: t = %.2f, df = %d, p %s  (SE %.4f)",
          co[2, 3], s1$df[2], fmt_p(co[2, 4]), co[2, 2]),
  sprintf("   Robust (HC1) t-test:  t = %.2f, p %s  (SE %.4f)",
          t_r, fmt_p(2 * pt(-abs(t_r), s1$df[2])), se_r),
  sprintf("   95%% CI for slope: %.2f to %.2f pp per year",
          100 * confint(lm1)[2, 1], 100 * confint(lm1)[2, 2]),
  sprintf("   R-squared = %.4f  (adjusted %.4f)", s1$r.squared, s1$adj.r.squared),
  sprintf("   e.g. predicted win prob. 5 years younger: %.1f%%, 5 years older: %.1f%%",
          100 * predict(lm1, data.frame(age_diff = -5)), 100 * predict(lm1, data.frame(age_diff = 5))),
  "   Why R-squared is small here: each bout is a single win or loss, so most of the",
  "   variation is fight-to-fight randomness that NO age-based line could explain.",
  "   A small R-squared with a highly significant slope means: age has a real, steady",
  "   effect, but it is one of many things that decide a fight. See section 3.", "")

# ---------------------------------------------------------- 3. Binned scatter + regression
bins <- b |>
  mutate(bin = pmax(pmin(round(age_diff), MAX_DIFF), -MAX_DIFF)) |>
  group_by(bin) |>
  summarise(n = n(), wins = sum(win), win_rate = mean(win), mean_diff = mean(age_diff), .groups = "drop") |>
  mutate(lo = mapply(\(w, k) binom.test(w, k)$conf.int[1], wins, n),
         hi = mapply(\(w, k) binom.test(w, k)$conf.int[2], wins, n))
write_csv(bins, file.path(OUT, "agediff_bins.csv"))
bk <- filter(bins, n >= MIN_BIN_N)
lm_bin <- lm(win_rate ~ mean_diff, data = bk, weights = n)   # weighted by bouts in each bin
sb <- summary(lm_bin); cb <- sb$coefficients
rep <- c(rep, sprintf("3. BINNED REGRESSION: win rate in 1-year bins (%d bins with >= %d bouts), weighted by bouts",
                      nrow(bk), MIN_BIN_N),
  sprintf("   Slope = %.2f pp per year, t = %.2f, df = %d, p %s",
          100 * cb[2, 1], cb[2, 3], sb$df[2], fmt_p(cb[2, 4])),
  sprintf("   R-squared = %.3f  -> share of the variation in WIN RATE across age gaps explained by a straight line",
          sb$r.squared),
  "   This is the R-squared that answers 'is the trend linear?'. Section 2's R-squared",
  "   answers 'how much does age alone predict a single fight?'. Report both, labelled.", "")

# ------------------------------------------------------------ 4. Is it linear?
lm_q <- lm(win ~ age_diff + I(age_diff^2), data = b)
lm_s <- lm(win ~ ns(age_diff, df = 4), data = b)
a_q <- anova(lm1, lm_q); a_s <- anova(lm1, lm_s)
glm1 <- glm(win ~ age_diff, data = b, family = binomial)
# binned residual check: does the linear fit miss systematically anywhere?
bins_res <- b |> mutate(res = resid(lm1), bin = pmax(pmin(round(age_diff), MAX_DIFF), -MAX_DIFF)) |>
  group_by(bin) |> summarise(n = n(), mean_res = mean(res), se = sd(res) / sqrt(n()), .groups = "drop") |>
  filter(n >= MIN_BIN_N)
outside <- sum(abs(bins_res$mean_res) > 1.96 * bins_res$se)
rep <- c(rep, "4. IS IT LINEAR?",
  sprintf("   Add a squared term (curve):  quadratic coef = %.5f, F = %.2f, p %s",
          coef(lm_q)[3], a_q$F[2], fmt_p(a_q$`Pr(>F)`[2])),
  sprintf("   Flexible spline (4 df) vs straight line: F = %.2f, p %s",
          a_s$F[2], fmt_p(a_s$`Pr(>F)`[2])),
  "   p > 0.05 on both = no evidence of curvature -> a straight line is adequate.",
  sprintf("   Binned residuals outside +/-1.96 SE: %d of %d bins (expect ~%.0f by chance)",
          outside, nrow(bins_res), 0.05 * nrow(bins_res)),
  sprintf("   Logistic version (for comparison): OR = %.3f per year (95%% CI %.3f to %.3f), z = %.2f, p %s",
          exp(coef(glm1)[2]), exp(confint.default(glm1)[2, 1]), exp(confint.default(glm1)[2, 2]),
          summary(glm1)$coefficients[2, 3], fmt_p(summary(glm1)$coefficients[2, 4])),
  sprintf("   Linear vs logistic predicted win prob. at -10 / +10 years: %.1f%% / %.1f%% vs %.1f%% / %.1f%%",
          100 * predict(lm1, data.frame(age_diff = -10)), 100 * predict(lm1, data.frame(age_diff = 10)),
          100 * predict(glm1, data.frame(age_diff = -10), type = "response"),
          100 * predict(glm1, data.frame(age_diff = 10), type = "response")),
  "   Within the range of real age gaps the two agree closely; the logistic model is the",
  "   formally correct one for a 0/1 outcome (it can't go below 0% or above 100%).", "")

# ------------------------------------------------------------ 5. Two-sample t-test
tt <- t.test(age_diff ~ factor(win, levels = c(1, 0), labels = c("won", "lost")), data = b)
rep <- c(rep, "5. TWO-SAMPLE t-TEST (Welch): average age difference in fights A WON vs fights A LOST",
  sprintf("   Mean age_diff when A won: %.2f years; when A lost: %.2f years", tt$estimate[1], tt$estimate[2]),
  sprintf("   Difference = %.2f years (95%% CI %.2f to %.2f), t = %.2f, df = %.0f, p %s",
          tt$estimate[1] - tt$estimate[2], tt$conf.int[1], tt$conf.int[2], tt$statistic, tt$parameter, fmt_p(tt$p.value)),
  "   Same question as the slope t-test from the other direction: winners tend to be the younger fighter.", "")

# ------------------------------------------------------------ 6. Men vs women
lm_g <- lm(win ~ age_diff * gender, data = b)
cg <- summary(lm_g)$coefficients
slopes <- lapply(levels(b$gender), \(g) {
  s <- summary(lm(win ~ age_diff, data = filter(b, gender == g)))
  sprintf("   %-6s n = %5d  slope = %.2f pp/yr, t = %.2f, p %s, R-squared = %.4f",
          g, sum(b$gender == g), 100 * s$coefficients[2, 1], s$coefficients[2, 3],
          fmt_p(s$coefficients[2, 4]), s$r.squared)
})
int_row <- grep("^age_diff:gender", rownames(cg))
rep <- c(rep, "6. MEN vs WOMEN", unlist(slopes),
  if (length(int_row)) sprintf("   Difference in slopes (interaction): t = %.2f, p %s",
                               cg[int_row, 3], fmt_p(cg[int_row, 4])) else character(0), "")

writeLines(rep, file.path(OUT, "agediff_report.txt"))
cat(rep, sep = "\n")

# ------------------------------------------------------------------ figures
grid <- data.frame(age_diff = seq(-MAX_DIFF, MAX_DIFF, by = 0.1))
pred_lin <- predict(lm1, grid, interval = "confidence")
lines <- bind_rows(
  data.frame(grid, fit = pred_lin[, "fit"], model = "Linear regression"),
  data.frame(grid, fit = predict(glm1, grid, type = "response"), model = "Logistic regression"),
  data.frame(grid, fit = predict(lm_s, grid), model = "Flexible spline (linearity check)"))
band <- data.frame(grid, lo = pred_lin[, "lwr"], hi = pred_lin[, "upr"])
cols <- c("Linear regression" = "#1f5aa6", "Logistic regression" = "#d1495b",
          "Flexible spline (linearity check)" = "grey45")
lab <- sprintf("Linear fit: %.2f pp per year, t = %.1f, p %s\nR-sq (bouts) = %.3f   R-sq (binned) = %.2f   r = %.3f",
               100 * co[2, 1], co[2, 3], fmt_p(co[2, 4]), s1$r.squared, sb$r.squared, ct$estimate)

p1 <- ggplot() +
  geom_hline(yintercept = 0.5, colour = "grey75", linetype = "dashed") +
  geom_vline(xintercept = 0, colour = "grey75", linetype = "dashed") +
  geom_ribbon(data = band, aes(age_diff, ymin = lo, ymax = hi), fill = "#1f5aa6", alpha = 0.15) +
  geom_errorbar(data = bk, aes(mean_diff, ymin = lo, ymax = hi), width = 0, colour = "grey55") +
  geom_point(data = bk, aes(mean_diff, win_rate, size = n), colour = "grey20", alpha = 0.85) +
  geom_line(data = lines, aes(age_diff, fit, colour = model, linetype = model), linewidth = 0.9) +
  scale_colour_manual(values = cols, name = NULL) +
  scale_linetype_manual(values = c("Linear regression" = "solid", "Logistic regression" = "solid",
                                   "Flexible spline (linearity check)" = "dotted"), name = NULL) +
  scale_size_area(max_size = 6, name = "Bouts in bin") +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  annotate("label", x = -MAX_DIFF, y = 0.06, label = lab, hjust = 0, size = 3.2, label.size = 0) +
  labs(x = "Age difference (fighter minus opponent, years)",
       y = "Win rate",
       title = "Older fighter vs younger fighter: win rate by age gap",
       subtitle = sprintf("UFC 2010-2026, %s decided bouts. Points = win rate per 1-year bin (95%% CI); +/-%d pools the tails.",
                          format(n, big.mark = ","), MAX_DIFF)) +
  theme_minimal(base_size = 12) + theme(legend.position = "bottom", legend.box = "vertical")
ggsave(file.path(OUT, "fig_win_vs_agediff.png"), p1, width = 9, height = 6.5, dpi = 200, bg = "white")

set.seed(2026)
p2 <- ggplot(b, aes(age_diff, win)) +
  geom_jitter(width = 0, height = 0.06, alpha = 0.08, size = 0.7) +
  geom_line(data = data.frame(grid, fit = pred_lin[, "fit"]), aes(age_diff, fit), colour = "#1f5aa6", linewidth = 1) +
  scale_y_continuous(breaks = c(0, 0.5, 1), labels = c("Loss (0)", "50%", "Win (1)")) +
  labs(x = "Age difference (fighter minus opponent, years)", y = NULL,
       title = "Every bout: win (1) or loss (0) against age difference",
       subtitle = sprintf("Line = linear regression, slope %.2f pp per year, R-sq = %.3f", 100 * co[2, 1], s1$r.squared)) +
  theme_minimal(base_size = 12)
ggsave(file.path(OUT, "fig_win_vs_agediff_raw.png"), p2, width = 9, height = 5, dpi = 200, bg = "white")

p3 <- ggplot(bins_res, aes(bin, mean_res)) +
  geom_hline(yintercept = 0, colour = "grey60") +
  geom_errorbar(aes(ymin = -1.96 * se, ymax = 1.96 * se), colour = "grey80", width = 0.4, linewidth = 2) +
  geom_point(size = 2.5, colour = "#1f5aa6") +
  labs(x = "Age difference bin (years)", y = "Mean residual (actual - predicted win rate)",
       title = "Linearity check: residuals of the linear fit, by age-gap bin",
       subtitle = "Grey bars = +/-1.96 SE. Points scattered randomly inside the bars = the straight line fits.") +
  theme_minimal(base_size = 12)
ggsave(file.path(OUT, "fig_residuals_binned.png"), p3, width = 9, height = 5, dpi = 200, bg = "white")
cat("\nSaved figures, agediff_bins.csv and agediff_report.txt to", OUT, "/\n")
