# models_win_prediction.R — predict win/loss before the fight and compare models
#
# Input : data/bouts_prefight.csv   (from build_prefight_features.py)
# Output: prediction/
#   model_comparison.csv       test-set accuracy, AUC, log loss, Brier score per model
#   rf_importance.csv          random forest variable importance (Gini + permutation)
#   fig_model_comparison.png   accuracy (95% CI) and AUC by model
#   fig_rf_importance.png      mean decrease in Gini and in accuracy
#   prediction_report.txt      everything above + odds ratios + nested-model tests
#
# Design
#   - One row per bout. Fighters randomly assigned to side A / B; target = A wins.
#     So a model with no information scores ~50%.
#   - Train on bouts before 2024, test on 2024 onward. Testing on LATER fights than
#     the model has seen is the honest version of "measure of accuracy on labeled data".
#   - Models (logistic regressions are nested, so each adds one idea):
#       M1  age only               a_win ~ age_diff
#       M2  age + weight class      + age_diff:weight_class  (does age matter more in some divisions?)
#       M3  age + class + record    + diff_prior_win_pct     (three explanatory variables)
#       M4  all pre-fight features  + experience, streak, layoff, prior performance averages ...
#       RF  random forest           same features as M4, plus each side's age
#   - Weight class enters as an INTERACTION with age, not on its own: both fighters are
#     in the same division, so the division alone can't say which of them wins.
#
# Usage (from ~/ufc_scraper):
#   Rscript models_win_prediction.R
# One-time setup:
#   Rscript -e 'install.packages(c("randomForest","dplyr","readr","tidyr","ggplot2"), repos="https://cloud.r-project.org")'

suppressPackageStartupMessages({
  library(randomForest); library(dplyr); library(readr); library(tidyr); library(ggplot2)
})
set.seed(2026)
OUT <- "prediction"; dir.create(OUT, showWarnings = FALSE)
TEST_FROM <- as.Date("2024-01-01")

b <- read_csv("data/bouts_prefight.csv", show_col_types = FALSE) |>
  mutate(event_date = as.Date(event_date),
         weight_class = factor(weight_class),
         gender = factor(gender),
         a_win_f = factor(ifelse(a_win == 1, "A", "B"), levels = c("B", "A")))
train <- b |> filter(event_date <  TEST_FROM)
test  <- b |> filter(event_date >= TEST_FROM)
# test-set classes must exist in train
test  <- test |> filter(weight_class %in% unique(train$weight_class)) |>
  mutate(weight_class = factor(weight_class, levels = levels(train$weight_class)))

diff_vars <- c("age_diff", "diff_prior_win_pct", "diff_ufc_experience", "diff_prior_fights",
               "diff_last3_win_pct", "diff_streak", "diff_prior_ko_losses", "diff_prior_finish_wins",
               "diff_layoff_days", "diff_debut_in_data",
               "diff_prior_sig_landed_pm", "diff_prior_sig_absorbed_pm", "diff_prior_sig_acc",
               "diff_prior_sig_def", "diff_prior_td_landed_p15", "diff_prior_td_acc",
               "diff_prior_td_def", "diff_prior_ctrl_share", "diff_prior_kd_p15", "diff_prior_sub_att_p15")

nice <- c(age_diff = "Age difference", a_age = "Age (side A)", b_age = "Age (side B)",
          weight_class = "Weight class", diff_prior_win_pct = "Prior win %",
          diff_ufc_experience = "UFC experience (fights)", diff_prior_fights = "Fights in data",
          diff_last3_win_pct = "Win % last 3", diff_streak = "Current streak",
          diff_prior_ko_losses = "Prior KO/TKO losses", diff_prior_finish_wins = "Prior finish wins",
          diff_layoff_days = "Layoff (days)", diff_debut_in_data = "UFC debut",
          diff_prior_sig_landed_pm = "Sig. strikes landed/min", diff_prior_sig_absorbed_pm = "Sig. strikes absorbed/min",
          diff_prior_sig_acc = "Sig. strike accuracy", diff_prior_sig_def = "Sig. strike defense",
          diff_prior_td_landed_p15 = "Takedowns/15 min", diff_prior_td_acc = "Takedown accuracy",
          diff_prior_td_def = "Takedown defense", diff_prior_ctrl_share = "Control time share",
          diff_prior_kd_p15 = "Knockdowns/15 min", diff_prior_sub_att_p15 = "Sub. attempts/15 min")

cat(sprintf("Train: %d bouts (%s to %s) | Test: %d bouts (%s to %s)\n",
            nrow(train), min(train$event_date), max(train$event_date),
            nrow(test), min(test$event_date), max(test$event_date)))

# ------------------------------------------------------------ logistic models
forms <- list(
  M1 = a_win ~ age_diff,
  M2 = a_win ~ age_diff + age_diff:weight_class,
  M3 = a_win ~ age_diff + age_diff:weight_class + diff_prior_win_pct,
  M4 = as.formula(paste("a_win ~ age_diff:weight_class +", paste(diff_vars, collapse = " + ")))
)
labels <- c(M1 = "M1 Logistic: age", M2 = "M2 Logistic: age + class",
            M3 = "M3 Logistic: age + class + record", M4 = "M4 Logistic: all features",
            RF = "RF Random forest: all features")
fits <- lapply(forms, \(f) glm(f, data = train, family = binomial))
probs <- lapply(fits, \(f) predict(f, newdata = test, type = "response"))

# -------------------------------------------------------------- random forest
rf_x <- c(diff_vars, "a_age", "b_age", "weight_class")
# choose mtry by out-of-bag error on the training data
mtrys <- c(2, 3, 4, 6, 8)
oob <- sapply(mtrys, \(m) {
  f <- randomForest(x = train[, rf_x], y = train$a_win_f, ntree = 300, mtry = m)
  tail(f$err.rate[, "OOB"], 1)
})
best_mtry <- mtrys[which.min(oob)]
rf <- randomForest(x = train[, rf_x], y = train$a_win_f, ntree = 1000, mtry = best_mtry,
                   importance = TRUE)
probs$RF <- predict(rf, newdata = test[, rf_x], type = "prob")[, "A"]

# ---------------------------------------------------------------- metrics
auc <- function(p, y) {                       # Mann-Whitney AUC
  r <- rank(p); n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
y <- test$a_win
metrics <- bind_rows(lapply(names(probs), \(m) {
  p <- pmin(pmax(probs[[m]], 1e-6), 1 - 1e-6)
  correct <- sum((p > 0.5) == (y == 1))
  ci <- binom.test(correct, length(y))$conf.int
  tibble(model = m, label = labels[[m]], accuracy = correct / length(y), acc_lo = ci[1], acc_hi = ci[2],
         auc = auc(p, y), log_loss = -mean(y * log(p) + (1 - y) * log(1 - p)),
         brier = mean((p - y)^2), n_test = length(y))
}))
# simple rule baselines
young_rule <- ifelse(test$age_diff < 0, 1, ifelse(test$age_diff > 0, 0, NA))
rec_rule   <- ifelse(test$diff_prior_win_pct > 0, 1, ifelse(test$diff_prior_win_pct < 0, 0, NA))
baselines <- c(
  sprintf("Coin flip: 50.0%%"),
  sprintf("Rule 'younger fighter wins': %.1f%% (on %d bouts with an age difference)",
          100 * mean(young_rule == y, na.rm = TRUE), sum(!is.na(young_rule))),
  sprintf("Rule 'better prior record wins': %.1f%% (on %d bouts where records differ)",
          100 * mean(rec_rule == y, na.rm = TRUE), sum(!is.na(rec_rule))))
write_csv(metrics, file.path(OUT, "model_comparison.csv"))

# McNemar: is the best model's accuracy really different from M1 (age only)?
best <- metrics$model[which.max(metrics$auc)]
c_best <- (probs[[best]] > 0.5) == (y == 1); c_m1 <- (probs$M1 > 0.5) == (y == 1)
mcn <- mcnemar.test(table(c_best, c_m1))

# Nested logistic models on the training data: does each addition help?
nest <- anova(fits$M1, fits$M2, fits$M3, fits$M4, test = "Chisq")

# ------------------------------------------------------------- importance
imp <- importance(rf)
imp_df <- tibble(variable = rownames(imp), gini = imp[, "MeanDecreaseGini"],
                 accuracy = imp[, "MeanDecreaseAccuracy"]) |>
  mutate(name = ifelse(variable %in% names(nice), nice[variable], variable)) |>
  arrange(desc(gini))
write_csv(imp_df, file.path(OUT, "rf_importance.csv"))

# ---------------------------------------------------------------- figures
blue <- "#2a78d6"; ink2 <- "#52514e"; grid <- "#e4e3df"
th <- theme_minimal(base_size = 10) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line(colour = grid), plot.title.position = "plot",
        axis.text = element_text(colour = ink2), strip.text = element_text(hjust = 0, face = "bold"))

cmp <- metrics |> mutate(label = factor(label, levels = rev(labels[model])))
p1 <- ggplot(cmp, aes(y = label)) +
  geom_vline(xintercept = 50, colour = ink2, linewidth = 0.4, linetype = "dashed") +
  geom_errorbarh(aes(xmin = 100 * acc_lo, xmax = 100 * acc_hi), height = 0, colour = blue, linewidth = 0.8) +
  geom_point(aes(x = 100 * accuracy), colour = blue, size = 2.6) +
  geom_text(aes(x = 100 * acc_hi, label = sprintf("%.1f%%  (AUC %.3f)", 100 * accuracy, auc)),
            hjust = -0.1, size = 3.2, colour = "#0b0b0b") +
  scale_x_continuous(limits = c(45, 72), breaks = seq(45, 70, 5), labels = \(v) paste0(v, "%")) +
  labs(x = "Test-set accuracy (95% CI)", y = NULL,
       title = "Predicting UFC winners before the fight",
       subtitle = sprintf("Trained on bouts before %s, tested on %d later bouts; dashed line = coin flip",
                          format(TEST_FROM, "%Y"), nrow(test))) + th
ggsave(file.path(OUT, "fig_model_comparison.png"), p1, width = 8, height = 3.6, dpi = 150, bg = "white")

top <- imp_df |> slice_max(gini, n = 15)
imp_long <- top |> select(name, gini, accuracy) |>
  pivot_longer(c(gini, accuracy), names_to = "measure", values_to = "value") |>
  mutate(measure = factor(measure, c("gini", "accuracy"),
                          c("Mean decrease in Gini", "Mean decrease in accuracy (permutation)")),
         name = factor(name, levels = rev(top$name)))
p2 <- ggplot(imp_long, aes(value, name)) +
  geom_col(fill = blue, width = 0.65) +
  facet_wrap(~measure, scales = "free_x") +
  labs(x = NULL, y = NULL, title = "Random forest: which pre-fight variables matter most",
       subtitle = sprintf("%d trees, mtry = %d; top 15 by mean decrease in Gini", rf$ntree, best_mtry)) + th
ggsave(file.path(OUT, "fig_rf_importance.png"), p2, width = 9, height = 4.8, dpi = 150, bg = "white")

# ---------------------------------------------------------------- report
or <- function(fit, terms) {
  ci <- suppressMessages(confint.default(fit))[terms, , drop = FALSE]
  sprintf("  %-28s OR = %.3f  [%.3f, %.3f]", terms, exp(coef(fit)[terms]), exp(ci[, 1]), exp(ci[, 2]))
}
report <- c(
  sprintf("Train: %d bouts before %s | Test: %d bouts from %s", nrow(train), TEST_FROM, nrow(test), TEST_FROM),
  "", "Baselines on the test set:", paste0("  ", baselines),
  "", "Model comparison on the test set:",
  capture.output(print(as.data.frame(metrics |> transmute(
    model = label, accuracy = sprintf("%.1f%% [%.1f-%.1f]", 100 * accuracy, 100 * acc_lo, 100 * acc_hi),
    AUC = round(auc, 3), log_loss = round(log_loss, 4), brier = round(brier, 4))), row.names = FALSE)),
  "", sprintf("McNemar test, %s vs M1 (age only): chi2 = %.2f, p = %.3g", best, mcn$statistic, mcn$p.value),
  "", "Nested logistic models on the training data (does each addition improve fit?):",
  capture.output(print(nest)),
  sprintf("AIC: M1 %.0f | M2 %.0f | M3 %.0f | M4 %.0f", AIC(fits$M1), AIC(fits$M2), AIC(fits$M3), AIC(fits$M4)),
  "", "M1 odds ratio per year of age difference (A minus B):", or(fits$M1, "age_diff"),
  "  (OR < 1: being older lowers the odds of winning)",
  "", "M3 odds ratios (age effect shown for the reference division; see M2 interaction terms for others):",
  or(fits$M3, c("age_diff", "diff_prior_win_pct")),
  "  diff_prior_win_pct is on a 0-1 scale: OR is for a 100-point record difference; per 10 points = OR^0.1",
  "", "Age effect by division (M2): odds ratio per year older, with 95% CI",
  {
    v <- vcov(fits$M2); cf <- coef(fits$M2)
    lv <- levels(train$weight_class)
    sapply(lv, \(w) {
      term <- paste0("age_diff:weight_class", w)
      if (term %in% names(cf) && !is.na(cf[term])) {
        est <- cf["age_diff"] + cf[term]
        se <- sqrt(v["age_diff", "age_diff"] + v[term, term] + 2 * v["age_diff", term])
      } else { est <- cf["age_diff"]; se <- sqrt(v["age_diff", "age_diff"]) }
      n <- sum(train$weight_class == w)
      sprintf("  %-22s OR = %.3f [%.3f, %.3f]  (n train = %d)", w, exp(est), exp(est - 1.96 * se),
              exp(est + 1.96 * se), n)
    })
  },
  "", sprintf("Random forest: %d trees, mtry = %d (OOB error by mtry: %s)", rf$ntree, best_mtry,
              paste(sprintf("%d=%.3f", mtrys, oob), collapse = ", ")),
  "Variable importance (sorted by mean decrease in Gini):",
  capture.output(print(as.data.frame(imp_df |> transmute(variable = name, gini = round(gini, 1),
                                                         accuracy = round(accuracy, 2))), row.names = FALSE)),
  "", "Note: Gini importance favors continuous variables with many split points; the permutation",
  "measure (mean decrease in accuracy) does not, so report both.")
writeLines(report, file.path(OUT, "prediction_report.txt"))
cat(paste(report, collapse = "\n"), "\n")
