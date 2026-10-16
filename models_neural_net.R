# models_neural_net.R - neural-network win prediction, compared with the logistic models
#                       (Dr. Johnson, Oct 15 2026 meeting: "another model to look at: neural network")
#
# Input : data/bouts_prefight.csv   (from build_prefight_features.py)
# Output: prediction/
#   nn_comparison.csv        test-set accuracy, AUC, log loss, Brier for NN and logistic models
#   nn_tuning.csv            cross-validated log loss for each network size / weight decay tried
#   fig_nn_comparison.png    accuracy (95% CI) and AUC, NN vs logistic
#   fig_nn_age_curve.png     win probability vs age difference: what the network learned vs logistic M1
#   nn_report.txt            everything above + McNemar tests vs logistic M3
#
# Design (same as models_win_prediction.R so the numbers are directly comparable)
#   - One row per bout, random sides, target = A wins. Train < 2024, test 2024+.
#   - Network: one hidden layer (feed-forward, logistic output), R package nnet.
#   - Three feature sets, each matched to a logistic model:
#       NN-age  : A's age, B's age                         vs  M1 (age difference only)
#       NN-M3   : ages, age_diff, prior win % diff, class  vs  M3 (age + class + record)
#       NN-all  : every pre-fight feature                  vs  M4 (all features)
#     NN-age asks: does a flexible model find anything about age that a straight line misses?
#   - Hidden units and weight decay chosen by 5-fold cross-validation on the TRAINING bouts only.
#   - Inputs standardised with training means/SDs. Each final network is the average of
#     10 networks with different random starting weights (single nnet fits are unstable).
#
# Usage (from ~/ufc_scraper):
#   Rscript models_neural_net.R          (takes a few minutes)
# One-time setup (nnet ships with R, but just in case):
#   Rscript -e 'install.packages(c("nnet","dplyr","readr","tidyr","ggplot2"), repos="https://cloud.r-project.org")'

suppressPackageStartupMessages({
  library(nnet); library(dplyr); library(readr); library(tidyr); library(ggplot2)
})
set.seed(2026)
OUT <- "prediction"; dir.create(OUT, showWarnings = FALSE)
TEST_FROM <- as.Date("2024-01-01")
SIZES  <- c(1, 2, 3, 5, 8)          # hidden units to try
DECAYS <- c(0.01, 0.1, 0.5, 1)      # weight decay (L2 penalty) to try
K <- 5                              # CV folds
N_ENSEMBLE <- 10                    # random restarts averaged in the final network

b <- read_csv("data/bouts_prefight.csv", show_col_types = FALSE) |>
  mutate(event_date = as.Date(event_date), weight_class = factor(weight_class))
train <- b |> filter(event_date <  TEST_FROM)
test  <- b |> filter(event_date >= TEST_FROM) |>
  filter(weight_class %in% unique(train$weight_class)) |>
  mutate(weight_class = factor(weight_class, levels = levels(train$weight_class)))

diff_vars <- c("age_diff", "diff_prior_win_pct", "diff_ufc_experience", "diff_prior_fights",
               "diff_last3_win_pct", "diff_streak", "diff_prior_ko_losses", "diff_prior_finish_wins",
               "diff_layoff_days", "diff_debut_in_data",
               "diff_prior_sig_landed_pm", "diff_prior_sig_absorbed_pm", "diff_prior_sig_acc",
               "diff_prior_sig_def", "diff_prior_td_landed_p15", "diff_prior_td_acc",
               "diff_prior_td_def", "diff_prior_ctrl_share", "diff_prior_kd_p15", "diff_prior_sub_att_p15")
sets <- list(
  `NN-age` = list(num = c("a_age", "b_age"), class = FALSE),
  `NN-M3`  = list(num = c("a_age", "b_age", "age_diff", "diff_prior_win_pct"), class = TRUE),
  `NN-all` = list(num = c("a_age", "b_age", diff_vars), class = TRUE))
logit_forms <- list(
  M1 = a_win ~ age_diff,
  M3 = a_win ~ age_diff + age_diff:weight_class + diff_prior_win_pct,
  M4 = as.formula(paste("a_win ~ age_diff:weight_class +", paste(diff_vars, collapse = " + "))))
pair <- c(`NN-age` = "M1", `NN-M3` = "M3", `NN-all` = "M4")

cat(sprintf("Train: %d bouts | Test: %d bouts\n", nrow(train), nrow(test)))

# ---------------------------------------------------------------- helpers
make_x <- function(d, s, ref) {                   # standardise with TRAINING stats
  x <- as.matrix(d[, s$num])
  x <- sweep(sweep(x, 2, ref$mu), 2, ref$sd, "/")
  if (s$class) x <- cbind(x, model.matrix(~ weight_class - 1, d))
  x
}
fit_nn <- function(x, y, size, decay, seed)
  { set.seed(seed); nnet(x, y, size = size, decay = decay, maxit = 1000, entropy = TRUE, trace = FALSE) }
logloss <- function(p, y) { p <- pmin(pmax(p, 1e-6), 1 - 1e-6); -mean(y * log(p) + (1 - y) * log(1 - p)) }
auc <- function(p, y) { r <- rank(p); n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0) }

# ------------------------------------------------- tune size & decay by 5-fold CV
folds <- sample(rep(1:K, length.out = nrow(train)))
grid <- expand.grid(size = SIZES, decay = DECAYS)
tuning <- list(); best <- list(); nn_prob <- list(); nets <- list(); refs <- list()
for (nm in names(sets)) {
  s <- sets[[nm]]
  cv <- sapply(seq_len(nrow(grid)), \(g) {
    mean(sapply(1:K, \(k) {
      tr <- train[folds != k, ]; va <- train[folds == k, ]
      ref <- list(mu = colMeans(tr[, s$num]), sd = apply(tr[, s$num], 2, sd))
      m <- fit_nn(make_x(tr, s, ref), tr$a_win, grid$size[g], grid$decay[g], seed = k)
      logloss(predict(m, make_x(va, s, ref)), va$a_win)
    }))
  })
  tuning[[nm]] <- cbind(model = nm, grid, cv_log_loss = cv)
  best[[nm]] <- grid[which.min(cv), ]
  cat(sprintf("%s: best size = %d, decay = %g (CV log loss %.4f)\n",
              nm, best[[nm]]$size, best[[nm]]$decay, min(cv)))
  # final: refit on all training bouts, average N_ENSEMBLE random starts
  ref <- list(mu = colMeans(train[, s$num]), sd = apply(train[, s$num], 2, sd)); refs[[nm]] <- ref
  xtr <- make_x(train, s, ref); xte <- make_x(test, s, ref)
  nets[[nm]] <- lapply(1:N_ENSEMBLE, \(r) fit_nn(xtr, train$a_win, best[[nm]]$size, best[[nm]]$decay, seed = 100 + r))
  nn_prob[[nm]] <- rowMeans(sapply(nets[[nm]], \(m) predict(m, xte)))
}
write_csv(bind_rows(tuning), file.path(OUT, "nn_tuning.csv"))

# ------------------------------------------------------- logistic comparisons
lg_prob <- lapply(logit_forms, \(f) predict(glm(f, data = train, family = binomial), newdata = test, type = "response"))
probs <- c(lg_prob, nn_prob)
labels <- c(M1 = "Logistic M1: age", M3 = "Logistic M3: age + class + record", M4 = "Logistic M4: all features",
            `NN-age` = "Neural net: ages only", `NN-M3` = "Neural net: M3 features", `NN-all` = "Neural net: all features")
y <- test$a_win
metrics <- bind_rows(lapply(names(probs), \(m) {
  p <- probs[[m]]; correct <- sum((p > 0.5) == (y == 1)); ci <- binom.test(correct, length(y))$conf.int
  tibble(model = m, label = labels[[m]], type = ifelse(grepl("^NN", m), "Neural network", "Logistic regression"),
         accuracy = correct / length(y), acc_lo = ci[1], acc_hi = ci[2],
         auc = auc(p, y), log_loss = logloss(p, y), brier = mean((p - y)^2), n_test = length(y))
}))
write_csv(metrics, file.path(OUT, "nn_comparison.csv"))

mcnemar_vs <- function(a, b) {               # do models a and b get different bouts right?
  ca <- (probs[[a]] > 0.5) == (y == 1); cb <- (probs[[b]] > 0.5) == (y == 1)
  t <- mcnemar.test(table(factor(ca, c(FALSE, TRUE)), factor(cb, c(FALSE, TRUE))))
  sprintf("   %-7s vs %-3s: %.1f%% vs %.1f%%  (%d bouts only %s right, %d only %s right), McNemar p = %.3f",
          a, b, 100 * mean(ca), 100 * mean(cb), sum(ca & !cb), a, sum(!ca & cb), b, t$p.value)
}

rep <- c("NEURAL NETWORK vs LOGISTIC REGRESSION - test set (2024+)",
  sprintf("Train %d bouts (< 2024), test %d bouts (2024+). One hidden layer (nnet), %d-fold CV tuning, %d-net ensemble.",
          nrow(train), nrow(test), K, N_ENSEMBLE), "",
  "Chosen network settings:",
  sapply(names(best), \(nm) sprintf("   %-7s hidden units = %d, weight decay = %g, inputs = %d",
                                    nm, best[[nm]]$size, best[[nm]]$decay, nets[[nm]][[1]]$n[1])),
  "", "Test-set results:",
  sprintf("   %-36s acc %5.1f%% (95%% CI %4.1f-%4.1f)  AUC %.3f  log loss %.4f  Brier %.4f",
          metrics$label, 100 * metrics$accuracy, 100 * metrics$acc_lo, 100 * metrics$acc_hi,
          metrics$auc, metrics$log_loss, metrics$brier),
  "", "Each network vs its matching logistic model (same inputs), and vs M3 (current best):",
  sapply(names(pair), \(nm) mcnemar_vs(nm, pair[[nm]])),
  mcnemar_vs("NN-all", "M3"),
  "", "How to read this:",
  "   - If a network does NOT beat its matching logistic model, the relationships are close to",
  "     linear on the logit scale, and the simpler, interpretable logistic model is preferred.",
  "   - NN-age vs M1 is the direct test of 'is the age effect linear?' from a different angle.",
  "   - Lower log loss / Brier = better-calibrated probabilities; AUC = ranking ability.")
writeLines(rep, file.path(OUT, "nn_report.txt"))
cat(rep, sep = "\n")

# ------------------------------------------------------------------ figures
metrics$label <- factor(metrics$label, levels = rev(labels))
p1 <- metrics |>
  select(label, type, accuracy, acc_lo, acc_hi, auc) |>
  pivot_longer(c(accuracy, auc), names_to = "measure", values_to = "value") |>
  mutate(lo = ifelse(measure == "accuracy", acc_lo, NA), hi = ifelse(measure == "accuracy", acc_hi, NA),
         measure = recode(measure, accuracy = "Test accuracy (95% CI)", auc = "Test AUC")) |>
  ggplot(aes(value, label, colour = type)) +
  geom_vline(xintercept = 0.5, colour = "grey70", linetype = "dashed") +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0, na.rm = TRUE) +
  geom_point(size = 3) +
  facet_wrap(~measure, scales = "free_x") +
  scale_colour_manual(values = c("Logistic regression" = "#1f5aa6", "Neural network" = "#d1495b"), name = NULL) +
  labs(x = NULL, y = NULL, title = "Neural network vs logistic regression: predicting the winner",
       subtitle = sprintf("Trained on bouts before 2024, tested on %d bouts from 2024 on", nrow(test))) +
  theme_minimal(base_size = 12) + theme(legend.position = "bottom")
ggsave(file.path(OUT, "fig_nn_comparison.png"), p1, width = 10, height = 4.5, dpi = 200, bg = "white")

# What did NN-age learn about age? Win prob vs age difference, opponent fixed at 30.
m1 <- glm(logit_forms$M1, data = train, family = binomial)
g <- data.frame(a_age = seq(20, 42, by = 0.25), b_age = 30) |> mutate(age_diff = a_age - b_age)
xg <- make_x(g, sets$`NN-age`, refs$`NN-age`)
curve <- bind_rows(
  data.frame(age_diff = g$age_diff, p = rowMeans(sapply(nets$`NN-age`, \(m) predict(m, xg))), model = "Neural net (ages only)"),
  data.frame(age_diff = g$age_diff, p = predict(m1, g, type = "response"), model = "Logistic M1 (straight line on logit scale)"))
p2 <- ggplot(curve, aes(age_diff, p, colour = model)) +
  geom_hline(yintercept = 0.5, colour = "grey75", linetype = "dashed") +
  geom_line(linewidth = 1) +
  scale_colour_manual(values = c("#d1495b", "#1f5aa6"), name = NULL) +
  scale_y_continuous(labels = scales::percent) +
  labs(x = "Age difference (fighter minus a 30-year-old opponent, years)", y = "Predicted win probability",
       title = "Does a flexible model find a non-linear age effect?",
       subtitle = "Opponent age fixed at 30. Curves that overlap = the linear (logistic) age effect is adequate.") +
  theme_minimal(base_size = 12) + theme(legend.position = "bottom")
ggsave(file.path(OUT, "fig_nn_age_curve.png"), p2, width = 9, height = 5.5, dpi = 200, bg = "white")
cat("\nSaved nn_comparison.csv, nn_tuning.csv, nn_report.txt and 2 figures to", OUT, "/\n")
