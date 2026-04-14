# =============================================================================
# Bayesian Logistic Regression — Citation Impact Prediction
# DS 4420 Project, Nidhi Jadhav and Triya Basu 
#
# Model:   p(y=1 | x) = sigmoid(x'β)
# Prior:   βⱼ ~ N(0, σ²=25)  (weakly informative)
# Sampler: Metropolis–Hastings MCMC 
# Target:  high_impact  (1 = top-10% cited within publication year)
# =============================================================================

set.seed(42)

# ── 0. packages (base R + standard linear-algebra only) ──────────────────────
# No rstan / brms / rstanarm used. Only these utility packages:
library(ggplot2)   # plots
library(reshape2)  # melt for heatmap

# ── Prepare data ───────────────────────────────────────────────────
df <- read.csv("data/processed/dataset.csv")

FEATURE_COLS <- c(
  "log_num_authors",
  "log_num_institutions",
  "log_num_countries",
  "is_multi_institution",
  "is_international",
  "has_elite_affiliation",
  "has_industry",
  "has_gov_nonprofit",
  "has_us_institution",
  "is_open_access",
  "is_journal"
)
TARGET_COL <- "high_impact"

X_raw <- as.matrix(df[, FEATURE_COLS])
y     <- df[[TARGET_COL]]

# standardize (same as MLP)
col_means <- colMeans(X_raw)
col_sds   <- apply(X_raw, 2, sd)
col_sds[col_sds == 0] <- 1
X_scaled  <- sweep(sweep(X_raw, 2, col_means, "-"), 2, col_sds, "/")

# add intercept column
X <- cbind(intercept = 1, X_scaled)
p <- ncol(X)   # 12 (intercept + 11 features)
n <- nrow(X)

# 80 / 20 train-test split (same permutation seed as MLP)
idx       <- sample(n, size = floor(0.8 * n), replace = FALSE)
X_train   <- X[idx, ];   y_train <- y[idx]
X_test    <- X[-idx, ];  y_test  <- y[-idx]

cat(sprintf("Train: %d  |  Test: %d\n", nrow(X_train), nrow(X_test)))
cat(sprintf("Positive class (train): %.3f\n\n", mean(y_train)))

# ── 2. model helpers ─────────────────────────────────────────────────────────
sigmoid <- function(z) 1 / (1 + exp(-z))

# log-likelihood:  sum_i [ y_i * log(p_i) + (1-y_i) * log(1-p_i) ]
log_likelihood <- function(beta, X, y) {
  eta <- X %*% beta
  p   <- sigmoid(eta)
  p   <- pmax(p, 1e-10); p <- pmin(p, 1 - 1e-10)   # numerical safety
  sum(y * log(p) + (1 - y) * log(1 - p))
}

# log-prior:  βⱼ ~ N(0, prior_var)
PRIOR_VAR <- 25
log_prior <- function(beta) {
  -sum(beta^2) / (2 * PRIOR_VAR)
}

# log-posterior (unnormalised)
log_posterior <- function(beta, X, y) {
  log_likelihood(beta, X, y) + log_prior(beta)
}

# ── 3. Metropolis–Hastings MCMC ──────────────────────────────────────────────
# Random-walk MH: propose β* = β + ε,  ε ~ N(0, step_size² I)

N_ITER    <- 15000   # total iterations
BURNIN    <- 5000    # discard first 5 000
STEP_SIZE <- 0.025   # proposal std-dev (tuned for ~25% acceptance)

beta_chain  <- matrix(NA_real_, nrow = N_ITER, ncol = p)
colnames(beta_chain) <- colnames(X)

beta_current  <- rep(0, p)               # start at zero
lp_current    <- log_posterior(beta_current, X_train, y_train)
n_accepted    <- 0L

cat("Running MCMC...\n")
pb <- txtProgressBar(min = 0, max = N_ITER, style = 3)

for (iter in seq_len(N_ITER)) {
  # propose
  beta_proposed <- beta_current + rnorm(p, mean = 0, sd = STEP_SIZE)
  lp_proposed   <- log_posterior(beta_proposed, X_train, y_train)

  # accept / reject (log scale)
  log_alpha <- lp_proposed - lp_current
  if (log(runif(1)) < log_alpha) {
    beta_current <- beta_proposed
    lp_current   <- lp_proposed
    n_accepted   <- n_accepted + 1L
  }

  beta_chain[iter, ] <- beta_current
  setTxtProgressBar(pb, iter)
}
close(pb)

accept_rate <- n_accepted / N_ITER
cat(sprintf("\nAcceptance rate: %.3f  (target ≈ 0.20–0.40)\n\n", accept_rate))

# discard burn-in
posterior_samples <- beta_chain[(BURNIN + 1):N_ITER, ]

# ── 4. posterior summaries ───────────────────────────────────────────────────
post_mean   <- colMeans(posterior_samples)
post_sd     <- apply(posterior_samples, 2, sd)
ci_lower    <- apply(posterior_samples, 2, quantile, probs = 0.025)
ci_upper    <- apply(posterior_samples, 2, quantile, probs = 0.975)

summary_df <- data.frame(
  feature   = colnames(X),
  post_mean = round(post_mean, 4),
  post_sd   = round(post_sd,   4),
  ci_lower  = round(ci_lower,  4),
  ci_upper  = round(ci_upper,  4)
)
print(summary_df, row.names = FALSE)

# ── 5. convergence diagnostics ───────────────────────────────────────────────
# trace plots for 4 key coefficients
key_features <- c("has_elite_affiliation", "log_num_authors",
                  "is_international",      "is_open_access")

trace_df <- data.frame(
  iter = rep(seq_len(N_ITER - BURNIN), length(key_features)),
  value = as.vector(posterior_samples[, key_features]),
  feature = rep(key_features, each = N_ITER - BURNIN)
)

p_trace <- ggplot(trace_df, aes(x = iter, y = value)) +
  geom_line(alpha = 0.5, color = "#2c7fb8", linewidth = 0.3) +
  facet_wrap(~feature, scales = "free_y", ncol = 2) +
  labs(title = "MCMC Trace Plots (post burn-in)",
       x = "Iteration", y = "β value") +
  theme_minimal(base_size = 11)

ggsave("trace_plots.png", p_trace, width = 8, height = 5, dpi = 150)

# ── 6. posterior coefficient plot ────────────────────────────────────────────
coef_df <- summary_df[summary_df$feature != "intercept", ]
coef_df$feature <- factor(coef_df$feature, levels = coef_df$feature[order(coef_df$post_mean)])

p_coef <- ggplot(coef_df, aes(x = post_mean, y = feature)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  geom_errorbarh(aes(xmin = ci_lower, xmax = ci_upper),
                 height = 0.3, color = "#2c7fb8", linewidth = 0.8) +
  geom_point(size = 2.5, color = "#d7301f") +
  labs(title = "Posterior Means with 95% Credible Intervals",
       x = "Coefficient (β)", y = NULL) +
  theme_minimal(base_size = 11)

ggsave("posterior_coefs.png", p_coef, width = 7, height = 5, dpi = 150)

# ── 7. prediction & evaluation ───────────────────────────────────────────────
# use posterior mean as point estimate for prediction
beta_hat   <- post_mean
prob_test  <- sigmoid(X_test %*% beta_hat)
pred_test  <- as.integer(prob_test >= 0.5)

accuracy   <- mean(pred_test == y_test)
tp <- sum(pred_test == 1 & y_test == 1)
tn <- sum(pred_test == 0 & y_test == 0)
fp <- sum(pred_test == 1 & y_test == 0)
fn <- sum(pred_test == 0 & y_test == 1)

precision  <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
recall     <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
f1         <- ifelse((precision + recall) > 0,
                     2 * precision * recall / (precision + recall), 0)

cat("\n── Evaluation (posterior mean β̂) ──────────────────────────\n")
cat(sprintf("Accuracy  : %.4f\n", accuracy))
cat(sprintf("Precision : %.4f\n", precision))
cat(sprintf("Recall    : %.4f\n", recall))
cat(sprintf("F1 Score  : %.4f\n\n", f1))
cat("Confusion Matrix:\n")
cat(sprintf("              Pred 0   Pred 1\n"))
cat(sprintf("  Actual 0    %5d    %5d\n", tn, fp))
cat(sprintf("  Actual 1    %5d    %5d\n", fn, tp))

# ── 8. ROC curve & AUC ───────────────────────────────────────────────────────
roc_data <- function(probs, labels) {
  thresholds <- sort(unique(c(0, probs, 1)), decreasing = TRUE)
  tpr <- fpr <- numeric(length(thresholds))
  for (i in seq_along(thresholds)) {
    pred <- as.integer(probs >= thresholds[i])
    tpr[i] <- sum(pred == 1 & labels == 1) / max(sum(labels == 1), 1)
    fpr[i] <- sum(pred == 1 & labels == 0) / max(sum(labels == 0), 1)
  }
  list(fpr = fpr, tpr = tpr)
}

roc   <- roc_data(as.vector(prob_test), y_test)
# trapezoidal AUC
auc   <- sum(diff(roc$fpr) * (head(roc$tpr, -1) + tail(roc$tpr, -1)) / 2)
auc   <- abs(auc)   # ensure positive (direction)
cat(sprintf("\nAUC: %.4f\n", auc))

roc_df <- data.frame(fpr = roc$fpr, tpr = roc$tpr)
p_roc  <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
  geom_line(color = "#2c7fb8", linewidth = 1) +
  geom_abline(linetype = "dashed", color = "grey50") +
  annotate("text", x = 0.6, y = 0.15,
           label = sprintf("AUC = %.3f", auc), size = 4) +
  labs(title = "ROC Curve — Bayesian Logistic Regression",
       x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal(base_size = 11)

ggsave("roc_curve.png", p_roc, width = 6, height = 5, dpi = 150)

cat("\nSaved: trace_plots.png, posterior_coefs.png, roc_curve.png\n")
