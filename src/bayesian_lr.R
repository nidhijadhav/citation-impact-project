# Bayesian Logistic Regression — Citation Impact Prediction
# DS 4420 Project, Nidhi Jadhav and Triya Basu
# We use Bayesian logistic regression to predict whether a paper
# falls in the top 10% of citations within its publication year.
# The posterior for the coefficients is sampled using
# Metropolis-Hastings MCMC.

set.seed(42)

# plotting libraries
library(ggplot2)
library(reshape2)

# load data
df <- read.csv("data/processed/papers.csv")

feature_cols <- c(
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

target_col <- "high_impact"

x_raw <- as.matrix(df[, feature_cols])
y <- df[[target_col]]

# standardize predictors
feature_means <- colMeans(x_raw)
feature_sds <- apply(x_raw, 2, sd)
feature_sds[feature_sds == 0] <- 1

x_scaled <- sweep(sweep(x_raw, 2, feature_means, "-"), 2, feature_sds, "/")

# add intercept column
x <- cbind(intercept = 1, x_scaled)
n <- nrow(x)
p <- ncol(x)

# train-test split
train_idx <- sample(n, size = floor(0.8 * n), replace = FALSE)

x_train <- x[train_idx, ]
y_train <- y[train_idx]

x_test <- x[-train_idx, ]
y_test <- y[-train_idx]

cat(sprintf("Train: %d | Test: %d\n", nrow(x_train), nrow(x_test)))
cat(sprintf("Positive class proportion (train): %.3f\n\n", mean(y_train)))

# helper function
sigmoid_fn <- function(z) {
  1 / (1 + exp(-z))
}

# log-likelihood for logistic regression
log_lik <- function(beta, x, y) {
  eta <- x %*% beta
  probs <- sigmoid_fn(eta)

  probs <- pmax(probs, 1e-10)
  probs <- pmin(probs, 1 - 1e-10)

  sum(y * log(probs) + (1 - y) * log(1 - probs))
}

# Gaussian prior: beta_j ~ N(0, 25)
prior_var <- 25

log_prior <- function(beta) {
  -sum(beta^2) / (2 * prior_var)
}

# log-posterior
log_post <- function(beta, x, y) {
  log_lik(beta, x, y) + log_prior(beta)
}

# MCMC settings
n_iter <- 15000
burn_in <- 5000
step_size <- 0.025

beta_samples <- matrix(NA_real_, nrow = n_iter, ncol = p)
colnames(beta_samples) <- colnames(x)

beta_curr <- rep(0, p)
log_post_curr <- log_post(beta_curr, x_train, y_train)
n_accept <- 0L
pb <- txtProgressBar(min = 0, max = n_iter, style = 3)

for (iter in seq_len(n_iter)) {
  beta_prop <- beta_curr + rnorm(p, mean = 0, sd = step_size)
  log_post_prop <- log_post(beta_prop, x_train, y_train)

  log_alpha <- log_post_prop - log_post_curr

  if (log(runif(1)) < log_alpha) {
    beta_curr <- beta_prop
    log_post_curr <- log_post_prop
    n_accept <- n_accept + 1L
  }

  beta_samples[iter, ] <- beta_curr
  setTxtProgressBar(pb, iter)
}

close(pb)

accept_rate <- n_accept / n_iter
cat(sprintf("\nAcceptance rate: %.3f (target ≈ 0.20–0.40)\n\n", accept_rate))

# remove burn-in
post_samples <- beta_samples[(burn_in + 1):n_iter, ]

# posterior summary stats
post_mean <- colMeans(post_samples)
post_sd <- apply(post_samples, 2, sd)
ci_lower <- apply(post_samples, 2, quantile, probs = 0.025)
ci_upper <- apply(post_samples, 2, quantile, probs = 0.975)

summary_df <- data.frame(
  feature = colnames(x),
  post_mean = round(post_mean, 4),
  post_sd = round(post_sd, 4),
  ci_lower = round(ci_lower, 4),
  ci_upper = round(ci_upper, 4)
)

print(summary_df, row.names = FALSE)

# trace plots for selected coefficients
selected_features <- c(
  "has_elite_affiliation",
  "log_num_authors",
  "is_international",
  "is_open_access"
)

trace_df <- data.frame(
  iter = rep(seq_len(n_iter - burn_in), length(selected_features)),
  value = as.vector(post_samples[, selected_features]),
  feature = rep(selected_features, each = n_iter - burn_in)
)

trace_plot <- ggplot(trace_df, aes(x = iter, y = value)) +
  geom_line(alpha = 0.5, color = "#2c7fb8", linewidth = 0.3) +
  facet_wrap(~feature, scales = "free_y", ncol = 2) +
  labs(
    title = "MCMC Trace Plots - Post Burn-in",
    x = "Iteration",
    y = expression(beta)
  ) +
  theme_minimal(base_size = 11)

ggsave("results/blr_trace_plots.png", trace_plot, width = 8, height = 5, dpi = 150)

# coefficient plot with 95% credible intervals
coef_df <- summary_df[summary_df$feature != "intercept", ]
coef_df$feature <- factor(
  coef_df$feature,
  levels = coef_df$feature[order(coef_df$post_mean)]
)

coef_plot <- ggplot(coef_df, aes(x = post_mean, y = feature)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  geom_errorbarh(
    aes(xmin = ci_lower, xmax = ci_upper),
    height = 0.3,
    color = "#2c7fb8",
    linewidth = 0.8
  ) +
  geom_point(size = 2.5, color = "#d7301f") +
  labs(
    title = "Posterior Means with 95% Credible Intervals",
    x = expression(beta),
    y = NULL
  ) +
  theme_minimal(base_size = 11)

ggsave("results/blr_posterior_coefs.png", coef_plot, width = 7, height = 5, dpi = 150)

# prediction on test set using posterior mean
beta_hat <- post_mean
test_probs <- sigmoid_fn(x_test %*% beta_hat)
test_preds <- as.integer(test_probs >= 0.5)

accuracy <- mean(test_preds == y_test)

tp <- sum(test_preds == 1 & y_test == 1)
tn <- sum(test_preds == 0 & y_test == 0)
fp <- sum(test_preds == 1 & y_test == 0)
fn <- sum(test_preds == 0 & y_test == 1)

precision <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
recall <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
f1_score <- ifelse((precision + recall) > 0,
                   2 * precision * recall / (precision + recall), 0)

cat("\n--- Evaluation using posterior mean ---\n")
cat(sprintf("Accuracy  : %.4f\n", accuracy))
cat(sprintf("Precision : %.4f\n", precision))
cat(sprintf("Recall    : %.4f\n", recall))
cat(sprintf("F1 Score  : %.4f\n\n", f1_score))

cat("Confusion Matrix:\n")
cat(sprintf("              Pred 0   Pred 1\n"))
cat(sprintf("  Actual 0    %5d    %5d\n", tn, fp))
cat(sprintf("  Actual 1    %5d    %5d\n", fn, tp))

# ROC + AUC
get_roc_data <- function(probs, labels) {
  thresholds <- sort(unique(c(0, probs, 1)), decreasing = TRUE)
  tpr <- numeric(length(thresholds))
  fpr <- numeric(length(thresholds))

  for (i in seq_along(thresholds)) {
    pred <- as.integer(probs >= thresholds[i])

    tpr[i] <- sum(pred == 1 & labels == 1) / max(sum(labels == 1), 1)
    fpr[i] <- sum(pred == 1 & labels == 0) / max(sum(labels == 0), 1)
  }

  list(fpr = fpr, tpr = tpr)
}

roc_vals <- get_roc_data(as.vector(test_probs), y_test)
auc <- sum(diff(roc_vals$fpr) *
             (head(roc_vals$tpr, -1) + tail(roc_vals$tpr, -1)) / 2)
auc <- abs(auc)

cat(sprintf("\nAUC: %.4f\n", auc))

roc_df <- data.frame(fpr = roc_vals$fpr, tpr = roc_vals$tpr)

roc_plot <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
  geom_line(color = "#2c7fb8", linewidth = 1) +
  geom_abline(linetype = "dashed", color = "grey50") +
  annotate("text", x = 0.6, y = 0.15,
           label = sprintf("AUC = %.3f", auc), size = 4) +
  labs(
    title = "ROC Curve — Bayesian Logistic Regression",
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  theme_minimal(base_size = 11)

ggsave("results/blr_roc_curve.png", roc_plot, width = 6, height = 5, dpi = 150)

cat("\nFinished saving.\n")