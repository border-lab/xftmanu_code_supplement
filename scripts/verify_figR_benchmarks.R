#!/usr/bin/env Rscript
# Verify: Benchmarks figure
# Prints computed numbers for comparison against original figure

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"

scale_n <- read.csv(file.path(BASE_DIR, "processed/figR_benchmarks_scale_n.csv"))
scale_m <- read.csv(file.path(BASE_DIR, "processed/figR_benchmarks_scale_m.csv"))
scenarios <- read.csv(file.path(BASE_DIR, "processed/figR_benchmarks_scenarios.csv"))

cat("=== Benchmarks Verification ===\n\n")

# --- Panel a: scaling with n ---
cat("--- Panel a: Scaling with n (m=4000) ---\n")
cat("Rows:", nrow(scale_n), "\n")
cat("n values (thousands):", paste(sort(unique(scale_n$n_k)), collapse = ", "), "\n")
agg_n <- aggregate(minutes ~ n_k, data = scale_n, FUN = function(x)
  c(mean = mean(x), sd = sd(x), n = length(x)))
agg_n <- do.call(data.frame, agg_n)
names(agg_n) <- c("n_k", "mean_min", "sd_min", "n_reps")
cat("Mean runtime by n:\n")
for (i in 1:nrow(agg_n)) {
  cat(sprintf("  n=%6.0fk: %.2f +/- %.2f min (n=%d)\n",
              agg_n$n_k[i], agg_n$mean_min[i], agg_n$sd_min[i],
              agg_n$n_reps[i]))
}

# --- Panel b: scaling with m ---
cat("\n--- Panel b: Scaling with m ---\n")
cat("Rows:", nrow(scale_m), "\n")
for (nl in unique(scale_m$n_label)) {
  sub <- scale_m[scale_m$n_label == nl, ]
  cat("\n  ", nl, ":\n")
  cat("  m values (thousands):", paste(sort(unique(sub$m_k)), collapse = ", "), "\n")
  agg_m <- aggregate(minutes ~ m_k, data = sub, FUN = mean)
  for (j in 1:nrow(agg_m)) {
    cat(sprintf("    m=%5.1fk: %.2f min\n", agg_m$m_k[j], agg_m$minutes[j]))
  }
}

# --- Panel c: per-scenario ---
cat("\n--- Panel c: Per-scenario runtime ---\n")
cat("Rows:", nrow(scenarios), "\n")
cat("Scenarios:", paste(unique(scenarios$scenario_label), collapse = ", "), "\n")
agg_s <- aggregate(minutes ~ scenario_label, data = scenarios, FUN = function(x)
  c(mean = mean(x), sd = sd(x), n = length(x)))
agg_s <- do.call(data.frame, agg_s)
names(agg_s) <- c("scenario", "mean_min", "sd_min", "n_reps")
for (i in 1:nrow(agg_s)) {
  cat(sprintf("  %-12s: %.2f +/- %.2f min (n=%d)\n",
              agg_s$scenario[i], agg_s$mean_min[i], agg_s$sd_min[i],
              agg_s$n_reps[i]))
}

cat("\nVerification complete.\n")
