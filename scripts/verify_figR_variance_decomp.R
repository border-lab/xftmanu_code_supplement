#!/usr/bin/env Rscript
# Verify: Variance Decomposition figure
# Prints computed numbers for comparison against original figure

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"

agg <- read.csv(file.path(BASE_DIR, "processed/figR_variance_decomp.csv"))

cat("=== Variance Decomposition Verification ===\n\n")

cat("Dimensions:", nrow(agg), "rows x", ncol(agg), "cols\n")
cat("Scenarios:", paste(unique(agg$scenario), collapse = ", "), "\n")
cat("Quantities:", paste(unique(agg$quantity), collapse = ", "), "\n")
cat("Generations:", paste(sort(unique(agg$gen)), collapse = ", "), "\n")
cat("Reps per cell (n):", unique(agg$n), "\n\n")

# Show gen 0 and gen 5 values for each scenario x quantity
cat("--- Generation 0 values (should all start near h2=0.5) ---\n")
g0 <- agg[agg$gen == 0, ]
for (s in levels(factor(g0$scenario))) {
  cat("\n  Scenario:", s, "\n")
  sub <- g0[g0$scenario == s, ]
  for (i in 1:nrow(sub)) {
    cat(sprintf("    %-25s mean=%.4f  sd=%.4f\n",
                sub$quantity[i], sub$mean_h2[i], sub$sd_h2[i]))
  }
}

cat("\n--- Generation 5 values (divergence expected) ---\n")
g5 <- agg[agg$gen == max(agg$gen), ]
for (s in levels(factor(g5$scenario))) {
  cat("\n  Scenario:", s, "\n")
  sub <- g5[g5$scenario == s, ]
  for (i in 1:nrow(sub)) {
    cat(sprintf("    %-25s mean=%.4f  sd=%.4f\n",
                sub$quantity[i], sub$mean_h2[i], sub$sd_h2[i]))
  }
}

cat("\n--- Key checks ---\n")
# Panmictic should remain stable across generations (invariant to AM)
pan <- agg[agg$quantity == "Panmictic (causal var.)", ]
cat("Panmictic h2 range across all: ",
    sprintf("%.4f - %.4f\n", min(pan$mean_h2), max(pan$mean_h2)))

# Pop estimated should inflate most under AM scenarios
est_5xam <- agg[agg$quantity == "Pop. estimated (LDSC)" &
                agg$scenario == "5xAM" & agg$gen == max(agg$gen), ]
cat("Pop. estimated h2 at gen 5 (5xAM):", sprintf("%.4f\n", est_5xam$mean_h2))

cat("\nVerification complete.\n")
