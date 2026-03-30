#!/usr/bin/env Rscript
# Verify: Correlated Noise figure
# Prints computed numbers for comparison against original figure

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"

all_cn <- read.csv(file.path(BASE_DIR, "processed/figR_cn.csv"))

cat("=== Correlated Noise Verification ===\n\n")

cat("Dimensions:", nrow(all_cn), "rows x", ncol(all_cn), "cols\n")
cat("CN levels:", paste(sort(unique(all_cn$CN)), collapse = ", "), "\n")
cat("Scenarios:", paste(unique(all_cn$scenario), collapse = ", "), "\n")
cat("Generations:", paste(sort(unique(all_cn$gen)), collapse = ", "), "\n\n")

cat("--- Rows per CN x scenario ---\n")
print(table(all_cn$CN, all_cn$scenario))

cat("\n--- Mean he_rg at gen 0 by CN level (should be near 0 for all) ---\n")
g0 <- all_cn[all_cn$gen == 0, ]
agg0 <- aggregate(he_rg ~ CN + scenario, data = g0, FUN = mean)
print(agg0)

cat("\n--- Mean he_rg at max gen by CN level ---\n")
gmax <- all_cn[all_cn$gen == max(all_cn$gen), ]
aggmax <- aggregate(he_rg ~ CN + scenario, data = gmax, FUN = mean)
print(aggmax)

cat("\n--- Key checks ---\n")
# Higher CN should produce higher spurious rg
cat("Mean he_rg at max gen for 5xAM by CN:\n")
sub <- aggmax[aggmax$scenario == "5xAM", ]
for (i in 1:nrow(sub)) {
  cat(sprintf("  CN=%s: he_rg=%.4f\n", sub$CN[i], sub$he_rg[i]))
}

cat("\nMean he_rg at max gen for 5xAM + VT (5%) by CN:\n")
sub <- aggmax[aggmax$scenario == "5xAM + VT (5%)", ]
for (i in 1:nrow(sub)) {
  cat(sprintf("  CN=%s: he_rg=%.4f\n", sub$CN[i], sub$he_rg[i]))
}

# Also check true rg for reference
cat("\n--- Mean rg_true at max gen by CN x scenario ---\n")
agg_true <- aggregate(rg_true ~ CN + scenario, data = gmax, FUN = mean)
print(agg_true)

cat("\nVerification complete.\n")
