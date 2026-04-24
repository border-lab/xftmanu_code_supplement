#!/usr/bin/env Rscript
# Verify: h2=0.25 sensitivity figure
# Prints computed numbers for comparison against original figure

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"

h2_long <- read.csv(file.path(BASE_DIR, "processed/figR_h2_025_h2.csv"))
rg_long <- read.csv(file.path(BASE_DIR, "processed/figR_h2_025_rg.csv"))
fp_long <- read.csv(file.path(BASE_DIR, "processed/figR_h2_025_fp.csv"))

cat("=== h2=0.25 Sensitivity Verification ===\n\n")

cat("h2 data:", nrow(h2_long), "rows\n")
cat("rg data:", nrow(rg_long), "rows\n")
cat("FP data:", nrow(fp_long), "rows\n")
cat("Scenarios:", paste(unique(h2_long$scenario), collapse = ", "), "\n")
cat("Generations:", paste(sort(unique(h2_long$gen)), collapse = ", "), "\n\n")

# --- Panel b: h2 estimates ---
cat("--- Panel b: h2 at gen 0 (should start near 0.25) ---\n")
g0_h2 <- h2_long[h2_long$gen == 0, ]
agg <- aggregate(h2 ~ scenario + type, data = g0_h2, FUN = mean)
print(agg)

cat("\n--- Panel b: h2 at max gen ---\n")
gmax_h2 <- h2_long[h2_long$gen == max(h2_long$gen), ]
agg <- aggregate(h2 ~ scenario + type, data = gmax_h2, FUN = mean)
print(agg)

# --- Panel c: rg estimates ---
cat("\n--- Panel c: rg at gen 0 (should be near 0) ---\n")
g0_rg <- rg_long[rg_long$gen == 0, ]
agg <- aggregate(rg ~ scenario + type, data = g0_rg, FUN = mean)
print(agg)

cat("\n--- Panel c: rg at max gen ---\n")
gmax_rg <- rg_long[rg_long$gen == max(rg_long$gen), ]
agg <- aggregate(rg ~ scenario + type, data = gmax_rg, FUN = mean)
print(agg)

# --- Panel d: GWAS false positives ---
cat("\n--- Panel d: FPR at gen 0 (should be near 0.05) ---\n")
g0_fp <- fp_long[fp_long$gen == 0, ]
agg <- aggregate(fpr ~ scenario + gwas_type, data = g0_fp, FUN = mean)
print(agg)

cat("\n--- Panel d: FPR at max gen ---\n")
gmax_fp <- fp_long[fp_long$gen == max(fp_long$gen), ]
agg <- aggregate(fpr ~ scenario + gwas_type, data = gmax_fp, FUN = mean)
print(agg)

cat("\n--- Key checks ---\n")
# AM scenarios should show inflated population GWAS FP
cat("Population GWAS FPR at max gen:\n")
sub <- gmax_fp[gmax_fp$gwas_type == "Population GWAS", ]
agg <- aggregate(fpr ~ scenario, data = sub, FUN = mean)
for (i in 1:nrow(agg)) {
  cat(sprintf("  %-20s: FPR=%.4f\n", agg$scenario[i], agg$fpr[i]))
}

cat("\nVerification complete.\n")
