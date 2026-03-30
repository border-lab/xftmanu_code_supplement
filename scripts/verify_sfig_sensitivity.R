## verify_sfig_sensitivity.R
## Prints key summary statistics for supplementary figures S5 and S6

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
PROC_DIR <- file.path(BASE_DIR, "processed")

## ── S5: rg sensitivity to rmate ────────────────────────────────────────────────

cat("===== S5: rg under 5xAM with r=0.1 vs r=0.2 =====\n\n")

s5 <- read.csv(file.path(PROC_DIR, "sfig_s5_rg_sensitivity.csv"))

cat("Data dimensions:", nrow(s5), "rows x", ncol(s5), "cols\n")
cat("Scenarios:", paste(unique(s5$scenario), collapse = ", "), "\n")
cat("rmate values:", paste(sort(unique(s5$rmate)), collapse = ", "), "\n")
cat("Generations:", paste(sort(unique(s5$gen)), collapse = ", "), "\n\n")

## Mean he_rg at gen=5 by scenario and rmate
cat("Mean estimated rg at generation 5:\n")
agg5 <- aggregate(he_rg ~ scenario + rmate, data = s5[s5$gen == 5, ], mean)
agg5$he_rg <- round(agg5$he_rg, 4)
print(agg5)

cat("\nMean true rg at generation 5:\n")
agg5t <- aggregate(rg_true ~ scenario + rmate, data = s5[s5$gen == 5, ], mean)
agg5t$rg_true <- round(agg5t$rg_true, 4)
print(agg5t)

cat("\nBias (he_rg - rg_true) at generation 5:\n")
agg5b <- merge(agg5, agg5t)
agg5b$bias <- round(agg5b$he_rg - agg5b$rg_true, 4)
print(agg5b[, c("scenario", "rmate", "bias")])

## ── S6: fixed latent correlation ────────────────────────────────────────────────

cat("\n\n===== S6: rg under xAM with fixed latent correlation =====\n\n")

s6 <- read.csv(file.path(PROC_DIR, "sfig_s6_fixed_latent.csv"))

cat("Data dimensions:", nrow(s6), "rows x", ncol(s6), "cols\n")
cat("Scenario labels:", paste(unique(s6$scenario_label), collapse = ", "), "\n")
cat("Latent r values:", paste(sort(unique(s6$latent_r)), collapse = ", "), "\n\n")

## Mean he_rg at gen=5 by scenario
cat("Mean estimated rg at generation 5:\n")
agg6 <- aggregate(he_rg ~ scenario_label + latent_r,
                  data = s6[s6$gen == 5, ], mean)
agg6$he_rg <- round(agg6$he_rg, 4)
print(agg6)

cat("\nMean rg bias at generation 5:\n")
agg6b <- aggregate(cbind(he_rg, rg_true) ~ scenario_label + latent_r,
                   data = s6[s6$gen == 5, ], mean)
agg6b$bias <- round(agg6b$he_rg - agg6b$rg_true, 4)
print(agg6b[, c("scenario_label", "latent_r", "bias")])

cat("\nDone: verify_sfig_sensitivity.R\n")
