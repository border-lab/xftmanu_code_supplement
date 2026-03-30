## verify_sfig_analytic.R
## Prints key summary statistics for supplementary figures S9, S10, S11, S12.
## Run after process_sfig_analytic.R to sanity-check the processed data.

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
PROC_DIR <- file.path(BASE_DIR, "processed")

## ── S9: Incremental on-target vs off-target associations ──────────────────────

cat("===== S9: Incremental on-target vs off-target by sample size =====\n\n")

s9 <- read.csv(file.path(PROC_DIR, "sfig_s9_incremental_hits.csv"))

cat("Data dimensions:", nrow(s9), "rows x", ncol(s9), "cols\n")
cat("Phenotype counts (kpheno):", paste(sort(unique(s9$kpheno)), collapse = ", "), "\n")
cat("Causal variant counts (m_causal):", paste(sort(unique(s9$m_causal)), collapse = ", "), "\n")
cat("Sample size bins:", paste(unique(s9$delta_N[order(s9$N)]), collapse = " | "), "\n")
cat("Association types:", paste(unique(s9$type), collapse = ", "), "\n\n")

## Show incremental hits at largest sample size increment
cat("At largest sample-size bin (N =", format(max(s9$N), big.mark = ","), "), by kpheno x m_causal:\n")
s9_large <- s9[s9$N == max(s9$N), ]
agg9 <- aggregate(value ~ kpheno + m_causal + type, data = s9_large, sum)
agg9$value <- round(agg9$value, 2)
print(agg9)

## On-target fraction at first vs last sample size bin
cat("\nOn-target fraction of incremental hits (first vs last bin):\n")
for (kp in sort(unique(s9$kpheno))) {
  for (mc in sort(unique(s9$m_causal))) {
    sub <- s9[s9$kpheno == kp & s9$m_causal == mc, ]
    on <- sub[sub$type == "On target associations", ]
    off <- sub[sub$type == "Off target associations", ]
    on <- on[order(on$N), ]
    off <- off[order(off$N), ]
    total_first <- on$value[1] + off$value[1]
    total_last  <- on$value[nrow(on)] + off$value[nrow(off)]
    frac_first <- if (total_first > 0) on$value[1] / total_first else NA
    frac_last  <- if (total_last > 0) on$value[nrow(on)] / total_last else NA
    cat(sprintf("  K=%d, M=%d: first bin=%.4f, last bin=%.4f\n",
                kp, mc, frac_first, frac_last))
  }
}

## Key check: at small N, nearly all new hits should be on-target
cat("\nSanity check: at smallest N, on-target should dominate.\n")

## ── S10: Type-I error inflation vs sample size ───────────────────────────────

cat("\n\n===== S10: Type-I error inflation vs sample size =====\n\n")

s10 <- read.csv(file.path(PROC_DIR, "sfig_s10_t1e_inflation.csv"))

cat("Data dimensions:", nrow(s10), "rows x", ncol(s10), "cols\n")
cat("Generations:", paste(sort(unique(s10$gen)), collapse = ", "), "\n")
cat("Causal variant counts:", paste(sort(unique(s10$m_causal)), collapse = ", "), "\n")
cat("Phenotype counts:", paste(sort(unique(s10$kpheno)), collapse = ", "), "\n")
cat("Sample sizes:", paste(format(sort(unique(s10$N)), big.mark = ","),
                           collapse = ", "), "\n\n")

## Relative T1E at gen=5, largest N
cat("Relative T1E (pred_t1e / 5e-8) at gen=5, largest N:\n")
s10_g5 <- s10[s10$gen == 5 & s10$N == max(s10$N), ]
agg10 <- aggregate(relative_t1e ~ kpheno + m_causal, data = s10_g5, mean)
agg10$relative_t1e <- round(agg10$relative_t1e, 1)
print(agg10)

## T1E at gen=0 should be ~1 (no inflation under random mating)
cat("\nRelative T1E at gen=0 (should be ~1.0):\n")
s10_g0 <- s10[s10$gen == 0, ]
cat("  Range:", round(min(s10_g0$relative_t1e), 4), "to",
    round(max(s10_g0$relative_t1e), 4), "\n")

## Monotonicity checks
cat("\nMonotonicity checks:\n")
cat("  T1E increases with N at gen=5: ")
for (kp in sort(unique(s10$kpheno))) {
  for (mc in c(2000, 8000)) {
    sub <- s10[s10$gen == 5 & s10$kpheno == kp & s10$m_causal == mc, ]
    sub <- sub[order(sub$N), ]
    is_mono <- all(diff(sub$relative_t1e) >= 0)
    cat(sprintf("K=%d,M=%d:%s ", kp, mc, ifelse(is_mono, "YES", "NO")))
  }
}
cat("\n")

cat("  T1E increases with gen at fixed N/M: ")
for (kp in sort(unique(s10$kpheno))) {
  sub <- s10[s10$kpheno == kp & s10$m_causal == 4000 & s10$N == max(s10$N), ]
  sub <- sub[order(sub$gen), ]
  is_mono <- all(diff(sub$relative_t1e) >= 0)
  cat(sprintf("K=%d:%s ", kp, ifelse(is_mono, "YES", "NO")))
}
cat("\n")

cat("  More phenotypes -> larger T1E at gen=5: ")
for (mc in c(2000, 8000)) {
  sub <- s10[s10$gen == 5 & s10$m_causal == mc & s10$N == max(s10$N), ]
  k2 <- sub$relative_t1e[sub$kpheno == 2]
  k5 <- sub$relative_t1e[sub$kpheno == 5]
  cat(sprintf("M=%d: K=5(%.0f) > K=2(%.0f)? %s  ",
              mc, k5, k2, ifelse(k5 > k2, "YES", "NO")))
}
cat("\n")

## ── S11: Cumulative on-target vs off-target associations ──────────────────────

cat("\n\n===== S11: Cumulative on-target vs off-target by sample size =====\n\n")

s11 <- read.csv(file.path(PROC_DIR, "sfig_s11_cumulative_hits.csv"))

cat("Data dimensions:", nrow(s11), "rows x", ncol(s11), "cols\n")
cat("Phenotype counts:", paste(sort(unique(s11$kpheno)), collapse = ", "), "\n")
cat("Causal variant counts:", paste(sort(unique(s11$m_causal)), collapse = ", "), "\n")
cat("Association types:", paste(unique(s11$type), collapse = ", "), "\n\n")

## Show cumulative hits at largest sample size
cat("Cumulative hits at N =", format(max(s11$N), big.mark = ","), ":\n")
s11_max <- s11[s11$N == max(s11$N), ]
agg11 <- aggregate(value ~ kpheno + m_causal + type, data = s11_max, sum)
print(agg11)

## Off-target fraction at largest N
cat("\nOff-target fraction at largest N:\n")
for (kp in sort(unique(s11$kpheno))) {
  for (mc in sort(unique(s11$m_causal))) {
    sub <- s11[s11$kpheno == kp & s11$m_causal == mc & s11$N == max(s11$N), ]
    on  <- sub$value[sub$type == "On target"]
    off <- sub$value[sub$type == "Off target"]
    fp  <- sub$value[sub$type == "Conventional false positives"]
    total <- on + off + fp
    cat(sprintf("  K=%d, M=%d: on=%d, off=%d, FP=%d, off_frac=%.4f\n",
                kp, mc, on, off, fp, off / total))
  }
}

## Key check: on-target should approach M at large N
cat("\nSanity check: on-target hits should approach M at large N.\n")
for (kp in sort(unique(s11$kpheno))) {
  for (mc in sort(unique(s11$m_causal))) {
    sub <- s11[s11$kpheno == kp & s11$m_causal == mc & s11$N == max(s11$N), ]
    on <- sub$value[sub$type == "On target"]
    cat(sprintf("  K=%d, M=%d: on-target=%d (M=%d, ratio=%.3f)\n",
                kp, mc, on, mc, on / mc))
  }
}

## ── S12: False positive rate by effect magnitude ──────────────────────────────

cat("\n\n===== S12: False positive rate at off-target loci by effect size =====\n\n")

s12 <- read.csv(file.path(PROC_DIR, "sfig_s12_persnp_t1e.csv"))

cat("Data dimensions:", nrow(s12), "rows x", ncol(s12), "cols\n")
cat("Sample sizes:", paste(format(sort(unique(s12$N)), big.mark = ","),
                           collapse = ", "), "\n")
cat("M causal:", unique(s12$m_causal), "\n")
cat("K pheno:", unique(s12$kpheno), "\n\n")

## T1E range by sample size
cat("T1E range by sample size:\n")
for (nn in sort(unique(s12$N))) {
  sub <- s12[s12$N == nn, ]
  cat(sprintf("  N=%s: min=%.2e, median=%.2e, max=%.2e, mean=%.2e\n",
              format(nn, big.mark = ","),
              min(sub$T1E), median(sub$T1E), max(sub$T1E), mean(sub$T1E)))
}

## Fraction exceeding meaningful thresholds
cat("\nFraction of off-target loci with T1E > 10x nominal (5e-7) at each N:\n")
for (nn in sort(unique(s12$N))) {
  sub <- s12[s12$N == nn, ]
  frac <- mean(sub$T1E > 5e-7)
  cat(sprintf("  N=%s: %.4f\n", format(nn, big.mark = ","), frac))
}

cat("\nFraction of off-target loci with T1E > 1000x nominal (5e-5) at each N:\n")
for (nn in sort(unique(s12$N))) {
  sub <- s12[s12$N == nn, ]
  frac <- mean(sub$T1E > 5e-5)
  cat(sprintf("  N=%s: %.4f\n", format(nn, big.mark = ","), frac))
}

## Check that T1E increases with effect size (beta_quantile)
cat("\nT1E at 10th vs 90th percentile of effect size:\n")
for (nn in sort(unique(s12$N))) {
  sub <- s12[s12$N == nn, ]
  low <- median(sub$T1E[sub$beta_quantile <= 0.1])
  high <- median(sub$T1E[sub$beta_quantile >= 0.9])
  cat(sprintf("  N=%s: q10=%.2e, q90=%.2e, ratio=%.1f\n",
              format(nn, big.mark = ","), low, high, high / low))
}

cat("\nDone: verify_sfig_analytic.R\n")
