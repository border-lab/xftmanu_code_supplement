## verify_sfig_cca.R
## Prints key summary statistics for supplementary figures S1, S2, S3

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
PROC_DIR <- file.path(BASE_DIR, "processed")

cat("============================================================\n")
cat("  SUPPLEMENTARY FIGURES S1-S3: VERIFICATION\n")
cat("============================================================\n\n")

## ---------------------------------------------------------------
## S1: Multivariate vs multidimensional mating (synthetic data)
## ---------------------------------------------------------------
cat("===== S1: Synthetic mating regimes =====\n\n")

regime_labels <- c(u1 = "univariate/unidimensional",
                   r1 = "multivariate/unidimensional",
                   r3 = "multivariate/multidimensional")

for (regime in c("u1", "r1", "r3")) {
  Xdat <- read.csv(file.path(PROC_DIR, paste0("sfig_s1_pheno_", regime, ".csv")))
  Cdat <- read.csv(file.path(PROC_DIR, paste0("sfig_s1_cv_", regime, ".csv")))

  cat(sprintf("  Regime: %s (%s)\n", regime, regime_labels[regime]))
  cat("    Phenotype data:", nrow(Xdat), "x", ncol(Xdat), "\n")
  cat("    CV data:", nrow(Cdat), "x", ncol(Cdat), "\n")

  ## Cross-mate phenotype correlations (mate 1 pheno vs mate 2 pheno)
  x1_cols <- grep("Mate\\.1", names(Xdat))
  x2_cols <- grep("Mate\\.2", names(Xdat))
  pheno_cors <- cor(Xdat[, x1_cols], Xdat[, x2_cols])
  cat("    Phenotype cross-mate correlations (diag):",
      paste(round(diag(pheno_cors), 3), collapse = ", "), "\n")

  ## Cross-mate CV correlations (CV1 mate 1 vs CV1 mate 2, etc.)
  c1_cols <- grep("Mate\\.1", names(Cdat))
  c2_cols <- grep("Mate\\.2", names(Cdat))
  cv_cors <- cor(Cdat[, c1_cols], Cdat[, c2_cols])
  cat("    CV cross-mate correlations (diag):",
      paste(round(diag(cv_cors), 3), collapse = ", "), "\n")
  cat("    CV off-diag max abs:",
      round(max(abs(cv_cors[row(cv_cors) != col(cv_cors)])), 4), "\n\n")
}

## Expected patterns:
## - u1: only pheno 1 correlated across mates; CV1 captures it, CVs 2-3 ~0
## - r1: all phenotypes correlated (rank 1); CV1 captures it, CVs 2-3 ~0
## - r3: all phenotypes correlated (rank 3); CVs 1-3 all show nonzero correlations
cat("  Expected patterns:\n")
cat("    u1: only pheno 1 cross-correlated; CV1 only\n")
cat("    r1: all phenos correlated via rank-1; CV1 dominant\n")
cat("    r3: all phenos correlated via rank-3; all CVs active\n\n")

## ---------------------------------------------------------------
## S2: Nonlinear mating CCA (synthetic data)
## ---------------------------------------------------------------
cat("===== S2: Nonlinear mating CCA =====\n\n")

nl_labels <- c(pw = "piecewise linear",
               at = "inverse tangent",
               q  = "quadratic")

for (regime in c("pw", "at", "q")) {
  Xdat <- read.csv(file.path(PROC_DIR, paste0("sfig_s2_pheno_", regime, ".csv")))
  Cdat <- read.csv(file.path(PROC_DIR, paste0("sfig_s2_cv_", regime, ".csv")))

  cat(sprintf("  Regime: %s (%s)\n", regime, nl_labels[regime]))
  cat("    Phenotype data:", nrow(Xdat), "x", ncol(Xdat), "\n")

  ## Cross-mate phenotype correlations
  x1_cols <- grep("Mate\\.1", names(Xdat))
  x2_cols <- grep("Mate\\.2", names(Xdat))
  pheno_cors <- cor(Xdat[, x1_cols], Xdat[, x2_cols])
  cat("    Phenotype cross-mate correlations (diag):",
      paste(round(diag(pheno_cors), 3), collapse = ", "), "\n")

  ## Cross-mate CV correlations
  c1_cols <- grep("Mate\\.1", names(Cdat))
  c2_cols <- grep("Mate\\.2", names(Cdat))
  cv_cors <- cor(Cdat[, c1_cols], Cdat[, c2_cols])
  cat("    CV cross-mate correlations (diag):",
      paste(round(diag(cv_cors), 3), collapse = ", "), "\n")
  cat("    CV off-diag max abs:",
      round(max(abs(cv_cors[row(cv_cors) != col(cv_cors)])), 4), "\n\n")
}

cat("  Expected patterns:\n")
cat("    All nonlinear: CCA still captures linear component in CV1\n")
cat("    pw:  asymmetric scatter; CCA less efficient\n")
cat("    at:  compressed tails; reduced apparent correlation\n")
cat("    q:   quadratic (no linear signal); CCA finds near-zero correlation\n\n")

## ---------------------------------------------------------------
## S3: CCA scree plot (MICE PMM imputed)
## ---------------------------------------------------------------
cat("===== S3: CCA scree (MICE PMM imputed) =====\n\n")

scree <- read.csv(file.path(PROC_DIR, "sfig_s3_cca_scree.csv"))
verify <- read.csv(file.path(PROC_DIR, "sfig_s3_verify.csv"))

cat("  Scree data:", nrow(scree), "rows (including CV=0 baseline)\n")
cat("  Total CVs:", verify$value[verify$metric == "total_cvs"], "\n")

n_sig <- verify$value[verify$metric == "sig_05"]
n_bonf <- verify$value[verify$metric == "sig_bonf"]
cat("  Significant CVs at alpha=0.05:", n_sig, "\n")
cat("  Significant CVs at Bonferroni:", n_bonf, "\n")

dims_90 <- verify$value[verify$metric == "dims_90"]
dims_95 <- verify$value[verify$metric == "dims_95"]
cumred_90 <- verify$value[verify$metric == "cumred_90"]
cumred_95 <- verify$value[verify$metric == "cumred_95"]

cat("  CVs for 90% redundancy:", dims_90,
    "(cumulative =", round(cumred_90, 4), ")\n")
cat("  CVs for 95% redundancy:", dims_95,
    "(cumulative =", round(cumred_95, 4), ")\n")

## Manuscript states 12 CVs for 95% -- verify
cat("\n  Manuscript check: 12 CVs for 95%?\n")
cv12_val <- scree$cumulative_redundancy[scree$cv == 12]
cat("    Cumulative redundancy at CV12:", round(cv12_val, 4), "\n")
if (cv12_val >= 0.95) {
  cat("    PASS: CV12 exceeds 0.95\n")
} else {
  cat("    NOTE: CV12 =", round(cv12_val, 4), "< 0.95; threshold CV =", dims_95, "\n")
}

## Print first 15 CVs for inspection
cat("\n  First 15 CVs:\n")
print(scree[1:min(16, nrow(scree)),
            c("CanonicalVariate", "cv", "cumulative_redundancy")],
      row.names = FALSE)

cat("\nDone: verify_sfig_cca.R\n")
