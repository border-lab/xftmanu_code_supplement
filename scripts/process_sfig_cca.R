## process_sfig_cca.R
## Generates/extracts data for supplementary figures S1, S2, S3:
##   S1: Multivariate vs multidimensional mating (synthetic data)
##   S2: Nonlinear mating CCA (synthetic data)
##   S3: CCA scree plot (MICE PMM imputed)
## Saves intermediate CSVs to processed/

library(MASS)
library(yacca)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
PROCESSED_DIR <- file.path(BASE_DIR, "processed")
dir.create(PROCESSED_DIR, showWarnings = FALSE, recursive = TRUE)

cat("=== Processing supplementary CCA figure data ===\n\n")

## ---------------------------------------------------------------
## S1: Multivariate vs multidimensional mating (synthetic data)
## Three regimes:
##   (a) univariate/unidimensional  - single trait drives mating
##   (b) multivariate/unidimensional - rank-1 weight matrix (all traits, 1 dim)
##   (c) multivariate/multidimensional - full-rank weight matrix (all traits, 3 dims)
## ---------------------------------------------------------------
cat("--- S1: Synthetic mating regimes ---\n")

set.seed(1)
N <- 1000
K <- 3
z_cor <- diag(rep(1, K))

## Helper: generate mate pairs under a weight matrix W
generate_mating <- function(W, N, K, z_cor) {
  Z <- MASS::mvrnorm(N, rep(0, K), z_cor)
  X1 <- scale(Z %*% W + MASS::mvrnorm(N, rep(0, K), z_cor))
  X2 <- scale(Z %*% W + MASS::mvrnorm(N, rep(0, K), z_cor))
  ccares <- yacca::cca(X1, X2)
  C1 <- ccares$canvarx
  C2 <- ccares$canvary
  list(X1 = X1, X2 = X2, C1 = C1, C2 = C2)
}

## (a) Univariate/unidimensional: W = e1 e1^T * 2
a <- c(1, 0, 0) * 2
W_u1 <- a %o% a
res_u1 <- generate_mating(W_u1, N, K, z_cor)

## (b) Multivariate/unidimensional: W = a a^T * 1.3 (rank 1, nonzero everywhere)
a <- rnorm(K)
a <- a + min(a)  # shift to make all positive-ish
W_r1 <- a %o% a * 1.3
res_r1 <- generate_mating(W_r1, N, K, z_cor)

## (c) Multivariate/multidimensional: W = a a^T + b b^T + c c^T (rank 3)
a <- rnorm(K)
b <- rnorm(K)
cc <- rnorm(K)
W_r3 <- a %o% a + b %o% b + cc %o% cc
res_r3 <- generate_mating(W_r3, N, K, z_cor)

## Save phenotype and CV matrices for each regime
for (regime in c("u1", "r1", "r3")) {
  res <- get(paste0("res_", regime))

  Xdat <- data.frame(res$X1, res$X2)
  colnames(Xdat) <- paste0("Mate.", rep(1:2, each = K), "..pheno.", rep(1:K, times = 2))

  Cdat <- data.frame(res$C1, res$C2)
  colnames(Cdat) <- paste0("Mate.", rep(1:2, each = K), "..CV.", rep(1:K, times = 2))

  write.csv(Xdat, file.path(PROCESSED_DIR, paste0("sfig_s1_pheno_", regime, ".csv")),
            row.names = FALSE)
  write.csv(Cdat, file.path(PROCESSED_DIR, paste0("sfig_s1_cv_", regime, ".csv")),
            row.names = FALSE)
  cat("  Saved: sfig_s1_pheno_", regime, ".csv, sfig_s1_cv_", regime, ".csv\n", sep = "")
}

## ---------------------------------------------------------------
## S2: Nonlinear mating CCA (synthetic data)
## Three nonlinear transforms applied to mate 1's latent:
##   (a) piecewise linear: asymmetric scaling of positive/negative halves
##   (b) inverse tangent (atan): compressive nonlinearity
##   (c) quadratic: z^2 shift
## ---------------------------------------------------------------
cat("\n--- S2: Nonlinear mating CCA ---\n")

set.seed(1)
N_nl <- 1200
a_nl <- c(1, 0, 0) * 2
W_nl <- a_nl %o% a_nl

## Shared base latent draw
Z_base <- MASS::mvrnorm(N_nl, rep(0, K), z_cor)

## (a) Piecewise linear
Z_pw <- Z_base
Z_pw[, 1] <- scale((Z_base[, 1] * (Z_base[, 1] < 0))) * 2 +
              scale((Z_base[, 1] * (Z_base[, 1] >= 0))) * 0.15
X1_pw <- Z_pw %*% W_nl + MASS::mvrnorm(N_nl, rep(0, K), z_cor)
X2_pw <- Z_base %*% W_nl + MASS::mvrnorm(N_nl, rep(0, K), z_cor)

ccares_pw <- yacca::cca(X1_pw, X2_pw)
res_pw <- list(X1 = X1_pw, X2 = X2_pw,
               C1 = ccares_pw$canvarx, C2 = ccares_pw$canvary)

## (b) Inverse tangent (atan)
Z_at <- Z_base
Z_at[, 1] <- scale(atan(Z_base[, 1] * 2)) * 1.5
X1_at <- Z_at %*% W_nl + MASS::mvrnorm(N_nl, rep(0, K), z_cor)
X2_at <- Z_base %*% W_nl + MASS::mvrnorm(N_nl, rep(0, K), z_cor)

ccares_at <- yacca::cca(X1_at, X2_at)
res_at <- list(X1 = X1_at, X2 = X2_at,
               C1 = ccares_at$canvarx, C2 = ccares_at$canvary)

## (c) Quadratic
Z_q <- Z_base
Z_q[, 1] <- scale(Z_base[, 1]^2) - 5
X1_q <- Z_base %*% W_nl + MASS::mvrnorm(N_nl, rep(0, K), z_cor)
X2_q <- Z_q %*% W_nl + MASS::mvrnorm(N_nl, rep(0, K), z_cor)

ccares_q <- yacca::cca(X1_q, X2_q)
res_q <- list(X1 = X1_q, X2 = X2_q,
              C1 = ccares_q$canvarx, C2 = ccares_q$canvary)

## Save phenotype and CV matrices for each nonlinear regime
for (regime in c("pw", "at", "q")) {
  res <- get(paste0("res_", regime))

  Xdat <- data.frame(res$X1, res$X2)
  colnames(Xdat) <- paste0("Mate.", rep(1:2, each = K), "..pheno.", rep(1:K, times = 2))

  Cdat <- data.frame(res$C1, res$C2)
  colnames(Cdat) <- paste0("Mate.", rep(1:2, each = K), "..CV.", rep(1:K, times = 2))

  write.csv(Xdat, file.path(PROCESSED_DIR, paste0("sfig_s2_pheno_", regime, ".csv")),
            row.names = FALSE)
  write.csv(Cdat, file.path(PROCESSED_DIR, paste0("sfig_s2_cv_", regime, ".csv")),
            row.names = FALSE)
  cat("  Saved: sfig_s2_pheno_", regime, ".csv, sfig_s2_cv_", regime, ".csv\n", sep = "")
}

## ---------------------------------------------------------------
## S3: CCA scree plot (MICE PMM imputed)
## Load imp_cca.Rdata -> `out` (yacca CCA result)
## Compute relative canonical redundancy per CV
## ---------------------------------------------------------------
cat("\n--- S3: CCA scree (MICE PMM imputed) ---\n")

load(file.path(BASE_DIR, "data/cca/imp_cca.Rdata"), verbose = TRUE)

## Relative canonical redundancy: average of x and y sides
cca_vars <- (out$xvrd / out$xrd + out$yvrd / out$yrd) / 2
cum_red <- cumsum(cca_vars)

n_cv <- length(cca_vars)
cat("  Total CVs:", n_cv, "\n")

## Significance test
pvals <- 1 - pchisq(out$chisq, out$df)
n_sig_05 <- sum(pvals < 0.05)
n_sig_bonf <- sum(pvals < 0.05 / length(out$df))
cat("  Significant at alpha=0.05:", n_sig_05, "\n")
cat("  Significant at Bonferroni:", n_sig_bonf, "\n")

## Build scree data frame
KK <- min(n_cv, 44)
scree_df <- data.frame(
  CanonicalVariate = c("", paste("CV", 1:KK)),
  cv = 0:KK,
  cumulative_redundancy = c(0, cum_red[1:KK]),
  individual_redundancy = c(0, cca_vars[1:KK]),
  stringsAsFactors = FALSE
)

## Determine dimensions for 90% and 95%
dims_90 <- min(which(cum_red >= 0.90))
dims_95 <- min(which(cum_red >= 0.95))
cat("  CVs for 90% redundancy:", dims_90, "(val =",
    round(cum_red[dims_90], 4), ")\n")
cat("  CVs for 95% redundancy:", dims_95, "(val =",
    round(cum_red[dims_95], 4), ")\n")

write.csv(scree_df, file.path(PROCESSED_DIR, "sfig_s3_cca_scree.csv"),
          row.names = FALSE)
cat("  Saved: sfig_s3_cca_scree.csv\n")

## Save verification metadata
verify_df <- data.frame(
  metric = c("total_cvs", "sig_05", "sig_bonf", "dims_90", "dims_95",
             "cumred_90", "cumred_95", "xrd", "yrd"),
  value = c(n_cv, n_sig_05, n_sig_bonf, dims_90, dims_95,
            cum_red[dims_90], cum_red[dims_95], out$xrd, out$yrd),
  stringsAsFactors = FALSE
)
write.csv(verify_df, file.path(PROCESSED_DIR, "sfig_s3_verify.csv"),
          row.names = FALSE)
cat("  Saved: sfig_s3_verify.csv\n")

cat("\nDone: process_sfig_cca.R\n")
