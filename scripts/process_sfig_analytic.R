## process_sfig_analytic.R
## Processes data for supplementary figures S9, S10, S11, S12:
##   S9:  Incremental on-target vs off-target associations by sample size
##   S10: Type-I error inflation vs sample size (across generations & M)
##   S11: Cumulative on-target vs off-target associations by sample size
##   S12: False positive rate at off-target loci by effect magnitude
##
## Data sources:
##   - newhits.rdata            -> S9 (incremental new-hit counts)
##   - powerSims2.rdata (fres)  -> S10 (per-locus T1E predictions across N/gen/M/K)
##   - powerSims3.rdata (frac_res) -> S11 (cumulative on/off-target counts)
##   - merged_tabla_redux_results_011024.csv -> covariance matrices for S12
##
## The per-SNP T1E calculation (S12) uses functions from
## xftmanu_code_supplement/gwas_pwr_t1e.r applied to average simulation
## covariance matrices (K=5, gen=5, averaged across seeds).
##
## Saves intermediate CSVs to processed/

library(reshape2)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
DATA_DIR <- file.path(BASE_DIR, "data", "sim_results")
CODE_DIR <- file.path(BASE_DIR, "xftmanu_code_supplement")
OUT_DIR  <- file.path(BASE_DIR, "processed")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

## ── Helper functions (from gwas_pwr_t1e.r + rb_testing notebook) ──────────────
## These implement the analytic non-centrality parameter calculations for
## GWAS under cross-trait assortative mating (xAM).

construct_betas <- function(Mk, Vg0, K) {
  ## Sample random effect sizes for Mk causal variants per phenotype.
  ## Returns a (Mk*K x K) matrix where column k has nonzero betas in
  ## rows [(k-1)*Mk+1 .. k*Mk], each drawn ~ N(0, Vg0[k,k]/Mk).
  B <- matrix(0, Mk * K, K)
  for (k in 0:(K - 1)) {
    B[(1 + Mk * k):(Mk + Mk * k), k + 1] <- scale(rnorm(Mk)) * sqrt(Vg0[k + 1, k + 1] / Mk)
  }
  B
}

chi2_upper_tail <- function(alpha, lambda) {
  ## P(chi2_1(ncp=lambda) > chi2_1_crit(alpha))
  crit <- qchisq(1 - alpha, df = 1)
  1 - pchisq(crit, df = 1, ncp = lambda)
}

## Off-target (type I error) rate: expected false positive rate at loci
## causal for other phenotypes but not the focal phenotype.
t1e <- function(Vgt, Vg0, Ve, Mk, NN, alpha = 0.05, NMC = 2000) {
  nrep <- max(1, floor(NMC / Mk))
  mean(replicate(nrep, {
    K <- dim(Vgt)[1]
    NMC_use <- min(NMC, Mk)
    B <- construct_betas(Mk = Mk, Vg0 = Vg0, K = K)
    ## Off-target loci: indices > Mk (causal for phenotypes 2..K)
    subinds <- sort(sample((Mk + 1):(Mk * K), NMC_use))
    A <- solve(Vg0, Vgt) %*% solve(Vg0) - solve(Vg0)
    Vy <- Vgt + Ve
    lambda_null <- sapply(subinds, function(j) {
      b1 <- B[j, , drop = FALSE]
      beta <- B[j, 1]
      sx2 <- 1 + b1 %*% A %*% t(b1)
      bias <- b1 %*% A %*% t(B[-j, ]) %*% B[-j, 1]
      Vresid <- Vy[1, 1] - beta^2 * sx2
      NN * sx2 * (beta + bias)^2 / Vresid
    })
    mean(chi2_upper_tail(alpha, lambda_null))
  }))
}

## On-target (true positive) rate: expected detection rate at loci
## truly causal for the focal phenotype.
tpr <- function(Vgt, Vg0, Ve, Mk, NN, alpha = 0.05, NMC = 2000) {
  nrep <- max(1, floor(NMC / Mk))
  mean(replicate(nrep, {
    K <- dim(Vgt)[1]
    NMC_use <- min(NMC, Mk)
    B <- construct_betas(Mk = Mk, Vg0 = Vg0, K = K)
    ## On-target loci: indices 1..Mk (causal for phenotype 1)
    subinds <- sort(sample(1:Mk, NMC_use))
    A <- solve(Vg0, Vgt) %*% solve(Vg0) - solve(Vg0)
    Vy <- Vgt + Ve
    lambda_true <- sapply(subinds, function(j) {
      b1 <- B[j, , drop = FALSE]
      beta <- B[j, 1]
      sx2 <- 1 + b1 %*% A %*% t(b1)
      bias <- b1 %*% A %*% t(B[-j, ]) %*% B[-j, 1]
      Vresid <- Vy[1, 1] - beta^2 * sx2
      NN * sx2 * (beta + bias)^2 / Vresid
    })
    mean(chi2_upper_tail(alpha, lambda_true))
  }))
}

## Per-SNP T1E: returns (beta, T1E) for each sampled off-target locus.
## Used for S12 (effect-size-resolved false positive rates).
t1e_per_SNP <- function(Vgt, Vg0, Ve, Mk, NN, alpha = 0.05, NMC = 8000) {
  K <- dim(Vgt)[1]
  NMC_use <- min(NMC, Mk)
  B <- construct_betas(Mk = Mk, Vg0 = Vg0, K = K)
  subinds <- sort(sample((Mk + 1):(Mk * K), NMC_use))
  A <- solve(Vg0, Vgt) %*% solve(Vg0) - solve(Vg0)
  Vy <- Vgt + Ve
  lambda_null <- sapply(subinds, function(j) {
    NN * ((B[j, ]) %*% A %*% t(B[-j, ]) %*% B[-j, 1])^2 /
      (Vy[1, 1] * (1 + t(B[j, ]) %*% A %*% B[j, ]))
  })
  data.frame(
    beta = apply(B[subinds, ], 1, function(x) x[x != 0]),
    T1E = chi2_upper_tail(alpha, lambda_null)
  )
}

## Format large numbers as K/M labels (for sample size axis)
format_K_N <- function(x) {
  y <- as.character(x)
  idx_k <- log10(x) >= 3 & log10(x) < 6
  idx_m <- log10(x) >= 6
  y[idx_k] <- paste0(format(x[idx_k] / 1000, big.mark = ","), "K")
  y[idx_m] <- paste0(format(x[idx_m] / 1000000, big.mark = ","), "M")
  trimws(y)
}

## ── 1. Load simulation covariance matrices ───────────────────────────────────
## These are extracted from the merged simulation results, filtered to
## the 2xAM and 5xAM scenarios with assortative mating correlation r=0.2.
## Used for the per-SNP T1E calculation in S12.

cat("=== Processing S9-S12 analytic prediction data ===\n\n")

res <- read.csv(file.path(DATA_DIR, "merged_tabla_redux_results_011024.csv"))
res <- res[res$scenario %in% c("2xAM", "5xAM"), ]
sres <- res[res$args_rmate == 0.2, ]

cat("Loaded merged_tabla:", nrow(sres), "rows after filtering to 2xAM/5xAM, r=0.2\n")

## Build covariance matrices from each simulation run:
##   Vgt = true genetic covariance (inflated by AM)
##   Vg0 = random-mating genetic covariance (diagonal)
##   Ve  = residual covariance (= Vg0 for unit phenotypic variance)
ttt <- lapply(1:nrow(sres), function(j) {
  z <- sres[j, ]
  ss <- diag(rep(sqrt(z$vg_true), z$args_kphen))
  rr <- matrix(z$rg_true, z$args_kphen, z$args_kphen)
  diag(rr) <- 1
  Vgt <- ss %*% rr %*% ss
  Vg0 <- diag(z$vbeta_true, z$args_kphen)
  Mk <- z$args_m_causal
  N <- z$args_subsample
  K <- z$args_kphen
  list(Vgt = Vgt, Vg0 = Vg0, Ve = Vg0, Mk = Mk, N = N, K = K, gen = z$gen,
       pgwas_false_positives_0.05 = z$pgwas_false_positives_0.05,
       pgwas_false_positives_0.5 = z$pgwas_false_positives_0.5,
       pgwas_true_positives_0.05 = z$pgwas_true_positives_0.05,
       pgwas_true_positives_0.5 = z$pgwas_true_positives_0.5)
})
cat("Built", length(ttt), "covariance matrix sets\n\n")

## ── 2. S9: Incremental on-target vs off-target (from newhits.rdata) ──────────
## newhits contains pre-computed incremental (delta) expected on-target
## and off-target GWAS hits at each sample size step, derived from the
## predict_t1e_rate_alpha() function applied to averaged covariance matrices.

cat("--- S9: Incremental hits ---\n")
load(file.path(DATA_DIR, "newhits.rdata"))
cat("Loaded newhits.rdata:", nrow(newhits), "rows\n")

nhdat <- melt(newhits,
              id.vars = c("delta_N", "kpheno", "m_causal", "N", "gen"),
              measure.vars = c("delta_ex_on_target", "delta_ex_off_target"))
nhdat$type <- NA
nhdat$type[nhdat$variable == "delta_ex_off_target"] <- "Off target associations"
nhdat$type[nhdat$variable == "delta_ex_on_target"]  <- "On target associations"
nhdat$type <- factor(nhdat$type, levels = sort(unique(nhdat$type)))
nhdat$kpheno_label <- as.factor(nhdat$kpheno)
nhdat$m_causal_label <- nhdat$m_causal

write.csv(nhdat, file.path(OUT_DIR, "sfig_s9_incremental_hits.csv"),
          row.names = FALSE)
cat("Saved sfig_s9_incremental_hits.csv:", nrow(nhdat), "rows\n\n")

## ── 3. S10: Type-I error inflation vs sample size ────────────────────────────
## powerSims2.rdata (fres) contains the predicted off-target T1E rate
## at alpha=5e-8, computed via the t1e() function across a grid of
## (m_causal, gen, kpheno, N) values.

cat("--- S10: T1E inflation ---\n")
load(file.path(DATA_DIR, "powerSims2.rdata"))
cat("Loaded powerSims2.rdata:", nrow(fres), "rows\n")

## Filter to generations {0,1,3,5}, standard M values, alpha=5e-8
GENZ <- c(0, 1, 3, 5)
MZ <- c(1000, 2000, 4000, 8000, 16000)
pfdat <- fres[fres$alpha == 5e-8 & fres$gen %in% GENZ & fres$m_causal %in% MZ, ]
pfdat$kpheno_label <- as.factor(pfdat$kpheno)
pfdat$m_causal_label <- pfdat$m_causal
## Relative T1E = ratio of predicted off-target rate to nominal alpha
pfdat$relative_t1e <- pfdat$pred_t1e / 5e-8

write.csv(pfdat, file.path(OUT_DIR, "sfig_s10_t1e_inflation.csv"),
          row.names = FALSE)
cat("Saved sfig_s10_t1e_inflation.csv:", nrow(pfdat), "rows\n\n")

## ── 4. S11: Cumulative on-target vs off-target ───────────────────────────────
## powerSims3.rdata (frac_res) contains cumulative expected on-target,
## off-target, and conventional false positive counts at each sample size,
## computed via predict_t1e_rate_alpha() at alpha=5e-8, gen=5.

cat("--- S11: Cumulative hits ---\n")
load(file.path(DATA_DIR, "powerSims3.rdata"))
cat("Loaded powerSims3.rdata:", nrow(frac_res), "rows\n")

fpdat <- melt(frac_res,
              id.vars = c("N", "kpheno", "m_causal", "gen"),
              measure.vars = c("ex_on_target", "ex_off_target", "ex_FP"))
fpdat$type <- NA
fpdat$type[fpdat$variable == "ex_on_target"]  <- "On target"
fpdat$type[fpdat$variable == "ex_off_target"] <- "Off target"
fpdat$type[fpdat$variable == "ex_FP"]         <- "Conventional false positives"
fpdat$type <- factor(fpdat$type,
                     levels = c("Conventional false positives",
                                "Off target", "On target"))
fpdat$kpheno_label <- as.factor(fpdat$kpheno)
fpdat$m_causal_label <- fpdat$m_causal
fpdat$value <- ceiling(fpdat$value)

write.csv(fpdat, file.path(OUT_DIR, "sfig_s11_cumulative_hits.csv"),
          row.names = FALSE)
cat("Saved sfig_s11_cumulative_hits.csv:", nrow(fpdat), "rows\n\n")

## ── 5. S12: False positive rate at off-target loci by effect magnitude ───────
## For each off-target SNP, compute the probability of a false positive
## as a function of its effect size magnitude and GWAS sample size.
## Uses averaged covariance matrices from K=5, gen=5 simulations.

cat("--- S12: Per-SNP T1E by effect size ---\n")

## Average covariance matrices across simulation seeds for K=5, gen=5
SS <- ttt[sapply(ttt, function(x) (x$K == 5) & (x$gen == 5))]
if (length(SS) == 0) {
  cat("WARNING: No K=5, gen=5 simulations found. Trying K=5, gen>=4.\n")
  SS <- ttt[sapply(ttt, function(x) (x$K == 5) & (x$gen >= 4))]
}
cat("Averaging over", length(SS), "simulation runs for S12\n")

XX <- lapply(names(SS[[1]]), function(vv) {
  VV <- lapply(SS, function(x) x[[vv]])
  Reduce(`+`, VV) / length(VV)
})
names(XX) <- names(SS[[1]])

## Compute per-SNP T1E for M=4000, K=5, gen=5 across sample sizes
set.seed(42)
NN_grid <- c(5e4, 1e5, 2e5, 4e5, 8e5, 1.6e6, 3.2e6, 6.4e6)
cat("Computing per-SNP T1E for M=4000, K=5 at",
    length(NN_grid), "sample sizes...\n")

perSNPres <- do.call(rbind.data.frame, lapply(NN_grid, function(NNN) {
  out <- t1e_per_SNP(Vgt = XX$Vgt, Vg0 = XX$Vg0, Ve = XX$Ve,
                     Mk = 4000, NN = NNN, alpha = 5e-8)
  out$N <- NNN
  out$m_causal <- 4000
  out$kpheno <- 5
  out$gen <- 5
  out$alpha <- 5e-8
  out
}))

## Sort by absolute beta and compute quantiles within each N
NGRID <- 8000
s12_list <- lapply(split(perSNPres, perSNPres$N), function(sub) {
  sub <- sub[order(abs(sub$beta)), ]
  idx <- seq(1, nrow(sub), length.out = min(NGRID, nrow(sub)))
  sub <- sub[idx, ]
  sub$beta_quantile <- seq_len(nrow(sub)) / nrow(sub)
  sub$N_label <- format(sub$N, big.mark = ",", trim = TRUE)
  sub
})
s12_dat <- do.call(rbind.data.frame, s12_list)

## Factor N_label in decreasing order of N (for legend ordering)
n_levels <- unique(s12_dat$N_label[order(s12_dat$N, decreasing = TRUE)])
s12_dat$N_label <- factor(s12_dat$N_label, levels = n_levels)

write.csv(s12_dat, file.path(OUT_DIR, "sfig_s12_persnp_t1e.csv"),
          row.names = FALSE)
cat("Saved sfig_s12_persnp_t1e.csv:", nrow(s12_dat), "rows\n\n")

cat("Done: process_sfig_analytic.R\n")
