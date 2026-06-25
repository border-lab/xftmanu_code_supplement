#!/usr/bin/env Rscript
## process_fig3_vdecomp_r5.R   (ROUND 5 — non-destructive copy of scripts/process_fig3_vdecomp.R)
## Only substantive change vs the round-4 original:
##   * the middle decomposition tier is now the EXACT apparent (BLP) quantity
##     (Cᵀ G⁻¹ C) instead of predictive R²_g (h²) / r_cross r× (rg);
##   * "Population h²" tier relabelled "Direct h²"; "rscore" relabelled "Direct r_g".
## Reads the SAME round-4 per-seed data; writes only to round5/fig3_r5/ (originals untouched).
##
## Output: round5/fig3_r5/fig3_vdecomp_r5_h2_long.csv
##         round5/fig3_r5/fig3_vdecomp_r5_rg_long.csv
##         round5/fig3_r5/fig3_vdecomp_r5_fp.csv
##         round5/fig3_r5/fig3_vdecomp_r5_summary.csv

library(reshape2)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"          # inputs (round-4 data)
OUT_DIR  <- "/home/rsb/Dropbox/ftsim/round4/round5/fig3_r5"   # outputs (round-5, new)
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

## ── exact apparent (BLP) heritability / genetic correlation ────────────────
## Compound-symmetric reconstruction, identical to round5/compute_apparent.py.
## a = h2_true (Var(direct PGI) scale), b = rg_true*h2_true (off-diag of G),
## c = cov_g_y (diag of C), d = cov_g_y_cross (off-diag of C).
exact_app <- function(K, h2, rg, cgy, cgyx) {
  tryCatch({
    a <- h2; b <- rg * h2; cc <- cgy; d <- cgyx
    G <- matrix(b, K, K); diag(G) <- a
    C <- matrix(d, K, K); diag(C) <- cc
    M  <- crossprod(C, solve(G, C))          # t(C) %*% G^{-1} %*% C
    dg <- diag(M)
    Rm <- M / sqrt(outer(dg, dg))
    c(mean(dg), mean(Rm[upper.tri(Rm)]))
  }, error = function(e) c(NA_real_, NA_real_))
}

## ── 1. Load and merge per-seed CSVs (unchanged) ────────────────────────────
load_vdecomp <- function(data_dir, h2_label) {
  files <- list.files(data_dir, pattern = "_parsed\\.csv$", full.names = TRUE)
  cat("Loading", length(files), "files from", basename(data_dir), "\n")
  dfs <- lapply(files, function(f) { d <- read.csv(f); d$h2_param <- h2_label; d })
  common_cols <- Reduce(intersect, lapply(dfs, names))
  dfs <- lapply(dfs, function(d) d[, common_cols])
  do.call(rbind, dfs)
}

d05  <- load_vdecomp(file.path(BASE_DIR, "data/vdecomp_lane"),     0.5)
d025 <- load_vdecomp(file.path(BASE_DIR, "data/vdecomp_h2_0_25"), 0.25)
common <- intersect(names(d05), names(d025))
dat <- rbind(d05[, common], d025[, common])
cat("Total rows:", nrow(dat), "\n")

## ── 2. Scenario labels (unchanged) ─────────────────────────────────────────
dat$scenario <- NA_character_
dat$scenario[dat$args_rmate == 0   & dat$args_theta == 0    & dat$args_kphen == 5] <- "RM"
dat$scenario[dat$args_rmate == 0   & dat$args_theta == 0.05 & dat$args_kphen == 5] <- "RM + VT"
dat$scenario[dat$args_rmate == 0.2 & dat$args_theta == 0    & dat$args_kphen == 2] <- "2xAM"
dat$scenario[dat$args_rmate == 0.2 & dat$args_theta == 0.05 & dat$args_kphen == 2] <- "2xAM + VT"
dat$scenario[dat$args_rmate == 0.2 & dat$args_theta == 0    & dat$args_kphen == 5] <- "5xAM"
dat$scenario[dat$args_rmate == 0.2 & dat$args_theta == 0.05 & dat$args_kphen == 5] <- "5xAM + VT"
stopifnot(!any(is.na(dat$scenario)))
dat$scenario <- factor(dat$scenario,
  levels = c("RM", "RM + VT", "2xAM", "2xAM + VT", "5xAM", "5xAM + VT"))

## ── 3. Derived columns + EXACT APPARENT (new) ──────────────────────────────
dat$h2_he    <- dat$he_h2
dat$rbeta_HE <- dat$he_rg
app <- mapply(exact_app, dat$args_kphen, dat$h2_true, dat$rg_true,
              dat$cov_g_y, dat$cov_g_y_cross)
dat$apparent_h2_exact <- app[1, ]
dat$apparent_rg_exact <- app[2, ]

## sanity check vs known values (5xAM+VT, h2=0.5, gen 5): apparent h2≈0.613, rg≈0.405
chk <- dat[dat$scenario == "5xAM + VT" & dat$h2_param == 0.5 & dat$gen == 5, ]
cat(sprintf("CHECK 5xAM+VT gen5: apparent h2 = %.3f (exp 0.613), apparent rg = %.3f (exp 0.405)\n",
            mean(chk$apparent_h2_exact), mean(chk$apparent_rg_exact)))

## ── 4. h² decomposition long (Direct, Apparent, LDSC) ──────────────────────
h2_vars <- c("h2_true", "apparent_h2_exact", "h2_he")
h2_long <- melt(dat[, c("gen", "seed", "scenario", "h2_param", h2_vars)],
                id.vars = c("gen", "seed", "scenario", "h2_param"),
                variable.name = "quantity", value.name = "h2")
h2_long$quantity <- factor(h2_long$quantity,
  levels = c("h2_true", "apparent_h2_exact", "h2_he"),
  labels = c("Direct h²", "Apparent h²", "LDSC h²"))
write.csv(h2_long, file.path(OUT_DIR, "fig3_vdecomp_r5_h2_long.csv"), row.names = FALSE)
cat("Saved fig3_vdecomp_r5_h2_long.csv:", nrow(h2_long), "rows\n")

## ── 5. rg decomposition long (Direct, Apparent, LDSC) ──────────────────────
rg_vars <- c("rg_true", "apparent_rg_exact", "rbeta_HE")
rg_long <- melt(dat[, c("gen", "seed", "scenario", "h2_param", rg_vars)],
                id.vars = c("gen", "seed", "scenario", "h2_param"),
                variable.name = "quantity", value.name = "rg")
rg_long$quantity <- factor(rg_long$quantity,
  levels = c("rg_true", "apparent_rg_exact", "rbeta_HE"),
  labels = c("Direct r_g", "Apparent r_g", "LDSC r_g"))
write.csv(rg_long, file.path(OUT_DIR, "fig3_vdecomp_r5_rg_long.csv"), row.names = FALSE)
cat("Saved fig3_vdecomp_r5_rg_long.csv:", nrow(rg_long), "rows\n")

## ── 6. GWAS FP data (unchanged copy) ───────────────────────────────────────
orig <- read.csv(file.path(BASE_DIR, "data/sim_results/merged_tabla_redux_results_011024.csv"))
fig3_fp_scens <- c("RM", "RM + VT", "5xAM", "5xAM + VT")
orig_fp <- orig[orig$scenario %in% fig3_fp_scens, ]
fp_vars <- c("sgwas_false_positives_0.05", "pgwas_false_positives_0.05")
idvars_fp <- c("seed", "gen", "args_kmate", "args_rmate", "args_theta",
               "args_phi", "args_m_causal", "power", "scenario")
fp_melt <- melt(orig_fp, id.vars = idvars_fp, measure.vars = fp_vars)
fp_melt$relative_T1R <- fp_melt$value / 0.05
fp_melt$GWAS <- ifelse(grepl("^sgwas", fp_melt$variable), "Sibship", "Population")
write.csv(fp_melt[, c("gen", "seed", "scenario", "power", "GWAS", "relative_T1R")],
          file.path(OUT_DIR, "fig3_vdecomp_r5_fp.csv"), row.names = FALSE)
cat("Saved fig3_vdecomp_r5_fp.csv:", nrow(fp_melt), "rows\n")

## ── 7. Gen-5 summary (Direct / Apparent / LDSC) ────────────────────────────
gen5 <- dat[dat$gen == 5, ]
summ <- aggregate(
  gen5[, c("h2_true", "apparent_h2_exact", "h2_he",
           "rg_true", "apparent_rg_exact", "rbeta_HE")],
  gen5[, c("scenario", "h2_param")], function(x) c(mean = mean(x), sd = sd(x)))
summ <- do.call(data.frame, summ)
write.csv(summ, file.path(OUT_DIR, "fig3_vdecomp_r5_summary.csv"), row.names = FALSE)
cat("Saved fig3_vdecomp_r5_summary.csv:", nrow(summ), "rows\n")

cat("\n=== process_fig3_vdecomp_r5.R complete ===\n")
