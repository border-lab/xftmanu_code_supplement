## verify_sfig_theory_gxe.R
## Prints key summary statistics for supplementary figures S8, S13-S16

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
PROC_DIR <- file.path(BASE_DIR, "processed")

## ── S8: Theory vs observed ─────────────────────────────────────────────────────

cat("===== S8: Theory vs observed GWAS false positives =====\n\n")

gwas_fp <- read.csv(file.path(PROC_DIR, "sfig_s8_gwas_fp.csv"))
math_sim <- read.csv(file.path(PROC_DIR, "sfig_s8_mathematica_sim.csv"))

cat("gwas_fp dimensions:", nrow(gwas_fp), "rows x", ncol(gwas_fp), "cols\n")
cat("math_sim dimensions:", nrow(math_sim), "rows x", ncol(math_sim), "cols\n")
cat("Scenarios in gwas_fp:", paste(unique(gwas_fp$params), collapse = ", "), "\n")
cat("GWAS types:", paste(unique(gwas_fp$GWAS), collapse = ", "), "\n")
cat("Generations:", paste(sort(unique(gwas_fp$gen)), collapse = ", "), "\n")

## FP rate at alpha=0.05 for gen=3, population GWAS
fp_pop_g3 <- gwas_fp[gwas_fp$gen == 3 & gwas_fp$GWAS == "Population" &
                      gwas_fp$alpha == 0.05, ]
if (nrow(fp_pop_g3) > 0) {
  cat("\nFalse positive rate at alpha=0.05, gen=3, Population GWAS:\n")
  agg <- aggregate(FalsePositive ~ params, data = fp_pop_g3, mean)
  agg$FalsePositive <- round(agg$FalsePositive, 4)
  print(agg)
}

cat("\nmathematica_sim columns:", paste(names(math_sim), collapse = ", "), "\n")
cat("K values:", paste(sort(unique(math_sim$K)), collapse = ", "), "\n")
cat("Valence values:", paste(sort(unique(math_sim$valence)), collapse = ", "), "\n")

## ── S13/S14: h2 grid ───────────────────────────────────────────────────────────

cat("\n\n===== S13/S14: h2 across GxE/VT grid =====\n\n")

h2g <- read.csv(file.path(PROC_DIR, "sfig_s13s14_h2_grid.csv"))
cat("Dimensions:", nrow(h2g), "rows x", ncol(h2g), "cols\n")
cat("GxE levels:", paste(unique(h2g$GxE_level), collapse = ", "), "\n")
cat("VT levels:", paste(unique(h2g$VT_level), collapse = ", "), "\n")

cat("\nMean h2 estimated (he_h2) at gen=5:\n")
agg_h2 <- aggregate(he_h2 ~ GxE_level + VT_level,
                     data = h2g[h2g$gen == 5, ], mean)
agg_h2$he_h2 <- round(agg_h2$he_h2, 4)
print(agg_h2)

cat("\nMean h2 bias at gen=5:\n")
agg_bias <- aggregate(h2_bias ~ GxE_level + VT_level,
                       data = h2g[h2g$gen == 5, ], mean)
agg_bias$h2_bias <- round(agg_bias$h2_bias, 4)
print(agg_bias)

## ── S15: rg grid ────────────────────────────────────────────────────────────────

cat("\n\n===== S15: rg across GxE/VT grid =====\n\n")

rgg <- read.csv(file.path(PROC_DIR, "sfig_s15_rg_grid.csv"))
cat("Dimensions:", nrow(rgg), "rows x", ncol(rgg), "cols\n")

cat("\nMean estimated rg at gen=5:\n")
agg_rg <- aggregate(he_rg ~ GxE_level + VT_level,
                     data = rgg[rgg$gen == 5, ], mean)
agg_rg$he_rg <- round(agg_rg$he_rg, 4)
print(agg_rg)

cat("\nMean rg bias at gen=5:\n")
agg_rgb <- aggregate(rg_bias ~ GxE_level + VT_level,
                      data = rgg[rgg$gen == 5, ], mean)
agg_rgb$rg_bias <- round(agg_rgb$rg_bias, 4)
print(agg_rgb)

## ── S16: PGI correlation grid ───────────────────────────────────────────────────

cat("\n\n===== S16: PGI cross-phenotype slope with GxE/VT =====\n\n")

pgi <- read.csv(file.path(PROC_DIR, "sfig_s16_pgi_grid.csv"))
cat("Dimensions:", nrow(pgi), "rows x", ncol(pgi), "cols\n")

cat("\nMean rbeta_hat_pgwas (pop GWAS) at gen=5:\n")
agg_pgi <- aggregate(rbeta_hat_pgwas ~ GxE_level + VT_level,
                      data = pgi[pgi$gen == 5, ], mean)
agg_pgi$rbeta_hat_pgwas <- round(agg_pgi$rbeta_hat_pgwas, 4)
print(agg_pgi)

cat("\nMean rbeta_hat_sgwas (sib GWAS) at gen=5:\n")
agg_sib <- aggregate(rbeta_hat_sgwas ~ GxE_level + VT_level,
                      data = pgi[pgi$gen == 5, ], mean)
agg_sib$rbeta_hat_sgwas <- round(agg_sib$rbeta_hat_sgwas, 4)
print(agg_sib)

cat("\nDone: verify_sfig_theory_gxe.R\n")
