## process_sfig_sibcomp.R
## Processes data for supplementary figure S17:
##   S17a: Relative T1E inflation for pop vs sib GWAS across scenarios
##   S17b: Cross-phenotype slope correlations for pop vs sib GWAS
## Saves intermediate CSVs to processed/

library(reshape2)
library(stringr)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
DATA_DIR <- file.path(BASE_DIR, "data", "sim_results")
OUT_DIR  <- file.path(BASE_DIR, "processed")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

## ── Load data ──────────────────────────────────────────────────────────────────

res <- read.csv(file.path(DATA_DIR, "merged_tabla_redux_results_011024.csv"))
cat("Loaded merged_tabla:", nrow(res), "rows\n")

## ── S17a: Relative Type-I error (T1E) inflation ────────────────────────────────
## false_positives columns: sgwas_false_positives_0.05, pgwas_false_positives_0.05
## Relative T1E = observed FP rate / nominal alpha (0.05)

mvars_fp <- c("sgwas_false_positives_0.05", "pgwas_false_positives_0.05")
idvars <- c("seed", "gen", "args_kmate", "args_rmate",
            "args_theta", "args_phi", "args_m_causal", "power", "scenario")

mdat_fp <- melt(res, id.vars = idvars, measure.vars = mvars_fp)
mdat_fp$relative_T1R <- mdat_fp$value / 0.05
mdat_fp$GWAS <- "Population"
mdat_fp$GWAS[grep("^sgwas", mdat_fp$variable)] <- "Sibship"
mdat_fp$alpha <- 0.05
mdat_fp$power_label <- paste(mdat_fp$power, "power at alpha=0.05")

## Keep scenarios of interest (including RM+VT and 2xAM for broader context)
keep_scenarios <- c("2xAM", "RM + VT", "5xAM", "5xAM + GxE",
                    "5xAM + VT", "5xAM + VT + GxE")
mdat_fp <- mdat_fp[mdat_fp$scenario %in% keep_scenarios, ]

## Order scenarios
mdat_fp$scenario <- factor(mdat_fp$scenario,
  levels = c("RM + VT", "2xAM", "5xAM", "5xAM + GxE",
             "5xAM + VT", "5xAM + VT + GxE"))

write.csv(mdat_fp, file.path(OUT_DIR, "sfig_s17a_t1e_comparison.csv"),
          row.names = FALSE)
cat("Saved sfig_s17a_t1e_comparison.csv:", nrow(mdat_fp), "rows\n")

## ── S17b: Cross-phenotype slope correlations ────────────────────────────────────
## rbeta_hat_pgwas (pop GWAS), rbeta_hat_sgwas (sib GWAS)

mvars_slope <- c("rbeta_hat_sgwas", "rbeta_hat_pgwas")
idvars_slope <- c("seed", "gen", "args_kmate", "args_rmate",
                  "args_theta", "args_phi", "scenario")

mdat_slope <- melt(res, id.vars = idvars_slope, measure.vars = mvars_slope)
mdat_slope$GWAS <- "Population"
mdat_slope$GWAS[grep("sgwas", mdat_slope$variable)] <- "Sibship"

mdat_slope <- mdat_slope[mdat_slope$scenario %in% keep_scenarios, ]
mdat_slope$scenario <- factor(mdat_slope$scenario,
  levels = c("RM + VT", "2xAM", "5xAM", "5xAM + GxE",
             "5xAM + VT", "5xAM + VT + GxE"))

write.csv(mdat_slope, file.path(OUT_DIR, "sfig_s17b_slope_comparison.csv"),
          row.names = FALSE)
cat("Saved sfig_s17b_slope_comparison.csv:", nrow(mdat_slope), "rows\n")

cat("Done: process_sfig_sibcomp.R\n")
