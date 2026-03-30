## process_sfig_theory_gxe.R
## Processes data for supplementary figures S8 and S13-S16:
##   S8:  Theory vs observed: false positive rates and power
##   S13: h2 across generations, GxE rows x VT cols, under 5xAM
##   S14: h2 across generations, VT rows (alternative arrangement)
##   S15: rg across generations with GxE/VT grid
##   S16: PGI cross-phenotype slope correlation with GxE/VT grid
## Saves intermediate CSVs to processed/

library(reshape2)
library(stringr)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
DATA_DIR <- file.path(BASE_DIR, "data", "sim_results")
OUT_DIR  <- file.path(BASE_DIR, "processed")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

## ── 1. S8: Theory vs observed (gwas_fp + mathematica_sim) ──────────────────────

gwas_fp <- read.csv(file.path(DATA_DIR, "gwas_fp.csv"))
math_sim <- read.csv(file.path(DATA_DIR, "mathematica_sim.csv"))

## Clean gwas_fp: parse scenario labels
gwas_fp$xAM <- "5-variate"
gwas_fp$xAM[gwas_fp$params == "2xAM"] <- "2-variate"
gwas_fp$nm_ratio <- gwas_fp$args_subsample / gwas_fp$args_m
gwas_fp$params <- str_replace_all(gwas_fp$params, "_", " + ")

## Reorder scenario factor to match notebook
gwas_fp$params <- factor(gwas_fp$params,
  levels = c("2xAM", "5xAM", "5xAM + gXe", "5xAM + eVT",
             "5xAM + eVT + gXe", "5xAM + pVT", "5xAM + pVT + gXe"))

write.csv(gwas_fp, file.path(OUT_DIR, "sfig_s8_gwas_fp.csv"),
          row.names = FALSE)
cat("Saved sfig_s8_gwas_fp.csv:", nrow(gwas_fp), "rows\n")

write.csv(math_sim, file.path(OUT_DIR, "sfig_s8_mathematica_sim.csv"),
          row.names = FALSE)
cat("Saved sfig_s8_mathematica_sim.csv:", nrow(math_sim), "rows\n")

## ── 2. S13-S16: GxE/VT parameter grid from merged_tabla_redux_results_0524 ────
## The 0524 dataset has the full 3x3 grid: theta in {0, 0.05, 0.2},
## phi in {0, 0.05, 0.2} with explicit scenario labels for all combinations.
## (The older 011024 only had theta/phi in {0, 0.05}.)

redux <- read.csv(file.path(DATA_DIR, "merged_tabla_redux_results_0524.csv"))

## Drop RM scenarios -- S13-S16 concern 5xAM only
redux <- redux[!grepl("^RM", redux$scenario), ]

## Build facet labels from numeric theta/phi values
redux$GxE_level <- paste0("GxE=", redux$args_phi)
redux$VT_level  <- paste0("VT=", redux$args_theta)

## Create factor versions for faceting
redux$GxE_level <- factor(redux$GxE_level,
  levels = c("GxE=0", "GxE=0.05", "GxE=0.2"))
redux$VT_level <- factor(redux$VT_level,
  levels = c("VT=0", "VT=0.05", "VT=0.2"))

## Scenario label combining GxE + VT
redux$grid_scenario <- paste0(redux$args_kphen, "xAM, ",
                               redux$GxE_level, ", ", redux$VT_level)

## Computed quantities
redux$h2_bias <- redux$he_h2 - redux$h2_true
redux$rg_bias <- redux$he_rg - redux$rg_true

## ── S13/S14: h2 across generations ──────────────────────────────────────────────
## Keep kphen=5 for 5xAM grid
h2_grid <- redux[redux$args_kphen == 5,
  c("seed", "gen", "args_theta", "args_phi", "GxE_level", "VT_level",
    "grid_scenario", "he_h2", "h2_true", "h2_bias")]

write.csv(h2_grid, file.path(OUT_DIR, "sfig_s13s14_h2_grid.csv"),
          row.names = FALSE)
cat("Saved sfig_s13s14_h2_grid.csv:", nrow(h2_grid), "rows\n")

## ── S15: rg across generations ──────────────────────────────────────────────────
rg_grid <- redux[redux$args_kphen == 5,
  c("seed", "gen", "args_theta", "args_phi", "GxE_level", "VT_level",
    "grid_scenario", "he_rg", "rg_true", "rg_bias")]

write.csv(rg_grid, file.path(OUT_DIR, "sfig_s15_rg_grid.csv"),
          row.names = FALSE)
cat("Saved sfig_s15_rg_grid.csv:", nrow(rg_grid), "rows\n")

## ── S16: PGI cross-phenotype slope with GxE/VT ─────────────────────────────────
## rbeta_hat_pgwas = cross-phenotype slope from population GWAS
pgi_grid <- redux[redux$args_kphen == 5,
  c("seed", "gen", "args_theta", "args_phi", "GxE_level", "VT_level",
    "grid_scenario", "rbeta_hat_pgwas", "rbeta_hat_sgwas",
    "rbeta_true", "rg_true")]

write.csv(pgi_grid, file.path(OUT_DIR, "sfig_s16_pgi_grid.csv"),
          row.names = FALSE)
cat("Saved sfig_s16_pgi_grid.csv:", nrow(pgi_grid), "rows\n")

cat("Done: process_sfig_theory_gxe.R\n")
