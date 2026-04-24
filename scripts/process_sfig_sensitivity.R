## process_sfig_sensitivity.R
## Processes data for supplementary figures S5 and S6:
##   S5: rg under xAM with r=0.1 vs r=0.2, with/without VT
##   S6: rg under unidimensional xAM with fixed latent correlation (~0.5)
## Saves intermediate CSVs to processed/

library(reshape2)
library(stringr)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
DATA_DIR <- file.path(BASE_DIR, "data", "sim_results")
OUT_DIR  <- file.path(BASE_DIR, "processed")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

## ── 1. Load data ───────────────────────────────────────────────────────────────

## merged_tabla: has rmate=0.2 scenarios (used for S5 r=0.2 panel)
tabla <- read.csv(file.path(DATA_DIR, "merged_tabla_redux_results_011024.csv"))

## merged_res_redux: has rmate=0.1, varied theta/phi (used for S5 r=0.1 panel + S6)
redux <- read.csv(file.path(DATA_DIR, "merged_res_redux_120725.csv"))

## ── 2. S5: rg under 5xAM with r=0.1 vs r=0.2 ────────────────────────────────
## Compare he_rg across generations for kphen=5, rmate in {0.1, 0.2}
## with/without VT and GxE

## r=0.2 panel (from merged_tabla, kphen=5)
s5_r02 <- tabla[tabla$args_kphen == 5, ]
s5_r02 <- s5_r02[s5_r02$scenario %in% c("5xAM", "5xAM + GxE",
                                          "5xAM + VT", "5xAM + VT + GxE"), ]
s5_r02$rmate <- 0.2

## r=0.1 panel (from merged_res_redux, kphen=5, standard scenarios only)
## Filter to the standard theta/phi combos (0 and 0.05)
s5_r01 <- redux[redux$args_kphen == 5 &
                redux$args_theta %in% c(0, 0.05) &
                redux$args_phi %in% c(0, 0.05), ]
## Build scenario labels matching the merged_tabla convention
s5_r01$scenario <- "5xAM"
s5_r01$scenario[s5_r01$args_theta != 0 & s5_r01$args_phi == 0]  <- "5xAM + VT"
s5_r01$scenario[s5_r01$args_theta == 0 & s5_r01$args_phi != 0]  <- "5xAM + GxE"
s5_r01$scenario[s5_r01$args_theta != 0 & s5_r01$args_phi != 0]  <- "5xAM + VT + GxE"
s5_r01$rmate <- 0.1

## Also include 2xAM as baseline from merged_res_redux
s5_r01_2xam <- redux[redux$args_kphen == 2 &
                      redux$args_theta == 0 &
                      redux$args_phi == 0, ]
s5_r01_2xam$scenario <- "2xAM"
s5_r01_2xam$rmate <- 0.1

## Also include 2xAM baseline from merged_tabla
s5_r02_2xam <- tabla[tabla$scenario == "2xAM", ]
s5_r02_2xam$rmate <- 0.2

## Shared columns for combining
keep_cols <- c("seed", "gen", "scenario", "rmate",
               "args_kphen", "args_rmate", "args_theta", "args_phi",
               "he_rg", "rg_true", "he_h2", "h2_true")

## Ensure columns exist
safe_select <- function(df, cols) {
  existing <- cols[cols %in% names(df)]
  df[, existing, drop = FALSE]
}

s5_combined <- rbind(
  safe_select(s5_r02, keep_cols),
  safe_select(s5_r02_2xam, keep_cols),
  safe_select(s5_r01, keep_cols),
  safe_select(s5_r01_2xam, keep_cols)
)

write.csv(s5_combined, file.path(OUT_DIR, "sfig_s5_rg_sensitivity.csv"),
          row.names = FALSE)
cat("Saved sfig_s5_rg_sensitivity.csv:", nrow(s5_combined), "rows\n")

## ── 3. S6: rg under xAM with fixed latent correlation ≈ 0.5 ──────────────────
## latent_r = rmate * kphen, so for latent_r ≈ 0.5:
##   2xAM with rmate=0.25 -> nope, only rmate=0.1 in redux
## Actually from FigSX_reconstruction.R: scenarios are named like
##   "2xAM.1", "2xAM.25", "3xAM.167", "4xAM.125", "5xAM.1"
## meaning kphen=kmate and latent_r = rmate * kphen
## With rmate=0.1:  2xAM -> latent_r = 0.2, 5xAM -> latent_r = 0.5
## So the "fixed latent r=0.5" comparison uses 5xAM baseline with different kphen
## The data available has kphen=2 (latent=0.2) and kphen=5 (latent=0.5)

## For S6 we show how rg evolves under different kphen with rmate=0.1
## Filter merged_res_redux to baseline (no VT, no GxE)
s6_dat <- redux[redux$args_theta == 0 & redux$args_phi == 0, ]
s6_dat$latent_r <- s6_dat$args_rmate * s6_dat$args_kphen

## Create scenario label: KxAM.rmate
s6_dat$scenario_label <- paste0(s6_dat$args_kphen, "xAM.",
                                 s6_dat$args_rmate)
s6_dat$scenario_label <- paste0(s6_dat$scenario_label,
                                 " (latent r = ",
                                 round(s6_dat$latent_r, 2), ")")

s6_keep <- c("seed", "gen", "scenario_label", "latent_r",
             "args_kphen", "args_kmate", "args_rmate",
             "he_rg", "rg_true", "he_h2", "h2_true")
s6_out <- s6_dat[, s6_keep[s6_keep %in% names(s6_dat)]]

write.csv(s6_out, file.path(OUT_DIR, "sfig_s6_fixed_latent.csv"),
          row.names = FALSE)
cat("Saved sfig_s6_fixed_latent.csv:", nrow(s6_out), "rows\n")

cat("Done: process_sfig_sensitivity.R\n")
