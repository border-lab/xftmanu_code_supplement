## process_fig3.R
## Loads BOTH merged simulation CSVs and produces:
##   - Figure panels b-e data from 011024 (4 basic scenarios, VT 5% only)
##   - Verification data from 0524 (matches manuscript P46/P48 numbers)
##   - Expanded P48 data from 0524 (VT 20%, GxE 20% scenarios)
##
## Datasets:
##   011024 (~750 seeds): 5xAM, 5xAM + GxE, 5xAM + VT, 5xAM + VT + GxE
##   0524   (~950 seeds): above + VT(20%), GxE(20%) combos; VT(5%) numbers
##                        match the manuscript
##
## Source notebook: manu/figure_nb/mFigComplexity.ipynb

library(reshape2)
library(stringr)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
DATA_DIR <- file.path(BASE_DIR, "data", "sim_results")
OUT_DIR  <- file.path(BASE_DIR, "processed")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

## ── 1. Load BOTH datasets ──────────────────────────────────────────────────

res_011024 <- read.csv(file.path(DATA_DIR, "merged_tabla_redux_results_011024.csv"))
cat("Loaded 011024:", nrow(res_011024), "rows,", ncol(res_011024), "columns\n")
cat("  Scenarios:", paste(unique(res_011024$scenario), collapse = ", "), "\n")

res_0524 <- read.csv(file.path(DATA_DIR, "merged_tabla_redux_results_0524.csv"))
cat("Loaded 0524:", nrow(res_0524), "rows,", ncol(res_0524), "columns\n")
cat("  Scenarios:", paste(unique(res_0524$scenario), collapse = ", "), "\n")

## ── 2. Derived columns for 011024 (figure panels) ──────────────────────────

add_derived <- function(df) {
  df$v_tot    <- df$h2_true / (1 - df$h2_true) * 0.5 + 0.5
  df$vbeta    <- df$vbeta_true * df$v_tot
  df$h2_he    <- df$he_h2
  df$rbeta_HE <- df$he_rg
  df$h2_bias  <- df$h2_he - df$h2_true
  df$rg_bias  <- df$he_rg - df$rg_true
  df$sim      <- apply(str_split_fixed(df$X, "_", 5)[, -2], 1,
                       paste, collapse = "_")
  df
}

res_011024 <- add_derived(res_011024)

## Derived columns for 0524 (verification + expanded scenarios)
add_derived_0524 <- function(df) {
  df$h2_he    <- df$he_h2
  df$rbeta_HE <- df$he_rg
  df$h2_bias  <- df$h2_he - df$h2_true
  df$rg_bias  <- df$he_rg - df$rg_true
  df
}

res_0524 <- add_derived_0524(res_0524)

## ── 3. Figure panel data from 011024 (4 basic scenarios) ────────────────────

target_scenarios_011024 <- c("5xAM", "5xAM + GxE", "5xAM + VT",
                             "5xAM + VT + GxE")
gdat <- res_011024[res_011024$scenario %in% target_scenarios_011024, ]
cat("011024 figure data:", nrow(gdat), "rows\n")

## ── 4. Summary table of h2/rg by gen and scenario (for figure) ──────────────

summary_stats <- aggregate(
  gdat[, c("h2_true", "h2_he", "rg_true", "he_rg", "h2_bias", "rg_bias")],
  gdat[, c("gen", "scenario")],
  mean
)

summary_medians <- aggregate(
  gdat[, c("h2_bias", "rg_bias")],
  gdat[, c("gen", "scenario")],
  median
)
names(summary_medians)[3:4] <- c("h2_bias_median", "rg_bias_median")

summary_all <- merge(summary_stats, summary_medians, by = c("gen", "scenario"))

write.csv(summary_all, file.path(OUT_DIR, "fig3_summary.csv"), row.names = FALSE)
cat("Saved fig3_summary.csv:", nrow(summary_all), "rows\n")

## ── 5. Panel b data: h2 true vs estimated ───────────────────────────────────

h2_long <- melt(gdat,
                id.vars = c("gen", "X", "seed", "args_m", "scenario"),
                measure.vars = c("h2_he", "h2_true"))

h2_long$var_label <- as.character(h2_long$variable)
h2_long$var_label[h2_long$variable == "h2_he"]   <- "hat(italic(h))^2"
h2_long$var_label[h2_long$variable == "h2_true"] <- "italic(h)^2"
h2_long$intercept <- 0.5

h2_plot <- aggregate(
  h2_long["value"],
  h2_long[c("gen", "scenario", "variable", "var_label")],
  function(x) c(median = quantile(x, 0.5),
                q10 = quantile(x, 0.1),
                q90 = quantile(x, 0.9),
                mean = mean(x),
                se = sd(x) / sqrt(length(x)))
)
h2_plot <- do.call(data.frame, h2_plot)
names(h2_plot) <- gsub("value\\.", "", names(h2_plot))

write.csv(h2_plot, file.path(OUT_DIR, "fig3_h2_plot.csv"), row.names = FALSE)
cat("Saved fig3_h2_plot.csv:", nrow(h2_plot), "rows\n")

h2_raw <- h2_long[, c("gen", "seed", "scenario", "variable", "var_label",
                       "value", "intercept")]
write.csv(h2_raw, file.path(OUT_DIR, "fig3_h2_raw.csv"), row.names = FALSE)
cat("Saved fig3_h2_raw.csv:", nrow(h2_raw), "rows\n")

## ── 6. Panel c data: rg true vs estimated ───────────────────────────────────

rg_long <- melt(gdat,
                id.vars = c("gen", "X", "seed", "args_m", "scenario"),
                measure.vars = c("rbeta_HE", "rg_true"))

rg_long$var_label <- as.character(rg_long$variable)
rg_long$var_label[rg_long$variable == "rbeta_HE"] <- "hat(italic(r))[beta]"
rg_long$var_label[rg_long$variable == "rg_true"]  <- "italic(r)[score]"
rg_long$intercept <- 0

rg_plot <- aggregate(
  rg_long["value"],
  rg_long[c("gen", "scenario", "variable", "var_label")],
  function(x) c(median = quantile(x, 0.5),
                q10 = quantile(x, 0.1),
                q90 = quantile(x, 0.9),
                mean = mean(x),
                se = sd(x) / sqrt(length(x)))
)
rg_plot <- do.call(data.frame, rg_plot)
names(rg_plot) <- gsub("value\\.", "", names(rg_plot))

write.csv(rg_plot, file.path(OUT_DIR, "fig3_rg_plot.csv"), row.names = FALSE)
cat("Saved fig3_rg_plot.csv:", nrow(rg_plot), "rows\n")

rg_raw <- rg_long[, c("gen", "seed", "scenario", "variable", "var_label",
                       "value", "intercept")]
write.csv(rg_raw, file.path(OUT_DIR, "fig3_rg_raw.csv"), row.names = FALSE)
cat("Saved fig3_rg_raw.csv:", nrow(rg_raw), "rows\n")

## ── 7. Panel d data: GWAS false positive inflation (from 011024) ────────────

fp_vars <- c("sgwas_false_positives_0.05", "pgwas_false_positives_0.05")
idvars  <- c("seed", "gen", "args_kmate", "args_rmate", "args_theta",
             "args_phi", "args_gamma", "args_m_causal", "power", "scenario")

fp_melt <- melt(res_011024[res_011024$scenario %in% target_scenarios_011024, ],
                id.vars = idvars, measure.vars = fp_vars)
fp_melt$relative_T1R <- fp_melt$value / 0.05
fp_melt$GWAS <- ifelse(grepl("^sgwas", fp_melt$variable), "Sibship", "Population")

fp_melt$power_label <- paste(fp_melt$power, "power at alpha=0.05")

fp_summary <- aggregate(
  fp_melt["relative_T1R"],
  fp_melt[c("gen", "scenario", "power", "GWAS")],
  function(x) c(mean = mean(x),
                se = sd(x) / sqrt(length(x)),
                se_1.96 = 1.96 * sd(x) / sqrt(length(x)),
                median = median(x))
)
fp_summary <- do.call(data.frame, fp_summary)
names(fp_summary) <- gsub("relative_T1R\\.", "", names(fp_summary))

write.csv(fp_summary, file.path(OUT_DIR, "fig3_fp_summary.csv"), row.names = FALSE)
cat("Saved fig3_fp_summary.csv:", nrow(fp_summary), "rows\n")

fp_raw <- fp_melt[, c("gen", "seed", "scenario", "power", "GWAS",
                       "relative_T1R")]
write.csv(fp_raw, file.path(OUT_DIR, "fig3_fp_raw.csv"), row.names = FALSE)
cat("Saved fig3_fp_raw.csv:", nrow(fp_raw), "rows\n")

## ── 8. Cross-scenario FP averages at gen 5 (for in-text numbers) ────────────

fp_gen5 <- fp_melt[fp_melt$gen == 5 & fp_melt$GWAS == "Population", ]

fp_gen5_all <- aggregate(relative_T1R ~ scenario, data = fp_gen5, FUN = mean)
names(fp_gen5_all)[2] <- "fp_rate_all_power"

fp_gen5_bypower <- dcast(fp_gen5, scenario ~ power,
                         value.var = "relative_T1R", fun.aggregate = mean)

fp_gen5_merged <- merge(fp_gen5_all, fp_gen5_bypower, by = "scenario")

write.csv(fp_gen5_merged, file.path(OUT_DIR, "fig3_fp_gen5.csv"), row.names = FALSE)
cat("Saved fig3_fp_gen5.csv:", nrow(fp_gen5_merged), "rows\n")

## ── 9. Verification data from 0524 (ALL manuscript numbers) ─────────────────
## The 0524 dataset has scenario names like "5xAM + VT (5%)" etc.

## Scenarios needed for verification (P46 + P48)
verify_scenarios_0524 <- c("5xAM", "5xAM + VT (5%)", "5xAM + VT (5%) + GxE (5%)",
                           "5xAM + GxE (5%)",
                           "5xAM + VT (20%)", "5xAM + VT (20%) + GxE (5%)",
                           "5xAM + VT (20%) + GxE (20%)",
                           "5xAM + VT (5%) + GxE (20%)",
                           "5xAM + GxE (20%)")
vdat <- res_0524[res_0524$scenario %in% verify_scenarios_0524, ]
cat("0524 verify data:", nrow(vdat), "rows across",
    length(unique(vdat$scenario)), "scenarios\n")

## Gen 5 summary for verification
vgen5 <- vdat[vdat$gen == 5, ]
verify_data <- aggregate(
  vgen5[, c("h2_true", "h2_he", "rg_true", "he_rg", "h2_bias", "rg_bias")],
  vgen5[, "scenario", drop = FALSE],
  function(x) c(mean = mean(x), median = median(x))
)
verify_data <- do.call(data.frame, verify_data)
names(verify_data) <- gsub("\\.mean", "_mean", gsub("\\.median", "_median", names(verify_data)))

## Gen 0 for initial conditions
vgen0 <- vdat[vdat$gen == 0, ]
verify_gen0 <- aggregate(
  vgen0[, c("h2_true", "h2_he")],
  vgen0[, "scenario", drop = FALSE],
  mean
)
names(verify_gen0)[2:3] <- c("h2_true_gen0", "h2_he_gen0")

verify_merged <- merge(verify_data, verify_gen0, by = "scenario")

write.csv(verify_merged, file.path(OUT_DIR, "fig3_verify_0524.csv"), row.names = FALSE)
cat("Saved fig3_verify_0524.csv:", nrow(verify_merged), "rows\n")

## ── 10. P48 expanded scenarios from 0524 ────────────────────────────────────
## VT(20%) and GxE(20%) scenarios for the expanded comparison table

p48_scenarios <- c("5xAM",
                   "5xAM + VT (5%)", "5xAM + VT (20%)",
                   "5xAM + GxE (5%)", "5xAM + GxE (20%)",
                   "5xAM + VT (5%) + GxE (5%)",
                   "5xAM + VT (5%) + GxE (20%)",
                   "5xAM + VT (20%) + GxE (5%)",
                   "5xAM + VT (20%) + GxE (20%)")

p48dat <- res_0524[res_0524$scenario %in% p48_scenarios, ]

## Summary by gen and scenario
p48_stats <- aggregate(
  p48dat[, c("h2_true", "h2_he", "rg_true", "he_rg", "h2_bias", "rg_bias")],
  p48dat[, c("gen", "scenario")],
  mean
)

p48_medians <- aggregate(
  p48dat[, c("h2_bias", "rg_bias")],
  p48dat[, c("gen", "scenario")],
  median
)
names(p48_medians)[3:4] <- c("h2_bias_median", "rg_bias_median")

p48_all <- merge(p48_stats, p48_medians, by = c("gen", "scenario"))

write.csv(p48_all, file.path(OUT_DIR, "fig3_p48_expanded.csv"), row.names = FALSE)
cat("Saved fig3_p48_expanded.csv:", nrow(p48_all), "rows\n")

cat("\n=== process_fig3.R complete ===\n")
