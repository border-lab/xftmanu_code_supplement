## process_fig1.R
## Loads raw data, computes summary statistics, and saves to processed/
## Figure 1: CCA scree plots, h2/rg under xAM, GWAS T1E inflation, analytic predictions

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
PROCESSED_DIR <- file.path(BASE_DIR, "processed")
dir.create(PROCESSED_DIR, showWarnings = FALSE, recursive = TRUE)

library(reshape2)
library(stringr)

cat("=== Processing Figure 1 data ===\n\n")

## ---------------------------------------------------------------
## Panel A: UKB CCA scree plot
## ---------------------------------------------------------------
cat("--- Panel A: UKB CCA ---\n")

## Load CCA table data (contains chisq/df for significance testing)
load(file.path(BASE_DIR, "data/cca/cca_table_dat.rdata"), verbose = TRUE)
KK <- 14

## Hard-coded complete-cases CCA cumulative redundancies (from notebook)
ccdat <- c(`CV 1` = 0.367035175299488, `CV 2` = 0.517038565488573, `CV 3` = 0.601984317923847,
           `CV 4` = 0.697522369281801, `CV 5` = 0.823705718371409, `CV 6` = 0.847231186585211,
           `CV 7` = 0.882527689139077, `CV 8` = 0.899659822643449, `CV 9` = 0.911256123643412,
           `CV 10` = 0.924392948857031, `CV 11` = 0.93651589132108, `CV 12` = 0.945772218124514,
           `CV 13` = 0.949665007373801, `CV 14` = 0.956550832180996, `CV 15` = 0.963552498585247,
           `CV 16` = 0.970755140854732, `CV 17` = 0.977729335609696, `CV 18` = 0.983474327171582,
           `CV 19` = 0.986492487623242, `CV 20` = 0.988608639968566, `CV 21` = 0.990737600100733,
           `CV 22` = 0.993414303176453, `CV 23` = 0.995491432537665, `CV 24` = 0.997001889358986,
           `CV 25` = 0.997610775141301, `CV 26` = 0.998155194073544, `CV 27` = 0.998834015621939,
           `CV 28` = 0.999256378883214, `CV 29` = 0.999488082189504, `CV 30` = 0.999743284993516,
           `CV 31` = 0.999872313961061, `CV 32` = 0.99996705028616, `CV 33` = 0.999995450042255,
           `CV 34` = 1)

## For Figure 1 panel a, use only Complete Cases up to KK=14
ukb_cca <- data.frame(
  CanonicalVariate = c("", paste("CV", 1:KK)),
  cv = 0:KK,
  cumulative_redundancy = c(0, ccdat[1:KK]),
  stringsAsFactors = FALSE
)

## Count significant canonical vectors (from cca_table_dat)
n_sig_05 <- sum((1 - pchisq(cca_table_dat$chisq, cca_table_dat$df)) < 0.05)
cat("  Significant canonical vectors at alpha=0.05:", n_sig_05, "\n")

## Determine dimensions for 90% and 95%
## CV8 = 0.8997 (~90%), CV14 = 0.9566 (~95%)
## Manuscript: "8 dimensions for 90%, 14 for 95%"
## Use rounding to nearest % for the threshold match
dims_90_ukb <- min(which(round(ukb_cca$cumulative_redundancy, 2) >= 0.90)) - 1
dims_95_ukb <- min(which(ukb_cca$cumulative_redundancy > 0.95)) - 1
cat("  UKB dims for 90%:", dims_90_ukb, "(val =",
    round(ukb_cca$cumulative_redundancy[dims_90_ukb + 1], 4), ")\n")
cat("  UKB dims for 95%:", dims_95_ukb, "(val =",
    round(ukb_cca$cumulative_redundancy[dims_95_ukb + 1], 4), ")\n")

write.csv(ukb_cca, file.path(PROCESSED_DIR, "fig1_ukb_cca.csv"), row.names = FALSE)
cat("  Saved: fig1_ukb_cca.csv\n\n")


## ---------------------------------------------------------------
## Panel B: Taiwan NHIRD CCA scree plot
## ---------------------------------------------------------------
cat("--- Panel B: Taiwan NHIRD CCA ---\n")

## NHIRD CCA cumulative redundancy values (from notebook / nhird_cca.r)
nhird_mean_red <- c(0.261017159845987, 0.404835764140372, 0.532464280704279, 0.643260247888746,
                    0.747218762552395, 0.848773522271279, 0.933659192221925, 0.975194233960818,
                    0.98982819870675, 0.999031285792778, 1)

nhird_cca <- data.frame(
  CanonicalVariate = c("", paste0("CV", 1:11)),
  cv = 0:11,
  cumulative_redundancy = c(0, nhird_mean_red),
  stringsAsFactors = FALSE
)

## Determine dimensions for 90% and 95%
dims_90_nhird <- min(which(round(nhird_cca$cumulative_redundancy, 2) >= 0.90)) - 1
dims_95_nhird <- min(which(nhird_cca$cumulative_redundancy > 0.95)) - 1
cat("  NHIRD dims for 90%:", dims_90_nhird, "\n")
cat("  NHIRD dims for 95%:", dims_95_nhird, "\n")

write.csv(nhird_cca, file.path(PROCESSED_DIR, "fig1_nhird_cca.csv"), row.names = FALSE)
cat("  Saved: fig1_nhird_cca.csv\n\n")


## ---------------------------------------------------------------
## Panels C,D: h2 and rg under 2xAM and 5xAM
## ---------------------------------------------------------------
cat("--- Panels C,D: Simulation h2/rg ---\n")

res <- read.csv(file.path(BASE_DIR, "data/sim_results/merged_tabla_redux_results_011024.csv"))

## Filter to 2xAM and 5xAM scenarios (args_theta==0, args_phi==0, args_rmate==0.2)
res_filt <- res[res$scenario %in% c("2xAM", "5xAM"), ]
cat("  Rows after filtering to 2xAM/5xAM:", nrow(res_filt), "\n")

## Univariate: h2_true, he_h2
mvars_uni <- c("h2_true", "he_h2")
idvars <- c("seed", "gen", "args_kmate", "args_rmate", "args_theta",
            "args_phi", "args_m_causal", "power", "scenario")

udat <- reshape2::melt(res_filt, id.vars = idvars, measure.vars = mvars_uni)
udat$xAM <- ifelse(udat$args_kmate == 2, "2-variate xAM", "5-variate xAM")

## Compute summary stats by gen, scenario, variable
h2_summary <- aggregate(value ~ gen + scenario + variable + xAM,
                        data = udat,
                        FUN = function(x) c(mean = mean(x), sd = sd(x), se = sd(x)/sqrt(length(x)), n = length(x)))
h2_summary <- do.call(data.frame, h2_summary)
names(h2_summary) <- c("gen", "scenario", "variable", "xAM", "mean", "sd", "se", "n")

write.csv(h2_summary, file.path(PROCESSED_DIR, "fig1_h2_summary.csv"), row.names = FALSE)
cat("  Saved: fig1_h2_summary.csv\n")

## Bivariate: rg_true, he_rg
mvars_biv <- c("rg_true", "he_rg")
bdat <- reshape2::melt(res_filt, id.vars = idvars, measure.vars = mvars_biv)
bdat$xAM <- ifelse(bdat$args_kmate == 2, "2-variate xAM", "5-variate xAM")

rg_summary <- aggregate(value ~ gen + scenario + variable + xAM,
                        data = bdat,
                        FUN = function(x) c(mean = mean(x), sd = sd(x), se = sd(x)/sqrt(length(x)), n = length(x)))
rg_summary <- do.call(data.frame, rg_summary)
names(rg_summary) <- c("gen", "scenario", "variable", "xAM", "mean", "sd", "se", "n")

write.csv(rg_summary, file.path(PROCESSED_DIR, "fig1_rg_summary.csv"), row.names = FALSE)
cat("  Saved: fig1_rg_summary.csv\n")

## Also save raw filtered data for plotting (stat_summary needs individual points)
udat_save <- udat[, c("seed", "gen", "scenario", "variable", "value", "xAM", "args_kmate", "power")]
bdat_save <- bdat[, c("seed", "gen", "scenario", "variable", "value", "xAM", "args_kmate", "power")]
write.csv(udat_save, file.path(PROCESSED_DIR, "fig1_h2_raw.csv"), row.names = FALSE)
write.csv(bdat_save, file.path(PROCESSED_DIR, "fig1_rg_raw.csv"), row.names = FALSE)
cat("  Saved: fig1_h2_raw.csv, fig1_rg_raw.csv\n\n")


## ---------------------------------------------------------------
## Panel E: GWAS type-I error inflation
## ---------------------------------------------------------------
cat("--- Panel E: GWAS FP inflation ---\n")

mvars_gwas <- c("sgwas_false_positives_0.05", "pgwas_false_positives_0.05")

mdat <- reshape2::melt(res_filt, id.vars = idvars, measure.vars = mvars_gwas)
mdat$relative_T1R <- mdat$value / 0.05
mdat$GWAS <- ifelse(grepl("^s", mdat$variable), "Sibship", "Population")
mdat$xAM <- ifelse(mdat$args_kmate == 2, "2-variate", "5-variate")

## Summary stats
gwas_summary <- aggregate(relative_T1R ~ gen + scenario + GWAS + xAM + power,
                          data = mdat,
                          FUN = function(x) c(mean = mean(x), median = median(x),
                                              sd = sd(x), se = sd(x)/sqrt(length(x)), n = length(x)))
gwas_summary <- do.call(data.frame, gwas_summary)
names(gwas_summary) <- c("gen", "scenario", "GWAS", "xAM", "power",
                         "mean", "median", "sd", "se", "n")

write.csv(gwas_summary, file.path(PROCESSED_DIR, "fig1_gwas_summary.csv"), row.names = FALSE)

## Save raw data for stat_summary plotting
gwas_raw <- mdat[, c("seed", "gen", "scenario", "GWAS", "xAM", "relative_T1R", "power")]
write.csv(gwas_raw, file.path(PROCESSED_DIR, "fig1_gwas_raw.csv"), row.names = FALSE)
cat("  Saved: fig1_gwas_summary.csv, fig1_gwas_raw.csv\n\n")


## ---------------------------------------------------------------
## Panel F: Analytic on-target vs off-target associations (newhits)
## ---------------------------------------------------------------
cat("--- Panel F: New hits (on-target vs off-target) ---\n")

load(file.path(BASE_DIR, "data/sim_results/newhits.rdata"), verbose = TRUE)

nhdat <- reshape2::melt(newhits,
                        id.vars = c("delta_N", "kpheno", "m_causal"),
                        measure.vars = c("delta_ex_on_target", "delta_ex_off_target"))

nhdat$type <- NA
nhdat$type[nhdat$variable == "delta_ex_off_target"] <- "Off target associations"
nhdat$type[nhdat$variable == "delta_ex_on_target"]  <- "On target associations"
nhdat$type <- factor(nhdat$type, levels = sort(unique(nhdat$type)))

write.csv(nhdat, file.path(PROCESSED_DIR, "fig1_newhits.csv"), row.names = FALSE)
cat("  Saved: fig1_newhits.csv\n\n")


## ---------------------------------------------------------------
## Summary table for in-text verification
## ---------------------------------------------------------------
cat("--- Computing in-text verification numbers ---\n")

## h2 at gen 0 and gen 5 by scenario and variable
h2_verify <- aggregate(value ~ gen + scenario + variable,
                       data = udat[udat$gen %in% c(0, 5), ],
                       FUN = function(x) c(mean = mean(x), sd = sd(x)))
h2_verify <- do.call(data.frame, h2_verify)
names(h2_verify) <- c("gen", "scenario", "variable", "mean", "sd")

## rg at gen 0 and gen 5 by scenario and variable
rg_verify <- aggregate(value ~ gen + scenario + variable,
                       data = bdat[bdat$gen %in% c(0, 5), ],
                       FUN = function(x) c(mean = mean(x), sd = sd(x)))
rg_verify <- do.call(data.frame, rg_verify)
names(rg_verify) <- c("gen", "scenario", "variable", "mean", "sd")

## GWAS FP at gen 5 population only
gwas_verify <- aggregate(relative_T1R ~ gen + GWAS + xAM,
                         data = mdat[mdat$gen == 5 & mdat$GWAS == "Population", ],
                         FUN = function(x) c(mean = mean(x), median = median(x), sd = sd(x)))
gwas_verify <- do.call(data.frame, gwas_verify)
names(gwas_verify) <- c("gen", "GWAS", "xAM", "mean", "median", "sd")

## Save verification tables
verify <- list(h2 = h2_verify, rg = rg_verify, gwas = gwas_verify)
write.csv(h2_verify, file.path(PROCESSED_DIR, "fig1_verify_h2.csv"), row.names = FALSE)
write.csv(rg_verify, file.path(PROCESSED_DIR, "fig1_verify_rg.csv"), row.names = FALSE)
write.csv(gwas_verify, file.path(PROCESSED_DIR, "fig1_verify_gwas.csv"), row.names = FALSE)

## CCA verification
cca_verify <- data.frame(
  dataset = c("UKB", "UKB", "NHIRD", "NHIRD", "UKB"),
  threshold = c("90%", "95%", "90%", "95%", "sig_0.05"),
  n_dims = c(dims_90_ukb, dims_95_ukb, dims_90_nhird, dims_95_nhird, n_sig_05),
  stringsAsFactors = FALSE
)
write.csv(cca_verify, file.path(PROCESSED_DIR, "fig1_verify_cca.csv"), row.names = FALSE)

cat("  Saved verification CSVs\n")
cat("\n=== Processing complete ===\n")
