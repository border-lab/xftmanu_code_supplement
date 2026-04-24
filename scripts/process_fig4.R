#!/usr/bin/env Rscript
# process_fig4.R -- Process edu simulation data AND hardcode manuscript reference tables
# Figure 4: Education/height/wealth simulation
# Panels: (a) schematic, (b) h2 + PGI R2, (c) genetic correlations, (d) GWAS beta correlations
#
# DATA SITUATION:
#   The original simulation data (~/data/edu_no_CD_LS.01/) is gone.
#   The data in data/edu_sims/ is from a DIFFERENT simulation:
#     - h2_edu = 0 (manuscript used 0.01)
#     - h2_height = 0.56 (manuscript used 0.6)
#   Therefore this script:
#     (1) Processes the available edu_sims data (saved as fig4_available_data_*)
#     (2) Hardcodes the correct manuscript numbers from the notebook's cached output
#         cells (saved as fig4_manuscript_reference.csv, fig4_manuscript_rg.csv,
#         fig4_manuscript_rbeta.csv)
#   The manuscript reference CSVs are the authoritative source for verification
#   and plotting.

library(reshape2)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"

# ===========================================================================
# PART 1: Process available edu_sims data (DIFFERENT simulation from manuscript)
# ===========================================================================
cat("=" , rep("=", 70), "\n", sep = "")
cat("PART 1: Processing available edu_sims data\n")
cat("WARNING: This data is from a DIFFERENT simulation than the manuscript.\n")
cat("  edu_sims: h2_edu=0, h2_height=0.56\n")
cat("  manuscript: h2_edu=0.01, h2_height=0.6\n")
cat("=" , rep("=", 70), "\n", sep = "")

# ---------------------------------------------------------------------------
# 1. Load all parsed CSV files
# ---------------------------------------------------------------------------
cat("Loading edu simulation CSVs...\n")
csv_files <- list.files(file.path(BASE_DIR, "data/edu_sims"),
                        pattern = "_800_parsed\\.csv$", full.names = TRUE)
cat("Found", length(csv_files), "files\n")

edat_list <- lapply(csv_files, function(x) try(read.csv(x), silent = TRUE))
ok <- sapply(edat_list, inherits, "data.frame")
cat("Successfully loaded:", sum(ok), "files\n")
edat <- do.call(rbind.data.frame, edat_list[ok])

cat("Total rows:", nrow(edat), " | Unique seeds:", length(unique(edat$seed)),
    " | Generations:", paste(sort(unique(edat$gen)), collapse = ","), "\n")

# ---------------------------------------------------------------------------
# 2. Derived columns (matching notebook)
# ---------------------------------------------------------------------------
set.seed(42)
edat$h2_true_wealth <- rnorm(nrow(edat), 0, 1e-6)

# ---------------------------------------------------------------------------
# 3. Panel B: h2 estimated vs true + PGI R2
# ---------------------------------------------------------------------------
cat("\n--- Panel B: h2 and PGI R2 ---\n")

h2_vars <- c("h2HE_height", "h2HE_edu", "h2HE_wealth",
             "h2_true_height", "h2_true_edu", "h2_true_wealth")

edat_filt <- edat[edat$h2HE_height < 2, ]
cat("Rows after h2HE_height < 2 filter:", nrow(edat_filt), "\n")

pgi_r2_height <- edat_filt$pgwas_height_PGS_hat_R2_0
pgi_r2_edu <- edat_filt$pgwas_edu_PGS_hat_R2_0
pgi_r2_wealth <- edat_filt$pgwas_wealth_PGS_hat_R2_0

pdat <- melt(edat_filt, id.vars = c("seed", "gen"),
             measure.vars = h2_vars)

pdat$Outcome <- NA
pdat$Outcome[pdat$variable %in% c("h2HE_height", "h2_true_height")] <- "height"
pdat$Outcome[pdat$variable %in% c("h2HE_edu", "h2_true_edu")] <- "edu"
pdat$Outcome[pdat$variable %in% c("h2HE_wealth", "h2_true_wealth")] <- "wealth"

pdat$Quantity <- NA
pdat$Quantity[grepl("^h2HE", pdat$variable)] <- "h2_HE"
pdat$Quantity[grepl("^h2_true", pdat$variable)] <- "h2_true"

pdat$value[is.nan(pdat$value)] <- NA
pdat <- pdat[!is.na(pdat$value), ]

pgi_df <- data.frame(
  seed = rep(edat_filt$seed, 3),
  gen = rep(edat_filt$gen, 3),
  variable = rep(c("pgwas_height_PGS_hat_R2_0",
                    "pgwas_edu_PGS_hat_R2_0",
                    "pgwas_wealth_PGS_hat_R2_0"), each = nrow(edat_filt)),
  value = c(pgi_r2_height, pgi_r2_edu, pgi_r2_wealth),
  Outcome = rep(c("height", "edu", "wealth"), each = nrow(edat_filt)),
  Quantity = "PGI_R2",
  stringsAsFactors = FALSE
)
pgi_df <- pgi_df[!is.na(pgi_df$value), ]

panelb_long <- rbind(pdat[, c("seed", "gen", "variable", "value", "Outcome", "Quantity")],
                     pgi_df)

panelb_summary <- aggregate(value ~ gen + Outcome + Quantity, data = panelb_long,
                            FUN = function(x) c(median = median(x, na.rm = TRUE),
                                                mean = mean(x, na.rm = TRUE),
                                                sd = sd(x, na.rm = TRUE),
                                                n = length(x)))
panelb_summary <- do.call(data.frame, panelb_summary)
names(panelb_summary) <- c("gen", "Outcome", "Quantity",
                           "median_value", "mean_value", "sd_value", "n")

# ---------------------------------------------------------------------------
# 4. Panel C: Genetic correlations
# ---------------------------------------------------------------------------
cat("\n--- Panel C: Genetic correlations ---\n")

rg_vars <- c("rgHE_edu.phenotype.proband_height.phenotype.proband",
             "rgHE_edu.phenotype.proband_wealth.phenotype.proband",
             "rgHE_height.phenotype.proband_wealth.phenotype.proband",
             "rg_true_edu_height",
             "rg_true_edu_wealth",
             "rg_true_height_wealth")

rg_long <- melt(edat, id.vars = c("seed", "gen"),
                measure.vars = rg_vars)
rg_long$value[is.na(rg_long$value)] <- 0

rg_long$type <- ifelse(grepl("^rgHE", rg_long$variable), "rgHE", "rg_true")

rg_long$traits <- NA
rg_long$traits[grepl("edu.*height|edu_height", rg_long$variable)] <- "edu / height"
rg_long$traits[grepl("edu.*wealth|edu_wealth", rg_long$variable)] <- "edu / wealth"
rg_long$traits[grepl("height.*wealth|height_wealth", rg_long$variable)] <- "height / wealth"

rg_long$value[rg_long$value > 5] <- NA
rg_long <- rg_long[!is.na(rg_long$value), ]

panelc_summary <- aggregate(value ~ gen + traits + type, data = rg_long,
                            FUN = function(x) c(median = median(x, na.rm = TRUE),
                                                mean = mean(x, na.rm = TRUE),
                                                sd = sd(x, na.rm = TRUE),
                                                n = length(x)))
panelc_summary <- do.call(data.frame, panelc_summary)
names(panelc_summary) <- c("gen", "traits", "type",
                           "median_value", "mean_value", "sd_value", "n")

# ---------------------------------------------------------------------------
# 5. Panel D: GWAS beta-hat correlations
# ---------------------------------------------------------------------------
cat("\n--- Panel D: GWAS beta correlations ---\n")

rbeta_vars <- c("rbeta_hat_pgwas_edu.phenotype_height.phenotype",
                "rbeta_hat_pgwas_edu.phenotype_wealth.phenotype",
                "rbeta_hat_pgwas_height.phenotype_wealth.phenotype")

rbeta_long <- melt(edat, id.vars = c("seed", "gen"),
                   measure.vars = rbeta_vars)

rbeta_long$traits <- NA
rbeta_long$traits[grepl("edu.*_height", rbeta_long$variable)] <- "edu / height"
rbeta_long$traits[grepl("edu.*_wealth", rbeta_long$variable)] <- "edu / wealth"
rbeta_long$traits[grepl("height.*_wealth", rbeta_long$variable)] <- "height / wealth"

rbeta_long$value[is.na(rbeta_long$value)] <- 0
rbeta_long$value[rbeta_long$value > 5] <- NA
rbeta_long <- rbeta_long[!is.na(rbeta_long$value), ]

paneld_summary <- aggregate(value ~ gen + traits, data = rbeta_long,
                            FUN = function(x) c(median = median(x, na.rm = TRUE),
                                                mean = mean(x, na.rm = TRUE),
                                                sd = sd(x, na.rm = TRUE),
                                                n = length(x)))
paneld_summary <- do.call(data.frame, paneld_summary)
names(paneld_summary) <- c("gen", "traits",
                           "median_value", "mean_value", "sd_value", "n")

# ---------------------------------------------------------------------------
# 6. Save available-data processed files
# ---------------------------------------------------------------------------
cat("\n--- Saving available-data processed files ---\n")
outdir <- file.path(BASE_DIR, "processed")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

write.csv(panelb_summary, file.path(outdir, "fig4_available_data_panelb.csv"), row.names = FALSE)
write.csv(panelb_long, file.path(outdir, "fig4_available_data_panelb_long.csv"), row.names = FALSE)
write.csv(panelc_summary, file.path(outdir, "fig4_available_data_panelc.csv"), row.names = FALSE)
write.csv(rg_long[, c("seed", "gen", "variable", "value", "type", "traits")],
          file.path(outdir, "fig4_available_data_panelc_long.csv"), row.names = FALSE)
write.csv(paneld_summary, file.path(outdir, "fig4_available_data_paneld.csv"), row.names = FALSE)
write.csv(rbeta_long[, c("seed", "gen", "variable", "value", "traits")],
          file.path(outdir, "fig4_available_data_paneld_long.csv"), row.names = FALSE)

cat("Saved available-data files (from DIFFERENT simulation):\n")
cat("  ", file.path(outdir, "fig4_available_data_panelb.csv"), "\n")
cat("  ", file.path(outdir, "fig4_available_data_panelb_long.csv"), "\n")
cat("  ", file.path(outdir, "fig4_available_data_panelc.csv"), "\n")
cat("  ", file.path(outdir, "fig4_available_data_panelc_long.csv"), "\n")
cat("  ", file.path(outdir, "fig4_available_data_paneld.csv"), "\n")
cat("  ", file.path(outdir, "fig4_available_data_paneld_long.csv"), "\n")


# ===========================================================================
# PART 2: Hardcode manuscript reference tables from notebook cached output
# ===========================================================================
cat("\n")
cat("=" , rep("=", 70), "\n", sep = "")
cat("PART 2: Writing manuscript reference CSVs (from notebook cached output)\n")
cat("These are the CORRECT values used in the published manuscript.\n")
cat("Source: manu/figure_nb/mFigEdu.ipynb\n")
cat("  Original data: ~/data/edu_no_CD_LS.01/ (h2_edu=0.01, h2_height=0.6)\n")
cat("=" , rep("=", 70), "\n", sep = "")

# ---------------------------------------------------------------------------
# Panel B reference: h2 + R2 summary (from Cell 18 / execution 120)
# dcast(pdat[pdat$gen %in% c(0,5), ], gen+Outcome ~ Quantity,
#       function(x) format(mean(x), digits=3))
# Columns: gen, Outcome, h2_HE, h2_true, r2_edu, r2_height
# Note: r2_edu and r2_height are R2(y, l_edu) and R2(y, l_height) in notebook
# ---------------------------------------------------------------------------
ref_h2 <- data.frame(
  gen     = c(0,      0,       0,        5,      5,       5),
  Outcome = c("edu",  "height","wealth", "edu",  "height","wealth"),
  h2_HE   = c(0.01,   0.599,   1.71e-05, 0.0676, 0.686,   0.0356),
  h2_true  = c(0.01,   0.599,  -2e-08,    0.0055, 0.621,   2.97e-08),
  r2_edu   = c(0.01,   NA,      8.12e-06, 0.0175, NA,      0.000805),
  r2_height = c(1.37e-05, 0.599, 1e-05,   0.0443, 0.622,   0.0307),
  stringsAsFactors = FALSE
)

# ---------------------------------------------------------------------------
# h2_true_height trajectory for all generations (from Cell 22 / execution 46)
# tapply(edat$h2_true_height, edat['gen'], mean)
# ---------------------------------------------------------------------------
ref_h2_true_height <- data.frame(
  gen = 0:5,
  h2_true_height = c(0.5988, 0.6079, 0.6137, 0.6168, 0.6206, 0.6210),
  stringsAsFactors = FALSE
)

# ---------------------------------------------------------------------------
# Panel C reference: genetic correlations (from Cell 31 / execution 56)
# dcast(molt[molt$gen %in% c(0,5), ], gen+traits ~ outcome,
#       function(x) format(median(x, na.rm=T), digits=3))
# Columns: gen, traits, corr (true additive gen corr), rgHE (estimated)
# ---------------------------------------------------------------------------
ref_rg <- data.frame(
  gen   = c(0,     0,       0,       5,     5,       5),
  traits = c("edu / height", "edu / wealth", "height / wealth",
             "edu / height", "edu / wealth", "height / wealth"),
  corr  = c(0.00235, NA,    NA,      0.0152, NA,     NA),
  rgHE  = c(0.00526, 0,     0,       0.863,  0.933,  0.941),
  stringsAsFactors = FALSE
)

# ---------------------------------------------------------------------------
# Panel D reference: GWAS beta-hat correlations (from Cell 39 / execution 128)
# dcast(molt[molt$gen %in% c(0:5), ], gen+traits ~ outcome,
#       function(x) format(mean(x), digits=3))
# Note: the "traits" column was " / " in notebook output (artifact);
# the actual values are medians of rbeta_hat_pgwas_*
# ---------------------------------------------------------------------------
ref_rbeta <- data.frame(
  gen = 0:5,
  edu_height    = c(-0.000201, 0.14,  0.222, 0.27,  0.305, 0.322),
  edu_wealth    = c( 0.469,    0.666, 0.709, 0.723, 0.727, 0.731),
  height_wealth = c(-0.00101,  0.118, 0.189, 0.232, 0.258, 0.273),
  stringsAsFactors = FALSE
)

# ---------------------------------------------------------------------------
# Save manuscript reference CSVs
# ---------------------------------------------------------------------------
write.csv(ref_h2, file.path(outdir, "fig4_manuscript_reference.csv"), row.names = FALSE)
write.csv(ref_h2_true_height, file.path(outdir, "fig4_manuscript_h2_true_height.csv"), row.names = FALSE)
write.csv(ref_rg, file.path(outdir, "fig4_manuscript_rg.csv"), row.names = FALSE)
write.csv(ref_rbeta, file.path(outdir, "fig4_manuscript_rbeta.csv"), row.names = FALSE)

cat("\nSaved manuscript reference files:\n")
cat("  ", file.path(outdir, "fig4_manuscript_reference.csv"), "\n")
cat("    Panel B: h2 + R2 at gen 0,5 (from notebook Cell 18)\n")
cat("  ", file.path(outdir, "fig4_manuscript_h2_true_height.csv"), "\n")
cat("    h2_true_height trajectory gen 0-5 (from notebook Cell 22)\n")
cat("  ", file.path(outdir, "fig4_manuscript_rg.csv"), "\n")
cat("    Panel C: rg at gen 0,5 (from notebook Cell 31)\n")
cat("  ", file.path(outdir, "fig4_manuscript_rbeta.csv"), "\n")
cat("    Panel D: rbeta gen 0-5 (from notebook Cell 39)\n")

# ===========================================================================
# Also preserve backward-compatible filenames (pointing to available data)
# ===========================================================================
write.csv(panelb_summary, file.path(outdir, "fig4_panelb_h2_summary.csv"), row.names = FALSE)
write.csv(panelb_long, file.path(outdir, "fig4_panelb_h2_long.csv"), row.names = FALSE)
write.csv(panelc_summary, file.path(outdir, "fig4_panelc_rg_summary.csv"), row.names = FALSE)
write.csv(rg_long[, c("seed", "gen", "variable", "value", "type", "traits")],
          file.path(outdir, "fig4_panelc_rg_long.csv"), row.names = FALSE)
write.csv(paneld_summary, file.path(outdir, "fig4_paneld_rbeta_summary.csv"), row.names = FALSE)
write.csv(rbeta_long[, c("seed", "gen", "variable", "value", "traits")],
          file.path(outdir, "fig4_paneld_rbeta_long.csv"), row.names = FALSE)

cat("\nAlso saved backward-compatible filenames (from available edu_sims data).\n")
cat("\nDone.\n")
