#!/usr/bin/env Rscript
# Process: h2=0.25 sensitivity figure
# Loads 200 CSV files from h2_025/, assigns scenario labels,
# reshapes for 3 panels (h2, rg, GWAS FP)
# Saves intermediate CSVs to processed/

library(reshape2)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"

# --- Load data ---
datadir <- file.path(BASE_DIR, "data/h2_025")
flist <- list.files(datadir, pattern = "*.csv", full.names = TRUE)
cat("Loading", length(flist), "CSV files from", datadir, "\n")
dat <- do.call(rbind, lapply(flist, read.csv))
cat("Loaded data:", nrow(dat), "rows x", ncol(dat), "cols\n")

# --- Assign scenario labels ---
dat$scenario <- ifelse(dat$args_rmate == 0.0 & dat$args_theta == 0.0, "RM",
                ifelse(dat$args_rmate == 0.2 & dat$args_theta == 0.0, "5xAM",
                ifelse(dat$args_rmate == 0.0 & dat$args_theta == 0.05, "RM + VT (5%)",
                ifelse(dat$args_rmate == 0.2 & dat$args_theta == 0.05, "5xAM + VT (5%)",
                       NA))))
dat <- dat[!is.na(dat$scenario), ]
dat$scenario <- factor(dat$scenario,
                       levels = c("RM", "5xAM", "RM + VT (5%)", "5xAM + VT (5%)"))

cat("Rows per scenario:\n")
print(table(dat$scenario))

# --- Panel b: h2 (true and estimated) ---
h2_long <- melt(dat[, c("gen", "scenario", "seed", "h2_true", "he_h2")],
                id.vars = c("gen", "scenario", "seed"),
                variable.name = "type", value.name = "h2")
h2_long$type <- factor(h2_long$type,
                       levels = c("he_h2", "h2_true"),
                       labels = c("Estimated", "True"))

write.csv(h2_long, file.path(BASE_DIR, "processed/figR_h2_025_h2.csv"),
          row.names = FALSE)
cat("Saved h2 panel data\n")

# --- Panel c: rg (true and estimated) ---
rg_long <- melt(dat[, c("gen", "scenario", "seed", "rg_true", "he_rg")],
                id.vars = c("gen", "scenario", "seed"),
                variable.name = "type", value.name = "rg")
rg_long$type <- factor(rg_long$type,
                       levels = c("he_rg", "rg_true"),
                       labels = c("Estimated", "True"))

write.csv(rg_long, file.path(BASE_DIR, "processed/figR_h2_025_rg.csv"),
          row.names = FALSE)
cat("Saved rg panel data\n")

# --- Panel d: GWAS false positives ---
fp_long <- melt(dat[, c("gen", "scenario", "seed",
                         "pgwas_false_positives_0.05",
                         "sgwas_false_positives_0.05")],
                id.vars = c("gen", "scenario", "seed"),
                variable.name = "gwas_type", value.name = "fpr")
fp_long$gwas_type <- factor(fp_long$gwas_type,
                            levels = c("pgwas_false_positives_0.05",
                                       "sgwas_false_positives_0.05"),
                            labels = c("Population GWAS", "Sibling GWAS"))

write.csv(fp_long, file.path(BASE_DIR, "processed/figR_h2_025_fp.csv"),
          row.names = FALSE)
cat("Saved GWAS FP panel data\n")
