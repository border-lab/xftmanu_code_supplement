#!/usr/bin/env Rscript
# Process: Variance Decomposition figure
# Loads merged_res, filters to kphen=5 Figure 3 scenarios, reshapes h2 quantities
# Saves intermediate CSV to processed/

library(reshape2)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"

cat("Loading merged_res data...\n")
dat <- read.csv(file.path(BASE_DIR, "data/sim_results/merged_res_redux_120725.csv"))

# Filter: kphen=5, the 4 Figure 3 scenarios
fig3 <- dat[dat$args_kphen == 5 & !is.na(dat$scenario) &
            dat$scenario %in% c("5xAM", "5xAM + VT", "5xAM + GxE", "5xAM + VT + GxE"), ]

cat("Figure 3 data rows:", nrow(fig3), "\n")
cat("Scenarios:", unique(fig3$scenario), "\n")

fig3$scenario <- factor(fig3$scenario,
                        levels = c("5xAM", "5xAM + VT",
                                   "5xAM + GxE", "5xAM + VT + GxE"))

# Reshape: 3 h2 quantities into long format
h2_vars <- c("vbeta_true", "h2_true", "he_h2")
h2_long <- melt(fig3[, c("gen", "scenario", "seed", h2_vars)],
                id.vars = c("gen", "scenario", "seed"),
                variable.name = "quantity", value.name = "h2")

h2_long$quantity <- factor(h2_long$quantity,
  levels = c("vbeta_true", "h2_true", "he_h2"),
  labels = c("Panmictic (causal var.)",
             "True (Vg/Vy)",
             "Pop. estimated (LDSC)"))

# Compute summary stats: mean and sd per gen x scenario x quantity
agg <- aggregate(h2 ~ gen + scenario + quantity, data = h2_long,
                 FUN = function(x) c(mean = mean(x), sd = sd(x), n = length(x)))
agg <- do.call(data.frame, agg)
names(agg) <- c("gen", "scenario", "quantity", "mean_h2", "sd_h2", "n")

cat("Summary rows:", nrow(agg), "\n")

outfile <- file.path(BASE_DIR, "processed/figR_variance_decomp.csv")
write.csv(agg, outfile, row.names = FALSE)
cat("Saved:", outfile, "\n")

# Also save the raw long data for plotting with stat_summary
outfile2 <- file.path(BASE_DIR, "processed/figR_variance_decomp_long.csv")
write.csv(h2_long, outfile2, row.names = FALSE)
cat("Saved:", outfile2, "\n")
