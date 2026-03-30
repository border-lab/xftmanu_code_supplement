## process_tables.R
## Processes data for Supplementary Tables S2-S5
##   S2: Cross-mate CCA results (adequacies, redundancies, loadings)
##   S3: Cross-mate correlation estimates (UKB)
##   S4: Cross-mate correlation estimates (Taiwan NHIRD)
##   S5: Median true and estimated variance components by scenario
## Saves processed CSVs to processed/

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
PROCESSED_DIR <- file.path(BASE_DIR, "processed")
dir.create(PROCESSED_DIR, showWarnings = FALSE, recursive = TRUE)

cat("=== Processing Supplementary Tables S2-S5 ===\n\n")

## ===============================================================
## Table S2: Cross-mate CCA results
## ===============================================================
cat("--- Table S2: Cross-mate CCA ---\n")

load(file.path(BASE_DIR, "data/cca/cca_table_dat.rdata"), verbose = TRUE)
d <- cca_table_dat

phenotypes <- rownames(d$xcrosscorr)
cvs <- colnames(d$xcrosscorr)
n_cv <- length(cvs)

## S2a: Canonical correlations, adequacies, and redundancies per CV
s2_summary <- data.frame(
  CV         = cvs,
  corr       = d$corr,
  xcanvad    = d$xcanvad,
  ycanvad    = d$ycanvad,
  xvrd       = d$xvrd,
  yvrd       = d$yvrd,
  chisq      = d$chisq,
  df         = d$df,
  stringsAsFactors = FALSE
)
## Cumulative redundancies
s2_summary$xvrd_cum <- cumsum(d$xvrd)
s2_summary$yvrd_cum <- cumsum(d$yvrd)

write.csv(s2_summary, file.path(PROCESSED_DIR, "tableS2_cca_summary.csv"),
          row.names = FALSE)
cat("  Saved tableS2_cca_summary.csv (", nrow(s2_summary), " rows)\n")

## S2b: Cross-loadings (phenotype x CV)
## xcrosscorr = cross-loadings of X set on Y canonical variates (and vice versa)
s2_xcross <- as.data.frame(d$xcrosscorr)
s2_xcross$phenotype <- phenotypes
s2_xcross <- s2_xcross[, c("phenotype", cvs)]

s2_ycross <- as.data.frame(d$ycrosscorr)
s2_ycross$phenotype <- phenotypes
s2_ycross <- s2_ycross[, c("phenotype", cvs)]

write.csv(s2_xcross, file.path(PROCESSED_DIR, "tableS2_xcross_loadings.csv"),
          row.names = FALSE)
write.csv(s2_ycross, file.path(PROCESSED_DIR, "tableS2_ycross_loadings.csv"),
          row.names = FALSE)
cat("  Saved tableS2_xcross_loadings.csv (", nrow(s2_xcross), " rows)\n")
cat("  Saved tableS2_ycross_loadings.csv (", nrow(s2_ycross), " rows)\n")

## S2c: Structure correlations (phenotype x CV)
s2_xstruct <- as.data.frame(d$xstructcorr)
s2_xstruct$phenotype <- phenotypes
s2_xstruct <- s2_xstruct[, c("phenotype", cvs)]

s2_ystruct <- as.data.frame(d$ystructcorr)
s2_ystruct$phenotype <- phenotypes
s2_ystruct <- s2_ystruct[, c("phenotype", cvs)]

write.csv(s2_xstruct, file.path(PROCESSED_DIR, "tableS2_xstruct_corr.csv"),
          row.names = FALSE)
write.csv(s2_ystruct, file.path(PROCESSED_DIR, "tableS2_ystruct_corr.csv"),
          row.names = FALSE)
cat("  Saved tableS2_xstruct_corr.csv (", nrow(s2_xstruct), " rows)\n")
cat("  Saved tableS2_ystruct_corr.csv (", nrow(s2_ystruct), " rows)\n")

## S2d: Canonical coefficients (standardized)
s2_xcoef <- as.data.frame(d$xcoef)
s2_xcoef$phenotype <- phenotypes
s2_xcoef <- s2_xcoef[, c("phenotype", cvs)]

s2_ycoef <- as.data.frame(d$ycoef)
s2_ycoef$phenotype <- phenotypes
s2_ycoef <- s2_ycoef[, c("phenotype", cvs)]

write.csv(s2_xcoef, file.path(PROCESSED_DIR, "tableS2_xcoef.csv"),
          row.names = FALSE)
write.csv(s2_ycoef, file.path(PROCESSED_DIR, "tableS2_ycoef.csv"),
          row.names = FALSE)
cat("  Saved tableS2_xcoef.csv (", nrow(s2_xcoef), " rows)\n")
cat("  Saved tableS2_ycoef.csv (", nrow(s2_ycoef), " rows)\n\n")


## ===============================================================
## Table S3: Cross-mate correlation estimates (UKB)
## ===============================================================
cat("--- Table S3: Cross-mate correlations (UKB) ---\n")

s3_raw <- read.csv(file.path(BASE_DIR, "data/tables/xmate_correlations.csv"))

## Validate expected structure
expected_cols <- c("Trait1", "Trait2",
                   "r.xmate.ivw", "se.xmate.ivw", "p.xmate.ivw",
                   "r.wmate.ivw", "se.wmate.ivw", "p.wmate.ivw",
                   "r.female.female", "r.female.male", "r.male.female", "r.male.male",
                   "se.female.female", "se.female.male", "se.male.female", "se.male.male",
                   "p.female.female", "p.female.male", "p.male.female", "p.male.male")
missing_cols <- setdiff(expected_cols, colnames(s3_raw))
if (length(missing_cols) > 0) {
  warning("Missing expected columns in xmate_correlations.csv: ",
          paste(missing_cols, collapse = ", "))
}

## Drop the row-number column (X) and save clean version
s3 <- s3_raw[, expected_cols]
write.csv(s3, file.path(PROCESSED_DIR, "tableS3_ukb_xmate_correlations.csv"),
          row.names = FALSE)
cat("  Saved tableS3_ukb_xmate_correlations.csv (",
    nrow(s3), " rows x ", ncol(s3), " cols)\n")
cat("  Unique traits:", length(unique(s3$Trait1)), "\n")
cat("  Trait pairs:", nrow(s3), "\n\n")


## ===============================================================
## Table S4: Cross-mate correlation estimates (Taiwan NHIRD)
## ===============================================================
cat("--- Table S4: Cross-mate correlations (Taiwan NHIRD) ---\n")

## Load r and SE matrices
tw_r_raw <- read.csv(file.path(BASE_DIR, "data/taiwan/taiwan_mate_r.csv"))
tw_se_raw <- read.csv(file.path(BASE_DIR, "data/taiwan/taiwan_mate_se.csv"))

## Clean: use first column as row names, remove empty trailing rows
tw_r <- tw_r_raw[tw_r_raw$X != "", ]
rownames(tw_r) <- tw_r$X
tw_r$X <- NULL

tw_se <- tw_se_raw[tw_se_raw$X != "", ]
rownames(tw_se) <- tw_se$X
tw_se$X <- NULL

## Verify r and se have matching dimensions
traits <- rownames(tw_r)
cat("  Taiwan traits:", length(traits), "\n")
cat("  r matrix:", nrow(tw_r), "x", ncol(tw_r), "\n")
cat("  se matrix:", nrow(tw_se), "x", ncol(tw_se), "\n")

## Convert to long format: trait_row, trait_col, r, se
s4_long <- data.frame(
  Trait_row = character(),
  Trait_col = character(),
  r         = numeric(),
  se        = numeric(),
  stringsAsFactors = FALSE
)

for (i in seq_along(traits)) {
  for (j in seq_along(colnames(tw_r))) {
    tr <- traits[i]
    tc <- colnames(tw_r)[j]
    r_val <- as.numeric(tw_r[i, j])
    ## SE matrix may be smaller or have NAs
    se_val <- NA
    if (tr %in% rownames(tw_se) && tc %in% colnames(tw_se)) {
      se_val <- as.numeric(tw_se[tr, tc])
    }
    s4_long <- rbind(s4_long, data.frame(
      Trait_row = tr, Trait_col = tc, r = r_val, se = se_val,
      stringsAsFactors = FALSE
    ))
  }
}

write.csv(s4_long, file.path(PROCESSED_DIR, "tableS4_taiwan_xmate_correlations.csv"),
          row.names = FALSE)
cat("  Saved tableS4_taiwan_xmate_correlations.csv (",
    nrow(s4_long), " rows)\n")

## Also save the wide-format r and se matrices
write.csv(tw_r, file.path(PROCESSED_DIR, "tableS4_taiwan_r_matrix.csv"))
write.csv(tw_se, file.path(PROCESSED_DIR, "tableS4_taiwan_se_matrix.csv"))
cat("  Saved tableS4_taiwan_r_matrix.csv\n")
cat("  Saved tableS4_taiwan_se_matrix.csv\n\n")


## ===============================================================
## Table S5: Median true and estimated variance components
## ===============================================================
cat("--- Table S5: Variance component summaries by scenario ---\n")

sim <- read.csv(file.path(BASE_DIR,
                           "data/sim_results/merged_tabla_redux_results_0524.csv"))
cat("  Loaded sim data:", nrow(sim), "rows,", ncol(sim), "cols\n")
cat("  Scenarios:", paste(unique(sim$scenario), collapse = "; "), "\n")
cat("  Generations:", paste(sort(unique(sim$gen)), collapse = ", "), "\n")

## Key variables to summarize
key_vars <- c("h2_true", "he_h2", "rg_true", "he_rg", "vbeta_true")

## Compute median at gen 0 and gen 5 per scenario
gen_targets <- c(0, 5)

results_list <- list()
for (sc in unique(sim$scenario)) {
  for (g in gen_targets) {
    subset_dat <- sim[sim$scenario == sc & sim$gen == g, ]
    if (nrow(subset_dat) == 0) next
    row <- data.frame(scenario = sc, gen = g, n_sims = nrow(subset_dat),
                      stringsAsFactors = FALSE)
    for (v in key_vars) {
      vals <- subset_dat[[v]]
      row[[paste0(v, "_median")]] <- median(vals, na.rm = TRUE)
      row[[paste0(v, "_q025")]]   <- quantile(vals, 0.025, na.rm = TRUE)
      row[[paste0(v, "_q975")]]   <- quantile(vals, 0.975, na.rm = TRUE)
    }
    results_list[[length(results_list) + 1]] <- row
  }
}

s5 <- do.call(rbind, results_list)
rownames(s5) <- NULL

## Sort: RM first, then 2xAM, then 5xAM variants; within each gen 0 then gen 5
scenario_order <- c("RM", "RM + VT (5%)", "2xAM",
                    "5xAM", "5xAM + GxE (5%)", "5xAM + GxE (20%)",
                    "5xAM + VT (5%)", "5xAM + VT (5%) + GxE (5%)",
                    "5xAM + VT (5%) + GxE (20%)",
                    "5xAM + VT (20%)", "5xAM + VT (20%) + GxE (5%)",
                    "5xAM + VT (20%) + GxE (20%)")
s5$scenario <- factor(s5$scenario, levels = scenario_order)
s5 <- s5[order(s5$scenario, s5$gen), ]
s5$scenario <- as.character(s5$scenario)

write.csv(s5, file.path(PROCESSED_DIR, "tableS5_variance_components.csv"),
          row.names = FALSE)
cat("  Saved tableS5_variance_components.csv (",
    nrow(s5), " rows x ", ncol(s5), " cols)\n")

cat("\n=== Done processing supplementary tables ===\n")
