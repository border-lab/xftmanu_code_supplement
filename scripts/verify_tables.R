## verify_tables.R
## Reads processed table data and prints key values for comparison
## with manuscript Tables S2-S5

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
PROCESSED_DIR <- file.path(BASE_DIR, "processed")

cat("============================================================\n")
cat("  SUPPLEMENTARY TABLES S2-S5: VERIFICATION\n")
cat("============================================================\n\n")

## ===============================================================
## Table S2: Cross-mate CCA results
## ===============================================================
cat("============================================================\n")
cat("  TABLE S2: Cross-mate CCA\n")
cat("============================================================\n\n")

s2_sum <- read.csv(file.path(PROCESSED_DIR, "tableS2_cca_summary.csv"))
s2_xcross <- read.csv(file.path(PROCESSED_DIR, "tableS2_xcross_loadings.csv"))
s2_ycross <- read.csv(file.path(PROCESSED_DIR, "tableS2_ycross_loadings.csv"))
s2_xstruct <- read.csv(file.path(PROCESSED_DIR, "tableS2_xstruct_corr.csv"))
s2_ystruct <- read.csv(file.path(PROCESSED_DIR, "tableS2_ystruct_corr.csv"))
s2_xcoef <- read.csv(file.path(PROCESSED_DIR, "tableS2_xcoef.csv"))
s2_ycoef <- read.csv(file.path(PROCESSED_DIR, "tableS2_ycoef.csv"))

cat("--- Canonical correlations ---\n")
cat(sprintf("  %-8s  %8s  %8s  %8s  %8s  %8s  %10s  %6s  %10s  %10s\n",
            "CV", "corr", "xcanvad", "ycanvad", "xvrd", "yvrd",
            "chisq", "df", "xvrd_cum", "yvrd_cum"))
for (i in 1:min(14, nrow(s2_sum))) {
  cat(sprintf("  %-8s  %8.4f  %8.5f  %8.5f  %8.5f  %8.5f  %10.1f  %6.0f  %10.5f  %10.5f\n",
              s2_sum$CV[i], s2_sum$corr[i],
              s2_sum$xcanvad[i], s2_sum$ycanvad[i],
              s2_sum$xvrd[i], s2_sum$yvrd[i],
              s2_sum$chisq[i], s2_sum$df[i],
              s2_sum$xvrd_cum[i], s2_sum$yvrd_cum[i]))
}
cat("  ... (", nrow(s2_sum), " CVs total)\n\n")

cat("--- Total variance explained (cumulative redundancy) ---\n")
cat(sprintf("  X-set total redundancy: %.5f\n", sum(s2_sum$xvrd)))
cat(sprintf("  Y-set total redundancy: %.5f\n", sum(s2_sum$yvrd)))
cat("\n")

## Show top loadings for first 3 CVs
cat("--- Top 5 cross-loadings for CVs 1-3 (X set / wives) ---\n")
for (cv_idx in 1:3) {
  cv_col <- paste0("CV.", cv_idx)
  vals <- s2_xcross[[cv_col]]
  names(vals) <- s2_xcross$phenotype
  top5 <- sort(abs(vals), decreasing = TRUE)[1:5]
  cat(sprintf("  CV %d:\n", cv_idx))
  for (nm in names(top5)) {
    cat(sprintf("    %-50s  %8.4f\n", nm, vals[nm]))
  }
}
cat("\n")

cat("--- Top 5 cross-loadings for CVs 1-3 (Y set / husbands) ---\n")
for (cv_idx in 1:3) {
  cv_col <- paste0("CV.", cv_idx)
  vals <- s2_ycross[[cv_col]]
  names(vals) <- s2_ycross$phenotype
  top5 <- sort(abs(vals), decreasing = TRUE)[1:5]
  cat(sprintf("  CV %d:\n", cv_idx))
  for (nm in names(top5)) {
    cat(sprintf("    %-50s  %8.4f\n", nm, vals[nm]))
  }
}
cat("\n")

cat("--- Structure correlations sanity check (CV1, first 5 phenotypes) ---\n")
cat("  X-set structure correlations:\n")
for (i in 1:5) {
  cat(sprintf("    %-50s  %8.4f\n",
              s2_xstruct$phenotype[i], s2_xstruct$CV.1[i]))
}
cat("  Y-set structure correlations:\n")
for (i in 1:5) {
  cat(sprintf("    %-50s  %8.4f\n",
              s2_ystruct$phenotype[i], s2_ystruct$CV.1[i]))
}
cat("\n")


## ===============================================================
## Table S3: Cross-mate correlation estimates (UKB)
## ===============================================================
cat("============================================================\n")
cat("  TABLE S3: Cross-mate correlations (UKB)\n")
cat("============================================================\n\n")

s3 <- read.csv(file.path(PROCESSED_DIR, "tableS3_ukb_xmate_correlations.csv"))
cat("  Rows:", nrow(s3), " | Unique traits:", length(unique(s3$Trait1)), "\n\n")

## Show diagonal (same-trait cross-mate r)
diag_rows <- s3[s3$Trait1 == s3$Trait2, ]
cat("--- Same-trait cross-mate correlations (diagonal) ---\n")
cat(sprintf("  %-40s  %8s  %8s  %10s  %10s\n",
            "Trait", "r.xmate", "se", "r.f:m", "r.m:f"))
for (i in 1:min(15, nrow(diag_rows))) {
  cat(sprintf("  %-40s  %8.4f  %8.4f  %10.4f  %10.4f\n",
              diag_rows$Trait1[i],
              diag_rows$r.xmate.ivw[i], diag_rows$se.xmate.ivw[i],
              diag_rows$r.female.male[i], diag_rows$r.male.female[i]))
}
cat("  ... (", nrow(diag_rows), " diagonal entries total)\n\n")

## Show a few key off-diagonal correlations
cat("--- Select off-diagonal cross-mate correlations ---\n")
pairs_to_show <- list(
  c("eduyears", "bmi"),
  c("standing_height", "weight"),
  c("bmi", "body_fat_percentage"),
  c("alcohol_intake_frequency", "eduyears"),
  c("neuroticism_score", "self_reported_health")
)
cat(sprintf("  %-25s %-25s  %8s  %8s\n", "Trait1", "Trait2", "r.xmate", "se"))
for (pair in pairs_to_show) {
  row <- s3[s3$Trait1 == pair[1] & s3$Trait2 == pair[2], ]
  if (nrow(row) == 1) {
    cat(sprintf("  %-25s %-25s  %8.4f  %8.4f\n",
                pair[1], pair[2], row$r.xmate.ivw, row$se.xmate.ivw))
  } else {
    cat(sprintf("  %-25s %-25s  NOT FOUND\n", pair[1], pair[2]))
  }
}
cat("\n")

## Summary statistics
cat("--- Distribution of same-trait cross-mate r ---\n")
cat(sprintf("  Mean:   %.4f\n", mean(diag_rows$r.xmate.ivw, na.rm = TRUE)))
cat(sprintf("  Median: %.4f\n", median(diag_rows$r.xmate.ivw, na.rm = TRUE)))
cat(sprintf("  Range:  [%.4f, %.4f]\n",
            min(diag_rows$r.xmate.ivw, na.rm = TRUE),
            max(diag_rows$r.xmate.ivw, na.rm = TRUE)))
cat("\n")


## ===============================================================
## Table S4: Cross-mate correlations (Taiwan NHIRD)
## ===============================================================
cat("============================================================\n")
cat("  TABLE S4: Cross-mate correlations (Taiwan NHIRD)\n")
cat("============================================================\n\n")

s4 <- read.csv(file.path(PROCESSED_DIR, "tableS4_taiwan_xmate_correlations.csv"))
tw_r <- read.csv(file.path(PROCESSED_DIR, "tableS4_taiwan_r_matrix.csv"),
                 row.names = 1)
tw_se <- read.csv(file.path(PROCESSED_DIR, "tableS4_taiwan_se_matrix.csv"),
                  row.names = 1)

cat("  Long-format rows:", nrow(s4), "\n")
cat("  R matrix:", nrow(tw_r), "x", ncol(tw_r), "\n")
cat("  SE matrix:", nrow(tw_se), "x", ncol(tw_se), "\n\n")

## Show cross-sex block: female traits vs male traits
## Extract female trait names and male trait names
all_traits <- rownames(tw_r)
f_traits <- all_traits[grepl("^f_", all_traits)]
m_traits <- all_traits[grepl("^m_", all_traits)]

cat("--- Cross-sex same-disorder correlations (r +/- SE) ---\n")
## Match traits across sexes
for (ft in f_traits) {
  base <- sub("^f_", "", ft)
  mt <- paste0("m_", base)
  if (mt %in% m_traits && ft %in% rownames(tw_r) && mt %in% colnames(tw_r)) {
    r_val <- tw_r[ft, mt]
    se_val <- NA
    if (ft %in% rownames(tw_se) && mt %in% colnames(tw_se)) {
      se_val <- tw_se[ft, mt]
    }
    cat(sprintf("  %-15s  r = %7.4f  se = %s\n",
                base, as.numeric(r_val),
                ifelse(is.na(se_val), "  NA  ", sprintf("%.4f", as.numeric(se_val)))))
  }
}
cat("\n")

## Show a few cross-disorder entries
cat("--- Select cross-disorder correlations ---\n")
cross_pairs <- list(
  c("f_SCZ", "m_BPD"),
  c("f_MDD", "m_Anxiety"),
  c("f_ADHD", "m_SUD"),
  c("f_SCZ", "m_MDD"),
  c("f_DM1", "m_DM1")
)
for (pair in cross_pairs) {
  ft <- pair[1]; mt <- pair[2]
  if (ft %in% rownames(tw_r) && mt %in% colnames(tw_r)) {
    r_val <- as.numeric(tw_r[ft, mt])
    se_val <- NA
    if (ft %in% rownames(tw_se) && mt %in% colnames(tw_se)) {
      se_val <- as.numeric(tw_se[ft, mt])
    }
    cat(sprintf("  %-10s x %-12s  r = %7.4f  se = %s\n",
                ft, mt, r_val,
                ifelse(is.na(se_val), "  NA  ", sprintf("%.4f", se_val))))
  }
}
cat("\n")

## Compare with code supplement nhird_cors.csv
nhird_supp <- tryCatch(
  read.csv(file.path(BASE_DIR, "xftmanu_code_supplement/nhird_cors.csv")),
  error = function(e) NULL
)
if (!is.null(nhird_supp)) {
  cat("--- Concordance check: taiwan_mate_r vs nhird_cors.csv (code supplement) ---\n")
  ## nhird_cors has a subset of traits (no ASD, AN, DM2)
  nhird_traits <- colnames(nhird_supp)
  common_f <- intersect(nhird_traits[grepl("^f_", nhird_traits)], f_traits)
  n_match <- 0; n_mismatch <- 0
  for (ft in common_f) {
    for (mt in intersect(nhird_traits[grepl("^m_", nhird_traits)], m_traits)) {
      r_tw <- as.numeric(tw_r[ft, mt])
      idx_row <- which(nhird_traits == ft)
      r_nh <- as.numeric(nhird_supp[idx_row, mt])
      if (!is.na(r_tw) && !is.na(r_nh)) {
        if (abs(r_tw - r_nh) < 0.001) {
          n_match <- n_match + 1
        } else {
          n_mismatch <- n_mismatch + 1
          if (n_mismatch <= 5) {
            cat(sprintf("  MISMATCH: %s x %s  taiwan=%.4f  nhird_cors=%.4f\n",
                        ft, mt, r_tw, r_nh))
          }
        }
      }
    }
  }
  cat(sprintf("  Matching values: %d | Mismatches: %d\n\n", n_match, n_mismatch))
}


## ===============================================================
## Table S5: Variance component summaries
## ===============================================================
cat("============================================================\n")
cat("  TABLE S5: Variance component medians by scenario\n")
cat("============================================================\n\n")

s5 <- read.csv(file.path(PROCESSED_DIR, "tableS5_variance_components.csv"))

cat(sprintf("  %-40s %4s %5s  %7s  %7s  %7s  %7s  %7s\n",
            "Scenario", "Gen", "N",
            "h2_true", "he_h2", "rg_true", "he_rg", "vb_true"))
cat(paste0("  ", paste(rep("-", 115), collapse = ""), "\n"))

for (i in seq_len(nrow(s5))) {
  cat(sprintf("  %-40s %4d %5d  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f\n",
              s5$scenario[i], s5$gen[i], s5$n_sims[i],
              s5$h2_true_median[i], s5$he_h2_median[i],
              s5$rg_true_median[i], s5$he_rg_median[i],
              s5$vbeta_true_median[i]))
}
cat("\n")

## Print key comparisons for manuscript verification
cat("--- Key comparisons ---\n\n")

## RM: h2 should not change
rm_g0 <- s5[s5$scenario == "RM" & s5$gen == 0, ]
rm_g5 <- s5[s5$scenario == "RM" & s5$gen == 5, ]
cat("  RM baseline (gen 0):\n")
cat(sprintf("    h2_true=%.4f  he_h2=%.4f  rg_true=%.4f  he_rg=%.4f  vbeta=%.4f\n",
            rm_g0$h2_true_median, rm_g0$he_h2_median,
            rm_g0$rg_true_median, rm_g0$he_rg_median,
            rm_g0$vbeta_true_median))
cat("  RM after 5 gen:\n")
cat(sprintf("    h2_true=%.4f  he_h2=%.4f  rg_true=%.4f  he_rg=%.4f  vbeta=%.4f\n",
            rm_g5$h2_true_median, rm_g5$he_h2_median,
            rm_g5$rg_true_median, rm_g5$he_rg_median,
            rm_g5$vbeta_true_median))
cat("\n")

## 5xAM: h2 and rg should increase
am5_g0 <- s5[s5$scenario == "5xAM" & s5$gen == 0, ]
am5_g5 <- s5[s5$scenario == "5xAM" & s5$gen == 5, ]
cat("  5xAM baseline (gen 0):\n")
cat(sprintf("    h2_true=%.4f  he_h2=%.4f  rg_true=%.4f  he_rg=%.4f  vbeta=%.4f\n",
            am5_g0$h2_true_median, am5_g0$he_h2_median,
            am5_g0$rg_true_median, am5_g0$he_rg_median,
            am5_g0$vbeta_true_median))
cat("  5xAM after 5 gen:\n")
cat(sprintf("    h2_true=%.4f  he_h2=%.4f  rg_true=%.4f  he_rg=%.4f  vbeta=%.4f\n",
            am5_g5$h2_true_median, am5_g5$he_h2_median,
            am5_g5$rg_true_median, am5_g5$he_rg_median,
            am5_g5$vbeta_true_median))
cat("\n")

## 5xAM + VT (20%) + GxE (20%): most complex scenario
sc_complex <- "5xAM + VT (20%) + GxE (20%)"
cx_g0 <- s5[s5$scenario == sc_complex & s5$gen == 0, ]
cx_g5 <- s5[s5$scenario == sc_complex & s5$gen == 5, ]
if (nrow(cx_g0) > 0 && nrow(cx_g5) > 0) {
  cat(sprintf("  %s baseline (gen 0):\n", sc_complex))
  cat(sprintf("    h2_true=%.4f  he_h2=%.4f  rg_true=%.4f  he_rg=%.4f  vbeta=%.4f\n",
              cx_g0$h2_true_median, cx_g0$he_h2_median,
              cx_g0$rg_true_median, cx_g0$he_rg_median,
              cx_g0$vbeta_true_median))
  cat(sprintf("  %s after 5 gen:\n", sc_complex))
  cat(sprintf("    h2_true=%.4f  he_h2=%.4f  rg_true=%.4f  he_rg=%.4f  vbeta=%.4f\n",
              cx_g5$h2_true_median, cx_g5$he_h2_median,
              cx_g5$rg_true_median, cx_g5$he_rg_median,
              cx_g5$vbeta_true_median))
  cat("\n")
}

## 95% intervals for key quantities
cat("--- 95% intervals (2.5th - 97.5th percentile) ---\n\n")
for (sc in c("RM", "2xAM", "5xAM")) {
  for (g in c(0, 5)) {
    row <- s5[s5$scenario == sc & s5$gen == g, ]
    if (nrow(row) == 0) next
    cat(sprintf("  %s gen %d (n=%d):\n", sc, g, row$n_sims))
    cat(sprintf("    h2_true:  %.4f [%.4f, %.4f]\n",
                row$h2_true_median, row$h2_true_q025, row$h2_true_q975))
    cat(sprintf("    he_h2:    %.4f [%.4f, %.4f]\n",
                row$he_h2_median, row$he_h2_q025, row$he_h2_q975))
    cat(sprintf("    rg_true:  %.4f [%.4f, %.4f]\n",
                row$rg_true_median, row$rg_true_q025, row$rg_true_q975))
    cat(sprintf("    he_rg:    %.4f [%.4f, %.4f]\n",
                row$he_rg_median, row$he_rg_q025, row$he_rg_q975))
    cat(sprintf("    vbeta:    %.4f [%.4f, %.4f]\n",
                row$vbeta_true_median, row$vbeta_true_q025, row$vbeta_true_q975))
    cat("\n")
  }
}

cat("============================================================\n")
cat("  VERIFICATION COMPLETE\n")
cat("============================================================\n")
