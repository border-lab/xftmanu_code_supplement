## verify_fig1.R
## Reads processed data and prints every in-text number for comparison
## against manuscript values (P29, P34, P36, P37)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
PROCESSED_DIR <- file.path(BASE_DIR, "processed")

cat("============================================================\n")
cat("  FIGURE 1: IN-TEXT NUMBER VERIFICATION\n")
cat("============================================================\n\n")

## Helper function for formatted comparison
check_value <- function(label, manuscript_val, computed_val, computed_sd = NA, tol = 0.02) {
  match <- abs(manuscript_val - computed_val) < tol
  match_str <- ifelse(match, "YES", "NO <---")
  sd_str <- ifelse(is.na(computed_sd), "", sprintf(" (sd=%.4f)", computed_sd))
  cat(sprintf("  %-45s MANUSCRIPT: %.3f | COMPUTED: %.4f%s | MATCH: %s\n",
              label, manuscript_val, computed_val, sd_str, match_str))
}

check_int <- function(label, manuscript_val, computed_val) {
  match <- manuscript_val == computed_val
  match_str <- ifelse(match, "YES", "NO <---")
  cat(sprintf("  %-45s MANUSCRIPT: %d    | COMPUTED: %d    | MATCH: %s\n",
              label, manuscript_val, computed_val, match_str))
}

## ---------------------------------------------------------------
## CCA VERIFICATION (P29)
## ---------------------------------------------------------------
cat("--- CCA Dimensionality (P29) ---\n")
cca_verify <- read.csv(file.path(PROCESSED_DIR, "fig1_verify_cca.csv"))

sig_05 <- cca_verify$n_dims[cca_verify$dataset == "UKB" & cca_verify$threshold == "sig_0.05"]
check_int("UKB significant CVs at alpha=0.05", 22, sig_05)

ukb_90 <- cca_verify$n_dims[cca_verify$dataset == "UKB" & cca_verify$threshold == "90%"]
check_int("UKB dims for 90% variance", 8, ukb_90)

ukb_95 <- cca_verify$n_dims[cca_verify$dataset == "UKB" & cca_verify$threshold == "95%"]
check_int("UKB dims for 95% variance", 14, ukb_95)

nhird_90 <- cca_verify$n_dims[cca_verify$dataset == "NHIRD" & cca_verify$threshold == "90%"]
check_int("NHIRD dims for 90% variance", 7, nhird_90)

nhird_95 <- cca_verify$n_dims[cca_verify$dataset == "NHIRD" & cca_verify$threshold == "95%"]
check_int("NHIRD dims for 95% variance", 8, nhird_95)
cat("\n")

## ---------------------------------------------------------------
## h2 VERIFICATION (P34)
## ---------------------------------------------------------------
cat("--- h2 under xAM (P34) ---\n")
h2_verify <- read.csv(file.path(PROCESSED_DIR, "fig1_verify_h2.csv"))

## h2_true at gen 0 (baseline) - should be ~0.499
h2_true_gen0 <- h2_verify[h2_verify$gen == 0 & h2_verify$variable == "h2_true" &
                           h2_verify$scenario == "2xAM", ]
check_value("h2_true gen0 (baseline)", 0.499, h2_true_gen0$mean, h2_true_gen0$sd)

## h2_true at gen 5, 2xAM - should be ~0.527
h2_true_2xam_g5 <- h2_verify[h2_verify$gen == 5 & h2_verify$variable == "h2_true" &
                              h2_verify$scenario == "2xAM", ]
check_value("h2_true gen5 (2xAM, true)", 0.527, h2_true_2xam_g5$mean, h2_true_2xam_g5$sd)

## h2_true at gen 5, 5xAM - should be ~0.535
h2_true_5xam_g5 <- h2_verify[h2_verify$gen == 5 & h2_verify$variable == "h2_true" &
                              h2_verify$scenario == "5xAM", ]
check_value("h2_true gen5 (5xAM, true)", 0.535, h2_true_5xam_g5$mean, h2_true_5xam_g5$sd)

## he_h2 at gen 5, 2xAM - should be ~0.597
he_h2_2xam_g5 <- h2_verify[h2_verify$gen == 5 & h2_verify$variable == "he_h2" &
                            h2_verify$scenario == "2xAM", ]
check_value("he_h2 gen5 (2xAM, estimated)", 0.597, he_h2_2xam_g5$mean, he_h2_2xam_g5$sd)

## he_h2 at gen 5, 5xAM - should be ~0.658
he_h2_5xam_g5 <- h2_verify[h2_verify$gen == 5 & h2_verify$variable == "he_h2" &
                            h2_verify$scenario == "5xAM", ]
check_value("he_h2 gen5 (5xAM, estimated)", 0.658, he_h2_5xam_g5$mean, he_h2_5xam_g5$sd)
cat("\n")

## ---------------------------------------------------------------
## rg VERIFICATION (P36)
## ---------------------------------------------------------------
cat("--- rg under xAM (P36) ---\n")
rg_verify <- read.csv(file.path(PROCESSED_DIR, "fig1_verify_rg.csv"))

## rg_true at gen 5, 2xAM - should be ~0.107
rg_true_2xam_g5 <- rg_verify[rg_verify$gen == 5 & rg_verify$variable == "rg_true" &
                              rg_verify$scenario == "2xAM", ]
check_value("rg_true gen5 (2xAM)", 0.107, rg_true_2xam_g5$mean, rg_true_2xam_g5$sd)

## he_rg at gen 5, 2xAM - should be ~0.212
he_rg_2xam_g5 <- rg_verify[rg_verify$gen == 5 & rg_verify$variable == "he_rg" &
                            rg_verify$scenario == "2xAM", ]
check_value("he_rg gen5 (2xAM, estimated)", 0.212, he_rg_2xam_g5$mean, he_rg_2xam_g5$sd)

## he_rg at gen 5, 5xAM - should be ~0.296
he_rg_5xam_g5 <- rg_verify[rg_verify$gen == 5 & rg_verify$variable == "he_rg" &
                            rg_verify$scenario == "5xAM", ]
check_value("he_rg gen5 (5xAM, estimated)", 0.296, he_rg_5xam_g5$mean, he_rg_5xam_g5$sd)

## rg_true at gen 0 (baseline) - should be ~0
rg_true_gen0 <- rg_verify[rg_verify$gen == 0 & rg_verify$variable == "rg_true" &
                           rg_verify$scenario == "2xAM", ]
check_value("rg_true gen0 (baseline)", 0.000, rg_true_gen0$mean, rg_true_gen0$sd, tol = 0.05)
cat("\n")

## ---------------------------------------------------------------
## GWAS FP VERIFICATION (P37)
## ---------------------------------------------------------------
cat("--- GWAS Type-I Error Inflation (P37) ---\n")
gwas_verify <- read.csv(file.path(PROCESSED_DIR, "fig1_verify_gwas.csv"))

## FP inflation at gen 5, 5xAM, population GWAS - should be ~1.52
gwas_5xam_g5 <- gwas_verify[gwas_verify$gen == 5 & gwas_verify$xAM == "5-variate", ]
if (nrow(gwas_5xam_g5) > 0) {
  check_value("GWAS FP inflation gen5 (5xAM, pop, mean)", 1.52,
              gwas_5xam_g5$mean[1], gwas_5xam_g5$sd[1], tol = 0.05)
  cat(sprintf("  %-45s (median=%.4f for reference)\n",
              "", gwas_5xam_g5$median[1]))
}

## Also check 2xAM
gwas_2xam_g5 <- gwas_verify[gwas_verify$gen == 5 & gwas_verify$xAM == "2-variate", ]
if (nrow(gwas_2xam_g5) > 0) {
  cat(sprintf("  %-45s COMPUTED: %.4f (median) (sd=%.4f)\n",
              "GWAS FP inflation gen5 (2xAM, pop)",
              gwas_2xam_g5$median[1], gwas_2xam_g5$sd[1]))
}
cat("\n")

## ---------------------------------------------------------------
## LATENT CORRELATION (P37)
## ---------------------------------------------------------------
cat("--- Latent correlation ---\n")
cat(sprintf("  %-45s r*K = 0.2 * 5 = %.1f (manuscript: 1.0)\n",
            "Latent correlation (5xAM)", 0.2 * 5))
cat("\n")

## ---------------------------------------------------------------
## FULL TABLE DUMP
## ---------------------------------------------------------------
cat("--- Full h2 table (gen 0 and 5) ---\n")
print(h2_verify)
cat("\n--- Full rg table (gen 0 and 5) ---\n")
print(rg_verify)
cat("\n--- Full GWAS table (gen 5) ---\n")
print(gwas_verify)

cat("\n============================================================\n")
cat("  VERIFICATION COMPLETE\n")
cat("============================================================\n")
