## verify_fig3.R
## Reads processed fig3 tables and prints every in-text number
## alongside the computed value for comparison.
##
## All P46 and P48 VT/GxE numbers come from the 0524 dataset
## (fig3_verify_0524.csv and fig3_p48_expanded.csv).
## FP inflation numbers come from 011024 (fig3_fp_gen5.csv).
##
## Manuscript references: P46, P47, P48

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
PROC_DIR <- file.path(BASE_DIR, "processed")

## ── Load processed data ─────────────────────────────────────────────────────

verify   <- read.csv(file.path(PROC_DIR, "fig3_verify_0524.csv"))
p48      <- read.csv(file.path(PROC_DIR, "fig3_p48_expanded.csv"))
fp_gen5  <- read.csv(file.path(PROC_DIR, "fig3_fp_gen5.csv"))

## ── Helper ──────────────────────────────────────────────────────────────────

check <- function(label, manuscript_val, computed_val, tol = 0.002) {
  match_str <- ifelse(abs(computed_val - manuscript_val) < tol, "MATCH", "MISMATCH")
  cat(sprintf("%-60s MANUSCRIPT: %.3f | COMPUTED: %.4f | %s\n",
              label, manuscript_val, computed_val, match_str))
}

cat("==================================================================\n")
cat("  FIGURE 3 IN-TEXT NUMBER VERIFICATION\n")
cat("  All VT/GxE numbers from 0524 dataset\n")
cat("==================================================================\n\n")

## ── Convenience accessors ───────────────────────────────────────────────────

v <- function(scenario) verify[verify$scenario == scenario, ]
p <- function(scenario, gen) p48[p48$scenario == scenario & p48$gen == gen, ]

## ── P46: h2 under 5xAM + VT (5%) ──────────────────────────────────────────

cat("--- P46: h2 Results ---\n\n")

vt5  <- v("5xAM + VT (5%)")
am   <- v("5xAM")

## 1. 5xAM+VT(5%): true h2 fell from 0.500 (gen 0) ...
check("5xAM+VT(5%): true h2 at gen 0",
      0.500, vt5$h2_true_gen0)

## 2. ... to 0.396 at gen 5
check("5xAM+VT(5%): true h2 at gen 5",
      0.396, vt5$h2_true_mean)

## 3. 5xAM+VT(5%): estimated h2 increased to 0.749
check("5xAM+VT(5%): estimated h2 at gen 5",
      0.749, vt5$h2_he_mean)

## 4. 5xAM+VT(5%): median upward bias 0.353
check("5xAM+VT(5%): median h2 bias at gen 5",
      0.353, vt5$h2_bias_median)

## 5. Mean h2 bias (also referenced as 0.354)
check("5xAM+VT(5%): mean h2 bias at gen 5",
      0.354, vt5$h2_bias_mean)

## 6. 5xAM alone: h2 bias 0.124
check("5xAM: h2 bias at gen 5",
      0.124, am$h2_bias_mean, tol = 0.002)

cat("\n--- P46: rg Results ---\n\n")

## 7. 5xAM+VT(5%): true rg = 0.078
check("5xAM+VT(5%): true rg at gen 5",
      0.078, vt5$rg_true_mean)

## 8. 5xAM+VT(5%): estimated rg = 0.513
check("5xAM+VT(5%): estimated rg at gen 5",
      0.513, vt5$he_rg_mean)

## 9. 5xAM alone: estimated rg = 0.296 (P46)
check("5xAM: estimated rg at gen 5",
      0.296, am$he_rg_mean, tol = 0.002)

## 10. 5xAM+VT(5%): rg bias = 0.435
check("5xAM+VT(5%): rg bias at gen 5",
      0.435, vt5$rg_bias_mean)

## 11. 5xAM alone: rg bias = 0.163 (manu rounds to 0.164)
check("5xAM: rg bias at gen 5 (manu ~0.164)",
      0.164, am$rg_bias_mean, tol = 0.002)

cat("\n--- P48: Expanded VT/GxE Comparisons ---\n\n")

## ── P48: Stronger VT and GxE comparisons (from 0524) ───────────────────────

## All P48 numbers use gen 5 means from the 0524 dataset

vt20     <- v("5xAM + VT (20%)")
vt20gxe5 <- v("5xAM + VT (20%) + GxE (5%)")

## 12. VT(20%): rg bias = 0.600
check("5xAM+VT(20%): rg bias at gen 5",
      0.600, vt20$rg_bias_mean)

## 13. VT(5%): rg bias = 0.435 (same as item 10)
check("5xAM+VT(5%): rg bias at gen 5 (P48 ref)",
      0.435, vt5$rg_bias_mean)

## 14. 5xAM alone: rg bias = 0.163 (manu rounds to 0.164)
check("5xAM: rg bias at gen 5 (P48 ref)",
      0.164, am$rg_bias_mean, tol = 0.002)

## 15. VT(20%): h2 bias = 0.403
check("5xAM+VT(20%): h2 bias at gen 5",
      0.403, vt20$h2_bias_mean, tol = 0.002)

## 16. VT(20%)+GxE(5%): h2 bias = 0.295
check("5xAM+VT(20%)+GxE(5%): h2 bias at gen 5",
      0.295, vt20gxe5$h2_bias_mean)

## 17. VT(20%)+GxE(5%): rg bias = 0.624
check("5xAM+VT(20%)+GxE(5%): rg bias at gen 5",
      0.624, vt20gxe5$rg_bias_mean)

cat("\n--- GWAS False Positive Results (P48, from 011024) ---\n\n")

## ── P48: GWAS FP inflation (from 011024 dataset) ──────────────────────────

## FP numbers come from 011024 since the manuscript figure uses those
fp_vt <- fp_gen5[fp_gen5$scenario == "5xAM + VT", ]
fp_am <- fp_gen5[fp_gen5$scenario == "5xAM", ]

## 18. 5xAM+VT: FP inflation = 2.779 (averaged across power)
check("5xAM+VT: GWAS FP inflation (avg across power)",
      2.779, fp_vt$fp_rate_all_power)

## 19. 5xAM alone: FP inflation = 1.522
check("5xAM: GWAS FP inflation (avg across power)",
      1.522, fp_am$fp_rate_all_power)

## 20. 39% power -> 1.747 under 5xAM+VT
check("5xAM+VT: GWAS FP at 39% power",
      1.747, fp_vt$"X39.")

## 21. 66% power -> 4.251 under 5xAM+VT
check("5xAM+VT: GWAS FP at 66% power",
      4.251, fp_vt$"X66.")

cat("\n==================================================================\n")
cat("  VERIFICATION COMPLETE\n")
cat("  P46/P48 VT numbers: 0524 dataset (fig3_verify_0524.csv)\n")
cat("  FP inflation: 011024 dataset (fig3_fp_gen5.csv)\n")
cat("==================================================================\n")
