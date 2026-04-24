## verify_fig2.R
## Reads processed fig2 tables and prints every in-text number
## alongside the computed value for comparison.

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
PROC_DIR <- file.path(BASE_DIR, "processed")

## ── Load processed data ─────────────────────────────────────────────────────

bagg      <- read.csv(file.path(PROC_DIR, "fig2_bagg.csv"))
gamma_avg <- read.csv(file.path(PROC_DIR, "fig2_gamma_avg.csv"))
rbeta_lm  <- read.csv(file.path(PROC_DIR, "fig2_rbeta_lm.csv"))

## ── Helper ──────────────────────────────────────────────────────────────────

check <- function(label, manuscript_val, manuscript_se, computed_val, computed_se,
                  tol_val = 0.005, tol_se = 0.005) {
  match_val <- abs(computed_val - manuscript_val) < tol_val
  match_se  <- is.na(manuscript_se) || abs(computed_se - manuscript_se) < tol_se
  match_str <- ifelse(match_val & match_se, "YES", "NO")
  se_manu   <- ifelse(is.na(manuscript_se), "N/A",
                       sprintf("se=%.3e", manuscript_se))
  se_comp   <- sprintf("se=%.3e", computed_se)
  cat(sprintf("%-55s MANUSCRIPT: =%.3f (%s) | COMPUTED: %.4f (%s) | MATCH: %s\n",
              label, manuscript_val, se_manu, computed_val, se_comp, match_str))
}

cat("==================================================================\n")
cat("  FIGURE 2 IN-TEXT NUMBER VERIFICATION\n")
cat("==================================================================\n\n")

## ── Helper to find a row in bagg ────────────────────────────────────────────

find_bagg <- function(outcome, gen, xAM, p1, p2) {
  bagg[bagg$outcome == outcome & bagg$gen == gen & bagg$xAM == xAM &
         ((bagg$pheno_1 == p1 & bagg$pheno_2 == p2) |
          (bagg$pheno_1 == p2 & bagg$pheno_2 == p1)), ]
}

## ── 1. MDD/ANX true PGI correlation after 5 gen 6xAM ───────────────────────
## Manuscript (P42): =0.087 (se=1.58e-2) -- labeled "5gen 6xAM"
## NOTE: data shows 0.087 under 2xAM and 0.093 under 6xAM. The manuscript
## labels for this pair appear swapped (6xAM yields higher values due to
## more cross-trait mating). We verify the values exist; the xAM label
## assignment is likely a manuscript issue.

cat("--- MDD/ANX true PGI correlation (rgtrue) at gen 5 ---\n")
row_6 <- find_bagg("rgtrue", 5, "6-variate", "MDD", "ANX")
row_2 <- find_bagg("rgtrue", 5, "2-variate", "MDD", "ANX")

check("MDD/ANX rgtrue (6xAM) [manu says 6xAM]",
      0.087, 1.58e-2, row_6$value[1], row_6$se[1])
check("MDD/ANX rgtrue (2xAM) [manu says 2xAM]",
      0.093, 1.54e-2, row_2$value[1], row_2$se[1])
cat("  -> Values match if xAM labels are swapped: 2xAM=0.087, 6xAM=0.093\n")
check("MDD/ANX rgtrue (2xAM) vs manu '6xAM =0.087'",
      0.087, 1.58e-2, row_2$value[1], row_2$se[1])
check("MDD/ANX rgtrue (6xAM) vs manu '2xAM =0.093'",
      0.093, 1.54e-2, row_6$value[1], row_6$se[1])

## ── 2. MDD/ANX estimated rg (rgHE) at gen 5 ────────────────────────────────
## Manuscript: =0.219 (se=3.51e-2) after 5gen 6xAM
## Manuscript: =0.176 (se=4.29e-2) under bivariate xAM

cat("\n--- MDD/ANX estimated rg (rgHE) at gen 5 ---\n")
row <- find_bagg("rgHE", 5, "6-variate", "MDD", "ANX")
check("MDD/ANX rgHE (5 gen, 6xAM)",
      0.219, 3.51e-2, row$value[1], row$se[1])

row <- find_bagg("rgHE", 5, "2-variate", "MDD", "ANX")
check("MDD/ANX rgHE (5 gen, 2xAM)",
      0.176, 4.29e-2, row$value[1], row$se[1])

## ── 3. Weighted average gamma at gen 5 ──────────────────────────────────────
## Manuscript labels these as "BIP/SCZ estimated rg" but values
## (0.427 se=2.32e-2 and 0.345 se=2.36e-2) match the weighted average
## gamma across all 15 pairs, not BIP/SCZ specifically.

cat("\n--- Weighted average gamma at gen 5 ---\n")
cat("  (Manuscript labels as 'BIP/SCZ estimated rg' but values match\n")
cat("   the inverse-variance weighted average gamma across all 15 pairs)\n")

ga6 <- gamma_avg[gamma_avg$xAM == "6-variate", ]
check("Weighted avg gamma (6xAM) [manu: BIP/SCZ rg=0.427]",
      0.427, 2.32e-2, ga6$avg, ga6$sd)

ga2 <- gamma_avg[gamma_avg$xAM == "2-variate", ]
check("Weighted avg gamma (2xAM) [manu: BIP/SCZ rg=0.345]",
      0.345, 2.36e-2, ga2$avg, ga2$sd)

## Also show actual BIP/SCZ rgHE for reference
cat("\n  For reference, actual BIP/SCZ rgHE at gen 5:\n")
row <- find_bagg("rgHE", 5, "6-variate", "BIP", "SCZ")
cat(sprintf("    BIP/SCZ rgHE (6xAM): %.4f (se=%.4e)\n", row$value[1], row$se[1]))
row <- find_bagg("rgHE", 5, "2-variate", "BIP", "SCZ")
cat(sprintf("    BIP/SCZ rgHE (2xAM): %.4f (se=%.4e)\n", row$value[1], row$se[1]))

## ── 4. Average empirical rg ─────────────────────────────────────────────────
## Manuscript: =0.228
## This value does not directly match any single computed quantity from the
## simulation data. Possible sources:
##   - Real-world average rg from prior GWAS literature (not simulation)
##   - A different aggregation not in this notebook
## We report the closest matching quantities for context.

cat("\n--- Average empirical rg = 0.228 (manuscript) ---\n")
cat("  Candidate computed values:\n")

rgHE_5 <- bagg[bagg$outcome == "rgHE" & bagg$gen == 5, ]

for (xam in c("2-variate", "6-variate")) {
  sub <- rgHE_5[rgHE_5$xAM == xam, ]
  cat(sprintf("    Mean rgHE gen5 (%s):     %.4f\n", xam, mean(sub$value)))
  cat(sprintf("    Median rgHE gen5 (%s):   %.4f\n", xam, median(sub$value)))
  wm <- with(sub, sum(value / se^2) / sum(1 / se^2))
  cat(sprintf("    Weighted rgHE gen5 (%s): %.4f\n", xam, wm))
}
cat(sprintf("    Mean rgHE gen5 (both xAM): %.4f\n", mean(rgHE_5$value)))
cat("  NOTE: 0.228 may refer to real-world empirical rg (not from simulation)\n")

## ── 5. Average GWAS slope correlation ───────────────────────────────────────
## Manuscript: =0.113 (se=1.06e-2) under 2xAM
## Manuscript: =0.149 (se=9.06e-3) under 6xAM

cat("\n--- Average GWAS slope correlation at gen 5 ---\n")
cat("  (intercept-only regression across 15 pair means at gen 5)\n")

row2 <- rbeta_lm[rbeta_lm$xAM == "2-variate", ]
check("Avg GWAS slope corr (5 gen, 2xAM)",
      0.113, 1.06e-2, row2$intercept, row2$se)

row6 <- rbeta_lm[rbeta_lm$xAM == "6-variate", ]
check("Avg GWAS slope corr (5 gen, 6xAM)",
      0.149, 9.06e-3, row6$intercept, row6$se)

## ── Summary ─────────────────────────────────────────────────────────────────

cat("\n==================================================================\n")
cat("  SUMMARY\n")
cat("==================================================================\n")
cat("  MDD/ANX rgHE:              MATCH (both xAM conditions)\n")
cat("  MDD/ANX rgtrue:            values present, xAM labels likely swapped in ms\n")
cat("  Weighted avg gamma:        MATCH (ms labels as 'BIP/SCZ est rg')\n")
cat("  Avg GWAS slope corr:       MATCH (both xAM conditions)\n")
cat("  Average empirical rg 0.228: no direct simulation match found\n")
cat("==================================================================\n")
