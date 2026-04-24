## verify_sfig_sibcomp.R
## Prints key summary statistics for supplementary figure S17

library(reshape2)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
PROC_DIR <- file.path(BASE_DIR, "processed")

## ── S17a: Relative T1E inflation ────────────────────────────────────────────────

cat("===== S17a: Relative Type-I error for pop vs sib GWAS =====\n\n")

t1e <- read.csv(file.path(PROC_DIR, "sfig_s17a_t1e_comparison.csv"))
cat("Dimensions:", nrow(t1e), "rows x", ncol(t1e), "cols\n")
cat("Scenarios:", paste(unique(t1e$scenario), collapse = ", "), "\n")
cat("GWAS types:", paste(unique(t1e$GWAS), collapse = ", "), "\n")
cat("Power levels:", paste(unique(t1e$power), collapse = ", "), "\n\n")

## Mean relative T1E at gen=5 by scenario and GWAS type
cat("Mean relative T1E at generation 5 (pop vs sib):\n")
agg_t1e <- aggregate(relative_T1R ~ scenario + GWAS,
                      data = t1e[t1e$gen == 5, ], mean)
agg_t1e$relative_T1R <- round(agg_t1e$relative_T1R, 3)
## Wide format for easier comparison
w <- dcast(agg_t1e, scenario ~ GWAS, value.var = "relative_T1R")
w$ratio_pop_sib <- round(w$Population / w$Sibship, 3)
print(w)

## Same for gen=1
cat("\nMean relative T1E at generation 1 (pop vs sib):\n")
agg_t1e_g1 <- aggregate(relative_T1R ~ scenario + GWAS,
                          data = t1e[t1e$gen == 1, ], mean)
agg_t1e_g1$relative_T1R <- round(agg_t1e_g1$relative_T1R, 3)
w1 <- dcast(agg_t1e_g1, scenario ~ GWAS, value.var = "relative_T1R")
w1$ratio_pop_sib <- round(w1$Population / w1$Sibship, 3)
print(w1)

## ── S17b: Cross-phenotype slope ─────────────────────────────────────────────────

cat("\n\n===== S17b: Cross-phenotype slope (pop vs sib GWAS) =====\n\n")

slope <- read.csv(file.path(PROC_DIR, "sfig_s17b_slope_comparison.csv"))
cat("Dimensions:", nrow(slope), "rows x", ncol(slope), "cols\n\n")

## Mean slope at gen=5 by scenario and GWAS type
cat("Mean cross-phenotype slope at generation 5:\n")
agg_slope <- aggregate(value ~ scenario + GWAS,
                        data = slope[slope$gen == 5, ], mean)
agg_slope$value <- round(agg_slope$value, 4)
ws <- dcast(agg_slope, scenario ~ GWAS, value.var = "value")
print(ws)

## Show how slopes diverge from gen 0 to 5
cat("\nMean slope at generation 0:\n")
agg_slope0 <- aggregate(value ~ scenario + GWAS,
                          data = slope[slope$gen == 0, ], mean)
agg_slope0$value <- round(agg_slope0$value, 4)
ws0 <- dcast(agg_slope0, scenario ~ GWAS, value.var = "value")
print(ws0)

cat("\nDone: verify_sfig_sibcomp.R\n")
