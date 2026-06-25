#!/usr/bin/env Rscript
# Process: Correlated Noise figure
# Loads CN=0 (merged_res), CN=0.1 (corrnoise1), CN=0.2 (corrnoise),
# CN=0.4/0.8 (corrnoise/*.csv) and combines into one dataset
# Saves intermediate CSV to processed/

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"

keep_cols <- c("gen", "seed", "args_rmate", "args_kphen", "args_theta",
               "args_cnoise", "he_rg", "rg_true")

# ============================================================
# 1. CN=0 from merged_res
# ============================================================
cat("Loading CN=0 data from merged_res...\n")
merged <- read.csv(file.path(BASE_DIR, "data/sim_results/merged_res_redux_120725.csv"))
cn0 <- merged[merged$args_rmate == 0.1 & merged$args_kphen == 5 &
              merged$args_cnoise == 0 & merged$args_theta %in% c(0, 0.05), ]
cn0$args_cnoise <- 0
cn0 <- cn0[, keep_cols]
cat("  CN=0 rows:", nrow(cn0), "\n")

# ============================================================
# 2. CN=0.1 from corrnoise1
# ============================================================
cat("Loading CN=0.1 data from corrnoise1...\n")
cn1 <- read.csv(file.path(BASE_DIR, "data/sim_results/corrnoise1_res_120925.csv"))
cn1 <- cn1[cn1$args_rmate == 0.1 & cn1$args_kphen == 5 &
           cn1$args_theta %in% c(0, 0.05), ]
if (!"args_cnoise" %in% names(cn1)) cn1$args_cnoise <- 0.1
cn1 <- cn1[, keep_cols]
cat("  CN=0.1 rows:", nrow(cn1), "\n")

# ============================================================
# 3. CN=0.2 from corrnoise
# ============================================================
cat("Loading CN=0.2 data from corrnoise...\n")
cn2 <- read.csv(file.path(BASE_DIR, "data/sim_results/corrnoise_res_120925.csv"))
cn2 <- cn2[cn2$args_rmate == 0.1 & cn2$args_kphen == 5 &
           cn2$args_theta %in% c(0, 0.05), ]
cat("  CN=0.2 at rmate=0.1, kphen=5 rows:", nrow(cn2), "\n")
if (nrow(cn2) > 0) {
  cn2 <- cn2[, keep_cols]
} else {
  cat("  WARNING: No CN=0.2 data at rmate=0.1, kphen=5. Skipping CN=0.2.\n")
  cn2 <- NULL
}

# ============================================================
# 4. CN=0.4 and CN=0.8 from corrnoise directory
# ============================================================
cat("Loading CN=0.4/0.8 data from corrnoise directory...\n")
newdir <- file.path(BASE_DIR, "data/corrnoise")
flist <- list.files(newdir, pattern = "*.csv", full.names = TRUE)
cn_new <- do.call(rbind, lapply(flist, read.csv))
cn_new <- cn_new[cn_new$args_theta %in% c(0, 0.05), ]
cn_new <- cn_new[, keep_cols]
cat("  New CN data rows:", nrow(cn_new), "\n")
cat("  New CN values:", unique(cn_new$args_cnoise), "\n")

# ============================================================
# Combine all
# ============================================================
all_cn <- rbind(cn0, cn1)
if (!is.null(cn2)) all_cn <- rbind(all_cn, cn2)
all_cn <- rbind(all_cn, cn_new)

cat("Combined data:", nrow(all_cn), "rows\n")
cat("CN levels:", sort(unique(all_cn$args_cnoise)), "\n")
cat("Theta levels:", sort(unique(all_cn$args_theta)), "\n")

# Assign scenario labels
all_cn$scenario <- ifelse(all_cn$args_theta == 0, "5xAM", "5xAM + VT (5%)")
all_cn$scenario <- factor(all_cn$scenario, levels = c("5xAM", "5xAM + VT (5%)"))
all_cn$CN <- factor(all_cn$args_cnoise)

cat("Rows per CN x scenario:\n")
print(table(all_cn$CN, all_cn$scenario))

outfile <- file.path(BASE_DIR, "processed/figR_cn.csv")
write.csv(all_cn, outfile, row.names = FALSE)
cat("Saved:", outfile, "\n")
