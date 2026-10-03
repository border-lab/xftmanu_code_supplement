#!/usr/bin/env Rscript
# Process: Correlated Noise figure (Extended Data Figure 10)
# Mimics previous submission structure: 4 scenarios × 5 CN levels
#   5xAM, 5xAM + GxE, 5xAM + VT, 5xAM + VT + GxE
#   CN = 0, 0.1, 0.2, 0.4, 0.8
#
# Data sources:
#   CN = 0         : data/sim_results/merged_res_redux_120725.csv   (has all phi values)
#   CN = 0.1       : data/sim_results/corrnoise1_res_120925.csv     (has all phi values)
#   CN = 0.2       : data/sim_results/corrnoise_res_120925.csv      (has all phi values)
#   CN = 0.4, 0.8  : data/corrnoise/*.csv               (phi = 0   — from first revision)
#   CN = 0.4, 0.8  : data/corrnoise_gxe/*.csv           (phi = 0.05 — new G×E runs)

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
BASE_DIR <- dirname(dirname(normalizePath(sub("^--file=", "", script_arg[1]))))

keep_cols <- c("gen", "seed", "args_rmate", "args_kphen", "args_theta",
               "args_phi", "args_cnoise", "he_rg", "rg_true")

# ============================================================
# 1. CN = 0 from merged_res
# ============================================================
cat("Loading CN = 0 data from merged_res...\n")
merged <- read.csv(file.path(BASE_DIR, "data/sim_results/merged_res_redux_120725.csv"))
cn0 <- merged[merged$args_rmate == 0.1 & merged$args_kphen == 5 &
              merged$args_cnoise == 0 &
              merged$args_theta %in% c(0, 0.05) &
              merged$args_phi   %in% c(0, 0.05), ]
cn0$args_cnoise <- 0
cn0 <- cn0[, keep_cols]
cat("  CN = 0 rows:", nrow(cn0), "\n")

# ============================================================
# 2. CN = 0.1 from corrnoise1
# ============================================================
cat("Loading CN = 0.1 data from corrnoise1...\n")
cn1 <- read.csv(file.path(BASE_DIR, "data/sim_results/corrnoise1_res_120925.csv"))
cn1 <- cn1[cn1$args_rmate == 0.1 & cn1$args_kphen == 5 &
           cn1$args_theta %in% c(0, 0.05) &
           cn1$args_phi   %in% c(0, 0.05), ]
if (!"args_cnoise" %in% names(cn1)) cn1$args_cnoise <- 0.1
cn1 <- cn1[, keep_cols]
cat("  CN = 0.1 rows:", nrow(cn1), "\n")

# ============================================================
# 3. CN = 0.2 from corrnoise
# ============================================================
cat("Loading CN = 0.2 data from corrnoise...\n")
cn2 <- read.csv(file.path(BASE_DIR, "data/sim_results/corrnoise_res_120925.csv"))
cn2 <- cn2[cn2$args_rmate == 0.1 & cn2$args_kphen == 5 &
           cn2$args_theta %in% c(0, 0.05) &
           cn2$args_phi   %in% c(0, 0.05), ]
cn2 <- cn2[, keep_cols]
cat("  CN = 0.2 rows:", nrow(cn2), "\n")

# ============================================================
# 4. CN = 0.4 / 0.8, phi = 0 from data/corrnoise
# ============================================================
cat("Loading CN = 0.4/0.8 (phi = 0) from corrnoise directory...\n")
load_dir <- function(path) {
  flist <- list.files(path, pattern = "*_parsed\\.csv$", full.names = TRUE)
  do.call(rbind, lapply(flist, function(f) {
    d <- read.csv(f)
    d[, keep_cols]
  }))
}
cn_new <- load_dir(file.path(BASE_DIR, "data/corrnoise"))
cn_new <- cn_new[cn_new$args_theta %in% c(0, 0.05) &
                 cn_new$args_phi == 0, ]
cat("  CN = 0.4/0.8 phi = 0 rows:", nrow(cn_new), "\n")

# ============================================================
# 5. CN = 0.4 / 0.8, phi = 0.05 from data/corrnoise_gxe (NEW)
# ============================================================
cat("Loading CN = 0.4/0.8 (phi = 0.05) from corrnoise_gxe directory...\n")
cn_gxe_new <- load_dir(file.path(BASE_DIR, "data/corrnoise_gxe"))
# Only keep CN = 0.4, 0.8 from the new G×E runs (CN = 0/0.1/0.2 are redundant with merged_res)
cn_gxe_new <- cn_gxe_new[cn_gxe_new$args_cnoise %in% c(0.4, 0.8) &
                         cn_gxe_new$args_theta %in% c(0, 0.05) &
                         cn_gxe_new$args_phi == 0.05, ]
cat("  CN = 0.4/0.8 phi = 0.05 rows:", nrow(cn_gxe_new), "\n")

# ============================================================
# Combine all
# ============================================================
all_cn <- rbind(cn0, cn1, cn2, cn_new, cn_gxe_new)
cat("Combined data:", nrow(all_cn), "rows\n")
cat("CN levels:", sort(unique(all_cn$args_cnoise)), "\n")
cat("Theta levels:", sort(unique(all_cn$args_theta)), "\n")
cat("Phi levels:", sort(unique(all_cn$args_phi)), "\n")

# ============================================================
# Scenario labels (4 scenarios, mimicking previous submission)
# ============================================================
all_cn$scenario <- with(all_cn, ifelse(
  args_theta == 0 & args_phi == 0,       "5xAM",
  ifelse(args_theta == 0 & args_phi == 0.05,    "5xAM + GxE",
  ifelse(args_theta == 0.05 & args_phi == 0,    "5xAM + VT",
  ifelse(args_theta == 0.05 & args_phi == 0.05, "5xAM + VT + GxE", NA)))))

all_cn$scenario <- factor(all_cn$scenario,
  levels = c("5xAM", "5xAM + GxE", "5xAM + VT", "5xAM + VT + GxE"))
all_cn$CN <- factor(all_cn$args_cnoise)

cat("Rows per CN x scenario:\n")
print(table(all_cn$CN, all_cn$scenario))

outfile <- file.path(BASE_DIR, "processed/figR_cn.csv")
write.csv(all_cn, outfile, row.names = FALSE)
cat("Saved:", outfile, "\n")
