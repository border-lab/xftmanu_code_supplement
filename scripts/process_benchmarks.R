#!/usr/bin/env Rscript
# Process: Benchmarks figure
# Loads timing JSONL, trims JIT outliers, creates subsets for 3 panels
# Saves intermediate CSVs to processed/

library(jsonlite)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"

# --- Load all timing data ---
cat("Loading timing data...\n")
lines <- readLines(file.path(BASE_DIR, "data/timing_all_clean.jsonl"))
dat <- do.call(rbind, lapply(lines, function(l) as.data.frame(fromJSON(l))))
cat("Loaded", nrow(dat), "timing records\n")

dat$minutes <- dat$elapsed_seconds / 60
dat$n_k <- dat$n / 1000
dat$m_k <- dat$m / 1000

# --- Trim JIT outliers: drop the max per scenario ---
dat <- do.call(rbind, lapply(split(dat, dat$scenario), function(d) {
    d[d$minutes < max(d$minutes), ]
}))
cat("After trimming:", nrow(dat), "records\n")

# --- Panel a: n scaling at fixed m=4000 ---
scale_n <- dat[grepl("^n\\d", dat$scenario) & dat$m == 4000, ]
cat("Panel a (scale_n) rows:", nrow(scale_n), "\n")

# --- Panel b: m scaling ---
# m scaling at fixed n=256000
scale_m_256k <- dat[grepl("^m\\d", dat$scenario) & dat$n == 256000, ]
scale_m_256k$n_label <- "n = 256,000"

# m scaling at fixed n=64000
scale_m_64k <- dat[grepl("^n64k_m", dat$scenario), ]
scale_m_64k$n_label <- "n = 64,000"

scale_m <- rbind(scale_m_256k[, c("m_k", "minutes", "n_label")],
                 scale_m_64k[, c("m_k", "minutes", "n_label")])
cat("Panel b (scale_m) rows:", nrow(scale_m), "\n")

# --- Panel c: per-scenario ---
scenarios <- dat[dat$scenario %in% c("rm", "xam", "rm_vt", "xam_vt"), ]
scenarios$scenario_label <- factor(scenarios$scenario,
    levels = c("rm", "xam", "rm_vt", "xam_vt"),
    labels = c("RM", "5xAM", "RM + VT", "5xAM + VT"))
cat("Panel c (scenarios) rows:", nrow(scenarios), "\n")

# --- Save all subsets ---
write.csv(scale_n, file.path(BASE_DIR, "processed/figR_benchmarks_scale_n.csv"),
          row.names = FALSE)
write.csv(scale_m, file.path(BASE_DIR, "processed/figR_benchmarks_scale_m.csv"),
          row.names = FALSE)
write.csv(scenarios, file.path(BASE_DIR, "processed/figR_benchmarks_scenarios.csv"),
          row.names = FALSE)

cat("Saved 3 processed files for benchmarks figure\n")
