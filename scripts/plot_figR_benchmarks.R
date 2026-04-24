#!/usr/bin/env Rscript
# Plot: Benchmarks figure
# Reads ONLY from processed/, generates figure with 3 panels

library(ggplot2)
library(cowplot)
library(RColorBrewer)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
OUTDIR  <- file.path(BASE_DIR, "figures_output")

# Load processed data
scale_n   <- read.csv(file.path(BASE_DIR, "processed/figR_benchmarks_scale_n.csv"))
scale_m   <- read.csv(file.path(BASE_DIR, "processed/figR_benchmarks_scale_m.csv"))
scenarios <- read.csv(file.path(BASE_DIR, "processed/figR_benchmarks_scenarios.csv"))

# Restore factor for scenario labels
scenarios$scenario_label <- factor(scenarios$scenario_label,
    levels = c("RM", "5xAM", "RM + VT", "5xAM + VT"))

set1 <- brewer.pal(5, "Set1")

base_theme <- theme_bw() + theme(
    text = element_text(size = 14),
    legend.position = "bottom",
    legend.direction = "horizontal"
)

# =================================================================
# Panel a: Scaling with n (log-log, m=4000)
# =================================================================
pa <- ggplot(scale_n, aes(x = n_k, y = minutes)) +
    stat_summary(geom = "ribbon", fun.data = mean_sdl, fun.args = list(mult = 1),
                 alpha = 0.15, fill = set1[2]) +
    stat_summary(geom = "line", fun = mean, color = set1[2], linewidth = 0.8) +
    stat_summary(geom = "point", fun = mean, color = set1[2], size = 2.5) +
    scale_x_continuous(trans = "log2",
                       breaks = c(8, 16, 32, 64, 128, 256, 512)) +
    scale_y_continuous(trans = "log2",
                       breaks = c(1, 2, 4, 8, 16, 32)) +
    labs(x = "Sample size (thousands)",
         y = "Runtime (minutes)",
         title = expression(paste("Scaling with ", italic(n), " (", italic(m), " = 4,000)"))) +
    base_theme +
    theme(plot.title = element_text(size = 12))

# =================================================================
# Panel b: Scaling with m (log-log, two n values)
# =================================================================
pb <- ggplot(scale_m, aes(x = m_k, y = minutes, color = n_label, fill = n_label)) +
    stat_summary(geom = "ribbon", fun.data = mean_sdl, fun.args = list(mult = 1),
                 alpha = 0.1) +
    stat_summary(geom = "line", fun = mean, linewidth = 0.8) +
    stat_summary(geom = "point", fun = mean, size = 2.5) +
    scale_x_continuous(trans = "log2",
                       breaks = c(0.5, 1, 2, 4, 8, 16, 32, 64)) +
    scale_y_continuous(trans = "log2",
                       breaks = c(1, 2, 4, 8, 16, 32, 64)) +
    scale_color_manual(values = c("n = 256,000" = set1[1], "n = 64,000" = set1[4]),
                       name = NULL) +
    scale_fill_manual(values = c("n = 256,000" = set1[1], "n = 64,000" = set1[4]),
                      name = NULL) +
    labs(x = "Number of variants (thousands)",
         y = "Runtime (minutes)",
         title = expression(paste("Scaling with ", italic(m)))) +
    base_theme +
    theme(plot.title = element_text(size = 12))

# =================================================================
# Panel c: Per-scenario runtime
# =================================================================
pc <- ggplot(scenarios, aes(x = scenario_label, y = minutes)) +
    geom_jitter(width = 0.15, size = 1.5, alpha = 0.5, color = "grey40") +
    stat_summary(geom = "crossbar", fun = mean, fun.min = mean, fun.max = mean,
                 width = 0.5, linewidth = 0.6, color = "black") +
    stat_summary(geom = "errorbar", fun.data = mean_sdl, fun.args = list(mult = 1),
                 width = 0.25) +
    labs(x = NULL,
         y = "Runtime (minutes)",
         title = expression(paste("By scenario (", italic(n), " = 256,000; ", italic(m), " = 4,000)"))) +
    base_theme +
    theme(plot.title = element_text(size = 12))

# =================================================================
# Combined figure
# =================================================================
top_row <- plot_grid(pa, pb, ncol = 2, labels = c("a", "b"), label_size = 14)
fig <- plot_grid(top_row, pc, ncol = 1, labels = c("", "c"), label_size = 14,
                 rel_heights = c(1, 0.85))

ggsave(file.path(OUTDIR, "figR_benchmarks.pdf"), fig, width = 12, height = 9)
ggsave(file.path(OUTDIR, "figR_benchmarks.png"), fig, width = 12, height = 9, dpi = 300)
cat("Saved figR_benchmarks.pdf and figR_benchmarks.png\n")

# Also save scaling-only version
scaling_only <- plot_grid(pa, pb, ncol = 2, labels = c("a", "b"), label_size = 14)
ggsave(file.path(OUTDIR, "figR_benchmarks_scaling.pdf"), scaling_only, width = 12, height = 5)
ggsave(file.path(OUTDIR, "figR_benchmarks_scaling.png"), scaling_only, width = 12, height = 5, dpi = 300)
cat("Saved figR_benchmarks_scaling.pdf and figR_benchmarks_scaling.png\n")
