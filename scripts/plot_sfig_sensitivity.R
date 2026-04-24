## plot_sfig_sensitivity.R
## Generates supplementary figures S5 and S6 from simulation data.
##   S5: rg under xAM with r=0.1 vs r=0.2 (faceted)
##   S6: rg under unidimensional xAM with fixed latent correlation
##
## Verbatim port of:
##   - mergeall_120925.ipynb cells 19, 24, 26 (S5)
##   - FigSX_reconstruction.R  p_a (S6), p_b (S5 aesthetics), combined
##
## Data: full_ak_res121925.csv (pre-merged rr object from mergeall_120925.ipynb)

library(repr)
library(ggplot2)
library(reshape2)
library(cowplot)
library(stringr)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
FIG_DIR  <- file.path(BASE_DIR, "figures_output")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

default_theme <- theme_bw() + theme(text = element_text(size = 14))

## ── Load data ─────────────────────────────────────────────────────────────────
## rr is the pre-merged data written by mergeall_120925.ipynb cell 14
rr <- read.csv(file.path(BASE_DIR, "data/sim_results/full_ak_res121925.csv"))

## ── S6: Fixed Latent Correlation Experiments (FigSXa) ─────────────────────────

# from mergeall_120925.ipynb cell 19
rr$nxAM_expr <- rr$scenario %in% c('2xAM.1','2xAM.25','3xAM.167','4xAM.125','5xAM.1')

# Create custom labels with latent r values
rr$scenario_label_a <- factor(rr$scenario,
    levels = c("2xAM.1", "2xAM.25", "3xAM.167", "4xAM.125", "5xAM.1"),
    labels = c("2xAM.1 (latent r = 0.2)",
               "2xAM.25 (latent r = 0.5)",
               "3xAM.167 (latent r = 0.5)",
               "4xAM.125 (latent r = 0.5)",
               "5xAM.1 (latent r = 0.5)"))

p_a <- ggplot(rr[rr$nxAM_expr, ],
              aes(gen, he_rg, color = scenario_label_a,
                  linetype = scenario_label_a, shape = scenario_label_a)) +
    stat_summary(geom = 'linerange') +
    stat_summary(geom = 'line') +
    stat_summary(geom = 'point', size = 2.5) +
    scale_color_manual(values = c("2xAM.1 (latent r = 0.2)" = "#E41A1C",
                                  "2xAM.25 (latent r = 0.5)" = "#377EB8",
                                  "3xAM.167 (latent r = 0.5)" = "#4DAF4A",
                                  "4xAM.125 (latent r = 0.5)" = "#984EA3",
                                  "5xAM.1 (latent r = 0.5)" = "#A65628")) +
    scale_linetype_manual(values = c("2xAM.1 (latent r = 0.2)" = "solid",
                                     "2xAM.25 (latent r = 0.5)" = "dashed",
                                     "3xAM.167 (latent r = 0.5)" = "dotted",
                                     "4xAM.125 (latent r = 0.5)" = "dotdash",
                                     "5xAM.1 (latent r = 0.5)" = "longdash")) +
    scale_shape_manual(values = c("2xAM.1 (latent r = 0.2)" = 16,
                                  "2xAM.25 (latent r = 0.5)" = 17,
                                  "3xAM.167 (latent r = 0.5)" = 16,
                                  "4xAM.125 (latent r = 0.5)" = 18,
                                  "5xAM.1 (latent r = 0.5)" = 8)) +
    labs(x = "Generations of xAM",
         y = expression(Estimated~r[g]),
         color = NULL, linetype = NULL, shape = NULL) +
    default_theme +
    theme(legend.position = "bottom",
          legend.box = "horizontal") +
    guides(color = guide_legend(nrow = 2))

ggsave(file.path(FIG_DIR, "sfig_s6_fixed_latent.pdf"),
       p_a, width = 9, height = 5.5, bg = "white", dpi = 300)
cat("Saved sfig_s6_fixed_latent.pdf\n")

## ── S5: r = 0.1 vs r = 0.2 (FigSXb) ─────────────────────────────────────────

# from mergeall_120925.ipynb cell 26
# Filter for r = 0.1 and r = 0.2 scenarios
rr$xAM.1_r01_expr <- (rr$args_rmate == 0.1 | rr$args_rmate == 0.2) &
    ((rr$args_kphen == 5 & rr$args_phi != .2 & rr$args_theta != .2) |
     (rr$args_kphen == 2 & rr$args_phi == 0 & rr$args_theta == 0)) &
    rr$args_cnoise == 0

# Scenarios to include
scenarios_b <- c("2xAM.1", "5xAM.1", "5xAM.1 + VT.05", "5xAM.1 + GxE.05", "5xAM.1 + VT.05 + GxE.05",
                "2xAM.2", "5xAM.2", "5xAM.2 + VT.05", "5xAM.2 + GxE.05", "5xAM.2 + VT.05 + GxE.05")
rr$in_plot_b <- rr$xAM.1_r01_expr & rr$scenario %in% scenarios_b
zdat <- rr[rr$in_plot_b, ]

zdat$scenario <- gsub('xAM..','xAM',zdat$scenario)
p_b <- ggplot(zdat,
              aes(gen, he_rg, color = scenario, linetype = scenario, shape = scenario)) +
    stat_summary(geom = 'linerange') +
    stat_summary(geom = 'line') +
    stat_summary(geom = 'point', size = 2.5) +
    labs(x = "Generations of xAM",
         y = expression(Estimated~r[g]),
         color = NULL, linetype = NULL, shape = NULL) +
    default_theme + facet_wrap(~args_rmate) +
    theme(legend.position = "bottom",
          legend.box = "horizontal",
          legend.key.width = unit(1.5, 'cm')) +
    guides(color = guide_legend(nrow = 1))

ggsave(file.path(FIG_DIR, "sfig_s5_rg_sensitivity.pdf"),
       p_b, width = 12, height = 7, bg = "white", dpi = 300)
cat("Saved sfig_s5_rg_sensitivity.pdf\n")

## ── Combined S5 + S6 ──────────────────────────────────────────────────────────

p_combined <- plot_grid(
    p_a + theme(legend.position = "bottom"),
    p_b + theme(legend.position = "bottom"),
    ncol = 1,
    labels = c("A", "B"),
    label_size = 14,
    rel_heights = c(1, 1)
)

ggsave(file.path(FIG_DIR, "sfig_s5s6_sensitivity_combined.pdf"),
       p_combined, width = 12, height = 12, bg = "white", dpi = 300)
cat("Saved sfig_s5s6_sensitivity_combined.pdf\n")

cat("Done: plot_sfig_sensitivity.R\n")
