## plot_sfig_theory_gxe.R
## Generates supplementary figures S8 and S13-S16 from processed data ONLY.
##   S8:  Theory vs observed GWAS false positive calibration
##   S13: h2 across generations, GxE rows x VT cols
##   S14: h2 across generations, VT rows (alternative arrangement)
##   S15: rg across generations with GxE/VT grid
##   S16: PGI cross-phenotype slope with GxE/VT grid

library(ggplot2)
library(cowplot)
library(splines)
library(reshape2)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
PROC_DIR <- file.path(BASE_DIR, "processed")
FIG_DIR  <- file.path(BASE_DIR, "figures_output")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

default_theme  <- theme_bw() + theme(text = element_text(size = 14))
default_theme2 <- theme_bw() + theme(text = element_text(size = 12))
PAL <- RColorBrewer::brewer.pal(8, "Set1")

## ── S8: Theory vs observed false positive calibration ──────────────────────────

gwas_fp  <- read.csv(file.path(PROC_DIR, "sfig_s8_gwas_fp.csv"))
math_sim <- read.csv(file.path(PROC_DIR, "sfig_s8_mathematica_sim.csv"))

## Plot: log10(alpha) vs log10(FalsePositive) at gen=3, faceted by scenario
plot_s8 <- ggplot(gwas_fp[gwas_fp$gen == 3, ],
       aes(log10(alpha), log10(FalsePositive),
           color = GWAS, linetype = as.factor(nm_ratio))) +
  geom_line() +
  facet_grid(params ~ .) +
  geom_abline(slope = 1, intercept = 0, color = "grey50", linetype = "dashed") +
  labs(x = expression(log[10](alpha)),
       y = expression(log[10]("False positive rate")),
       color = "GWAS type",
       linetype = "n/m ratio") +
  default_theme +
  theme(legend.position = "top",
        legend.direction = "horizontal",
        legend.title.align = 0.5,
        strip.text.y = element_text(size = 9))

ggsave(file.path(FIG_DIR, "sfig_s8_theory_validation.pdf"),
       plot_s8, width = 8, height = 12)
cat("Saved sfig_s8_theory_validation.pdf\n")

## ── S13: h2 across generations, GxE rows x VT cols ─────────────────────────────

h2g <- read.csv(file.path(PROC_DIR, "sfig_s13s14_h2_grid.csv"))

## Melt to long form with estimated and true h2
h2g_long <- melt(h2g,
  id.vars = c("seed", "gen", "args_theta", "args_phi",
              "GxE_level", "VT_level", "grid_scenario"),
  measure.vars = c("he_h2", "h2_true"),
  variable.name = "quantity", value.name = "value")
h2g_long$quantity <- factor(h2g_long$quantity,
  levels = c("h2_true", "he_h2"),
  labels = c("True h2", "Estimated h2"))

## S13: GxE as rows, VT as columns
plot_s13 <- ggplot(h2g_long,
       aes(gen, value, color = quantity, linetype = quantity)) +
  stat_summary(geom = "linerange", position = position_dodge(width = 0.3),
               fun.max = function(x) quantile(x, 0.9),
               fun.min = function(x) quantile(x, 0.1),
               fun = function(x) quantile(x, 0.5)) +
  stat_summary(geom = "point", position = position_dodge(width = 0.3),
               fun = function(x) quantile(x, 0.5), size = 1.5) +
  stat_smooth(method = "gam",
              formula = y ~ bs(x, knots = 0:5, degree = 1), se = FALSE) +
  facet_grid(GxE_level ~ VT_level) +
  geom_hline(yintercept = 0.5, color = "grey50", linetype = "dotted") +
  scale_color_manual(values = c("True h2" = PAL[2], "Estimated h2" = PAL[1])) +
  scale_linetype_manual(values = c("True h2" = "dashed", "Estimated h2" = "solid")) +
  labs(x = "Generations of xAM",
       y = expression(h^2),
       color = NULL, linetype = NULL) +
  default_theme2 +
  theme(legend.position = "bottom",
        strip.text = element_text(size = 10))

ggsave(file.path(FIG_DIR, "sfig_s13_h2_gxe_grid.pdf"),
       plot_s13, width = 10, height = 10)
cat("Saved sfig_s13_h2_gxe_grid.pdf\n")

## ── S14: h2 across generations, VT rows (alternative arrangement) ──────────────
## Same data, but swap rows/cols: VT as rows, GxE as cols

plot_s14 <- ggplot(h2g_long,
       aes(gen, value, color = quantity, linetype = quantity)) +
  stat_summary(geom = "linerange", position = position_dodge(width = 0.3),
               fun.max = function(x) quantile(x, 0.9),
               fun.min = function(x) quantile(x, 0.1),
               fun = function(x) quantile(x, 0.5)) +
  stat_summary(geom = "point", position = position_dodge(width = 0.3),
               fun = function(x) quantile(x, 0.5), size = 1.5) +
  stat_smooth(method = "gam",
              formula = y ~ bs(x, knots = 0:5, degree = 1), se = FALSE) +
  facet_grid(VT_level ~ GxE_level) +
  geom_hline(yintercept = 0.5, color = "grey50", linetype = "dotted") +
  scale_color_manual(values = c("True h2" = PAL[2], "Estimated h2" = PAL[1])) +
  scale_linetype_manual(values = c("True h2" = "dashed", "Estimated h2" = "solid")) +
  labs(x = "Generations of xAM",
       y = expression(h^2),
       color = NULL, linetype = NULL) +
  default_theme2 +
  theme(legend.position = "bottom",
        strip.text = element_text(size = 10))

ggsave(file.path(FIG_DIR, "sfig_s14_h2_vt_grid.pdf"),
       plot_s14, width = 10, height = 10)
cat("Saved sfig_s14_h2_vt_grid.pdf\n")

## ── S15: rg across generations with GxE/VT grid ────────────────────────────────

rgg <- read.csv(file.path(PROC_DIR, "sfig_s15_rg_grid.csv"))

## Melt to long form
rg_long <- melt(rgg,
  id.vars = c("seed", "gen", "args_theta", "args_phi",
              "GxE_level", "VT_level", "grid_scenario"),
  measure.vars = c("he_rg", "rg_true"),
  variable.name = "quantity", value.name = "value")
rg_long$quantity <- factor(rg_long$quantity,
  levels = c("rg_true", "he_rg"),
  labels = c("True rg", "Estimated rg"))

plot_s15 <- ggplot(rg_long,
       aes(gen, value, color = quantity, linetype = quantity)) +
  stat_summary(geom = "linerange", position = position_dodge(width = 0.3),
               fun.max = function(x) quantile(x, 0.9),
               fun.min = function(x) quantile(x, 0.1),
               fun = function(x) quantile(x, 0.5)) +
  stat_summary(geom = "point", position = position_dodge(width = 0.3),
               fun = function(x) quantile(x, 0.5), size = 1.5) +
  stat_smooth(method = "gam",
              formula = y ~ bs(x, knots = 0:5, degree = 1), se = FALSE) +
  facet_grid(GxE_level ~ VT_level) +
  geom_hline(yintercept = 0, color = "grey50", linetype = "dotted") +
  scale_color_manual(values = c("True rg" = PAL[2], "Estimated rg" = PAL[1])) +
  scale_linetype_manual(values = c("True rg" = "dashed", "Estimated rg" = "solid")) +
  labs(x = "Generations of xAM",
       y = expression(r[g]),
       color = NULL, linetype = NULL) +
  default_theme2 +
  theme(legend.position = "bottom",
        strip.text = element_text(size = 10))

ggsave(file.path(FIG_DIR, "sfig_s15_rg_gxe_grid.pdf"),
       plot_s15, width = 10, height = 10)
cat("Saved sfig_s15_rg_gxe_grid.pdf\n")

## ── S16: PGI cross-phenotype slope with GxE/VT grid ────────────────────────────

pgi <- read.csv(file.path(PROC_DIR, "sfig_s16_pgi_grid.csv"))

## Melt to long form with pop and sib GWAS slope estimates
pgi_long <- melt(pgi,
  id.vars = c("seed", "gen", "args_theta", "args_phi",
              "GxE_level", "VT_level", "grid_scenario",
              "rbeta_true", "rg_true"),
  measure.vars = c("rbeta_hat_pgwas", "rbeta_hat_sgwas"),
  variable.name = "GWAS", value.name = "rbeta_hat")
pgi_long$GWAS <- factor(pgi_long$GWAS,
  levels = c("rbeta_hat_pgwas", "rbeta_hat_sgwas"),
  labels = c("Population GWAS", "Sibship GWAS"))

plot_s16 <- ggplot(pgi_long,
       aes(gen, rbeta_hat, color = GWAS, linetype = GWAS)) +
  stat_summary(geom = "linerange", position = position_dodge(width = 0.3),
               fun.max = function(x) quantile(x, 0.9),
               fun.min = function(x) quantile(x, 0.1),
               fun = function(x) quantile(x, 0.5)) +
  stat_summary(geom = "point", position = position_dodge(width = 0.3),
               fun = function(x) quantile(x, 0.5), size = 1.5) +
  stat_smooth(method = "gam",
              formula = y ~ bs(x, knots = 0:5, degree = 1), se = FALSE) +
  facet_grid(GxE_level ~ VT_level) +
  geom_hline(yintercept = 0, color = "grey50", linetype = "dotted") +
  scale_color_manual(values = c("Population GWAS" = PAL[1],
                                 "Sibship GWAS" = PAL[2])) +
  scale_linetype_manual(values = c("Population GWAS" = "solid",
                                    "Sibship GWAS" = "dashed")) +
  labs(x = "Generations of xAM",
       y = expression("Cross-phenotype slope " ~ hat(r)[beta]),
       color = NULL, linetype = NULL) +
  default_theme2 +
  theme(legend.position = "bottom",
        strip.text = element_text(size = 10))

ggsave(file.path(FIG_DIR, "sfig_s16_pgi_gxe_grid.pdf"),
       plot_s16, width = 10, height = 10)
cat("Saved sfig_s16_pgi_gxe_grid.pdf\n")

cat("Done: plot_sfig_theory_gxe.R\n")
