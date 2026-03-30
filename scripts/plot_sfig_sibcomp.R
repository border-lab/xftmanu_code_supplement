## plot_sfig_sibcomp.R
## Generates supplementary figure S17 from processed data ONLY.
##   S17a: Relative T1E inflation for pop vs sib GWAS
##   S17b: Cross-phenotype slope correlations for pop vs sib GWAS

library(ggplot2)
library(cowplot)
library(splines)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
PROC_DIR <- file.path(BASE_DIR, "processed")
FIG_DIR  <- file.path(BASE_DIR, "figures_output")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

default_theme <- theme_bw() + theme(text = element_text(size = 14))
PAL <- RColorBrewer::brewer.pal(8, "Set1")

## ── S17a: Relative T1E inflation ────────────────────────────────────────────────

t1e <- read.csv(file.path(PROC_DIR, "sfig_s17a_t1e_comparison.csv"))

## Reapply factor ordering (lost in CSV roundtrip)
t1e$scenario <- factor(t1e$scenario,
  levels = c("RM + VT", "2xAM", "5xAM", "5xAM + GxE",
             "5xAM + VT", "5xAM + VT + GxE"))

## Only plot gen > 0 (T1E at gen=0 is uninformative)
t1e_pos <- t1e[t1e$gen > 0, ]

plot_s17a <- ggplot(t1e_pos,
       aes(gen, relative_T1R, color = GWAS,
           linetype = power_label, shape = power_label)) +
  stat_summary(geom = "point", position = position_dodge(width = 0.1),
               fun.data = function(x) mean_se(x, mult = 1.96)) +
  stat_summary(geom = "linerange", position = position_dodge(width = 0.1),
               fun.data = function(x) mean_se(x, mult = 1.96)) +
  geom_smooth(formula = y ~ ns(x, df = 3), linewidth = 0.5,
              method = "gam",
              aes(gen, color = GWAS, linetype = power_label), se = FALSE) +
  facet_wrap(~ scenario) +
  geom_hline(color = "grey", linetype = 3, yintercept = 1) +
  scale_color_manual(values = c("Population" = PAL[1],
                                 "Sibship" = PAL[2])) +
  scale_linetype_manual(values = c(1, 2, 3)) +
  scale_shape_manual(values = c(16, 17, 15)) +
  labs(x = "Generations of xAM",
       y = expression(frac("Empirical Type-I error", alpha)),
       color = "GWAS type", linetype = NULL, shape = NULL) +
  default_theme +
  theme(legend.position = "bottom",
        legend.box = "vertical",
        strip.text = element_text(size = 10))

## ── S17b: Cross-phenotype slope ─────────────────────────────────────────────────

slope <- read.csv(file.path(PROC_DIR, "sfig_s17b_slope_comparison.csv"))

slope$scenario <- factor(slope$scenario,
  levels = c("RM + VT", "2xAM", "5xAM", "5xAM + GxE",
             "5xAM + VT", "5xAM + VT + GxE"))

plot_s17b <- ggplot(slope,
       aes(gen, value, color = GWAS)) +
  stat_summary(geom = "point", position = position_dodge(width = 0.2),
               fun.data = function(x) mean_se(x, mult = 1.96), size = 1.5) +
  stat_summary(geom = "linerange", position = position_dodge(width = 0.2),
               fun.data = function(x) mean_se(x, mult = 1.96)) +
  geom_smooth(formula = y ~ ns(x, df = 3), linewidth = 0.5,
              method = "gam", se = FALSE) +
  facet_wrap(~ scenario) +
  geom_hline(yintercept = 0, color = "grey", linetype = 3) +
  scale_color_manual(values = c("Population" = PAL[1],
                                 "Sibship" = PAL[2])) +
  labs(x = "Generations of xAM",
       y = expression("Cross-phenotype slope " ~ hat(r)[beta]),
       color = "GWAS type") +
  default_theme +
  theme(legend.position = "bottom",
        strip.text = element_text(size = 10))

## ── Combined S17 ────────────────────────────────────────────────────────────────

## Align on horizontal axis
aligned <- align_plots(
  plot_s17a + guides(shape = guide_legend(order = 1),
                     linetype = guide_legend(order = 1),
                     color = guide_legend(order = 2)) +
    theme(legend.box.just = "center"),
  plot_s17b + guides(color = guide_none()),
  axis = "b", align = "h"
)

plot_s17_combined <- plot_grid(
  aligned[[1]], aligned[[2]],
  nrow = 1,
  rel_widths = c(3, 2),
  labels = c("a", "b"),
  label_size = 14
)

ggsave(file.path(FIG_DIR, "sfig_s17_sibcomp_combined.pdf"),
       plot_s17_combined, width = 14, height = 7)
cat("Saved sfig_s17_sibcomp_combined.pdf\n")

## Also save individual panels
ggsave(file.path(FIG_DIR, "sfig_s17a_t1e.pdf"),
       plot_s17a, width = 8, height = 7)
cat("Saved sfig_s17a_t1e.pdf\n")

ggsave(file.path(FIG_DIR, "sfig_s17b_slope.pdf"),
       plot_s17b, width = 8, height = 7)
cat("Saved sfig_s17b_slope.pdf\n")

cat("Done: plot_sfig_sibcomp.R\n")
