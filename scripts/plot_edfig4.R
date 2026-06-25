#!/usr/bin/env Rscript
## plot_fig3_vdecomp_h025_r5.R  (ROUND 5 — non-destructive copy of scripts/plot_fig3_vdecomp_h025.R)
## ED Fig 4 (h²=0.25 counterpart of revised Figure 3 panels b–c), with the middle tier now the
## EXACT apparent (BLP) quantity and "Population h²"→"Direct h²". Reads round5/fig3_r5/.
## Writes round5/fig3_r5/sfig_vdecomp_h025_r5.{pdf,png}. Originals untouched.

library(ggplot2)
library(cowplot)

PROC_DIR <- "/home/rsb/Dropbox/ftsim/round4/round5/fig3_r5"
FIG_DIR  <- PROC_DIR

default_theme2 <- theme_minimal() + theme(text = element_text(size = 14, family = "Helvetica"))
PAL <- c("RM" = "#4DAF4A", "RM + VT" = "#984EA3", "5xAM" = "#E41A1C", "5xAM + VT" = "#377EB8")

h2_long <- read.csv(file.path(PROC_DIR, "fig3_vdecomp_r5_h2_long.csv"))
rg_long <- read.csv(file.path(PROC_DIR, "fig3_vdecomp_r5_rg_long.csv"))
fig3_scens <- c("RM", "RM + VT", "5xAM", "5xAM + VT")

h2_long$quantity <- factor(h2_long$quantity,
  levels = c("Direct h²", "Apparent h²", "LDSC h²"),
  labels = c("Direct~italic(h)^2", "Apparent~italic(h)^2", "LDSC~hat(italic(h))^2"))
rg_long$quantity <- factor(rg_long$quantity,
  levels = c("Direct r_g", "Apparent r_g", "LDSC r_g"),
  labels = c("Direct~italic(r)[g]", "Apparent~italic(r)[g]", "LDSC~hat(italic(r))['\U1D6FD']"))

h2_sub <- h2_long[h2_long$h2_param == 0.25 & h2_long$scenario %in% fig3_scens, ]
h2_sub$scenario <- factor(h2_sub$scenario, levels = fig3_scens)
plot_h2 <- ggplot(h2_sub, aes(x = gen, y = h2, color = scenario, linetype = scenario)) +
  default_theme2 +
  geom_hline(yintercept = 0.25, color = "grey", lty = 1) +
  stat_summary(geom = "linerange", position = position_dodge(width = 0.3),
               fun.max = function(x) quantile(x, 0.9), fun.min = function(x) quantile(x, 0.1),
               fun = function(x) quantile(x, 0.5)) +
  stat_summary(geom = "path", fun = median, linewidth = 1, position = position_dodge(width = 0.3)) +
  stat_summary(geom = "point", fun = median, size = 2, position = position_dodge(width = 0.3)) +
  facet_wrap(~ quantity, drop = TRUE, labeller = label_parsed) +
  scale_color_manual(values = PAL) +
  scale_linetype_manual(values = setNames(c(2:5), fig3_scens)) +
  ylab(NULL) + xlab("Generations of xAM")

rg_sub <- rg_long[rg_long$h2_param == 0.25 & rg_long$scenario %in% fig3_scens, ]
rg_sub$scenario <- factor(rg_sub$scenario, levels = fig3_scens)
plot_rg <- ggplot(rg_sub, aes(x = gen, y = rg, color = scenario, linetype = scenario)) +
  default_theme2 +
  geom_hline(yintercept = 0, color = "grey", lty = 1) +
  stat_summary(geom = "linerange", position = position_dodge(width = 0.3),
               fun.max = function(x) quantile(x, 0.9), fun.min = function(x) quantile(x, 0.1),
               fun = function(x) quantile(x, 0.5)) +
  stat_summary(geom = "path", fun = median, linewidth = 1, position = position_dodge(width = 0.3)) +
  stat_summary(geom = "point", fun = median, size = 2, position = position_dodge(width = 0.3)) +
  facet_wrap(~ quantity, drop = TRUE, labeller = label_parsed) +
  scale_color_manual(values = PAL) +
  scale_linetype_manual(values = setNames(c(2:5), fig3_scens)) +
  ylab(NULL) + xlab("Generations of xAM")

suppressWarnings({
  plts <- align_plots(
    plot_h2 + guides(color = guide_none(), linetype = guide_none()) +
      theme(axis.title.x.bottom = element_blank()),
    plot_rg + guides(color = guide_none(), linetype = guide_none()),
    align = "v", axis = "rl")
})
leg <- get_legend(plot_h2 + theme(legend.position = "bottom", legend.direction = "horizontal") +
  guides(color = guide_legend(nrow = 1), linetype = guide_legend(nrow = 1)))
fig <- plot_grid(plts[[1]], plts[[2]], leg, ncol = 1, rel_heights = c(1, 1, 0.1), labels = c("a", "b", ""))

ggsave(file.path(FIG_DIR, "sfig_vdecomp_h025_r5.pdf"), fig, width = 10, height = 8, device = cairo_pdf)
ggsave(file.path(FIG_DIR, "sfig_vdecomp_h025_r5.png"), fig, width = 10, height = 8, dpi = 300, bg = "white")
cat("Saved sfig_vdecomp_h025_r5.pdf/png\n")
