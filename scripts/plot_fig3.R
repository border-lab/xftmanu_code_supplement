#!/usr/bin/env Rscript
## plot_fig3_vdecomp_r5.R   (ROUND 5 — non-destructive copy of scripts/plot_fig3_vdecomp.R)
## Revised Figure 3 for the round-5 response: the middle decomposition tier is now the
## EXACT apparent (BLP) quantity (was predictive R²_g / r_cross r×); "Population h²"→"Direct h²".
## Reads round5/fig3_r5/ ; writes round5/fig3_r5/fig3_vdecomp_r5_full.{pdf,png}. Originals untouched.

library(ggplot2)
library(cowplot)
library(splines)
library(magick)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"                 # schematic input
PROC_DIR <- "/home/rsb/Dropbox/ftsim/round4/round5/fig3_r5"  # round-5 processed data
FIG_DIR  <- PROC_DIR
dir.create(FIG_DIR, showWarnings = FALSE)

## ── Themes (unchanged) ──────────────────────────────────────────────────────
default_theme  <- theme_bw()      + theme(text = element_text(size = 14, family = "Helvetica"))
default_theme2 <- theme_minimal() + theme(text = element_text(size = 14, family = "Helvetica"))
PAL <- c("RM" = "#4DAF4A", "RM + VT" = "#984EA3", "5xAM" = "#E41A1C", "5xAM + VT" = "#377EB8")

## ── Load round-5 data ───────────────────────────────────────────────────────
h2_long <- read.csv(file.path(PROC_DIR, "fig3_vdecomp_r5_h2_long.csv"))
rg_long <- read.csv(file.path(PROC_DIR, "fig3_vdecomp_r5_rg_long.csv"))
fp_raw  <- read.csv(file.path(PROC_DIR, "fig3_vdecomp_r5_fp.csv"))

fig3_scens <- c("RM", "RM + VT", "5xAM", "5xAM + VT")

## NEW tier labels: Direct, Apparent, LDSC
h2_long$quantity <- factor(h2_long$quantity,
  levels = c("Direct h²", "Apparent h²", "LDSC h²"),
  labels = c("Direct~italic(h)^2", "Apparent~italic(h)^2", "LDSC~hat(italic(h))^2"))
rg_long$quantity <- factor(rg_long$quantity,
  levels = c("Direct r_g", "Apparent r_g", "LDSC r_g"),
  labels = c("Direct~italic(r)[g]", "Apparent~italic(r)[g]", "LDSC~hat(italic(r))['\U1D6FD']"))

## ── Panel a: schematic (serves as legend) ──────────────────────────────────
schem_img <- image_read_svg(file.path(BASE_DIR, "colored new cdiags complexity gray.svg"), width = 3000)
panel_a <- ggdraw() + draw_image(schem_img)

## ── Panel b: h² decomposition ──────────────────────────────────────────────
h2_fig3 <- h2_long[h2_long$h2_param == 0.5 & h2_long$scenario %in% fig3_scens, ]
h2_fig3$scenario <- factor(h2_fig3$scenario, levels = fig3_scens)

plot_h2 <- ggplot(h2_fig3, aes(x = gen, y = h2, color = scenario, linetype = scenario)) +
  default_theme2 +
  geom_hline(yintercept = 0.5, color = "grey", lty = 1) +
  stat_summary(geom = "linerange", position = position_dodge(width = 0.3),
               fun.max = function(x) quantile(x, 0.9),
               fun.min = function(x) quantile(x, 0.1),
               fun     = function(x) quantile(x, 0.5)) +
  stat_summary(geom = "path", fun = median, linewidth = 1, position = position_dodge(width = 0.3)) +
  stat_summary(geom = "point", fun = median, size = 2, position = position_dodge(width = 0.3)) +
  facet_wrap(~ quantity, drop = TRUE, labeller = label_parsed) +
  scale_color_manual(values = PAL) +
  scale_linetype_manual(values = setNames(c(2:5), fig3_scens)) +
  ylab(NULL) + xlab("Generations of xAM")

## ── Panel c: rg decomposition ──────────────────────────────────────────────
rg_fig3 <- rg_long[rg_long$h2_param == 0.5 & rg_long$scenario %in% fig3_scens, ]
rg_fig3$scenario <- factor(rg_fig3$scenario, levels = fig3_scens)

plot_rg <- ggplot(rg_fig3, aes(x = gen, y = rg, color = scenario, linetype = scenario)) +
  default_theme2 +
  geom_hline(yintercept = 0, color = "grey", lty = 1) +
  stat_summary(geom = "linerange", position = position_dodge(width = 0.3),
               fun.max = function(x) quantile(x, 0.9),
               fun.min = function(x) quantile(x, 0.1),
               fun     = function(x) quantile(x, 0.5)) +
  stat_summary(geom = "path", fun = median, linewidth = 1, position = position_dodge(width = 0.3)) +
  stat_summary(geom = "point", fun = median, size = 2, position = position_dodge(width = 0.3)) +
  facet_wrap(~ quantity, drop = TRUE, labeller = label_parsed) +
  scale_color_manual(values = PAL) +
  scale_linetype_manual(values = setNames(c(2:5), fig3_scens)) +
  ylab(NULL) + xlab("Generations of xAM")

## ── Panel d: GWAS FP inflation (unchanged) ─────────────────────────────────
gwas_scens <- c("RM", "RM + VT", "5xAM", "5xAM + VT")
fp_pop <- fp_raw[fp_raw$GWAS == "Population" & fp_raw$scenario %in% gwas_scens, ]
fp_pop$scenario <- factor(fp_pop$scenario, levels = gwas_scens)
fp_pop$power_pct <- gsub("%", "", fp_pop$power)
fp_pop$power_label <- factor(paste0(fp_pop$power_pct, "*'%'~power~at~alpha==0.05"))

plot_fp <- ggplot(fp_pop, aes(gen, relative_T1R, color = scenario, lty = scenario, shape = scenario)) +
  default_theme +
  geom_hline(color = "grey", lty = 1, yintercept = 1) +
  stat_summary(geom = "point", position = position_dodge(width = 0.1), fun = median) +
  stat_summary(geom = "path", position = position_dodge(width = 0.1), fun = median, linewidth = 1) +
  facet_grid(~ power_label, labeller = label_parsed) +
  scale_color_manual(values = PAL[gwas_scens]) +
  scale_linetype_manual(values = setNames(c(2:5), gwas_scens)) +
  ylab(expression(over("Empirical Type-I Error Rate", "Theoretical Type-I Error Rate"))) +
  xlab("Generations of xAM") +
  theme(axis.title.x = element_blank()) +
  theme(legend.position = c(0.5, 0.9), legend.direction = "horizontal",
        legend.key.width = unit(1, "cm"))

## ── Assembly (unchanged) ────────────────────────────────────────────────────
suppressWarnings({
  plts <- align_plots(
    plot_h2 + guides(color = guide_none(), linetype = guide_none()) +
      theme(axis.title.x.bottom = element_blank()),
    plot_rg + guides(color = guide_none(), linetype = guide_none()),
    align = "v", axis = "rl")
})
st1 <- plot_grid(plts[[1]], plts[[2]], ncol = 1, rel_heights = c(1, 1), labels = c("b", "c"))
right_col <- plot_grid(
  st1,
  plot_fp + theme(legend.direction = "vertical", text = element_text(size = 13),
                  axis.title.y = element_text(size = 10), legend.position = c(0.18, 0.7)) +
    guides(shape = guide_none(), linetype = guide_none(), color = guide_none()),
  ncol = 1, rel_heights = c(6, 3), labels = c("", "d"))
fig3 <- plot_grid(panel_a, right_col, nrow = 1, rel_widths = c(0.7, 1.3), labels = c("a", ""))

ggsave(file.path(FIG_DIR, "fig3_vdecomp_r5_full.pdf"), fig3, width = 14, height = 8, device = cairo_pdf)
ggsave(file.path(FIG_DIR, "fig3_vdecomp_r5_full.png"), fig3, width = 14, height = 8, dpi = 300, bg = "white")
cat("Saved fig3_vdecomp_r5_full.pdf/png to", FIG_DIR, "\n")
cat("\n=== plot_fig3_vdecomp_r5.R complete ===\n")
