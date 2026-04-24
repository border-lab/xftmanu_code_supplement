#!/usr/bin/env Rscript
## plot_fig3_alt.R
## Alternative Figure 3 layouts incorporating variance decomposition (Comment 2)
## Does NOT overwrite any existing files — outputs to figures_output/fig3_alt_*

library(ggplot2)
library(cowplot)
library(splines)
library(magick)
library(reshape2)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
PROC_DIR <- file.path(BASE_DIR, "processed")
DIAG_DIR <- file.path(BASE_DIR, "data", "cdiags")
FIG_DIR  <- file.path(BASE_DIR, "figures_output")

## ── Theme ──────────────────────────────────────────────────────────────────
default_theme  <- theme_bw() + theme(text = element_text(size = 14))
default_theme2 <- theme_minimal() + theme(text = element_text(size = 14))

PAL <- RColorBrewer::brewer.pal(8, "Set1")[-(5:7)]  # 4 scenario colours

## ── Load all processed data ────────────────────────────────────────────────
h2_raw  <- read.csv(file.path(PROC_DIR, "fig3_h2_raw.csv"))
rg_raw  <- read.csv(file.path(PROC_DIR, "fig3_rg_raw.csv"))
fp_raw  <- read.csv(file.path(PROC_DIR, "fig3_fp_raw.csv"))
vd_long <- read.csv(file.path(PROC_DIR, "figR_variance_decomp_long.csv"))

## Factor levels
scenario_levels <- c("5xAM", "5xAM + VT", "5xAM + GxE", "5xAM + VT + GxE")
h2_raw$scenario  <- factor(h2_raw$scenario, levels = scenario_levels)
rg_raw$scenario  <- factor(rg_raw$scenario, levels = scenario_levels)
fp_raw$scenario  <- factor(fp_raw$scenario, levels = scenario_levels)
vd_long$scenario <- factor(vd_long$scenario, levels = scenario_levels)
vd_long$quantity <- factor(vd_long$quantity,
                           levels = c("Panmictic (causal var.)",
                                      "True (Vg/Vy)",
                                      "Pop. estimated (LDSC)"))

## ── Load schematics ───────────────────────────────────────────────────────
s5xAM       <- image_read_svg(file.path(DIAG_DIR, "cropped_diag5xAM.svg"), width = 2000)
s5xAMgXe    <- image_read_svg(file.path(DIAG_DIR, "cropped_diag5xAMgXE.svg"), width = 2000)
s5xAMpVT    <- image_read_svg(file.path(DIAG_DIR, "cropped_diag5xAMpVT.svg"), width = 2000)
s5xAMpVTgXe <- image_read_svg(file.path(DIAG_DIR, "cropped_diag5xAMgXEpVT.svg"), width = 2000)

## ── Reusable components ───────────────────────────────────────────────────

## Schematics (used in multiple layouts)
schem1 <- ggdraw() + draw_image(s5xAM, clip = TRUE)
schem2 <- ggdraw() + draw_image(s5xAMgXe)
schem3 <- ggdraw() + draw_image(s5xAMpVT)
schem4 <- ggdraw() + draw_image(s5xAMpVTgXe)

## Mini legend plots (same as original)
make_legend_plot <- function(scenario_name, color_idx, lty_val, label_text) {
  dd <- data.frame(x = 1, y = 1, scenario = scenario_name)
  p <- ggplot(dd, aes(x, y, color = scenario, lty = scenario)) +
    geom_line() + default_theme +
    scale_color_manual(values = PAL[color_idx], labels = label_text) +
    scale_linetype_manual(values = lty_val, labels = label_text)
  p
}

lsize <- 12; lkh <- 0.75; lkw <- 2
legend_theme <- theme(
  legend.justification = 0.5, legend.position = "left",
  legend.title = element_blank(),
  legend.key.width = unit(lkw, "cm"), legend.key.height = unit(lkh, "cm"),
  text = element_text(size = lsize)
)
leg_guide <- guides(
  color = guide_legend(label.position = "top"),
  linetype = guide_legend(label.position = "top")
)

p5xAM       <- make_legend_plot("5xAM",             1, 2, "5xAM")
p5xAMgxe    <- make_legend_plot("5xAM + GxE",       2, 3, expression("5xAM + G" * phantom() * times() * phantom() * "E"))
p5xAMpVT    <- make_legend_plot("5xAM + VT",        3, 4, "5xAM + VT")
p5xAMpVTgxe <- make_legend_plot("5xAM + VT + GxE",  4, 5, expression("5xAM + G" * phantom() * times() * phantom() * "E + VT"))

leg1 <- get_legend(p5xAM       + leg_guide + legend_theme)
leg2 <- get_legend(p5xAMgxe    + leg_guide + legend_theme)
leg3 <- get_legend(p5xAMpVT    + leg_guide + legend_theme)
leg4 <- get_legend(p5xAMpVTgxe + leg_guide + legend_theme)

panel_a <- plot_grid(schem1, schem2, schem3, schem4,
                     leg1, leg2, leg3, leg4,
                     rel_widths = c(9, 2), byrow = FALSE, ncol = 2,
                     labels = c("a", rep("", 7)))

## ── Reusable plot builders ─────────────────────────────────────────────────

## Variance decomposition panel: faceted by scenario, 3 h2 quantities
qty_pal <- c("Panmictic (causal var.)" = "#E41A1C",
             "True (Vg/Vy)"           = "#377EB8",
             "Pop. estimated (LDSC)"  = "#4DAF4A")
qty_lty <- c("Panmictic (causal var.)" = "dotted",
             "True (Vg/Vy)"           = "dashed",
             "Pop. estimated (LDSC)"  = "solid")

plot_vd <- ggplot(vd_long, aes(x = gen, y = h2, color = quantity, linetype = quantity)) +
  stat_summary(geom = "linerange", fun.data = mean_sdl, fun.args = list(mult = 1),
               position = position_dodge(width = 0.3), show.legend = FALSE) +
  stat_summary(geom = "line", fun = mean,
               position = position_dodge(width = 0.3)) +
  stat_summary(geom = "point", fun = mean, size = 2,
               position = position_dodge(width = 0.3)) +
  facet_wrap(~ scenario, nrow = 1) +
  scale_color_manual(values = qty_pal, name = "") +
  scale_linetype_manual(values = qty_lty, name = "") +
  labs(x = "Generation", y = expression(h^2)) +
  geom_hline(yintercept = 0.5, color = "#FF7F00", lty = 1) +
  default_theme2 +
  theme(legend.position = "bottom", legend.direction = "horizontal",
        strip.text = element_text(size = 11))

## Original h2 panel (true vs LDSC, faceted by variable, coloured by scenario)
h2_dat <- h2_raw[h2_raw$variable %in% c("h2_he", "h2_true"), ]
plot_h2 <- ggplot(h2_dat, aes(gen, value, color = scenario, linetype = scenario)) +
  default_theme2 +
  geom_hline(aes(yintercept = intercept), color = "#FF7F00", lty = 1,
             data = data.frame(intercept = 0.5)) +
  stat_summary(geom = "linerange", position = position_dodge(width = 0.3),
               fun.max = function(x) quantile(x, 0.9),
               fun.min = function(x) quantile(x, 0.1),
               fun     = function(x) quantile(x, 0.5)) +
  stat_smooth(method = "gam", formula = y ~ bs(x, knots = 0:5, degree = 1), se = FALSE) +
  ylab(expression(italic(h)^2)) + xlab("Generations of xAM") +
  scale_color_manual(values = PAL) +
  scale_linetype_manual(values = c(2:6)) +
  facet_wrap(~ var_label, drop = TRUE, labeller = label_parsed)

## Original rg panel
rg_dat <- rg_raw[rg_raw$variable %in% c("rbeta_HE", "rg_true"), ]
plot_rg <- ggplot(rg_dat, aes(gen, value, color = scenario, linetype = scenario)) +
  default_theme2 +
  geom_hline(aes(yintercept = intercept), color = "#FF7F00", lty = 1,
             data = data.frame(intercept = 0)) +
  stat_summary(geom = "linerange", position = position_dodge(width = 0.3),
               fun.max = function(x) quantile(x, 0.9),
               fun.min = function(x) quantile(x, 0.1),
               fun     = function(x) quantile(x, 0.5)) +
  stat_smooth(method = "gam", formula = y ~ bs(x, knots = 0:5, degree = 1), se = FALSE) +
  ylab(expression(italic(r)[g])) + xlab("Generations of xAM") +
  scale_color_manual(values = PAL) +
  scale_linetype_manual(values = c(2:6)) +
  facet_wrap(~ var_label, drop = TRUE, labeller = label_parsed)

## GWAS FP panel
fp_pop <- fp_raw[fp_raw$GWAS == "Population", ]
plot_fp <- ggplot(fp_pop, aes(gen, relative_T1R, color = scenario, lty = scenario, shape = scenario)) +
  default_theme +
  stat_summary(geom = "point", position = position_dodge(width = 0.1),
               fun.data = function(x) mean_se(x, mult = 1.96)) +
  stat_summary(geom = "linerange", position = position_dodge(width = 0.1),
               fun.data = function(x) mean_se(x, mult = 1.96)) +
  geom_smooth(formula = y ~ ns(x, df = 3), method = "gam",
              aes(gen, color = scenario, lty = scenario), se = FALSE) +
  facet_grid(~ power) +
  geom_hline(color = "grey", lty = 3, yintercept = 1) +
  scale_color_manual(values = PAL) +
  scale_linetype_manual(values = c(2:6)) +
  ylab(expression(over("Empirical Type-I Error Rate", "Theoretical Type-I Error Rate"))) +
  xlab("Generations of xAM")

## ========================================================================
## ALT A: Original layout + variance decomp panel added as new row
## Layout: [schematics | h2 + rg] / [variance decomp + gwas fp]
## ========================================================================

suppressWarnings({
  plts_A <- align_plots(
    plot_h2 + guides(color = guide_none(), linetype = guide_none()) +
      theme(axis.title.x.bottom = element_blank()),
    plot_rg + guides(color = guide_none(), linetype = guide_none()),
    align = "v", axis = "rl"
  )
})

right_top <- plot_grid(plts_A[[1]], plts_A[[2]],
                       ncol = 1, rel_heights = c(1, 1),
                       labels = c("b", "c"))

top_row <- plot_grid(panel_a, NULL, right_top,
                     nrow = 1, rel_widths = c(0.85, 0.05, 1.1))

bottom_row <- plot_grid(
  plot_vd + theme(legend.position = "bottom") +
    guides(color = guide_legend(nrow = 1), linetype = guide_legend(nrow = 1)),
  plot_fp +
    theme(legend.position = c(0.18, 0.7), legend.direction = "vertical",
          text = element_text(size = 13)) +
    guides(shape = guide_none(), linetype = guide_none(), color = guide_none()),
  nrow = 1, rel_widths = c(1.2, 1),
  labels = c("e", "d")
)

fig3_altA <- plot_grid(top_row, bottom_row, ncol = 1, rel_heights = c(1, 0.5))

ggsave(file.path(FIG_DIR, "fig3_altA.pdf"), fig3_altA, width = 16, height = 12)
ggsave(file.path(FIG_DIR, "fig3_altA.png"), fig3_altA, width = 16, height = 12, dpi = 300)
cat("Saved fig3_altA (original + variance decomp row)\n")

## ========================================================================
## ALT B: Row-per-scenario layout
## Each row: schematic | h2 decomp (3 quantities) | rg (true vs LDSC)
## No GWAS FP in main figure (move to supplement or add as separate panel)
## ========================================================================

## For this layout, we need per-scenario plots of h2 decomp and rg
make_row_h2 <- function(scen, show_xlab = FALSE, show_legend = FALSE) {
  d <- vd_long[vd_long$scenario == scen, ]
  p <- ggplot(d, aes(x = gen, y = h2, color = quantity, linetype = quantity)) +
    stat_summary(geom = "line", fun = mean) +
    stat_summary(geom = "point", fun = mean, size = 2) +
    stat_summary(geom = "linerange", fun.data = mean_sdl, fun.args = list(mult = 1),
                 show.legend = FALSE) +
    geom_hline(yintercept = 0.5, color = "#FF7F00", lty = 1) +
    scale_color_manual(values = qty_pal, name = "") +
    scale_linetype_manual(values = qty_lty, name = "") +
    labs(y = expression(h^2)) +
    default_theme2 +
    coord_cartesian(ylim = c(0.35, 0.75)) +
    theme(legend.position = if (show_legend) "bottom" else "none")
  if (!show_xlab) {
    p <- p + theme(axis.title.x = element_blank(), axis.text.x = element_blank())
  } else {
    p <- p + labs(x = "Generation")
  }
  p
}

make_row_rg <- function(scen, show_xlab = FALSE) {
  d <- rg_dat[rg_dat$scenario == scen, ]
  p <- ggplot(d, aes(gen, value, color = variable, linetype = variable)) +
    default_theme2 +
    geom_hline(yintercept = 0, color = "#FF7F00", lty = 1) +
    stat_summary(geom = "line", fun = mean) +
    stat_summary(geom = "point", fun = mean, size = 2) +
    stat_summary(geom = "linerange",
                 fun.max = function(x) quantile(x, 0.9),
                 fun.min = function(x) quantile(x, 0.1),
                 show.legend = FALSE) +
    scale_color_manual(values = c("rbeta_HE" = "#4DAF4A", "rg_true" = "#377EB8"),
                       labels = c("rbeta_HE" = expression(hat(r)[beta]),
                                  "rg_true" = expression(r[score])),
                       name = "") +
    scale_linetype_manual(values = c("rbeta_HE" = "solid", "rg_true" = "dashed"),
                          labels = c("rbeta_HE" = expression(hat(r)[beta]),
                                     "rg_true" = expression(r[score])),
                          name = "") +
    labs(y = expression(italic(r)[g])) +
    coord_cartesian(ylim = c(-0.05, 0.45)) +
    theme(legend.position = "none")
  if (!show_xlab) {
    p <- p + theme(axis.title.x = element_blank(), axis.text.x = element_blank())
  } else {
    p <- p + labs(x = "Generation")
  }
  p
}

scens  <- c("5xAM", "5xAM + VT", "5xAM + GxE", "5xAM + VT + GxE")
schems <- list(schem1, schem2, schem3, schem4)
legs   <- list(leg1, leg2, leg3, leg4)

rows <- list()
for (i in seq_along(scens)) {
  is_last <- (i == length(scens))

  schem_with_leg <- plot_grid(schems[[i]], legs[[i]],
                              ncol = 2, rel_widths = c(4, 1))

  h2_panel <- make_row_h2(scens[i], show_xlab = is_last, show_legend = FALSE)
  rg_panel <- make_row_rg(scens[i], show_xlab = is_last)

  row_label <- if (i == 1) letters[i] else ""
  rows[[i]] <- plot_grid(
    schem_with_leg, h2_panel, rg_panel,
    nrow = 1, rel_widths = c(0.8, 1, 0.8)
  )
}

## Add shared legends at the bottom
h2_legend <- get_legend(
  make_row_h2(scens[1], show_legend = TRUE) +
    guides(color = guide_legend(nrow = 1), linetype = guide_legend(nrow = 1))
)

rg_legend_plot <- ggplot(rg_dat[rg_dat$scenario == scens[1], ],
       aes(gen, value, color = variable, linetype = variable)) +
  geom_line() +
  scale_color_manual(values = c("rbeta_HE" = "#4DAF4A", "rg_true" = "#377EB8"),
                     labels = c("rbeta_HE" = expression(hat(r)[beta] ~ "(LDSC)"),
                                "rg_true" = expression(r[score] ~ "(true)")),
                     name = "") +
  scale_linetype_manual(values = c("rbeta_HE" = "solid", "rg_true" = "dashed"),
                        labels = c("rbeta_HE" = expression(hat(r)[beta] ~ "(LDSC)"),
                                   "rg_true" = expression(r[score] ~ "(true)")),
                        name = "") +
  theme(legend.position = "bottom")
rg_legend <- get_legend(rg_legend_plot)

legend_row <- plot_grid(NULL, h2_legend, rg_legend,
                        nrow = 1, rel_widths = c(0.8, 1, 0.8))

fig3_altB <- plot_grid(
  rows[[1]], rows[[2]], rows[[3]], rows[[4]], legend_row,
  ncol = 1, rel_heights = c(1, 1, 1, 1.15, 0.2),
  labels = c("a", "b", "c", "d", "")
)

ggsave(file.path(FIG_DIR, "fig3_altB.pdf"), fig3_altB, width = 16, height = 14)
ggsave(file.path(FIG_DIR, "fig3_altB.png"), fig3_altB, width = 16, height = 14, dpi = 300)
cat("Saved fig3_altB (row-per-scenario)\n")

## ========================================================================
## ALT C: Minimal change — keep original but replace panel b h2 facets
## with the 3-quantity variance decomposition (faceted by scenario)
## ========================================================================

suppressWarnings({
  plts_C <- align_plots(
    plot_vd + guides(color = guide_none(), linetype = guide_none()) +
      theme(axis.title.x.bottom = element_blank()),
    plot_rg + guides(color = guide_none(), linetype = guide_none()),
    align = "v", axis = "rl"
  )
})

right_col_C <- plot_grid(
  plts_C[[1]], plts_C[[2]],
  plot_fp +
    theme(legend.direction = "vertical", text = element_text(size = 13),
          legend.position = c(0.18, 0.7)) +
    guides(shape = guide_none(), linetype = guide_none(), color = guide_none()),
  ncol = 1, rel_heights = c(3, 3, 2.5),
  labels = c("b", "c", "d")
)

## Need a legend for the variance decomp quantities
vd_leg <- get_legend(
  plot_vd + theme(legend.position = "right") +
    guides(color = guide_legend(ncol = 1), linetype = guide_legend(ncol = 1))
)

right_with_leg <- plot_grid(right_col_C, vd_leg,
                            nrow = 1, rel_widths = c(5, 1))

fig3_altC <- plot_grid(
  panel_a, NULL, right_with_leg,
  nrow = 1, rel_widths = c(0.85, 0.05, 1.3)
)

ggsave(file.path(FIG_DIR, "fig3_altC.pdf"), fig3_altC, width = 16, height = 9)
ggsave(file.path(FIG_DIR, "fig3_altC.png"), fig3_altC, width = 16, height = 9, dpi = 300)
cat("Saved fig3_altC (replace h2 panel with variance decomp)\n")

cat("\n=== plot_fig3_alt.R complete ===\n")
