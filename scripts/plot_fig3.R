## plot_fig3.R
## Reads ONLY from processed/ and data/cdiags/ to generate Figure 3.
## 4-panel figure: (a) schematics, (b) h2, (c) rg, (d) GWAS FP
##
## Source notebook: manu/figure_nb/mFigComplexity.ipynb (cells 41, 39, 53, 63-72)

library(ggplot2)
library(cowplot)
library(splines)
library(magick)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
PROC_DIR <- file.path(BASE_DIR, "processed")
DIAG_DIR <- file.path(BASE_DIR, "data", "cdiags")
FIG_DIR  <- file.path(BASE_DIR, "figures_output")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

## ── Theme ───────────────────────────────────────────────────────────────────

default_theme  <- theme_bw() + theme(text = element_text(size = 14))
default_theme2 <- theme_minimal() + theme(text = element_text(size = 14))

PAL <- RColorBrewer::brewer.pal(8, "Set1")[-(5:7)]  # 4 colours for 4 scenarios

## ── Load processed data ─────────────────────────────────────────────────────

h2_raw <- read.csv(file.path(PROC_DIR, "fig3_h2_raw.csv"))
rg_raw <- read.csv(file.path(PROC_DIR, "fig3_rg_raw.csv"))
fp_raw <- read.csv(file.path(PROC_DIR, "fig3_fp_raw.csv"))

## ── Load SVG schematics ─────────────────────────────────────────────────────

s5xAM      <- image_read_svg(file.path(DIAG_DIR, "cropped_diag5xAM.svg"),
                              width = 2000)
s5xAMgXe   <- image_read_svg(file.path(DIAG_DIR, "cropped_diag5xAMgXE.svg"),
                              width = 2000)
s5xAMpVT   <- image_read_svg(file.path(DIAG_DIR, "cropped_diag5xAMpVT.svg"),
                              width = 2000)
s5xAMpVTgXe <- image_read_svg(file.path(DIAG_DIR, "cropped_diag5xAMgXEpVT.svg"),
                               width = 2000)

## ── Panel a: Schematics with per-scenario legends ───────────────────────────
## (notebook cells 61, 63-65)

schem1 <- ggdraw() + draw_image(s5xAM, clip = TRUE)
schem2 <- ggdraw() + draw_image(s5xAMgXe)
schem3 <- ggdraw() + draw_image(s5xAMpVT)
schem4 <- ggdraw() + draw_image(s5xAMpVTgXe)

## Build mini legend plots for each scenario (notebook cell 61)
## Use a dummy data approach to generate legends matching notebook style

make_legend_plot <- function(scenario_name, color_idx, lty_val, label_text) {
  dd <- data.frame(x = 1, y = 1, scenario = scenario_name)
  p <- ggplot(dd, aes(x, y, color = scenario, lty = scenario)) +
    geom_line() + default_theme +
    scale_color_manual(values = PAL[color_idx], labels = label_text) +
    scale_linetype_manual(values = lty_val, labels = label_text)
  p
}

p5xAM       <- make_legend_plot("5xAM",             1, 2, "5xAM")
p5xAMgxe    <- make_legend_plot("5xAM + GxE",       2, 3, expression("5xAM + G" * phantom() * times() * phantom() * "E"))
p5xAMpVT    <- make_legend_plot("5xAM + VT",        3, 4, "5xAM + VT")
p5xAMpVTgxe <- make_legend_plot("5xAM + VT + GxE",  4, 5, expression("5xAM + G" * phantom() * times() * phantom() * "E + VT"))

lsize <- 12
lkh <- 0.75
lkw <- 2

legend_theme <- theme(
  legend.justification = 0.5,
  legend.position = "left",
  legend.title = element_blank(),
  legend.key.width = unit(lkw, "cm"),
  legend.key.height = unit(lkh, "cm"),
  text = element_text(size = lsize)
)

leg_guide <- guides(
  color = guide_legend(label.position = "top"),
  linetype = guide_legend(label.position = "top")
)

leg1 <- get_legend(p5xAM       + leg_guide + legend_theme)
leg2 <- get_legend(p5xAMgxe    + leg_guide + legend_theme)
leg3 <- get_legend(p5xAMpVT    + leg_guide + legend_theme)
leg4 <- get_legend(p5xAMpVTgxe + leg_guide + legend_theme)

## Composite panel a: schematics stacked vertically with legends
## (notebook cell 72: schem1..schem4 + leg1..leg4 in 2-column byrow=F)
panel_a <- plot_grid(schem1, schem2, schem3, schem4,
                     leg1, leg2, leg3, leg4,
                     rel_widths = c(9, 2), byrow = FALSE, ncol = 2,
                     labels = c("a", rep("", 7)))

## ── Panel b: h2 true vs estimated (plot_uni_multi) ──────────────────────────
## (notebook cells 41, 45)

## Filter to h2 variables only
h2_dat <- h2_raw[h2_raw$variable %in% c("h2_he", "h2_true"), ]

(plot_uni_multi <- ggplot(
  h2_dat,
  aes(gen, value, color = scenario, linetype = scenario)) +
    default_theme2 +
    geom_hline(aes(yintercept = intercept), color = "#FF7F00", lty = 1,
               data = data.frame(intercept = 0.5)) +
    stat_summary(
      geom = "linerange", position = position_dodge(width = 0.3),
      fun.max = function(x) quantile(x, 0.9),
      fun.min = function(x) quantile(x, 0.1),
      fun     = function(x) quantile(x, 0.5)
    ) +
    stat_smooth(
      method = "gam",
      formula = y ~ bs(x, knots = 0:5, degree = 1),
      se = FALSE
    ) +
    ylab(expression(italic(h)^2)) +
    xlab("Generations of xAM") +
    scale_color_manual(values = PAL) +
    scale_linetype_manual(values = c(2:6)) +
    facet_wrap(~ var_label, drop = TRUE, labeller = label_parsed)
)

## ── Panel c: rg true vs estimated (plot_biv_multi) ──────────────────────────
## (notebook cell 39)

rg_dat <- rg_raw[rg_raw$variable %in% c("rbeta_HE", "rg_true"), ]

(plot_biv_multi <- ggplot(
  rg_dat,
  aes(gen, value, color = scenario, linetype = scenario)) +
    default_theme2 +
    geom_hline(aes(yintercept = intercept), color = "#FF7F00", lty = 1,
               data = data.frame(intercept = 0)) +
    stat_summary(
      geom = "linerange", position = position_dodge(width = 0.3),
      fun.max = function(x) quantile(x, 0.9),
      fun.min = function(x) quantile(x, 0.1),
      fun     = function(x) quantile(x, 0.5)
    ) +
    stat_smooth(
      method = "gam",
      formula = y ~ bs(x, knots = 0:5, degree = 1),
      se = FALSE
    ) +
    ylab(expression(italic(r)[g])) +
    xlab("Generations of xAM") +
    scale_color_manual(values = PAL) +
    scale_linetype_manual(values = c(2:6)) +
    facet_wrap(~ var_label, drop = TRUE, labeller = label_parsed)
)

## ── Panel d: GWAS FP inflation (plot_gwas2x5_alt) ──────────────────────────
## (notebook cell 53)
## Population GWAS only, faceted by power

fp_pop <- fp_raw[fp_raw$GWAS == "Population", ]
fp_pop$power_label <- paste(fp_pop$power, "power at \u03B1=0.05")

(plot_gwas_fp <- ggplot(
  fp_pop,
  aes(gen, relative_T1R, color = scenario, lty = scenario, shape = scenario)) +
    default_theme +
    stat_summary(
      geom = "point", position = position_dodge(width = 0.1),
      fun.data = function(x) mean_se(x, mult = 1.96)
    ) +
    stat_summary(
      geom = "linerange", position = position_dodge(width = 0.1),
      fun.data = function(x) mean_se(x, mult = 1.96)
    ) +
    geom_smooth(
      formula = y ~ ns(x, df = 3), method = "gam",
      aes(gen, color = scenario, lty = scenario), se = FALSE
    ) +
    facet_grid(~ power) +
    geom_hline(color = "grey", lty = 3, yintercept = 1) +
    scale_color_manual(values = PAL) +
    scale_linetype_manual(values = c(2:6)) +
    ylab(expression(over("Empirical Type-I Error Rate",
                         "Theoretical Type-I Error Rate"))) +
    xlab("Generations of xAM") +
    theme(axis.title.x = element_blank()) +
    theme(legend.position = c(0.5, 0.9),
          legend.direction = "horizontal",
          legend.key.width = unit(1, "cm"))
)

## ── Composite figure (notebook cell 72) ─────────────────────────────────────
## Layout: left = panel a (schematics), right = panels b/c stacked + d below

suppressWarnings({
  plts <- align_plots(
    plot_uni_multi + guides(color = guide_none(), linetype = guide_none()) +
      theme(axis.title.x.bottom = element_blank()),
    plot_biv_multi + guides(color = guide_none(), linetype = guide_none()),
    align = "v", axis = "rl"
  )
})

## Stack b and c, then add d below
st1 <- plot_grid(
  plts[[1]], plts[[2]],
  ncol = 1, rel_heights = c(1, 1),
  labels = c("b", "c")
)

right_col <- plot_grid(
  st1,
  plot_gwas_fp +
    theme(legend.direction = "vertical",
          text = element_text(size = 13),
          legend.position = c(0.18, 0.7)) +
    guides(shape = guide_none(), linetype = guide_none(),
           color = guide_none()),
  ncol = 1,
  rel_heights = c(6, 3),
  labels = c("", "d")
)

## Full figure
fig3 <- plot_grid(
  panel_a, NULL, right_col,
  nrow = 1,
  rel_widths = c(0.85, 0.05, 1.1)
)

## ── Save ────────────────────────────────────────────────────────────────────

ggsave(file.path(FIG_DIR, "fig3_complexity.pdf"),
       fig3, width = 14, height = 8)
ggsave(file.path(FIG_DIR, "fig3_complexity.png"),
       fig3, width = 14, height = 8, dpi = 300)

cat("Saved fig3_complexity.pdf and fig3_complexity.png to", FIG_DIR, "\n")

## Also save individual panels
ggsave(file.path(FIG_DIR, "fig3b_h2.pdf"),
       plot_uni_multi, width = 8, height = 5)
ggsave(file.path(FIG_DIR, "fig3c_rg.pdf"),
       plot_biv_multi, width = 8, height = 5)
ggsave(file.path(FIG_DIR, "fig3d_gwas_fp.pdf"),
       plot_gwas_fp, width = 8, height = 6)

cat("Saved individual panels to", FIG_DIR, "\n")
cat("\n=== plot_fig3.R complete ===\n")
