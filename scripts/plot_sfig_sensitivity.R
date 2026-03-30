## plot_sfig_sensitivity.R
## Generates supplementary figures S5 and S6 from processed data ONLY.
##   S5: rg under xAM with r=0.1 vs r=0.2
##   S6: rg under unidimensional xAM with fixed latent correlation

library(ggplot2)
library(cowplot)
library(splines)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
PROC_DIR <- file.path(BASE_DIR, "processed")
FIG_DIR  <- file.path(BASE_DIR, "figures_output")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

default_theme <- theme_bw() + theme(text = element_text(size = 14))
PAL <- RColorBrewer::brewer.pal(8, "Set1")

## ── S5: rg under 5xAM with r=0.1 vs r=0.2 ────────────────────────────────────

s5 <- read.csv(file.path(PROC_DIR, "sfig_s5_rg_sensitivity.csv"))

## Separate r=0.1 and r=0.2 panels
s5$rmate_label <- paste0("r = ", s5$rmate)

## Keep only 5xAM scenarios (plus 2xAM baseline)
s5_5x <- s5[s5$scenario %in% c("2xAM", "5xAM", "5xAM + VT",
                                 "5xAM + GxE", "5xAM + VT + GxE"), ]

## Set factor order
s5_5x$scenario <- factor(s5_5x$scenario,
                          levels = c("2xAM", "5xAM", "5xAM + VT",
                                     "5xAM + GxE", "5xAM + VT + GxE"))

plot_s5 <- ggplot(s5_5x, aes(gen, he_rg, color = scenario,
                               linetype = scenario, shape = scenario)) +
  stat_summary(geom = "linerange", position = position_dodge(width = 0.3),
               fun.max = function(x) quantile(x, 0.9),
               fun.min = function(x) quantile(x, 0.1),
               fun = function(x) quantile(x, 0.5)) +
  stat_summary(geom = "point", position = position_dodge(width = 0.3),
               fun = function(x) quantile(x, 0.5), size = 2.5) +
  stat_smooth(method = "gam",
              formula = y ~ bs(x, knots = 0:5, degree = 1), se = FALSE) +
  facet_wrap(~ rmate_label) +
  scale_color_manual(values = c("2xAM" = PAL[1],
                                 "5xAM" = PAL[1],
                                 "5xAM + VT" = PAL[2],
                                 "5xAM + GxE" = PAL[3],
                                 "5xAM + VT + GxE" = PAL[4])) +
  scale_linetype_manual(values = c("2xAM" = "dashed",
                                    "5xAM" = "solid",
                                    "5xAM + VT" = "solid",
                                    "5xAM + GxE" = "solid",
                                    "5xAM + VT + GxE" = "dashed")) +
  scale_shape_manual(values = c("2xAM" = 1, "5xAM" = 16,
                                 "5xAM + VT" = 16, "5xAM + GxE" = 17,
                                 "5xAM + VT + GxE" = 15)) +
  labs(x = "Generations of xAM",
       y = expression(Estimated ~ r[g]),
       color = NULL, linetype = NULL, shape = NULL) +
  default_theme +
  theme(legend.position = "bottom",
        legend.box = "horizontal",
        strip.text = element_text(size = 12)) +
  guides(color = guide_legend(nrow = 1))

ggsave(file.path(FIG_DIR, "sfig_s5_rg_sensitivity.pdf"),
       plot_s5, width = 12, height = 6)
cat("Saved sfig_s5_rg_sensitivity.pdf\n")

## ── S6: rg under xAM with fixed latent correlation ─────────────────────────────

s6 <- read.csv(file.path(PROC_DIR, "sfig_s6_fixed_latent.csv"))

## Color by scenario label
s6$scenario_label <- factor(s6$scenario_label)

## Use distinct colors for each kphen
n_scen <- length(unique(s6$scenario_label))
scen_colors <- PAL[seq_len(n_scen)]
names(scen_colors) <- levels(s6$scenario_label)

## Use distinct linetypes
scen_lty <- c("solid", "dashed", "dotted", "dotdash", "longdash")[seq_len(n_scen)]
names(scen_lty) <- levels(s6$scenario_label)

scen_shapes <- c(16, 17, 15, 18, 8)[seq_len(n_scen)]
names(scen_shapes) <- levels(s6$scenario_label)

plot_s6 <- ggplot(s6, aes(gen, he_rg, color = scenario_label,
                            linetype = scenario_label,
                            shape = scenario_label)) +
  stat_summary(geom = "linerange",
               fun.max = function(x) quantile(x, 0.9),
               fun.min = function(x) quantile(x, 0.1),
               fun = function(x) quantile(x, 0.5)) +
  stat_summary(geom = "line", fun = mean) +
  stat_summary(geom = "point", size = 2.5, fun = mean) +
  scale_color_manual(values = scen_colors) +
  scale_linetype_manual(values = scen_lty) +
  scale_shape_manual(values = scen_shapes) +
  labs(x = "Generations of xAM",
       y = expression(Estimated ~ r[g]),
       color = NULL, linetype = NULL, shape = NULL) +
  default_theme +
  theme(legend.position = "bottom",
        legend.box = "horizontal") +
  guides(color = guide_legend(nrow = 2))

ggsave(file.path(FIG_DIR, "sfig_s6_fixed_latent.pdf"),
       plot_s6, width = 9, height = 5.5)
cat("Saved sfig_s6_fixed_latent.pdf\n")

## ── Combined S5 + S6 ───────────────────────────────────────────────────────────

combined <- plot_grid(
  plot_s5 + theme(legend.position = "bottom"),
  plot_s6 + theme(legend.position = "bottom"),
  ncol = 1,
  labels = c("A", "B"),
  label_size = 14,
  rel_heights = c(1, 1)
)

ggsave(file.path(FIG_DIR, "sfig_s5s6_sensitivity_combined.pdf"),
       combined, width = 12, height = 12)
cat("Saved sfig_s5s6_sensitivity_combined.pdf\n")

cat("Done: plot_sfig_sensitivity.R\n")
