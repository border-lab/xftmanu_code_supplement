#!/usr/bin/env Rscript
## plot_fig_continuum_r5.R  (ROUND 5 — VT-structure continuum sensitivity figure, C1/C4)
## Same aesthetic as the revised Figure 3 (theme_minimal, 10–90% ribbons, median lines).
## Rows = mating (RM, 5xAM; free y per row + spacing → distinct axes); cols = tier (Direct, Apparent, LDSC);
## colour + line type = VT structure (Okabe–Ito; same-trait → constant cross-trait).
## Reads round5/fig3_r5/ ; writes round5/fig3_r5/fig_continuum_r5_*.{pdf,png}.

library(ggplot2)
library(cowplot)

PROC <- "/home/rsb/Dropbox/ftsim/round4/round5/fig3_r5"
default_theme2 <- theme_minimal() + theme(text = element_text(size = 13, family = "Helvetica"))

## distinct, colour-blind-safe (Okabe–Ito): grey reference for no-VT, then cool→warm
## (blue → green → orange → vermillion) as VT goes same-trait → constant cross-trait
STR_LEVELS <- c("no VT","uVT (same-trait)","weak (½ total)","weak (½ per-trait)","strong (constant)")
STR_PAL <- c("no VT" = "grey60", "uVT (same-trait)" = "#0072B2", "weak (½ total)" = "#009E73",
             "weak (½ per-trait)" = "#E69F00", "strong (constant)" = "#D55E00")
STR_LTY <- c("no VT" = "dotted", "uVT (same-trait)" = "solid", "weak (½ total)" = "longdash",
             "weak (½ per-trait)" = "dotdash", "strong (constant)" = "dashed")

h2l <- read.csv(file.path(PROC, "fig_continuum_r5_h2_long.csv"))
rgl <- read.csv(file.path(PROC, "fig_continuum_r5_rg_long.csv"))
h2l$structure <- factor(h2l$structure, levels = STR_LEVELS)
rgl$structure <- factor(rgl$structure, levels = STR_LEVELS)
h2l$mating <- factor(h2l$mating, levels = c("RM","5xAM"))
rgl$mating <- factor(rgl$mating, levels = c("RM","5xAM"))
h2l$quantity <- factor(h2l$quantity, levels = c("Direct h²","Apparent h²","LDSC h²"),
  labels = c("Direct~italic(h)^2","Apparent~italic(h)^2","LDSC~hat(italic(h))^2"))
rgl$quantity <- factor(rgl$quantity, levels = c("Direct r_g","Apparent r_g","LDSC r_g"),
  labels = c("Direct~italic(r)[italic(g)]","Apparent~italic(r)[italic(g)]","LDSC~hat(italic(r))[italic(g)]"))

panel <- function(df, yvar, ycol, hline) {
  ggplot(df, aes(x = gen, y = .data[[ycol]], color = structure, linetype = structure)) +
    default_theme2 +
    geom_hline(yintercept = hline, color = "grey80", lty = 1) +
    stat_summary(geom = "linerange", position = position_dodge(width = 0.4), linetype = 1,
                 fun.max = function(x) quantile(x, 0.9), fun.min = function(x) quantile(x, 0.1),
                 fun = function(x) quantile(x, 0.5), alpha = 0.6) +
    stat_summary(geom = "path", fun = median, linewidth = 0.9, position = position_dodge(width = 0.4)) +
    stat_summary(geom = "point", fun = median, size = 1.6, position = position_dodge(width = 0.4)) +
    facet_grid(mating ~ quantity, scales = "free_y",
               labeller = labeller(quantity = label_parsed)) +
    scale_color_manual(values = STR_PAL, name = "VT structure") +
    scale_linetype_manual(values = STR_LTY, name = "VT structure") +
    ylab(yvar) + xlab("Generations of xAM") +
    theme(legend.position = "bottom", panel.spacing.y = unit(1.1, "lines"))
}

make_fig <- function(h2v, tag, title_h2, title_rg) {
  ph <- panel(h2l[h2l$args_h2 == h2v, ], expression(italic(h)^2), "h2", h2v) +
        guides(color = guide_none(), linetype = guide_none()) +
        theme(axis.title.x = element_blank()) + ggtitle(title_h2)
  pr <- panel(rgl[rgl$args_h2 == h2v, ], expression(italic(r)[italic(g)]), "rg", 0) + ggtitle(title_rg)
  fig <- plot_grid(ph, pr, ncol = 1, rel_heights = c(1, 1.15), labels = c("a", "b"))
  ggsave(file.path(PROC, paste0("fig_continuum_r5_", tag, ".pdf")), fig,
         width = 11, height = 9, device = cairo_pdf)
  ggsave(file.path(PROC, paste0("fig_continuum_r5_", tag, ".png")), fig,
         width = 11, height = 9, dpi = 300, bg = "white")
  cat("Saved fig_continuum_r5_", tag, ".pdf/png\n", sep = "")
}

make_fig(0.5,  "h2_0.5",  "Heritability decomposition across VT structure (h² = 0.5)",
         "Genetic-correlation decomposition across VT structure (h² = 0.5)")
make_fig(0.25, "h2_0.25", "Heritability decomposition across VT structure (h² = 0.25)",
         "Genetic-correlation decomposition across VT structure (h² = 0.25)")
cat("=== plot_fig_continuum_r5.R complete ===\n")
