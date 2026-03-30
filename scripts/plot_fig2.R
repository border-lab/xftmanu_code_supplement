## plot_fig2.R
## Reads ONLY from processed/ and generates the 3-panel Figure 2.
## Panel a: Heatmap of true PGI correlations + estimated genetic correlations
## Panel b: Gamma ratio at generation 5 for all 15 pairs + weighted average
## Panel c: Prevalence trajectories across generations

library(ggplot2)
library(cowplot)
library(ggtext)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
PROC_DIR <- file.path(BASE_DIR, "processed")
FIG_DIR  <- file.path(BASE_DIR, "figures_output")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

PAL <- RColorBrewer::brewer.pal(8, "Set1")

default_theme2 <- theme_minimal() + theme(text = element_text(size = 14))

make_label <- function(number, digits = 2) {
  format(round(number, digits), nsmall = digits)
}

## ── Load processed data ─────────────────────────────────────────────────────

pdat_heat  <- read.csv(file.path(PROC_DIR, "fig2_heatmap.csv"))
gdat_gamma <- read.csv(file.path(PROC_DIR, "fig2_gamma.csv"))
gamma_avg  <- read.csv(file.path(PROC_DIR, "fig2_gamma_avg.csv"))
prdat      <- read.csv(file.path(PROC_DIR, "fig2_prevalence.csv"))

## ── Panel a: Heatmap ────────────────────────────────────────────────────────

pdat_heat$label <- make_label(pdat_heat$value)

## Restore factor levels for Outcome
pdat_heat$Outcome <- factor(pdat_heat$Outcome,
                            levels = rev(c("hat(italic(r))[HE]",
                                           "italic(r)[score]")))
pdat_heat$Generation <- factor(
  pdat_heat$Generation,
  levels = paste(c(0, 1, 3, 5), "gen.~xAM", sep = "~")
)

nf   <- 0.065
ns   <- 2.9
badj <- 0.03
xx   <- 0.1

plot_rgheat_psych <-
  ggplot(pdat_heat,
         aes(x = pheno_2, y = pheno_1, fill = value,
             color = (value > .012 & value < .21),
             label = label)) +
  facet_grid(Outcome ~ Generation, switch = "x", labeller = label_parsed) +
  geom_tile(color = 1) +
  scale_fill_distiller(palette = "Spectral", na.value = "white") +
  geom_text() +
  scale_color_grey(start = 1, end = .1, guide = guide_none(), na.value = "red") +
  default_theme2 +
  scale_x_discrete(position = "top") +
  geom_richtext(aes(x = "ADHD", y = "ADHD", label = "6xAM", fontface = "bold"),
                nudge_x = -nf - xx, nudge_y = nf, size = ns,
                label.padding = grid::unit(rep(0, 4), "pt"),
                color = 1, label.color = NA, fill = "white", angle = 45) +
  geom_richtext(aes(x = "SCZ", y = "SCZ", label = "2xAM", fontface = "bold"),
                nudge_x = nf + xx + badj, nudge_y = -nf - badj, size = ns,
                label.padding = grid::unit(rep(0, 4), "pt"),
                color = 1, label.color = NA, fill = "white", angle = 45) +
  geom_segment(aes(x = .75, y = .75, xend = 6.25, yend = 6.25),
               color = "darkgrey", lty = 2) +
  theme(axis.title = element_blank(),
        legend.position = "top",
        legend.title = element_blank(),
        legend.key.width = unit(1.5, "cm"))

## ── Panel b: Gamma ratio ────────────────────────────────────────────────────

av_string <- "\U1D656\U1D66B\U1D65A\U1D667\U1D656\U1D65C\U1D65A"

pldat <- gdat_gamma[gdat_gamma$gen == 5, ]
pldat$traits <- as.character(pldat$traits)

pldat2 <- pldat[pldat$xAM == "2-variate", ]
pldat6 <- pldat[pldat$xAM == "6-variate", ]

## Pre-pend rows for the weighted average
avg_rows <- pldat[1:2, ]
avg_rows$xAM    <- c("2-variate", "6-variate")
avg_rows$value  <- gamma_avg$avg
avg_rows$sd     <- gamma_avg$sd
avg_rows$lower  <- gamma_avg$lower
avg_rows$upper  <- gamma_avg$upper
avg_rows$traits <- av_string

pldat <- rbind.data.frame(avg_rows, pldat)

## Order by 6-variate gamma values
tmp_order <- gdat_gamma[gdat_gamma$gen == 5 & gdat_gamma$xAM == "6-variate", ]
pldat$traits <- factor(pldat$traits,
                       levels = c(av_string,
                                  as.character(tmp_order$traits)[order(tmp_order$value)]))

plot_gamma <-
  ggplot(pldat,
         aes(x = traits, y = value, ymax = upper, ymin = lower,
             color = xAM,
             linewidth = as.factor(traits == av_string))) +
  coord_flip() +
  geom_errorbar(position = position_dodge(width = .5)) +
  geom_point(position = position_dodge(width = .5)) +
  geom_hline(yintercept = 1, lty = 2) +
  geom_hline(yintercept = 0, lty = 3) +
  scale_color_manual(values = PAL) +
  default_theme2 +
  scale_linewidth_manual(values = c(.5, 1.2)) +
  guides(linewidth = guide_none()) +
  ylab(expression(italic(r)[xAM] / italic(r)[empirical] ~~ "(5 gen. xAM)")) +
  theme(axis.title.y = element_blank(),
        legend.position = "top",
        legend.title = element_blank(),
        legend.key.width = unit(1.5, "cm"))

## ── Panel c: Prevalence ─────────────────────────────────────────────────────

plot_prev <-
  ggplot(prdat, aes(gen, value, color = xAM)) +
  stat_summary(fun = median,
               geom = "point",
               position = position_dodge(width = .2)) +
  stat_summary(fun = median,
               geom = "line",
               position = position_dodge(width = .2)) +
  stat_summary(fun.max = function(z) median(z) + sd(z),
               fun.min = function(z) median(z) - sd(z),
               fun = median,
               geom = "linerange",
               position = position_dodge(width = .2)) +
  default_theme2 +
  xlab("Generations xAM") +
  ylab("Prevalence") +
  facet_wrap(~dx, scales = "free_y") +
  scale_color_manual(values = PAL)

## ── Compose 3-panel figure ──────────────────────────────────────────────────

suppressWarnings({

  aa <- align_plots(plot_rgheat_psych + theme(legend.position = "top"),
                    plot_prev + theme(legend.title.align = .5),
                    axis = "l", align = "v")
  plot_rgheat_psych_0 <- aa[[1]]
  plot_prev_0         <- aa[[2]]

  aa <- align_plots(plot_gamma + theme(legend.position = "bottom") +
                      guides(color = guide_none()),
                    plot_rgheat_psych_0,
                    axis = "b", align = "h")
  plot_gamma_1         <- aa[[1]]
  plot_rgheat_psych_1  <- aa[[2]]

  aa <- align_plots(plot_gamma_1, plot_prev_0,
                    axis = "b", align = "h")
  plot_gamma_2 <- aa[[1]]
  plot_prev_1  <- aa[[2]]

  c1 <- plot_grid(plot_rgheat_psych_1,
                  plot_prev_1,
                  nrow = 2, ncol = 1, rel_heights = c(6, 4),
                  labels = c("a", "c"), label_size = 20)

  c2 <- plot_grid(ggdraw(), plot_gamma_2,
                  ncol = 1, rel_heights = c(.2, 9))

  fig2 <- plot_grid(c1, c2,
                    nrow = 1, rel_widths = c(8, 4),
                    labels = c("", "b"), label_size = 20)
})

## ── Save ────────────────────────────────────────────────────────────────────

ggsave(file.path(FIG_DIR, "fig2_psych.pdf"),
       fig2, width = 18, height = 12)
ggsave(file.path(FIG_DIR, "fig2_psych.png"),
       fig2, width = 18, height = 12, dpi = 300)

cat("Saved fig2_psych.pdf and fig2_psych.png to", FIG_DIR, "\n")
