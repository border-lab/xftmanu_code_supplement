#!/usr/bin/env Rscript
# Plot: Correlated Noise figure (Extended Data Figure 10)
# Mimics previous submission FigSXc: 4 scenarios × 5 CN levels
# Reads from processed/figR_cn.csv

library(ggplot2)

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
BASE_DIR <- dirname(dirname(normalizePath(sub("^--file=", "", script_arg[1]))))
OUTDIR <- Sys.getenv("XFT_PLOT_OUTPUT", file.path(BASE_DIR, "figures_output"))
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)

default_theme <- theme_bw() + theme(text = element_text(size = 14, family = "Helvetica"))

# Load processed data
all_cn <- read.csv(file.path(BASE_DIR, "processed/figR_cn.csv"))

all_cn$scenario <- factor(all_cn$scenario,
  levels = c("5xAM", "5xAM + GxE", "5xAM + VT", "5xAM + VT + GxE"))
all_cn$CN <- factor(all_cn$CN)

# Sequential palette for CN levels
cn_levels <- levels(all_cn$CN)
n_cn <- length(cn_levels)
if (n_cn == 5) {
  cn_pal   <- c("#E41A1C", "#377EB8", "#4DAF4A", "#FF7F00", "#984EA3")
  cn_lty   <- c("solid",   "dashed",  "dotted",  "dotdash", "longdash")
  cn_shape <- c(16,         17,        15,        18,        8)
} else {
  cn_pal   <- RColorBrewer::brewer.pal(max(3, n_cn), "Set1")[1:n_cn]
  cn_lty   <- rep("solid", n_cn)
  cn_shape <- 16:(16 + n_cn - 1)
}
names(cn_pal) <- cn_levels
names(cn_lty) <- cn_levels
names(cn_shape) <- cn_levels

# Nice CN labels
cn_labels <- paste0("CN = ", cn_levels)
names(cn_labels) <- cn_levels

fig <- ggplot(all_cn, aes(gen, he_rg, color = CN, linetype = CN, shape = CN)) +
  stat_summary(geom = "linerange", fun.data = mean_sdl, fun.args = list(mult = 1)) +
  stat_summary(geom = "line",  fun = mean) +
  stat_summary(geom = "point", fun = mean, size = 2) +
  facet_wrap(~ scenario, ncol = 2) +
  scale_color_manual(values = cn_pal,   labels = cn_labels, name = NULL) +
  scale_linetype_manual(values = cn_lty, labels = cn_labels, name = NULL) +
  scale_shape_manual(values = cn_shape,  labels = cn_labels, name = NULL) +
  labs(x = "Generations of xAM",
       y = expression(Estimated ~ hat(italic(r))[beta])) +
  default_theme +
  theme(legend.position = "bottom",
        legend.box = "horizontal",
        strip.text = element_text(size = 11))

ggsave(file.path(OUTDIR, "figR_cn.pdf"), fig, width = 10, height = 8,
       device = cairo_pdf)
ggsave(file.path(OUTDIR, "figR_cn.png"), fig, width = 10, height = 8, dpi = 300,
       bg = "white")
cat("Saved figR_cn.pdf and figR_cn.png\n")
