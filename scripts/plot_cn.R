#!/usr/bin/env Rscript
# Plot: Correlated Noise figure
# Reads ONLY from processed/, generates figure

library(ggplot2)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
OUTDIR  <- file.path(BASE_DIR, "figures_output")

# Load processed data
all_cn <- read.csv(file.path(BASE_DIR, "processed/figR_cn.csv"))

# Restore factor levels
all_cn$scenario <- factor(all_cn$scenario, levels = c("5xAM", "5xAM + VT (5%)"))
all_cn$CN <- factor(all_cn$CN)

# --- Style ---
base_theme <- theme_bw() + theme(text = element_text(size = 14),
                                  legend.position = "bottom",
                                  legend.direction = "horizontal")

# Sequential blue-to-red palette for CN levels
cn_levels <- levels(all_cn$CN)
n_cn <- length(cn_levels)
if (n_cn == 5) {
  cn_pal <- c("#2166AC", "#67A9CF", "#D1E5F0", "#F4A582", "#B2182B")
} else if (n_cn == 4) {
  cn_pal <- c("#2166AC", "#67A9CF", "#F4A582", "#B2182B")
} else {
  cn_pal <- RColorBrewer::brewer.pal(max(3, n_cn), "RdBu")
  cn_pal <- rev(cn_pal[1:n_cn])
}
names(cn_pal) <- cn_levels

# --- Plot ---
fig <- ggplot(all_cn, aes(x = gen, y = he_rg, color = CN)) +
  stat_summary(geom = "linerange", fun.data = mean_sdl, fun.args = list(mult = 1),
               position = position_dodge(width = 0.3), show.legend = FALSE) +
  stat_summary(geom = "line", fun = mean,
               position = position_dodge(width = 0.3)) +
  stat_summary(geom = "point", fun = mean, size = 2.5,
               position = position_dodge(width = 0.3)) +
  facet_wrap(~ scenario) +
  scale_color_manual(values = cn_pal, name = "Correlated Noise (CN)") +
  labs(x = "Generation", y = expression(hat(r)[g])) +
  base_theme

ggsave(file.path(OUTDIR, "figR_cn.pdf"), fig, width = 10, height = 5)
ggsave(file.path(OUTDIR, "figR_cn.png"), fig, width = 10, height = 5, dpi = 300)
cat("Saved figR_cn.pdf and figR_cn.png\n")
