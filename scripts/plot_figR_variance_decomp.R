#!/usr/bin/env Rscript
# Plot: Variance Decomposition figure
# Reads ONLY from processed/, generates figure

library(ggplot2)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
OUTDIR  <- file.path(BASE_DIR, "figures_output")

# Load raw long data (needed for stat_summary with mean_sdl)
h2_long <- read.csv(file.path(BASE_DIR, "processed/figR_variance_decomp_long.csv"))

# Restore factor levels
h2_long$scenario <- factor(h2_long$scenario,
                           levels = c("5xAM", "5xAM + VT",
                                      "5xAM + GxE", "5xAM + VT + GxE"))
h2_long$quantity <- factor(h2_long$quantity,
                           levels = c("Panmictic (causal var.)",
                                      "True (Vg/Vy)",
                                      "Pop. estimated (LDSC)"))

# --- Style ---
qty_pal <- c(
  "Panmictic (causal var.)" = "#E41A1C",
  "True (Vg/Vy)"           = "#377EB8",
  "Pop. estimated (LDSC)"  = "#4DAF4A"
)

qty_lty <- c(
  "Panmictic (causal var.)" = "dotted",
  "True (Vg/Vy)"           = "dashed",
  "Pop. estimated (LDSC)"  = "solid"
)

base_theme <- theme_bw() + theme(text = element_text(size = 14),
                                  legend.position = "bottom",
                                  legend.direction = "horizontal",
                                  strip.text = element_text(size = 12))

# --- Plot ---
fig <- ggplot(h2_long, aes(x = gen, y = h2, color = quantity, linetype = quantity)) +
  stat_summary(geom = "linerange", fun.data = mean_sdl, fun.args = list(mult = 1),
               position = position_dodge(width = 0.3), show.legend = FALSE) +
  stat_summary(geom = "line", fun = mean,
               position = position_dodge(width = 0.3)) +
  stat_summary(geom = "point", fun = mean, size = 2.5,
               position = position_dodge(width = 0.3)) +
  facet_wrap(~ scenario, nrow = 1) +
  scale_color_manual(values = qty_pal, name = "") +
  scale_linetype_manual(values = qty_lty, name = "") +
  labs(x = "Generation", y = expression(h^2)) +
  base_theme +
  guides(color = guide_legend(nrow = 2),
         linetype = guide_legend(nrow = 2))

ggsave(file.path(OUTDIR, "figR_variance_decomp.pdf"), fig, width = 14, height = 5)
ggsave(file.path(OUTDIR, "figR_variance_decomp.png"), fig, width = 14, height = 5, dpi = 300)
cat("Saved figR_variance_decomp.pdf and figR_variance_decomp.png\n")
