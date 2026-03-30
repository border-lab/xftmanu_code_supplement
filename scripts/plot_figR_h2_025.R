#!/usr/bin/env Rscript
# Plot: h2=0.25 sensitivity figure
# Reads ONLY from processed/, generates figure with 3 panels (b, c, d)

library(ggplot2)
library(cowplot)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
OUTDIR  <- file.path(BASE_DIR, "figures_output")

# Load processed data
h2_long <- read.csv(file.path(BASE_DIR, "processed/figR_h2_025_h2.csv"))
rg_long <- read.csv(file.path(BASE_DIR, "processed/figR_h2_025_rg.csv"))
fp_long <- read.csv(file.path(BASE_DIR, "processed/figR_h2_025_fp.csv"))

# Restore factor levels
scen_levels <- c("RM", "5xAM", "RM + VT (5%)", "5xAM + VT (5%)")
h2_long$scenario <- factor(h2_long$scenario, levels = scen_levels)
rg_long$scenario <- factor(rg_long$scenario, levels = scen_levels)
fp_long$scenario <- factor(fp_long$scenario, levels = scen_levels)

h2_long$type <- factor(h2_long$type, levels = c("Estimated", "True"))
rg_long$type <- factor(rg_long$type, levels = c("Estimated", "True"))
fp_long$gwas_type <- factor(fp_long$gwas_type,
                            levels = c("Population GWAS", "Sibling GWAS"))

# --- Style ---
pal <- c("RM" = "#E41A1C", "5xAM" = "#377EB8",
         "RM + VT (5%)" = "#4DAF4A", "5xAM + VT (5%)" = "#984EA3")

base_theme <- theme_bw() + theme(text = element_text(size = 14),
                                  legend.position = "bottom",
                                  legend.direction = "horizontal")

# ============================================================
# Panel b: h2 estimates
# ============================================================
panel_b <- ggplot(h2_long, aes(x = gen, y = h2, color = scenario, linetype = type)) +
  stat_summary(geom = "linerange", fun.data = mean_sdl, fun.args = list(mult = 1),
               position = position_dodge(width = 0.3), show.legend = FALSE) +
  stat_summary(geom = "line", fun = mean,
               position = position_dodge(width = 0.3)) +
  stat_summary(geom = "point", fun = mean, size = 2.5,
               position = position_dodge(width = 0.3)) +
  scale_color_manual(values = pal, name = "Scenario") +
  scale_linetype_manual(values = c("Estimated" = "solid", "True" = "dashed"),
                        name = "Type") +
  labs(x = "Generation", y = expression(hat(h)^2)) +
  base_theme +
  guides(color = guide_legend(nrow = 2),
         linetype = guide_legend(nrow = 1))

# ============================================================
# Panel c: rg estimates
# ============================================================
panel_c <- ggplot(rg_long, aes(x = gen, y = rg, color = scenario, linetype = type)) +
  stat_summary(geom = "linerange", fun.data = mean_sdl, fun.args = list(mult = 1),
               position = position_dodge(width = 0.3), show.legend = FALSE) +
  stat_summary(geom = "line", fun = mean,
               position = position_dodge(width = 0.3)) +
  stat_summary(geom = "point", fun = mean, size = 2.5,
               position = position_dodge(width = 0.3)) +
  scale_color_manual(values = pal, name = "Scenario") +
  scale_linetype_manual(values = c("Estimated" = "solid", "True" = "dashed"),
                        name = "Type") +
  labs(x = "Generation", y = expression(r[g])) +
  base_theme +
  guides(color = guide_legend(nrow = 2),
         linetype = guide_legend(nrow = 1))

# ============================================================
# Panel d: GWAS false positive rates
# ============================================================
panel_d <- ggplot(fp_long, aes(x = gen, y = fpr, color = scenario, linetype = gwas_type)) +
  stat_summary(geom = "linerange", fun.data = mean_sdl, fun.args = list(mult = 1),
               position = position_dodge(width = 0.3), show.legend = FALSE) +
  stat_summary(geom = "line", fun = mean,
               position = position_dodge(width = 0.3)) +
  stat_summary(geom = "point", fun = mean, size = 2.5,
               position = position_dodge(width = 0.3)) +
  geom_hline(yintercept = 0.05, linetype = "dotted", color = "grey40") +
  scale_color_manual(values = pal, name = "Scenario") +
  scale_linetype_manual(values = c("Population GWAS" = "solid",
                                   "Sibling GWAS" = "dashed"),
                        name = "GWAS Type") +
  labs(x = "Generation", y = "False Positive Rate") +
  base_theme +
  guides(color = guide_legend(nrow = 2),
         linetype = guide_legend(nrow = 1))

# ============================================================
# Combine panels
# ============================================================
combined <- plot_grid(
  panel_b + theme(legend.position = "none"),
  panel_c + theme(legend.position = "none"),
  panel_d + theme(legend.position = "none"),
  nrow = 1, labels = c("b", "c", "d"), label_size = 16
)

# Extract shared legends
legend_b <- get_legend(panel_b)
legend_d <- get_legend(panel_d)
legends <- plot_grid(legend_b, legend_d, nrow = 1, rel_widths = c(1, 1))

fig <- plot_grid(combined, legends, ncol = 1, rel_heights = c(1, 0.2))

# --- Save ---
ggsave(file.path(OUTDIR, "figR_h2_025.pdf"), fig, width = 14, height = 5.5)
ggsave(file.path(OUTDIR, "figR_h2_025.png"), fig, width = 14, height = 5.5, dpi = 300)
cat("Saved figR_h2_025.pdf and figR_h2_025.png\n")
