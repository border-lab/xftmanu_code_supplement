#!/usr/bin/env Rscript
# plot_fig4.R -- Generate Figure 4 (education/height/wealth simulation)
# Uses MANUSCRIPT REFERENCE CSVs for plotting (not the edu_sims data),
# since the edu_sims data is from a different simulation than the manuscript.
#
# Panels: (a) schematic, (b) h2 + PGI R2, (c) genetic correlations, (d) GWAS beta correlations
#
# Data sources (all from notebook cached output, hardcoded in process_fig4.R):
#   processed/fig4_manuscript_reference.csv    -- Panel B h2/R2 at gen 0,5
#   processed/fig4_manuscript_h2_true_height.csv -- h2_true_height all gens
#   processed/fig4_manuscript_rg.csv           -- Panel C rg at gen 0,5
#   processed/fig4_manuscript_rbeta.csv        -- Panel D rbeta all gens

library(ggplot2)
library(cowplot)
library(reshape2)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"

default_theme <- theme_minimal() + theme(text = element_text(size = 14))
PAL <- RColorBrewer::brewer.pal(8, "Set1")

# ---------------------------------------------------------------------------
# Load manuscript reference data
# ---------------------------------------------------------------------------
cat("Loading manuscript reference data...\n")
ref_h2 <- read.csv(file.path(BASE_DIR, "processed/fig4_manuscript_reference.csv"))
ref_h2_traj <- read.csv(file.path(BASE_DIR, "processed/fig4_manuscript_h2_true_height.csv"))
ref_rg <- read.csv(file.path(BASE_DIR, "processed/fig4_manuscript_rg.csv"))
ref_rbeta <- read.csv(file.path(BASE_DIR, "processed/fig4_manuscript_rbeta.csv"))

# ---------------------------------------------------------------------------
# Panel A: Schematic (causal diagram)
# ---------------------------------------------------------------------------
cat("Loading schematic...\n")
schematic_path <- file.path(BASE_DIR, "data/cdiags/edu_cdiag_redux.png")
plot_diag_edu <- ggdraw() +
  draw_image(schematic_path, interpolate = TRUE)

# ---------------------------------------------------------------------------
# Panel B: h2 estimated vs true + PGI R2 (3 facets: height, edu, wealth)
# ---------------------------------------------------------------------------
cat("Building Panel B...\n")

# Build long-format data for panel B from the reference tables.
# ref_h2 has gen 0,5 with columns: gen, Outcome, h2_HE, h2_true, r2_edu, r2_height
# ref_h2_traj has gen 0-5 with h2_true_height

# Melt ref_h2 into long format: gen, Outcome, Quantity, value
panelb_rows <- list()

# h2_HE for all three outcomes (gen 0,5)
for (i in seq_len(nrow(ref_h2))) {
  panelb_rows[[length(panelb_rows) + 1]] <- data.frame(
    gen = ref_h2$gen[i], Outcome = ref_h2$Outcome[i],
    Quantity = "h2_HE", value = ref_h2$h2_HE[i],
    stringsAsFactors = FALSE)
}

# h2_true for all three outcomes (gen 0,5)
for (i in seq_len(nrow(ref_h2))) {
  panelb_rows[[length(panelb_rows) + 1]] <- data.frame(
    gen = ref_h2$gen[i], Outcome = ref_h2$Outcome[i],
    Quantity = "h2_true", value = ref_h2$h2_true[i],
    stringsAsFactors = FALSE)
}

# r2_edu (PGI from edu GWAS) for edu and wealth outcomes (gen 0,5)
for (i in seq_len(nrow(ref_h2))) {
  if (!is.na(ref_h2$r2_edu[i])) {
    panelb_rows[[length(panelb_rows) + 1]] <- data.frame(
      gen = ref_h2$gen[i], Outcome = ref_h2$Outcome[i],
      Quantity = "r2_edu", value = ref_h2$r2_edu[i],
      stringsAsFactors = FALSE)
  }
}

# r2_height (PGI from height GWAS) for all outcomes (gen 0,5)
for (i in seq_len(nrow(ref_h2))) {
  if (!is.na(ref_h2$r2_height[i])) {
    panelb_rows[[length(panelb_rows) + 1]] <- data.frame(
      gen = ref_h2$gen[i], Outcome = ref_h2$Outcome[i],
      Quantity = "r2_height", value = ref_h2$r2_height[i],
      stringsAsFactors = FALSE)
  }
}

# Override h2_true for height with the full trajectory (gen 0-5)
# First remove the gen 0,5 h2_true height entries we already added
panelb_df <- do.call(rbind, panelb_rows)
panelb_df <- panelb_df[!(panelb_df$Outcome == "height" &
                          panelb_df$Quantity == "h2_true"), ]

# Add full trajectory
for (g in 0:5) {
  panelb_df <- rbind(panelb_df, data.frame(
    gen = g, Outcome = "height", Quantity = "h2_true",
    value = ref_h2_traj$h2_true_height[ref_h2_traj$gen == g],
    stringsAsFactors = FALSE))
}

# Set factor levels
panelb_df$Outcome <- factor(panelb_df$Outcome,
                            levels = c("height", "edu", "wealth"))
panelb_df$Quantity <- factor(panelb_df$Quantity,
                             levels = c("h2_HE", "h2_true", "r2_edu", "r2_height"))

plot_uni <- ggplot(panelb_df,
                   aes(x = gen, y = value, color = Quantity, linetype = Quantity)) +
  default_theme +
  geom_point(size = 1.5) +
  geom_line(linewidth = 0.75) +
  scale_linetype_manual(
    values = c(1, 2, 3, 4),
    labels = c(expression(italic(hat(h))[HE]^2),
               expression(italic(h)[true]^2),
               expression(italic(R)[italic(G)[edu]]^2),
               expression(italic(R)[italic(G)[height]]^2))
  ) +
  scale_color_manual(
    values = PAL[1:4],
    labels = c(expression(italic(hat(h))[HE]^2),
               expression(italic(h)[true]^2),
               expression(italic(R)[italic(G)[edu]]^2),
               expression(italic(R)[italic(G)[height]]^2))
  ) +
  ylab("") +
  xlab("Generations of xAM") +
  facet_grid(Outcome ~ ., scales = "free_y") +
  theme(legend.key.width = unit(1.5, "cm"),
        legend.position = "top",
        legend.title = element_blank())

# ---------------------------------------------------------------------------
# Panel C: True vs estimated genetic correlations
# ---------------------------------------------------------------------------
cat("Building Panel C...\n")

# ref_rg: gen, traits, corr (true), rgHE (estimated)
# Melt into long format: gen, traits, type, value
panelc_corr <- data.frame(
  gen = ref_rg$gen, traits = ref_rg$traits,
  type = "corr", value = ref_rg$corr,
  stringsAsFactors = FALSE)
panelc_rghe <- data.frame(
  gen = ref_rg$gen, traits = ref_rg$traits,
  type = "rgHE", value = ref_rg$rgHE,
  stringsAsFactors = FALSE)
panelc_df <- rbind(panelc_corr, panelc_rghe)
panelc_df <- panelc_df[!is.na(panelc_df$value), ]
panelc_df$traits <- factor(panelc_df$traits,
                           levels = c("edu / height", "edu / wealth", "height / wealth"))

# Rename 'type' -> 'outcome' to match notebook convention
panelc_df$outcome <- panelc_df$type

plot_biv <- ggplot(panelc_df,
                   aes(x = gen, y = value, color = traits, linetype = outcome)) +
  default_theme +
  geom_point(size = 1.5) +
  geom_line(linewidth = 0.75) +
  geom_hline(yintercept = 0, linetype = 3, color = PAL[5]) +
  ylab(expression(italic(r)[g])) +
  xlab("Generations of xAM") +
  scale_color_manual(values = PAL[1:3]) +
  scale_linetype_manual(values = c(1, 2),
                        labels = c(expression(italic(r)[score]),
                                   expression(hat(italic(r))[beta]))) +
  guides(color = guide_legend(title = "", order = 2),
         linetype = guide_legend(title = "", order = 1,
                                 override.aes = list(color = 1))) +
  theme(legend.key.width = unit(1.5, "cm"),
        legend.title = element_blank())

# ---------------------------------------------------------------------------
# Panel D: GWAS beta-hat correlations
# ---------------------------------------------------------------------------
cat("Building Panel D...\n")

# ref_rbeta: gen, edu_height, edu_wealth, height_wealth
# Melt to long format
paneld_df <- melt(ref_rbeta, id.vars = "gen",
                  variable.name = "traits_raw", value.name = "value")
paneld_df$traits <- NA
paneld_df$traits[paneld_df$traits_raw == "edu_height"] <- "edu / height"
paneld_df$traits[paneld_df$traits_raw == "edu_wealth"] <- "edu / wealth"
paneld_df$traits[paneld_df$traits_raw == "height_wealth"] <- "height / wealth"
paneld_df$traits <- factor(paneld_df$traits,
                           levels = c("edu / height", "edu / wealth", "height / wealth"))

plot_r2 <- ggplot(paneld_df,
                  aes(x = gen, y = value, color = traits)) +
  default_theme +
  geom_point(size = 1.5) +
  geom_line(linewidth = 0.75) +
  geom_hline(yintercept = 0, linetype = 3, color = PAL[5]) +
  ylab(expression(italic(r)[hat(beta)])) +
  xlab("Generations of xAM") +
  scale_color_manual(values = PAL[1:3]) +
  guides(color = guide_legend(title = "")) +
  theme(legend.key.width = unit(1.5, "cm"),
        legend.position = c(0.8, 0.7),
        legend.title = element_blank())

# ---------------------------------------------------------------------------
# Assemble 4-panel figure (matching notebook cell 50 layout)
# ---------------------------------------------------------------------------
cat("Assembling figure...\n")
LS <- 20

suppressWarnings({
  aa <- align_plots(
    axis = "lr", align = "h",
    plot_uni + theme(legend.position = "top",
                     legend.key.width = unit(1, "cm")),
    plot_biv +
      theme(legend.position = c(0.663, 0.285),
            legend.box = "horizontal",
            legend.title = element_blank(),
            legend.key.width = unit(1, "cm")),
    plot_r2 +
      theme(legend.key.width = unit(1, "cm"))
  )

  c1 <- plot_grid(aa[[1]], ncol = 1, labels = c("b"), label_size = LS)
  c2 <- plot_grid(aa[[2]], aa[[3]], ncol = 1, labels = c("c", "d"), label_size = LS)
  r1 <- plot_grid(c1, c2, ncol = 2, rel_widths = c(3, 3), label_size = LS)
  fig4 <- plot_grid(
    plot_diag_edu, NULL, r1,
    nrow = 3, rel_heights = c(3.5, 0.15, 5),
    labels = c("a", "", ""), label_size = LS, hjust = 0
  )
})

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
outdir <- file.path(BASE_DIR, "figures_output")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

outfile_pdf <- file.path(outdir, "fig4_edu_sims.pdf")
outfile_png <- file.path(outdir, "fig4_edu_sims.png")

ggsave(outfile_pdf, fig4, width = 9, height = 10, dpi = 300)
cat("Saved:", outfile_pdf, "\n")

ggsave(outfile_png, fig4, width = 9, height = 10, dpi = 300)
cat("Saved:", outfile_png, "\n")

cat("Done.\n")
cat("\nNOTE: This figure uses manuscript reference data (from notebook cached output),\n")
cat("not the edu_sims data (which is from a different simulation).\n")
