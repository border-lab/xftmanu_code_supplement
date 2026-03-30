## plot_sfig_analytic.R
## Generates supplementary figures S9, S10, S11, S12 from processed data ONLY.
##   S9:  Incremental on-target vs off-target associations by sample size
##   S10: Type-I error inflation vs sample size
##   S11: Cumulative on-target vs off-target associations by sample size
##   S12: False positive rate at off-target loci by effect magnitude

library(ggplot2)
library(cowplot)
library(scales)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
PROC_DIR <- file.path(BASE_DIR, "processed")
FIG_DIR  <- file.path(BASE_DIR, "figures_output")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

default_theme <- theme_bw() + theme(text = element_text(size = 14))
PAL <- RColorBrewer::brewer.pal(8, "Set1")

## ── S9: Incremental on-target vs off-target associations ──────────────────────
## Stacked bar chart showing non-cumulative expected new associations
## per sample-size increment after 5 generations of xAM.
## Bivariate (K=2) vs 5-variate (K=5), M=2000/4000/8000 causal variants.

s9 <- read.csv(file.path(PROC_DIR, "sfig_s9_incremental_hits.csv"))

## Restore factor ordering for delta_N (sample size bins)
s9$delta_N <- factor(s9$delta_N,
                     levels = unique(s9$delta_N[order(s9$N)]))
s9$kpheno_label <- factor(paste("K =", s9$kpheno))
s9$m_causal_label <- factor(paste("M =", s9$m_causal),
                            levels = paste("M =", sort(unique(s9$m_causal))))

plot_s9 <- ggplot(s9, aes(delta_N, value, fill = type)) +
  geom_bar(stat = "identity") +
  facet_grid(kpheno_label ~ m_causal_label, labeller = label_value) +
  scale_fill_manual(values = PAL[c(1, 2)]) +
  default_theme +
  theme(
    legend.title = element_blank(),
    legend.position = "top",
    legend.key.width = unit(1.5, "cm"),
    axis.text.x = element_text(angle = -45, hjust = 0),
    strip.text = element_text(size = 12)
  ) +
  xlab("Incremental GWAS sample size") +
  ylab("Incremental new GWAS associations")

ggsave(file.path(FIG_DIR, "sfig_s9_incremental_hits.pdf"),
       plot_s9, width = 12, height = 8)
cat("Saved sfig_s9_incremental_hits.pdf\n")

## ── S10: Type-I error inflation vs sample size ───────────────────────────────
## Off-target false positive rate (relative to nominal alpha=5e-8) as a
## function of GWAS sample size.  Faceted by M causal (rows) and generation
## (columns), colored by number of phenotypes under xAM (K=2 vs K=5).

s10 <- read.csv(file.path(PROC_DIR, "sfig_s10_t1e_inflation.csv"))

s10$kpheno_label <- factor(paste0("K = ", s10$kpheno))
s10$m_causal_label <- factor(paste("M =", s10$m_causal),
                             levels = paste("M =", sort(unique(s10$m_causal))))
s10$gen_label <- factor(paste("Gen =", s10$gen),
                        levels = paste("Gen =", sort(unique(s10$gen))))

plot_s10 <- ggplot(s10, aes(N, relative_t1e,
                             color = kpheno_label,
                             linetype = kpheno_label)) +
  geom_hline(yintercept = 1, col = "orange", lty = 4, lwd = 1) +
  geom_point() +
  geom_path() +
  facet_grid(m_causal_label ~ gen_label, scales = "fixed") +
  scale_x_log10(breaks = unique(s10$N), labels = comma) +
  scale_y_log10(breaks = 10^(0:7)) +
  scale_color_brewer(palette = "Set1") +
  default_theme +
  theme(
    axis.text.x = element_text(angle = -45, hjust = 0),
    legend.position = "top",
    legend.key.width = unit(1, "cm"),
    strip.text = element_text(size = 11)
  ) +
  labs(x = "GWAS sample size",
       y = expression(frac("Off-target false positive rate",
                           "Nominal " * alpha * " = 5e-8")),
       color = "Number of phenotypes\nunder xAM:",
       linetype = "Number of phenotypes\nunder xAM:")

ggsave(file.path(FIG_DIR, "sfig_s10_t1e_inflation.pdf"),
       plot_s10, width = 14, height = 12)
cat("Saved sfig_s10_t1e_inflation.pdf\n")

## ── S11: Cumulative on-target vs off-target associations ──────────────────────
## Cumulative expected GWAS associations as a function of sample size,
## decomposed into on-target, off-target, and conventional false positives.
## Log-log scale; faceted by K (rows) and M (columns).

s11 <- read.csv(file.path(PROC_DIR, "sfig_s11_cumulative_hits.csv"))

s11$kpheno_label <- factor(paste("K =", s11$kpheno))
s11$m_causal_label <- factor(paste("M =", s11$m_causal),
                             levels = paste("M =", sort(unique(s11$m_causal))))

plot_s11 <- ggplot(s11, aes(N, value, color = type,
                             shape = type, linetype = type)) +
  geom_path(position = position_dodge(width = 0.05)) +
  geom_point(position = position_dodge(width = 0.05)) +
  facet_grid(kpheno_label ~ m_causal_label, labeller = label_value) +
  scale_color_manual(values = PAL[c(2, 1, 3)]) +
  scale_x_log10(breaks = unique(s11$N), labels = comma) +
  scale_y_log10(breaks = 10^(0:5), labels = comma) +
  default_theme +
  theme(
    legend.title = element_blank(),
    legend.position = "top",
    legend.key.width = unit(1.5, "cm"),
    axis.text.x = element_text(angle = -45, hjust = 0),
    strip.text = element_text(size = 12)
  ) +
  xlab("GWAS sample size") +
  ylab("Cumulative GWAS associations")

ggsave(file.path(FIG_DIR, "sfig_s11_cumulative_hits.pdf"),
       plot_s11, width = 12, height = 8)
cat("Saved sfig_s11_cumulative_hits.pdf\n")

## ── S12: False positive rate at off-target loci by effect magnitude ──────────
## Probability of detecting an off-target locus as a function of effect
## size quantile (from smallest to largest absolute effect) and GWAS sample
## size.  Each line represents a different N; y-axis is the per-locus false
## positive rate on log scale, with a secondary axis showing the ratio to
## nominal alpha=5e-8.  M=4000, K=5, gen=5.

s12 <- read.csv(file.path(PROC_DIR, "sfig_s12_persnp_t1e.csv"))

## Restore factor order for N_label (decreasing N)
n_levels <- unique(s12$N_label[order(s12$N, decreasing = TRUE)])
s12$N_label <- factor(s12$N_label, levels = n_levels)

plot_s12 <- ggplot(s12, aes(beta_quantile, T1E,
                             color = N_label, linetype = N_label)) +
  geom_path(lwd = 0.8) +
  geom_hline(yintercept = 5e-8, lty = 3, lwd = 0.8) +
  scale_y_log10(
    breaks = 5 * 10^-(8:1),
    name = expression("Off-target false positive rate at " *
                        alpha * " = 5e-8"),
    sec.axis = dup_axis(
      trans = ~ . / 5e-8,
      name = expression(
        frac("Off-target false positive rate",
             "Nominal " * alpha * " = 5e-8")
      ),
      breaks = 5 * 10^-(8:1) / 5e-8
    )
  ) +
  xlab("Effect size quantile") +
  guides(color = guide_legend(title = "GWAS\nsample size"),
         linetype = guide_legend(title = "GWAS\nsample size")) +
  default_theme +
  theme(
    legend.position = c(0.16, 0.67),
    legend.title = element_text(hjust = 0.5),
    legend.key.width = unit(1.5, "cm"),
    axis.title.y.right = element_text(margin = margin(l = 10))
  )

ggsave(file.path(FIG_DIR, "sfig_s12_persnp_t1e.pdf"),
       plot_s12, width = 9, height = 5)
cat("Saved sfig_s12_persnp_t1e.pdf\n")

## ── Combined figure ──────────────────────────────────────────────────────────
## All four panels in a 2x2 layout.

combined <- plot_grid(
  plot_s9  + theme(legend.position = "top"),
  plot_s11 + theme(legend.position = "top"),
  plot_s10 + theme(legend.position = "top"),
  plot_s12 + theme(legend.position = c(0.16, 0.67)),
  ncol = 2,
  labels = c("A (S9)", "B (S11)", "C (S10)", "D (S12)"),
  label_size = 14,
  rel_widths = c(1, 1),
  rel_heights = c(1, 1.2)
)

ggsave(file.path(FIG_DIR, "sfig_s9_s12_analytic_combined.pdf"),
       combined, width = 22, height = 20)
cat("Saved sfig_s9_s12_analytic_combined.pdf\n")

cat("Done: plot_sfig_analytic.R\n")
