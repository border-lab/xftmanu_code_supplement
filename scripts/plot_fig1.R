## plot_fig1.R
## Reads ONLY from processed/ directory, generates 6-panel Figure 1
## Panels: a (UKB CCA), b (NHIRD CCA), c (h2), d (rg), e (GWAS FP), f (newhits)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
PROCESSED_DIR <- file.path(BASE_DIR, "processed")
FIGURES_DIR <- file.path(BASE_DIR, "figures_output")
dir.create(FIGURES_DIR, showWarnings = FALSE, recursive = TRUE)

library(ggplot2)
library(reshape2)
library(cowplot)
library(RColorBrewer)

default_theme <- theme_bw() + theme(text = element_text(size = 14))

SS <- 12
LSIZE <- 20
WOFFSET <- 0.15
PAL <- RColorBrewer::brewer.pal(3, "Set1")

cat("=== Generating Figure 1 ===\n\n")

## ---------------------------------------------------------------
## Panel A: UKB CCA scree plot
## ---------------------------------------------------------------
cat("--- Panel A: UKB CCA scree ---\n")
ukb_cca <- read.csv(file.path(PROCESSED_DIR, "fig1_ukb_cca.csv"))
KK_ukb <- max(ukb_cca$cv)

plot_cca_var <- ggplot(ukb_cca, aes(cv, cumulative_redundancy)) +
  scale_y_continuous(position = "left", limits = 0:1) +
  scale_x_continuous(
    breaks = 0:KK_ukb,
    labels = gsub(" ", "", ukb_cca$CanonicalVariate),
    position = "bottom"
  ) +
  geom_hline(col = "purple", lty = 3, yintercept = 0.9) +
  geom_hline(col = "darkorange", lty = 3, yintercept = 0.95) +
  xlab("Canonical Variate") +
  ylab("Relative Canonical\nRedundancy") +
  geom_point() + geom_line() +
  theme_bw() +
  theme(text = element_text(size = 14), legend.position = "bottom")

## ---------------------------------------------------------------
## Panel B: Taiwan NHIRD CCA scree plot
## ---------------------------------------------------------------
cat("--- Panel B: NHIRD CCA scree ---\n")
nhird_cca <- read.csv(file.path(PROCESSED_DIR, "fig1_nhird_cca.csv"))
## Truncate to first 9 rows (CV0 through CV8) as in notebook
nhird_cca <- nhird_cca[1:9, ]
KK_nhird <- max(nhird_cca$cv)

plot_tcca_var <- ggplot(nhird_cca, aes(cv, cumulative_redundancy)) +
  scale_y_continuous(position = "left", limits = 0:1) +
  scale_x_continuous(
    breaks = 0:KK_nhird,
    labels = nhird_cca$CanonicalVariate,
    position = "bottom"
  ) +
  geom_hline(col = "purple", lty = 3, yintercept = 0.9) +
  geom_hline(col = "darkorange", lty = 3, yintercept = 0.95) +
  xlab("Canonical Variate") +
  ylab("") +
  geom_point() + geom_line() +
  theme_bw() +
  theme(
    text = element_text(size = 14),
    legend.position = "bottom",
    plot.margin = unit(c(0, 0, 0, 0), "cm")
  )

## ---------------------------------------------------------------
## Panel C: h2 under 2xAM vs 5xAM
## ---------------------------------------------------------------
cat("--- Panel C: h2 plot ---\n")
udat <- read.csv(file.path(PROCESSED_DIR, "fig1_h2_raw.csv"))

## Recode Quantity for legend labels (match notebook logic)
udat$Quantity <- factor(udat$variable, levels = c("he_h2", "h2_true"))

## Build the extra baseline segment data
## Add placeholder rows for the h0^2 baseline segment
utmp <- udat
utmp$Quantity <- factor(utmp$variable, levels = c("he_h2", "h2_true", "z"))
utmp <- rbind.data.frame(utmp[!duplicated(udat$gen), ], udat)
utmp$Quantity[1:6] <- "z"
utmp$value[1:6] <- 0.5

plot_h2_2x5 <- ggplot(utmp, aes(gen, value, color = Quantity, lty = xAM, shape = xAM)) +
  default_theme +
  geom_segment(y = 0.5, x = 0, yend = 0.5, xend = 5, col = PAL[3], lty = 1) +
  stat_summary(
    geom = "linerange",
    position = position_dodge(width = 0.1),
    fun.data = function(x) mean_se(x, mult = 1.96)
  ) +
  stat_summary(
    data = utmp,
    geom = "point",
    position = position_dodge(width = 0.1),
    fun.data = function(x) mean_se(x, mult = 1.96)
  ) +
  stat_summary(geom = "line", aes(gen, color = Quantity), fun = mean) +
  ylab(expression(italic(h)^2)) +
  xlab("Generations of xAM") +
  scale_color_manual(
    values = PAL,
    labels = c(
      expression(hat(italic(h))[HE]^2),
      expression(italic(h)[italic(t)]^2),
      expression(italic(h)[0]^2)
    )
  )

## ---------------------------------------------------------------
## Panel D: rg under 2xAM vs 5xAM
## ---------------------------------------------------------------
cat("--- Panel D: rg plot ---\n")
bdat <- read.csv(file.path(PROCESSED_DIR, "fig1_rg_raw.csv"))

bdat$Quantity <- factor(bdat$variable, levels = c("he_rg", "rg_true", "z"))

btmp <- rbind.data.frame(bdat[!duplicated(bdat$gen), ], bdat)
btmp$Quantity[1:6] <- "z"
btmp$value[1:6] <- 0

plot_rg_2x5 <- ggplot(btmp, aes(gen, value, color = Quantity, lty = xAM, shape = xAM)) +
  default_theme +
  geom_segment(y = 0, x = 0, yend = 0, xend = 5, col = PAL[3], lty = 1) +
  stat_summary(
    geom = "linerange",
    position = position_dodge(width = 0.1),
    fun.data = function(x) mean_se(x, mult = 1.96)
  ) +
  stat_summary(
    data = btmp,
    geom = "point",
    position = position_dodge(width = 0.1),
    fun.data = function(x) mean_se(x, mult = 1.96)
  ) +
  stat_summary(geom = "line", aes(gen, color = Quantity), fun = mean) +
  ylab(expression(italic(r[g]))) +
  xlab("Generations of xAM") +
  scale_color_manual(
    values = PAL,
    labels = c(
      expression(hat(italic(r))[HE]),
      expression(italic(r)[score]),
      expression(italic(r)["\u03B2"])
    )
  ) +
  guides(
    color = guide_legend(order = 2),
    linetype = guide_legend(order = 1),
    shape = guide_legend(order = 1)
  )

## ---------------------------------------------------------------
## Panel E: GWAS type-I error inflation
## ---------------------------------------------------------------
cat("--- Panel E: GWAS FP inflation ---\n")
gwas_raw <- read.csv(file.path(PROCESSED_DIR, "fig1_gwas_raw.csv"))

gwas_pop <- gwas_raw[gwas_raw$GWAS == "Population", ]

plot_gwas2x5_alt_v <- ggplot(gwas_pop, aes(gen, relative_T1R, color = xAM, shape = xAM)) +
  default_theme +
  stat_summary(geom = "path", position = position_dodge(width = 0.1), fun = median) +
  stat_summary(geom = "point", position = position_dodge(width = 0.1), fun = median) +
  facet_grid(power ~ ., as.table = TRUE) +
  geom_hline(color = "grey", lty = 3, yintercept = 1) +
  scale_color_manual(values = PAL) +
  scale_y_continuous(breaks = seq(1, 2, 0.25)) +
  geom_hline(color = "grey", lty = 3, yintercept = 1) +
  ylab(expression(over('Empirical Type-I Error Rate', 'Theoretical Type-I Error Rate'))) +
  xlab("Generations of xAM") +
  theme(
    legend.position = c(0.5, 0.9),
    legend.direction = "horizontal",
    legend.key.width = unit(1, "cm")
  )

## ---------------------------------------------------------------
## Panel F: On-target vs off-target new GWAS hits
## ---------------------------------------------------------------
cat("--- Panel F: New hits (on-target vs off-target) ---\n")
nhdat <- read.csv(file.path(PROCESSED_DIR, "fig1_newhits.csv"))

## Create facet columns matching notebook
M_col <- "\U1D440 causal / phenotype"
nhdat[["Number of phenotypes under xAM"]] <- as.factor(nhdat$kpheno)
nhdat[[M_col]] <- nhdat$m_causal

## Remap type labels to match notebook (strip " associations" suffix)
nhdat$type[nhdat$type == "Off target associations"] <- "Off target"
nhdat$type[nhdat$type == "On target associations"] <- "On target"
nhdat$type <- factor(nhdat$type, levels = sort(unique(nhdat$type)))

## Filter to m_causal==4000 only, keep ALL kpheno values
nhdat_sub <- nhdat[nhdat$m_causal == 4000, ]

## Build facet formula using the Unicode column name
facet_f <- as.formula(paste0("`Number of phenotypes under xAM` ~ `", M_col, "`"))

nhplot <- ggplot(nhdat_sub, aes(delta_N, value, fill = type, shape = type, linetype = type)) +
  geom_bar(stat = "identity") +
  theme_bw() +
  facet_grid(facet_f, label = label_both) +
  scale_fill_manual(values = RColorBrewer::brewer.pal(name = "Set1", 3)[c(1, 2)]) +
  theme(
    text = element_text(size = 12),
    plot.title = element_text(hjust = 0.5, vjust = -8),
    legend.title = element_blank(),
    legend.position = "top",
    legend.key.width = unit(1.5, "cm"),
    axis.text.x = element_text(angle = -45, hjust = 0),
    axis.title.y.right = element_text(margin = margin(l = 10))
  ) +
  xlab("Incremental GWAS sample size") +
  ylab("Incremental new GWAS hits")

## ---------------------------------------------------------------
## Composite figure assembly (matching resub_mFigHDxAM_r01 layout)
## ---------------------------------------------------------------
cat("--- Assembling composite figure ---\n")

ltheme <- theme(
  legend.position = c(0.1, 0.7),
  legend.title = element_blank(),
  legend.key.width = unit(1, "cm")
)

suppressWarnings({

  ## Build the legend from h2 plot
  leg <- get_legend(
    plot_h2_2x5 + scale_linetype_discrete() + guides(color = guide_none()) +
      theme(legend.key.width = unit(1, "cm"), legend.position = "right",
            legend.direction = "horizontal") +
      theme(text = element_text(size = SS))
  )

  ## Panels c and d (h2 and rg)
  a1 <- plot_h2_2x5 + guides(linetype = guide_none(), shape = guide_none()) + ltheme +
    theme(axis.title.x = element_blank()) +
    theme(legend.position = c(WOFFSET, 0.75)) +
    theme(text = element_text(size = SS))

  a2 <- plot_rg_2x5 + guides(linetype = guide_none(), shape = guide_none()) + ltheme +
    theme(legend.position = c(WOFFSET, 0.75)) +
    theme(text = element_text(size = SS))

  ## Align panels a (UKB CCA), c (h2), d (rg), b (NHIRD CCA)
  plist <- align_plots(
    plot_cca_var + xlab("Canonical Variate") + theme(text = element_text(size = SS)),
    a1,
    a2,
    plot_tcca_var + xlab("Canonical Variate") + theme(text = element_text(size = SS)),
    axis = "ltbr", align = "vh"
  )

  ## Align d with e and f
  plist2 <- align_plots(
    plist[[3]],
    plot_gwas2x5_alt_v +
      theme(legend.position = c(0.5, 0.62)) +
      xlab("Generations of xAM") +
      ylab(expression("\U1D6FC"[empircal] / "\U1D6FC"[theoretical])) +
      theme(axis.title.y = element_text(size = SS + 4), text = element_text(size = SS)),
    nhplot
  )

  ## Left column: legend + h2 (c) + rg (d)
  lcol <- plot_grid(
    plot_grid(
      get_legend(plot_h2_2x5 + guides(color = guide_none(), shape = guide_none(), linetype = guide_none())),
      leg,
      nrow = 1, rel_widths = c(1, 4)
    ),
    plist[[2]],
    plist2[[1]],
    rel_heights = c(0.75, 4, 4),
    ncol = 1,
    labels = c("", "c", "d"),
    label_size = LSIZE
  )

  ## Bottom row: left column + e + f
  brow <- plot_grid(
    lcol,
    plist2[[2]],
    nhplot,
    nrow = 1,
    labels = c("", "e", "f"),
    rel_widths = c(3.5, 3.2, 3.3),
    label_size = LSIZE
  )

  ## Top row: a (UKB CCA) + b (NHIRD CCA)
  rr1 <- plot_grid(
    plist[[1]], plist[[4]],
    rel_widths = c(5, 3, 2),
    labels = c("a", "b"),
    label_size = LSIZE
  )

  ## Full figure
  full_fig <- plot_grid(
    rr1, brow,
    rel_heights = c(2.2, 5),
    ncol = 1
  )
})

## ---------------------------------------------------------------
## Save outputs
## ---------------------------------------------------------------
cat("--- Saving figures ---\n")

## PNG
png(file.path(FIGURES_DIR, "fig1.png"),
    height = 11 * 0.9, width = 14 * 0.9, units = "in", res = 300)
print(full_fig)
dev.off()
cat("  Saved: fig1.png\n")

## PDF
cairo_pdf(file.path(FIGURES_DIR, "fig1.pdf"),
          width = 14 * 0.9, height = 11 * 0.9)
print(full_fig)
dev.off()
cat("  Saved: fig1.pdf\n")

cat("\n=== Figure 1 generation complete ===\n")
