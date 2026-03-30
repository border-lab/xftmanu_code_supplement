## plot_sfig_cca.R
## Generates supplementary figures S1, S2, S3 from processed data ONLY.
##   S1: Multivariate vs multidimensional mating (synthetic data)
##   S2: Nonlinear mating CCA (synthetic data)
##   S3: CCA scree plot (MICE PMM imputed)

library(ggplot2)
library(cowplot)
library(GGally)
library(grid)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
PROC_DIR <- file.path(BASE_DIR, "processed")
FIG_DIR  <- file.path(BASE_DIR, "figures_output")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

default_theme <- theme_bw() + theme(text = element_text(size = 12))

cat("=== Generating supplementary CCA figures ===\n\n")

## ---------------------------------------------------------------
## Helper: build ggduo pair plots for phenotypes and CVs
## ---------------------------------------------------------------
make_pair_grobs <- function(Xdat, Cdat, title_pheno, title_cv = " ") {
  ## Identify mate 1 and mate 2 columns
  m1_pheno <- grep("Mate\\.1", names(Xdat), value = TRUE)
  m2_pheno <- grep("Mate\\.2", names(Xdat), value = TRUE)
  m1_cv <- grep("Mate\\.1", names(Cdat), value = TRUE)
  m2_cv <- grep("Mate\\.2", names(Cdat), value = TRUE)

  ## Pretty column names for display
  names(Xdat) <- gsub("Mate\\.", "Mate ", names(Xdat))
  names(Xdat) <- gsub("\\.\\.", ": ", names(Xdat))
  names(Xdat) <- gsub("\\.", " ", names(Xdat))
  m1_pheno_nice <- grep("Mate 1", names(Xdat), value = TRUE)
  m2_pheno_nice <- grep("Mate 2", names(Xdat), value = TRUE)

  names(Cdat) <- gsub("Mate\\.", "Mate ", names(Cdat))
  names(Cdat) <- gsub("\\.\\.", ": ", names(Cdat))
  names(Cdat) <- gsub("\\.", " ", names(Cdat))
  m1_cv_nice <- grep("Mate 1", names(Cdat), value = TRUE)
  m2_cv_nice <- grep("Mate 2", names(Cdat), value = TRUE)

  ## Select first 2 of each mate for 2x2 pair plots (as in notebook)
  p_pheno <- ggduo(Xdat, m1_pheno_nice[1:2], m2_pheno_nice[1:2],
                   mapping = aes(alpha = 0.01),
                   types = list(continuous = "points")) +
    stat_smooth(color = 2, method = "lm", formula = y ~ x, se = FALSE) +
    default_theme + ggtitle(title_pheno)

  p_cv <- ggduo(Cdat, m1_cv_nice[1:2], m2_cv_nice[1:2],
                mapping = aes(alpha = 0.01),
                types = list(continuous = "points")) +
    stat_smooth(color = 4, method = "lm", formula = y ~ x, se = FALSE) +
    default_theme + ggtitle(title_cv)

  g_pheno <- grid::grid.grabExpr(print(p_pheno))
  g_cv    <- grid::grid.grabExpr(print(p_cv))

  list(pheno = g_pheno, cv = g_cv)
}

## ---------------------------------------------------------------
## S1: Multivariate vs multidimensional mating
## ---------------------------------------------------------------
cat("--- S1: Synthetic mating regimes ---\n")

## Load processed data
Xdat_u1 <- read.csv(file.path(PROC_DIR, "sfig_s1_pheno_u1.csv"))
Cdat_u1 <- read.csv(file.path(PROC_DIR, "sfig_s1_cv_u1.csv"))
Xdat_r1 <- read.csv(file.path(PROC_DIR, "sfig_s1_pheno_r1.csv"))
Cdat_r1 <- read.csv(file.path(PROC_DIR, "sfig_s1_cv_r1.csv"))
Xdat_r3 <- read.csv(file.path(PROC_DIR, "sfig_s1_pheno_r3.csv"))
Cdat_r3 <- read.csv(file.path(PROC_DIR, "sfig_s1_cv_r3.csv"))

grobs_u1 <- make_pair_grobs(Xdat_u1, Cdat_u1, "Univariate, unidimensional")
grobs_r1 <- make_pair_grobs(Xdat_r1, Cdat_r1, "Multivariate, unidimensional")
grobs_r3 <- make_pair_grobs(Xdat_r3, Cdat_r3, "Multivariate, multidimensional")

## Align and compose: 3 rows x 2 cols (phenotypes | canonical variates)
aa <- align_plots(grobs_u1$pheno, grobs_u1$cv,
                  grobs_r1$pheno, grobs_r1$cv,
                  grobs_r3$pheno, grobs_r3$cv,
                  axis = "tblr", align = "hv")

sfig_s1 <- plot_grid(aa[[1]], aa[[2]],
                     aa[[3]], aa[[4]],
                     aa[[5]], aa[[6]],
                     ncol = 2,
                     labels = c("a", "", "b", "", "c", ""))

ggsave(file.path(FIG_DIR, "sfig_s1_mating_cca.pdf"),
       sfig_s1, width = 10, height = 12)
ggsave(file.path(FIG_DIR, "sfig_s1_mating_cca.png"),
       sfig_s1, width = 10, height = 12, dpi = 300)
cat("  Saved: sfig_s1_mating_cca.pdf, sfig_s1_mating_cca.png\n\n")

## ---------------------------------------------------------------
## S2: Nonlinear mating CCA
## ---------------------------------------------------------------
cat("--- S2: Nonlinear mating CCA ---\n")

Xdat_pw <- read.csv(file.path(PROC_DIR, "sfig_s2_pheno_pw.csv"))
Cdat_pw <- read.csv(file.path(PROC_DIR, "sfig_s2_cv_pw.csv"))
Xdat_at <- read.csv(file.path(PROC_DIR, "sfig_s2_pheno_at.csv"))
Cdat_at <- read.csv(file.path(PROC_DIR, "sfig_s2_cv_at.csv"))
Xdat_q  <- read.csv(file.path(PROC_DIR, "sfig_s2_pheno_q.csv"))
Cdat_q  <- read.csv(file.path(PROC_DIR, "sfig_s2_cv_q.csv"))

grobs_pw <- make_pair_grobs(Xdat_pw, Cdat_pw, "Piecewise linear")
grobs_at <- make_pair_grobs(Xdat_at, Cdat_at, "Inverse tangent")
grobs_q  <- make_pair_grobs(Xdat_q, Cdat_q, "Quadratic")

## Notebook cell 12 orders: pw first, then at, then q
aa2 <- align_plots(grobs_pw$pheno, grobs_pw$cv,
                   grobs_at$pheno, grobs_at$cv,
                   grobs_q$pheno, grobs_q$cv,
                   axis = "tblr", align = "hv")

sfig_s2 <- plot_grid(aa2[[1]], aa2[[2]],
                     aa2[[3]], aa2[[4]],
                     aa2[[5]], aa2[[6]],
                     ncol = 2,
                     labels = c("a", "", "b", "", "c", ""))

ggsave(file.path(FIG_DIR, "sfig_s2_nonlinear_cca.pdf"),
       sfig_s2, width = 10, height = 12)
ggsave(file.path(FIG_DIR, "sfig_s2_nonlinear_cca.png"),
       sfig_s2, width = 10, height = 12, dpi = 300)
cat("  Saved: sfig_s2_nonlinear_cca.pdf, sfig_s2_nonlinear_cca.png\n\n")

## ---------------------------------------------------------------
## S3: CCA scree plot (MICE PMM imputed)
## ---------------------------------------------------------------
cat("--- S3: CCA scree (MICE PMM imputed) ---\n")

scree <- read.csv(file.path(PROC_DIR, "sfig_s3_cca_scree.csv"))

## Use KK=12 for display (as in notebook)
KK <- 12
scree_sub <- scree[scree$cv <= KK, ]
scree_sub$cv_factor <- factor(scree_sub$CanonicalVariate,
                              levels = scree_sub$CanonicalVariate)

sfig_s3 <- ggplot(scree_sub, aes(x = cv_factor, y = cumulative_redundancy)) +
  scale_y_continuous(position = "left", limits = 0:1) +
  geom_hline(col = "purple", lty = 3, yintercept = 0.9) +
  geom_hline(col = "darkorange", lty = 3, yintercept = 0.95) +
  geom_point() +
  geom_line(group = 1) +
  xlab("Canonical Variate") +
  ylab("Relative Canonical\nRedundancy") +
  theme_bw() +
  theme(text = element_text(size = 14), legend.position = "bottom")

ggsave(file.path(FIG_DIR, "sfig_s3_cca_scree.pdf"),
       sfig_s3, width = 8, height = 6)
ggsave(file.path(FIG_DIR, "sfig_s3_cca_scree.png"),
       sfig_s3, width = 8, height = 6, dpi = 300)
cat("  Saved: sfig_s3_cca_scree.pdf, sfig_s3_cca_scree.png\n\n")

cat("Done: plot_sfig_cca.R\n")
