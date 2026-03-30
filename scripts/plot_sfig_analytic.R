## plot_sfig_analytic.R
## Generates supplementary figures S9, S10, S11, S12 from pre-computed data.
##   S9:  Incremental on-target vs off-target associations by sample size
##   S10: Type-I error inflation vs sample size
##   S11: Cumulative on-target vs off-target associations
##   S12: False positive rate at off-target loci by effect magnitude
##
## Verbatim port of notebook code from:
##   rb_testing.ipynb (S10, S11, S12)
##   resub_mFigHDxAM.ipynb (S9 — finalized newhits panel)

library(ggplot2)
library(reshape2)
library(cowplot)
library(scales)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
FIG_DIR  <- file.path(BASE_DIR, "figures_output")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

PAL <- RColorBrewer::brewer.pal(8, 'Set1')

## Unicode label used in notebooks for M-italic
M_label <- "\U0001d440"

## ── S9: Incremental on-target vs off-target associations ────────────────────
## Source: resub_mFigHDxAM.ipynb cell 78

load(file.path(BASE_DIR, 'data/sim_results/newhits.rdata'), verbose = TRUE)

nhdat <- melt(newhits, id.vars=c('delta_N','kpheno', 'm_causal'),
              measure.vars=c('delta_ex_on_target','delta_ex_off_target'))

nhdat["Generations of xAM"] <- nhdat$gen
nhdat["Number of phenotypes under xAM"] <- as.factor(nhdat$kpheno)
nhdat[paste0(M_label, " causal / phenotype")] <- nhdat$m_causal
nhdat$type <- NA
nhdat$type[nhdat$variable=='delta_ex_off_target'] <- 'Off target'
nhdat$type[nhdat$variable=='delta_ex_on_target'] <- 'On target'
nhdat$type <- factor(nhdat$type, levels=sort(unique(nhdat$type, decreasing = F)))

M_facet_s9 <- paste0(M_label, " causal / phenotype")

plot_s9 <- ggplot(nhdat[nhdat$m_causal==4000,], aes(delta_N,value,fill=type, shape=type, linetype=type)) +
geom_bar(stat='identity')+
      theme_bw() + facet_grid(reformulate(paste0('`', M_facet_s9, '`'),
                                          '`Number of phenotypes under xAM`'),
                             label=label_both)+
scale_fill_manual(values=RColorBrewer::brewer.pal(name='Set1', 3)[c(1,2)]) +
  theme(
    text = element_text(size = 12),
    plot.title = element_text(hjust = .5, vjust = -8),
    legend.title = element_blank(),
    legend.position = 'top',
    legend.key.width = unit(1.5, 'cm'),
    axis.text.x= element_text(angle=-45, hjust=0),
    axis.title.y.right = element_text(margin = margin(l = 10))
  ) +
    xlab('Incremental GWAS sample size') +
    ylab('Incremental new GWAS hits')

ggsave(file.path(FIG_DIR, "sfig_s9_incremental_hits.pdf"),
       plot_s9, width = 8, height = 6, bg = "white", dpi = 300)
ggsave(file.path(FIG_DIR, "sfig_s9_incremental_hits.png"),
       plot_s9, width = 8, height = 6, bg = "white", dpi = 300)
cat("Saved sfig_s9_incremental_hits\n")


## ── S10: Type-I error inflation vs sample size ──────────────────────────────
## Source: rb_testing.ipynb cells 64, 66

load(file.path(BASE_DIR, 'data/sim_results/powerSims2.rdata'), verbose = TRUE)

hcol='orange'
GENZ = c(0,1,3,5)
MZ = c(1000, 2000,4000,8000,16000)
pfdat <- fres[fres$alpha ==5e-8 & fres$gen %in% GENZ & fres$m_causal %in% MZ,]
pfdat["Generations of xAM"] <- pfdat$gen
pfdat["Number of phenotypes under xAM:"] <- as.factor(pfdat$kpheno)
pfdat[paste0(M_label, " causal")] <- pfdat$m_causal

M_facet_s10 <- paste0(M_label, " causal")

plot_s10 <- ggplot(pfdat, aes((N),(pred_t1e/5e-8), color=`Number of phenotypes under xAM:`, linetype=`Number of phenotypes under xAM:`)) +
    geom_hline(yintercept = (1), col= hcol, lty=4,lwd=1) +
    geom_point() + geom_path() +
    facet_grid(reformulate('`Generations of xAM`', paste0('`', M_facet_s10, '`')),
               scales='fixed', label='label_both') + theme_bw() + theme(text=element_text(size=14)) +
    scale_x_log10(breaks=unique(pfdat$N),labels=comma) +
    scale_y_log10(breaks=(1*10^(0:7))) +
    xlab('GWAS sample size') +
    ylab(expression(frac('Empirical type I error rate at '*alpha*' = 5e-8',
                         'Theoretical type I error rate: '*alpha*' = 5e-8'))) +
    theme_bw() + scale_color_brewer(palette = 'Set1')+
    theme(text=element_text(size=14)) +
    theme(plot.title=element_text(hjust=.5, vjust=-8), axis.text.x=element_text(ang=-45,hjust=0),
          legend.position='top', legend.key.width=unit(1,'cm'))

ggsave(file.path(FIG_DIR, "sfig_s10_t1e_inflation.pdf"),
       plot_s10, width = 14, height = 12, bg = "white", dpi = 300)
ggsave(file.path(FIG_DIR, "sfig_s10_t1e_inflation.png"),
       plot_s10, width = 14, height = 12, bg = "white", dpi = 300)
cat("Saved sfig_s10_t1e_inflation\n")


## ── S11: Cumulative on-target vs off-target associations ────────────────────
## Source: rb_testing.ipynb cells 45, 48, 49, 50

load(file.path(BASE_DIR, 'data/sim_results/powerSims3.rdata'), verbose = TRUE)

fpdat <- melt(frac_res, id.vars=c('N','kpheno', 'm_causal'),
              measure.vars=c('ex_on_target','ex_off_target','ex_FP'))
fpdat["Generations of xAM"] <- fpdat$gen
fpdat["Number of phenotypes under xAM"] <- as.factor(fpdat$kpheno)
fpdat[paste0(M_label, " causal / phenotype")] <- fpdat$m_causal

fpdat$value <- ceiling(fpdat$value)

fpdat$type <- NA
fpdat$type[fpdat$variable=='ex_on_target'] <- 'On target'
fpdat$type[fpdat$variable=='ex_off_target'] <- 'Off target'
fpdat$type[fpdat$variable=='ex_FP'] <- 'Conventional false postives'
fpdat$type <- factor(fpdat$type, levels=sort(unique(fpdat$type, decreasing = T)))

M_facet_s11 <- paste0(M_label, " causal / phenotype")

plot_s11 <- ggplot(fpdat, aes(N, value, color=type, shape=type, linetype=type)) +
    geom_path(position=position_dodge(width=.05)) +
    geom_point(position=position_dodge(width=.05)) +
      theme_bw() + facet_grid(reformulate(paste0('`', M_facet_s11, '`'),
                                          '`Number of phenotypes under xAM`'),
                             label=label_both)+
scale_color_manual(values=RColorBrewer::brewer.pal(name='Set1', 3)[c(2,1,3)]) +
  theme(
    text = element_text(size = 14),
    plot.title = element_text(hjust = .5, vjust = -8),
    legend.title = element_blank(),
    legend.position = 'top',
    legend.key.width = unit(1.5, 'cm'),
    axis.text.x= element_text(angle=-45, hjust=0),
    axis.title.y.right = element_text(margin = margin(l = 10))
  ) +
    scale_x_log10(breaks=unique(frac_res$N),labels=comma) +
    scale_y_log10(breaks=1*10^(0:5),labels=comma) +
    xlab('GWAS sample size') +
    ylab('# GWAS hits')

ggsave(file.path(FIG_DIR, "sfig_s11_cumulative_hits.pdf"),
       plot_s11, width = 12, height = 8, bg = "white", dpi = 300)
ggsave(file.path(FIG_DIR, "sfig_s11_cumulative_hits.png"),
       plot_s11, width = 12, height = 8, bg = "white", dpi = 300)
cat("Saved sfig_s11_cumulative_hits\n")


## ── S12: False positive rate at off-target loci by effect magnitude ─────────
## Source: rb_testing.ipynb cells 28, 30, 31, 33
## NOTE: perSNPres requires heavy Monte Carlo simulation from ttt object.
##       We load the pre-processed result (XX) from the processed CSV,
##       which was extracted from the notebook output.

XX <- read.csv(file.path(BASE_DIR, 'processed/sfig_s12_persnp_t1e.csv'))

XX$NNN <- as.factor(format(XX$N,big.mark = ',',trim=TRUE))
XX$NNN <- factor(XX$NNN, levels=rev(levels(XX$NNN)[order(as.numeric(gsub(',','',levels(XX$NNN))))]))

plot_s12 <- ggplot(XX, aes(beta_quantile, T1E, color = NNN, linetype = NNN)) +
  geom_path(lwd = .8) +
  geom_hline(yintercept = 5e-8, lty = 3, lwd = .8) +
  scale_y_log10(
    breaks = 5 * 10^-(8:1),
    name = expression('Empirical type I error rate at '*alpha*' = 5e-8'),
    sec.axis = dup_axis(
      trans = ~ . / 5e-8,
      name = expression(frac('Empirical type I error rate at '*alpha*' = 5e-8',
                             'Theoretical type I error rate: '*alpha*' = 5e-8')),
      breaks = 5 * 10^-(8:1)/5e-8
    )
  ) +
  xlab('Effect size quantile') +
  guides(color = guide_legend(title = 'GWAS\nsample size'),
         linetype = guide_legend(title = 'GWAS\nsample size')) +
  theme_bw() +
  theme(
    legend.position=c(.16,.67),
    text = element_text(size = 14),
    plot.title = element_text(hjust = .5, vjust = -8),
    legend.title = element_text(hjust = .5),
    legend.key.width = unit(1.5, 'cm'),
    axis.title.y.right = element_text(margin = margin(l = 10))
  )

ggsave(file.path(FIG_DIR, "sfig_s12_persnp_t1e.pdf"),
       plot_s12, width = 9, height = 5, bg = "white", dpi = 300)
ggsave(file.path(FIG_DIR, "sfig_s12_persnp_t1e.png"),
       plot_s12, width = 9, height = 5, bg = "white", dpi = 300)
cat("Saved sfig_s12_persnp_t1e\n")


cat("Done: plot_sfig_analytic.R\n")
