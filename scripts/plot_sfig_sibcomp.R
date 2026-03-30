## plot_sfig_sibcomp.R
## Verbatim port of mFigSibComp.ipynb cells producing supplementary figure S17
## (population vs sibship GWAS comparison)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
FIG_DIR  <- file.path(BASE_DIR, "figures_output")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

## ── Cell 1: Initialize ─────────────────────────────────────────────────────────

library(repr)
library(ggplot2)
library(reshape2)
library(cowplot)
library(stringr)
library(splines)

default_theme <- theme_bw() + theme(text=element_text(size=14))
default_theme2 <- theme_minimal() + theme(text=element_text(size=14))

## ── Cell 2: Load data ──────────────────────────────────────────────────────────

res <- read.csv(file.path(BASE_DIR, 'data/sim_results/merged_tabla_redux_results_011024.csv')) ## from cleanAllGeneralSims2024
table(res$scenario)
# res <- res[res$scenario %in% c('2xAM', '5xAM'),]#,'RM'),]

res$v_tot <- res$h2_true/(1-res$h2_true) * .5 + .5
res$vbeta <- res$vbeta_true*res$v_tot

mvars_uni = c(#'pgwas_PGS_hat_R2_49',#'sgwas_PGS_hat_R2_49',
              #'pgwas_PGS_hat_R2_29','sgwas_PGS_hat_R2_29',
              # 'pgwas_PGS_hat_R2_39',#'sgwas                                                                                                                                                                                                                                                                       _PGS_hat_R2_39',
              'h2_true','he_h2')#,'vbeta')

mvars_biv = c(#'pgwas_PGS_hat_corr_49',#'sgwas_PGS_hat_corr_49',
              #'pgwas_PGS_hat_corr_29','sgwas_PGS_hat_corr_29',
              # 'pgwas_PGS_hat_corr_39',#'sgwas_PGS_hat_corr_39',
              # 'rbeta_true',
    'rg_true','he_rg')#,'rbeta_hat_pgwas','rbeta_hat_sgwas')
grep('^arg',names(res), val=T)
idvars = c('seed','gen','args_kmate','args_rmate', 'args_theta', 'args_phi', 'args_m_causal','power','scenario')

## ── Cell 48: Melt for T1E (panel a) ────────────────────────────────────────────

mvars_uni = c(#'pgwas_PGS_hat_R2_49',#'sgwas_PGS_hat_R2_49',
              #'pgwas_PGS_hat_R2_29','sgwas_PGS_hat_R2_29',
              # 'pgwas_PGS_hat_R2_39',#'sgwas                                                                                                                                                                                                                                                                       _PGS_hat_R2_39',
              'sgwas_false_positives_0.05','pgwas_false_positives_0.05')#,'vbeta')


idvars = c('seed','gen','args_kmate','args_rmate', 'args_theta', 'args_phi', 'args_m_causal','power','scenario')
mdat <- reshape2::melt(res, id.vars = idvars, measure.vars = mvars_uni)
mdat$relative_T1R<-mdat$value/0.05
mdat$GWAS <- 'Population'
mdat$GWAS[grep('^s',mdat$variable)] <- 'Sibship'
mdat$alpha=0.05
# # udat
# ggplot(udat, aes(as.factor(gen), value,
#                 col=as.factor(scenario), shape=variable)) + facet_wrap(~power)+
#     stat_summary()

tmpd <- mdat #within(mdat[mdat$GWAS=='Population',], {
#                # T1R[T1R==0] <- (5e-8)*2
#                relative_T1R <- T1R/alpha_bonferoni})
# tmpd$`over(n,m)` = as.factor(1/.8*tmpd$args_n/tmpd$args_m)
# tmpd$novm = paste0('italic(n)/italic(m) == ',tmpd$`over(n,m)`)
# tmpd$novm <- factor(tmpd$novm, levels = levels(as.factor(tmpd$novm))[c(2,1,3)])
# tmpd$xAM <-'2-variate'
# tmpd$xAM[tmpd$args_kmate==5] <-'5-variate'
tmpd$power = paste(tmpd$power, 'power at \U1D6FC=0.05')

## ── Cell 51: plot_sib1 (panel a) ───────────────────────────────────────────────

# mdat <- reshape2::dcast(molten,seed+gen+args_m+args_n+args_kmate +alpha +GWAS+alpha_bonferoni~ quantity)

# mdat$T1R <- mdat$FalsePositive/(.8)
# mdat$hits <- (mdat$FalsePositive + mdat$TruePositive)
options(repr.plot.width=7)
options(repr.plot.height=5)
unique(tmpd$scenario)
    options(repr.plot.width=6)
    options(repr.plot.height=6)

ppd <- tmpd[tmpd$scenario %in% c("2xAM","RM + VT",
                                                          "5xAM",
                                                          "5xAM + GxE",
                                                          "5xAM + VT",
                                                          "5xAM + VT + GxE"),]

ppd$scenario <- factor(ppd$scenario, levels=c("RM + VT", "2xAM",
                                                          "5xAM",
                                                          "5xAM + GxE",
                                                          "5xAM + VT",
                                                          "5xAM + VT + GxE"))
(plot_sib1 <-
 ggplot(ppd[ppd$gen>0,],
       aes(gen,relative_T1R, color=GWAS, lty=power,shape=power)) + default_theme+
    # geom_alpha_bonferoni*2))+
    # geom_jitter(height = 1)+
     stat_summary(geom='point',  position=position_dodge(width=.1),
                 fun.data = function(x) mean_se(x,mult=1.96)
                 ) +
     stat_summary(geom='linerange', position=position_dodge(width=.1),
                 fun.data =  function(x) mean_se(x,mult=1.96)
                 ) +
    geom_smooth(formula=y~ns(x,df =3), lwd=.5, method='gam', aes(gen, color=GWAS, lty=power), se=F ) +
    # geom_smooth(formula=y~bs(x,degree =3),method='gam', aes(gen, color=GWAS, lty=xAM), se=F ) +
    # stat_summary(geom='line', aes(gen,  color=GWAS, lty=xAM), fun = mean ) +
    facet_wrap(~scenario)+ #, labeller = label_parsed) +
    geom_hline(color='grey', lty=3,yintercept = 1) +
    # scale_color_manual(values=PAL) +
    # scale_y_continuous(breaks = seq(1,6,.25)) +
    geom_hline(color='grey', lty=3,yintercept = 1) +
    scale_color_manual(values = RColorBrewer::brewer.pal(8,'Set1')[-(5:6)]

                       # labels =c('xAM',
                       #           'xAM + G\u00d7E',
                       #           'xAM + VT',
                       #           'xAM + G\u00d7E + VT')
                     ) +
 guides(linetype=guide_legend(override.aes = list( color=1))) +
    scale_linetype_manual(values = c(2:7))+

    ylab(expression(over('Empirical Type-I Error Rate','Theoretical Type-I Error Rate'))) +
    #theme(axis.text.y.right = element_text(color=PAL[3]),
    #     axis.ticks.y.right=element_line(color=PAL[3])) +
    xlab('Generations of xAM') +
     # theme(axis.title.x=element_blank())+
     theme(legend.position='top', legend.direction='vertical', legend.title = element_blank(), legend.key.width=unit(1,'cm'), text=element_text(size=12))
 )

## ── Cell 52: Melt for rbeta + plot_sib2 (panel b) ──────────────────────────────

mvars_uni = c(#'pgwas_PGS_hat_R2_49',#'sgwas_PGS_hat_R2_49',
              #'pgwas_PGS_hat_R2_29','sgwas_PGS_hat_R2_29',
              # 'pgwas_PGS_hat_R2_39',#'sgwas                                                                                                                                                                                                                                                                       _PGS_hat_R2_39',
              'rbeta_hat_sgwas','rbeta_hat_pgwas')#,'vbeta')


idvars = c('seed','gen','args_kmate','args_rmate', 'args_theta', 'args_phi', 'scenario')
mdat <- reshape2::melt(res, id.vars = idvars, measure.vars = mvars_uni)
mdat$GWAS <- 'Population'
mdat$GWAS[grep('sgwas',mdat$variable)] <- 'Sibship'
mdat$alpha=0.05
# # udat
# ggplot(udat, aes(as.factor(gen), value,
#                 col=as.factor(scenario), shape=variable)) + facet_wrap(~power)+
#     stat_summary()

tmpd <- mdat #within(mdat[mdat$GWAS=='Population',], {
# mdat <- reshape2::dcast(molten,seed+gen+args_m+args_n+args_kmate +alpha +GWAS+alpha_bonferoni~ quantity)

# mdat$T1R <- mdat$FalsePositive/(.8)
# mdat$hits <- (mdat$FalsePositive + mdat$TruePositive)
options(repr.plot.width=7)
options(repr.plot.height=5)
unique(tmpd$scenario)
    options(repr.plot.width=6)
    options(repr.plot.height=6)

ppd <- tmpd[tmpd$scenario %in% c("2xAM","RM + VT",
                                                          "5xAM",
                                                          "5xAM + GxE",
                                                          "5xAM + VT",
                                                          "5xAM + VT + GxE"),]

ppd$scenario <- factor(ppd$scenario, levels=c("RM + VT", "2xAM",
                                                          "5xAM",
                                                          "5xAM + GxE",
                                                          "5xAM + VT",
                                                          "5xAM + VT + GxE"))
(plot_sib2 <-
 ggplot(ppd[ppd$gen>0,],
       aes(gen,value, color=GWAS)) + default_theme+
    # geom_alpha_bonferoni*2))+
    # geom_jitter(height = 1)+
     stat_summary(geom='point',  position=position_dodge(width=.1),
                 fun.data = function(x) mean_se(x,mult=1.96)
                 ) +
     stat_summary(geom='linerange', position=position_dodge(width=.1),
                 fun.data =  function(x) mean_se(x,mult=1.96)
                 ) +
    geom_smooth(formula=y~ns(x,df =3), lwd=.5, method='gam', aes(gen, color=GWAS), se=F ) +
    # geom_smooth(formula=y~bs(x,degree =3),method='gam', aes(gen, color=GWAS, lty=xAM), se=F ) +
    # stat_summary(geom='line', aes(gen,  color=GWAS, lty=xAM), fun = mean ) +
    facet_wrap(~scenario,ncol=2)+ #, labeller = label_parsed) +
    # scale_color_manual(values=PAL) +
    # scale_y_continuous(breaks = seq(1,6,.25)) +
    ylab(expression(italic(r[hat('\U1D6FD')]))) +

    scale_color_manual(values = RColorBrewer::brewer.pal(8,'Set1')[-(5:6)]

                       # labels =c('xAM',
                       #           'xAM + G\u00d7E',
                       #           'xAM + VT',
                       #           'xAM + G\u00d7E + VT')
                     ) +
 guides(linetype=guide_legend(override.aes = list( color=1))) +
    scale_linetype_manual(values = c(2:7))+

    #theme(axis.text.y.right = element_text(color=PAL[3]),
    #     axis.ticks.y.right=element_line(color=PAL[3])) +
    xlab('Generations of xAM') +
     # theme(axis.title.x=element_blank())+
     theme(legend.position='top', legend.direction='vertical', legend.title = element_blank(), legend.key.width=unit(1,'cm'), text=element_text(size=12))
 )

## ── Cell 53: Combine panels ────────────────────────────────────────────────────

options(repr.plot.width=10)
options(repr.plot.height=5)

suppressWarnings({

    aa <- align_plots(plot_sib1+guides(shape=guide_legend(order=1),
                           linetype=guide_legend(order=1, color=1),
                           color=guide_legend(order=2)) +
          theme(legend.box.just='center'),
          plot_sib2+guides(color=guide_none()),
                     axis='b', align='h')

    plot_s17_combined <- plot_grid(aa[[1]],aa[[2]], nrow=1, rel_widths = 3:2,
             labels=letters[1:2])

    })

## ── Save outputs ────────────────────────────────────────────────────────────────

ggsave(file.path(FIG_DIR, "sfig_s17_sibcomp_combined.pdf"),
       plot_s17_combined, width = 10, height = 5, bg = "white", dpi = 300)
ggsave(file.path(FIG_DIR, "sfig_s17_sibcomp_combined.png"),
       plot_s17_combined, width = 10, height = 5, bg = "white", dpi = 300)
cat("Saved sfig_s17_sibcomp_combined.pdf and .png\n")

ggsave(file.path(FIG_DIR, "sfig_s17a_t1e.pdf"),
       plot_sib1, width = 7, height = 5, bg = "white", dpi = 300)
ggsave(file.path(FIG_DIR, "sfig_s17a_t1e.png"),
       plot_sib1, width = 7, height = 5, bg = "white", dpi = 300)
cat("Saved sfig_s17a_t1e.pdf and .png\n")

ggsave(file.path(FIG_DIR, "sfig_s17b_slope.pdf"),
       plot_sib2, width = 6, height = 6, bg = "white", dpi = 300)
ggsave(file.path(FIG_DIR, "sfig_s17b_slope.png"),
       plot_sib2, width = 6, height = 6, bg = "white", dpi = 300)
cat("Saved sfig_s17b_slope.pdf and .png\n")

cat("Done: plot_sfig_sibcomp.R\n")
