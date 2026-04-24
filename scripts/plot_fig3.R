## plot_fig3.R
## Direct port of manu/figure_nb/mFigNewComplexity.ipynb
## Only changes: file paths adjusted for round4 layout, ggsave at end.
##
## Cells used: 2, 13, 19, 24, 36, 37, 40, 48, 52, 70

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
FIG_DIR  <- file.path(BASE_DIR, "figures_output")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

###############################################################################
## Cell 2
###############################################################################
# library(yacca)
library(repr)
library(ggplot2)
library(reshape2)
library(cowplot)
library(stringr)
library(splines)


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


default_theme <- theme_bw() + theme(text=element_text(size=14))
default_theme2 <- theme_minimal() + theme(text=element_text(size=14))

###############################################################################
## Cell 13
###############################################################################
res$sim = apply(str_split_fixed(res$X, '_',5)[,-2], 1, paste, collapse='_')
# res$scenario = apply(str_split_fixed(res$X, '_',5)[,-1], 1, paste, collapse='_')
# names(res)

###############################################################################
## Cell 19
###############################################################################
gdat <- res
gdat$h2_he <- gdat$he_h2

molt <-
reshape2::melt(gdat, id.vars = c('gen','X','seed','args_m_causal','scenario'),
               measure.vars = c("h2_he", "h2_true"))
# gdat

###############################################################################
## Cell 24
###############################################################################
gdat$rg_hat_pgwas <- gdat$pgwas_rPRS_hat
gdat$rg_hat_sgwas <- gdat$sgwas_rPRS_hat
molt <-
reshape2::melt(gdat, id.vars = c('gen','seed','args_m_causal','scenario','power'),
               measure.vars = c("h2_he", "h2_true"))

gdat2 <- molt#[molt$gen %in% c(0,1,3,5),]
# molt
# pdat <- dcast(molt, scenario+gen+seed+args_m_causal+power~variable)
# pdat
# pdat <- pdat[pdat$gen %in% c(0,1,3,5),]

# molt$scenario <- factor(molt$scenario, levels= c("2xAM","5xAM","5xAM + gXe", "5xAM + eVT", "5xAM + eVT + gXe",  "5xAM + pVT",
# "5xAM + pVT + gXe"))
# molt <- molt[molt$scenario!='2xAM',]
# gdat
# gdat2

###############################################################################
## Cell 36
###############################################################################
gdat$rbeta_HE <- gdat$he_rg
gdat$rg_hat_pgwas <- gdat$pgwas_rPRS_hat
gdat$rg_hat_sgwas <- gdat$sgwas_rPRS_hat
molt <-
reshape2::melt(gdat, id.vars = c('gen','X','seed','args_m','scenario'),
               measure.vars = c("h2_he", "rg_true", "rbeta_HE",
                                "h2_true"))

molt <- molt#[molt$gen %in% c(0,1,3,5),]
pdat <- dcast(molt, scenario+gen+X+seed+args_m~variable)
pdat <- pdat[pdat$gen %in% c(0,1,3,5),]
# molt$variable <- factor(molt$variable,
                       # levels=rev(c("rbeta_true",  "rbeta_HE", "rbeta_hat_sgwas", "rbeta_hat_pgwas", "rg_true",
                                    # "rg_hat_sgwas", "rg_hat_pgwas")))
# molt$scenario <- factor(molt$scenario, levels= c("2xAM","5xAM","5xAM + gXe", "5xAM + eVT", "5xAM + eVT + gXe",  "5xAM + pVT",
# "5xAM + pVT + gXe"))
# molt <- molt[molt$scenario!='2xAM',]

###############################################################################
## Cell 37: plot_biv_multi (scenarios: RM, RM + VT, 5xAM, 5xAM + VT)
###############################################################################
PAL = RColorBrewer::brewer.pal(8,'Set1')

options(repr.plot.height=5)
options(repr.plot.width=8)
pdat<-pdat2 <- molt[molt$variable %in% c('rbeta_HE','rg_true','h2_he','h2_true'),]
pdat$variable <- as.character(pdat$variable)
pdat$variable[pdat2$variable=='rbeta_HE'] <- 'hat(italic(r))[𝛽]'
pdat$variable[pdat2$variable=='rg_true'] <- 'italic(r)[score]'
pdat$variable[pdat2$variable=='h2_he'] <- 'hat(italic(h))^2'
pdat$variable[pdat2$variable=='h2_true'] <- 'italic(h)^2'
pdat$int <- 0
pdat$int[pdat2$variable=='h2_he' | pdat2$variable=='h2_true'] <- .5
    # expression(italic(r)[score])

pdat$variable <- as.factor(pdat$variable)
pdat$variable <- factor(pdat$variable, levels=rev(levels(pdat$variable)))
pdat$f1 <- pdat$f2 <- as.character(pdat$variable)
pdat$f1[pdat2$variable=='h2_he' | pdat2$variable=='h2_true'] <- NA
# pdat$f2[!pdat2$variable=='h2_he' & !pdat2$variable=='h2_true'] <- NA
(plot_biv_multi <- ggplot(pdat[pdat$scenario %in% c("RM", "RM + VT",
                                                          "5xAM",
                                                          # "5xAM + GxE",
                                                          "5xAM + VT"#,
                                                          # "5xAM + VT + GxE"
                                                   ) & !is.na(pdat$f1),],
                                aes(gen,value, color=scenario, linetype=scenario, shape=scenario)) +
     default_theme2 +
    geom_hline(aes(yintercept = int), color='grey', lty=1) +
    stat_summary(geom = 'point',position=position_dodge(width=.3),
                 fun = median) +
    stat_summary(geom = 'path',position=position_dodge(width=.3),lwd=1,
                 fun = median) +
    # stat_summary(data=utmp2,
                 # geom = 'point',position=position_dodge(width=.3),
                 # fun = function(x) quantile(x,.5)) +
    ylab(expression(italic(r)[g])) +
    xlab('Generations of xAM')       +
   scale_color_manual(values = PAL <- RColorBrewer::brewer.pal(8,'Set1')[-c(5:7)]#,
                       # labels =c('xAM',
                       #           'xAM + G×E',
                       #           'xAM + VT',
                       #           'xAM + G×E + VT')
                     ) +
 facet_wrap(~variable,drop = T, label='label_parsed') +
    scale_linetype_manual(values = c(2:6)#,
                       # labels =c('xAM',
                       #           'xAM + G×E',
                       #           'xAM + VT',
                       #           'xAM + G×E + VT')
                         )
                )

###############################################################################
## Cell 40: plot_uni_multi (scenarios: RM, RM + VT, 5xAM, 5xAM + VT)
###############################################################################
PAL = RColorBrewer::brewer.pal(8,'Set1')

options(repr.plot.height=5)
options(repr.plot.width=8)
pdat<-pdat2 <- molt[molt$variable %in% c('rbeta_HE','rg_true','h2_he','h2_true'),]
pdat$variable <- as.character(pdat$variable)
pdat$variable[pdat2$variable=='rbeta_HE'] <- 'hat(italic(r))[𝛽]'
pdat$variable[pdat2$variable=='rg_true'] <- 'italic(r)[score]'
pdat$variable[pdat2$variable=='h2_he'] <- 'hat(italic(h))^2'
pdat$variable[pdat2$variable=='h2_true'] <- 'italic(h)^2'
pdat$int <- 0
pdat$int[pdat2$variable=='h2_he' | pdat2$variable=='h2_true'] <- .5
    # expression(italic(r)[score])

pdat$variable <- as.factor(pdat$variable)
pdat$variable <- factor(pdat$variable, levels=rev(levels(pdat$variable)))
pdat$f1 <- pdat$f2 <- as.character(pdat$variable)
pdat$f1[pdat2$variable!='h2_he' & pdat2$variable!='h2_true'] <- NA
# pdat$f2[!pdat2$variable=='h2_he' & !pdat2$variable=='h2_true'] <- NA
(plot_uni_multi <- ggplot(pdat[pdat$scenario %in% c("RM","RM + VT",
                                                          "5xAM",
                                                          "5xAM + VT") & !is.na(pdat$f1),],
                                aes(gen,value, color=scenario, linetype=scenario, shape=scenario)) + default_theme2 +
    geom_hline(aes(yintercept = int), color='grey', lty=1) +
    stat_summary(geom = 'path',position=position_dodge(width=.3),lwd=1,
                 fun = median) +
     stat_summary(geom = 'point',position=position_dodge(width=.3),
                 fun = median) +
    # stat_summary(data=utmp2,
                 # geom = 'point',position=position_dodge(width=.3),
                 # fun = function(x) quantile(x,.5)) +
    ylab(expression(italic(h)^2)) +
    # ylab(expression(italic(r)[g])) +
    xlab('Generations of xAM')       +
   scale_color_manual(values = PAL <- RColorBrewer::brewer.pal(8,'Set1')[-(5:7)]#,
                       # labels =c('xAM',
                       #           'xAM + G×E',
                       #           'xAM + VT',
                       #           'xAM + G×E + VT')
                     ) +
 facet_wrap(~variable,drop = T, label='label_parsed') +
    scale_linetype_manual(values = c(2:6)#,
                       # labels =c('xAM',
                       #           'xAM + G×E',
                       #           'xAM + VT',
                       #           'xAM + G×E + VT')
                         )
                )

###############################################################################
## Cell 48: GWAS false positive computation
###############################################################################
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
tmpd$power = paste(tmpd$power, 'power at 𝛼=0.05')

###############################################################################
## Cell 52: plot_gwas2x5_alt (scenarios: RM, 5xAM, 5xAM + VT)
###############################################################################
# mdat <- reshape2::dcast(molten,seed+gen+args_m+args_n+args_kmate +alpha +GWAS+alpha_bonferoni~ quantity)

# mdat$T1R <- mdat$FalsePositive/(.8)
# mdat$hits <- (mdat$FalsePositive + mdat$TruePositive)
options(repr.plot.width=8)
options(repr.plot.height=6)
unique(tmpd$scenario)
(plot_gwas2x5_alt <-
 ggplot(tmpd[tmpd$GWAS=='Population'&tmpd$scenario %in% c("RM",
                                                          "5xAM",
                                                          # "5xAM + GxE",
                                                          "5xAM + VT"#,
                                                          # "5xAM + VT + GxE"
                                                         ),],
       aes(gen,relative_T1R, color=scenario, lty=scenario,shape=scenario)) + default_theme+
    # geom_alpha_bonferoni*2))+
    # geom_jitter(height = 1)+
    geom_hline(color='grey', lty=1,yintercept = 1) +
     stat_summary(geom='point',  position=position_dodge(width=.1),
                 fun=median
                 ) +
     stat_summary(geom='path',  position=position_dodge(width=.1),lwd=1,
                 fun=median
                 ) +
    # geom_smooth(formula=y~ns(x,df =3),method='gam', aes(gen, color=scenario, lty=scenario), se=F ) +
    # geom_smooth(formula=y~bs(x,degree =3),method='gam', aes(gen, color=GWAS, lty=xAM), se=F ) +
    # stat_summary(geom='line', aes(gen,  color=GWAS, lty=xAM), fun = mean ) +
    facet_grid(~power)+ #, labeller = label_parsed) +
    # scale_color_manual(values=PAL) +
    # scale_y_continuous(breaks = seq(1,6,.25)) +
    scale_color_manual(values = RColorBrewer::brewer.pal(8,'Set1')[-(5:7)]#,
                       # labels =c('xAM',
                       #           'xAM + G×E',
                       #           'xAM + VT',
                       #           'xAM + G×E + VT')
                     ) +
    scale_linetype_manual(values = c(2:6))+

    ylab(expression(over('Empirical Type-I Error Rate','Theoretical Type-I Error Rate'))) +
    #theme(axis.text.y.right = element_text(color=PAL[3]),
    #     axis.ticks.y.right=element_line(color=PAL[3])) +
    xlab('Generations of xAM') +
     theme(axis.title.x=element_blank())+
     theme(legend.position=c(.5,.9), legend.direction='horizontal', legend.key.width=unit(1,'cm'))
 )

###############################################################################
## Cell 70: Final assembly
###############################################################################
options(repr.plot.width=12,
        repr.plot.height=14)
options(repr.plot.res=300)
new_schem <- ggdraw() + draw_image(magick::image_read(file.path(BASE_DIR, 'data/cdiags/colored new cdiags complexity.png')))
suppressWarnings({
    plts <- align_plots(plot_uni_multi + guides(color=guide_none(),
                        linetype=guide_none(),shape=guide_none()) +
                            theme(axis.title.x.bottom = element_blank()),
                        plot_biv_multi + guides(color=guide_none(),
                        linetype=guide_none(),shape=guide_none()),align = 'v',axis = 'rl')

fig3 <- plot_grid(new_schem, rel_widths=c(2.5,6),
          plot_grid(plts[[1]],plts[[2]],
                    plot_gwas2x5_alt + theme(axis.title.y=element_text(size=10) )+
                                         guides(color=guide_none(), shape = guide_none(),
                        linetype=guide_none()),ncol=1,
                   labels=c('b','c','d')),
         labels=c('a',''))
})

###############################################################################
## Save outputs (dimensions from Cell 70: repr.plot.width=14, repr.plot.height=8)
###############################################################################
ggsave(file.path(FIG_DIR, "fig3_complexity.pdf"),
       fig3, width = 14, height = 8)
ggsave(file.path(FIG_DIR, "fig3_complexity.png"),
       fig3, width = 14, height = 8, dpi = 300, bg = "white")

cat("Saved fig3_complexity.pdf and fig3_complexity.png to", FIG_DIR, "\n")
cat("\n=== plot_fig3.R complete ===\n")
