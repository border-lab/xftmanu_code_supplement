## plot_fig3.R
## Direct port of manu/figure_nb/mFigComplexity.ipynb
## Only changes: file paths adjusted for round4 layout, ggsave at end.
##
## Cells used: 3, 20, 25, 38, 39, 41, 49, 53, 61, 63, 65, 72

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
FIG_DIR  <- file.path(BASE_DIR, "figures_output")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

###############################################################################
## Cell 3
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
## Cell 20
###############################################################################
gdat <- res
gdat$h2_he <- gdat$he_h2

molt <-
reshape2::melt(gdat, id.vars = c('gen','X','seed','args_m_causal','scenario'),
               measure.vars = c("h2_he", "h2_true"))
# gdat

###############################################################################
## Cell 25
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
## Cell 38
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
## Cell 39
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
(plot_biv_multi <- ggplot(pdat[pdat$scenario %in% c(#"RM + VT",
                                                          "5xAM",
                                                          "5xAM + GxE",
                                                          "5xAM + VT",
                                                          "5xAM + VT + GxE") & !is.na(pdat$f1),],
                                aes(gen,value, color=scenario, linetype=scenario)) + default_theme2 +
    geom_hline(aes(yintercept = int), color=PAL[5], lty=1) +
    stat_summary(geom = 'linerange',position=position_dodge(width=.3),
                 fun.max = function(x) quantile(x,.9),
                 fun.min = function(x) quantile(x,.1),
                 fun = function(x) quantile(x,.5)) +
    # stat_summary(data=utmp2,
                 # geom = 'point',position=position_dodge(width=.3),
                 # fun = function(x) quantile(x,.5)) +
    stat_smooth(method='gam', formula = y~bs(x, knots=0:5, degree = 1), se=F) +
    ylab(expression(italic(r)[g])) +
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
## Cell 41
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
(plot_uni_multi <- ggplot(pdat[pdat$scenario %in% c(#"RM + VT",
                                                          "5xAM",
                                                          "5xAM + GxE",
                                                          "5xAM + VT",
                                                          "5xAM + VT + GxE") & !is.na(pdat$f1),],
                                aes(gen,value, color=scenario, linetype=scenario)) + default_theme2 +
    geom_hline(aes(yintercept = int), color=PAL[5], lty=1) +
    stat_summary(geom = 'linerange',position=position_dodge(width=.3),
                 fun.max = function(x) quantile(x,.9),
                 fun.min = function(x) quantile(x,.1),
                 fun = function(x) quantile(x,.5)) +
    # stat_summary(data=utmp2,
                 # geom = 'point',position=position_dodge(width=.3),
                 # fun = function(x) quantile(x,.5)) +
    stat_smooth(method='gam', formula = y~bs(x, knots=0:5, degree = 1), se=F) +
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
## Cell 49
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
## Cell 53
###############################################################################
# mdat <- reshape2::dcast(molten,seed+gen+args_m+args_n+args_kmate +alpha +GWAS+alpha_bonferoni~ quantity)

# mdat$T1R <- mdat$FalsePositive/(.8)
# mdat$hits <- (mdat$FalsePositive + mdat$TruePositive)
options(repr.plot.width=8)
options(repr.plot.height=6)
unique(tmpd$scenario)
(plot_gwas2x5_alt <-
 ggplot(tmpd[tmpd$GWAS=='Population'&tmpd$scenario %in% c(#"RM + VT",
                                                          "5xAM",
                                                          "5xAM + GxE",
                                                          "5xAM + VT",
                                                          "5xAM + VT + GxE"),],
       aes(gen,relative_T1R, color=scenario, lty=scenario,shape=scenario)) + default_theme+
    # geom_alpha_bonferoni*2))+
    # geom_jitter(height = 1)+
     stat_summary(geom='point',  position=position_dodge(width=.1),
                 fun.data = function(x) mean_se(x,mult=1.96)
                 ) +
     stat_summary(geom='linerange',  position=position_dodge(width=.1),
                 fun.data =  function(x) mean_se(x,mult=1.96)
                 ) +
    geom_smooth(formula=y~ns(x,df =3),method='gam', aes(gen, color=scenario, lty=scenario), se=F ) +
    # geom_smooth(formula=y~bs(x,degree =3),method='gam', aes(gen, color=GWAS, lty=xAM), se=F ) +
    # stat_summary(geom='line', aes(gen,  color=GWAS, lty=xAM), fun = mean ) +
    facet_grid(~power)+ #, labeller = label_parsed) +
    geom_hline(color='grey', lty=3,yintercept = 1) +
    # scale_color_manual(values=PAL) +
    # scale_y_continuous(breaks = seq(1,6,.25)) +
    geom_hline(color='grey', lty=3,yintercept = 1) +
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
## Cell 61
###############################################################################
#### legends
utmp <- gdat2[gdat2$scenario%in%c("5xAM", "5xAM + GxE", "5xAM + VT", "5xAM + VT + GxE"),]

PAL <- RColorBrewer::brewer.pal(8,'Set1')
p5xAM <-
ggplot(uu <- utmp[utmp$scenario=='5xAM',],aes(gen,value, color=scenario, lty=scenario)) + default_theme +
    geom_hline(col=PAL[5], yintercept = 0) +
    stat_summary(geom = 'linerange',position=position_dodge(width=.3))+
    stat_summary(data=uu,geom = 'point',position=position_dodge(width=.3), fun = function(x) quantile(x,.5)) +
    stat_smooth(method='gam', formula = y~bs(x, knots=0:5, degree = 1), se=F) +
    scale_color_manual(values = PAL,
                       labels =c('5xAM'))+
    scale_linetype_manual(values = c(2:6),
                       labels =c('5xAM'))

p5xAMgxe <-
ggplot(uu <- utmp[utmp$scenario=='5xAM + GxE',],aes(gen,value, color=scenario, lty=scenario)) + default_theme +
    stat_summary(geom = 'linerange',position=position_dodge(width=.3))+
    stat_summary(data=uu,geom = 'point',position=position_dodge(width=.3), fun = function(x) quantile(x,.5)) +
    stat_smooth(method='gam', formula = y~bs(x, knots=0:5, degree = 1), se=F) +
    scale_color_manual(values = PAL[2:5],
                       labels =c('5xAM + G×E'))+
    scale_linetype_manual(values = c(3:6),
                       labels =c('5xAM + G×E'))

p5xAMpVTgxe <-
ggplot(uu <- utmp[utmp$scenario=='5xAM + VT + GxE',],aes(gen,value, color=scenario, lty=scenario)) + default_theme +
    stat_summary(geom = 'linerange',position=position_dodge(width=.3))+
    stat_summary(data=uu,geom = 'point',position=position_dodge(width=.3), fun = function(x) quantile(x,.5)) +
    stat_smooth(method='gam', formula = y~bs(x, knots=0:5, degree = 1), se=F) +
    scale_color_manual(values = PAL[4:5],
                       labels =c('5xAM + G×E + VT'))+
    scale_linetype_manual(values = c(5:6),
                       labels =c('5xAM + G×E + VT'))
p5xAMpVT <-
ggplot(uu <- utmp[utmp$scenario=='5xAM + VT',],aes(gen,value, color=scenario, lty=scenario)) + default_theme +
    stat_summary(geom = 'linerange',position=position_dodge(width=.3))+
    stat_summary(data=uu,geom = 'point',position=position_dodge(width=.3), fun = function(x) quantile(x,.5)) +
    stat_smooth(method='gam', formula = y~bs(x, knots=0:5, degree = 1), se=F) +
    scale_color_manual(values = PAL[3:5],
                       labels =c('5xAM + VT',
                                 '5xAM + G×E + VT'))+
    scale_linetype_manual(values = c(4:6),
                       labels =c('5xAM + VT',
                                 '5xAM + G×E + VT'))

###############################################################################
## Cell 63
###############################################################################
options(repr.plot.res=500)
s2xAM <- magick::image_read_svg(file.path(BASE_DIR, 'data/cdiags/cropped_diag2xAM.svg'))
s5xAM <- magick::image_read_svg(file.path(BASE_DIR, 'data/cdiags/cropped_diag5xAM.svg'),width = 2000)
s5xAMgXe <- magick::image_read_svg(file.path(BASE_DIR, 'data/cdiags/cropped_diag5xAMgXE.svg'),width = 2000)
# s5xAMeVT <- magick::image_read_svg('cdiags/diag5xAMeVT.svg',width = 2000)
# s5xAMeVTgXe <- magick::image_read_svg('cdiags/diag5xAMgXEeVT.svg',width = 2000)
s5xAMpVT <- magick::image_read_svg(file.path(BASE_DIR, 'data/cdiags/cropped_diag5xAMpVT.svg'),width = 2000)
s5xAMpVTgXe <- magick::image_read_svg(file.path(BASE_DIR, 'data/cdiags/cropped_diag5xAMgXEpVT.svg'),width = 2000)
# (plot_schematics_multiverse  <- plot_grid(#ggdraw() + draw_image(s2xAM),
#                                           ggdraw() + draw_image(s5xAM),
#                                           ggdraw() + draw_image(s5xAMgXe),
#                                           # ggdraw() + draw_image(s5xAMeVT),
#                                           # ggdraw() + draw_image(s5xAMeVTgXe),
#                                           ggdraw() + draw_image(s5xAMpVT),
#                                           ggdraw() + draw_image(s5xAMpVTgXe),
#          ncol=1)
# )
(plot_schematics_multiverse_row  <- plot_grid(#ggdraw() + draw_image(s2xAM,),
                                          ggdraw() + draw_image(s5xAM),
                                          ggdraw() + draw_image(s5xAMgXe),
                                          # ggdraw() + draw_image(s5xAMeVT),
                                          # ggdraw() + draw_image(s5xAMeVTgXe),
                                          ggdraw() + draw_image(s5xAMpVT),
                                          ggdraw() + draw_image(s5xAMpVTgXe),
         nrow=1)
)


###############################################################################
## Cell 65
###############################################################################
lsize=12
lkh=.75
lkw=2
rh=1
sp=.01

options(repr.plot.width=5,
        repr.plot.height=2)
l1 = plot_grid( schem1 <- ggdraw() + draw_image(s5xAM,clip = T), NULL,
          leg1 <- get_legend(p5xAM+guides(color=guide_legend(label.position='top'),
                                          linetype=guide_legend(label.position='top'))+
                             theme(legend.justification = .5,legend.position='left',legend.title = element_blank(), legend.key.width = unit(lkw,'cm'),legend.key.height = unit(lkh,'cm'),text=element_text(size=lsize))),
          ncol=2, rel_widths=c(5,sp,rh), nrow=1)
l2 = plot_grid(hjust = 0, schem2 <- ggdraw() + draw_image(s5xAMgXe), NULL,
          leg2 <- get_legend(p5xAMgxe+guides(color=guide_legend(label.position='top'),
                                             linetype=guide_legend(label.position='top'))+
                             theme(legend.justification = .5,legend.position='left',legend.title = element_blank(), legend.key.width = unit(lkw,'cm'),legend.key.height = unit(lkh,'cm'),text=element_text(size=lsize))),
          ncol=2, rel_widths=c(5,sp,rh), nrow=1)
l3 = plot_grid(hjust = 0, schem3 <- ggdraw() + draw_image(s5xAMpVT), NULL,
          leg3 <- get_legend(p5xAMpVT+guides(color=guide_legend(label.position='top'),
                                             linetype=guide_legend(label.position='top'))+
                             theme(legend.justification = .5,legend.position='left',legend.title = element_blank(), legend.key.width = unit(lkw,'cm'),legend.key.height = unit(lkh,'cm'),text=element_text(size=lsize))),
          ncol=2, rel_widths=c(5,sp,rh), nrow=1)
l4 = plot_grid(hjust = 0, schem4 <- ggdraw() + draw_image(s5xAMpVTgXe), NULL,
          leg4 <- get_legend(p5xAMpVTgxe+guides(color=guide_legend(label.position='top'),
                                                linetype=guide_legend(label.position='top'))+
                             theme(legend.justification = .5,legend.position='left',legend.title = element_blank(), legend.key.width = unit(lkw,'cm'),legend.key.height = unit(lkh,'cm'),text=element_text(size=lsize))),
          ncol=2, rel_widths=c(5,sp,rh), nrow=1)
# plot_grid(l1,l2,l3,l4, ncol=1)
# plot_grid(l1,l2,l3,l4, ncol=1,rel_widths = c(3,.5))
# plot_grid(schem1,leg1, rel_widths=c(9,2))
plot_grid(schem1,schem2,schem3,schem4,leg1,leg2,leg3,leg4, rel_widths=c(9,2), byrow = F, ncol=2)

###############################################################################
## Cell 72
###############################################################################
options(repr.plot.width=12,
        repr.plot.height=12)

suppressWarnings({
    plts <- align_plots(plot_uni_multi + guides(color=guide_none(),
                        linetype=guide_none()) +
                            theme(axis.title.x.bottom = element_blank()),
                        plot_biv_multi + guides(color=guide_none(),
                        linetype=guide_none()),align = 'v',axis = 'rl')
    # plot_grid(rel_heights = c(2,8,8),
    #           ggdraw(get_legend(plot_uni_multi +
    #                             theme(legend.title = element_blank(),
    #                                   legend.key.width = unit(1.5,'cm'),
    #                                   legend.position = 'top'))),
    #           plts[[1]],plts[[2]], ncol=1
    #         )



# plot_biv_multi


# plot_schematics_multiverse
# plot_gwas2x5 + default_theme2
st1 <- plot_grid(rel_heights =c(1,1),# c(2,8,8),
              # ggdraw(get_legend(plot_uni_multi +
              #                   theme(legend.title = element_blank(),
              #                         legend.key.width = unit(1.5,'cm'),
              #                         legend.position = 'top'))),
              plts[[1]],plts[[2]], ncol=1, labels=c('b','c')
            )
st2 <- plot_grid(rel_heights = c(8,8),

              plts[[1]],plts[[2]], ncol=1
            )
options(repr.plot.width=13)
options(repr.plot.height=8)
fig3 <- plot_grid(#plot_grid(hjust = 0, l1,l2,l3,l4, ncol=1,rel_widths = c(4,.5), labels=c('a','','','')),nrow=1, NULL,
          plot_grid(schem1,schem2,schem3,schem4,leg1,leg2,leg3,leg4, rel_widths=c(9,2), byrow = F, ncol=2, labels=c('a',rep('',7))), nrow=1, NULL,
          plot_grid(hjust = 0,rel_heights=c(6,3),labels=c('','d'),st1,plot_gwas2x5_alt+ theme(legend.direction = 'vertical',text=element_text(size=13),
                                                                                              legend.position = c(.18,.7)) +
                        guides(shape=guide_none(), linetype=guide_none(),
                                                 color=guide_none()),nrow=2), rel_widths = c(.85,.05,1.1))
# plot_grid(plot_grid(l1,l2,l3,l4, nrow=1),
#         # ggdraw(get_legend(plot_uni_multi +
#         #         theme(legend.spacing.x = unit(1,'cm'),
#         #               legend.title = element_blank(),
#         #               legend.key.width = unit(1.5,'cm'),
#         #               legend.position = 'top'))),
#           plot_grid(st2,plot_gwas2x5, nrow=1), ncol=1,
#         rel_heights=c(4,4))
    })

###############################################################################
## Save outputs (dimensions from Cell 72: repr.plot.width=13, repr.plot.height=8)
###############################################################################
ggsave(file.path(FIG_DIR, "fig3_complexity.pdf"),
       fig3, width = 13, height = 8)
ggsave(file.path(FIG_DIR, "fig3_complexity.png"),
       fig3, width = 13, height = 8, dpi = 300, bg = "white")

cat("Saved fig3_complexity.pdf and fig3_complexity.png to", FIG_DIR, "\n")
cat("\n=== plot_fig3.R complete ===\n")
