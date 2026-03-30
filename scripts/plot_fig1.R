## plot_fig1.R
## Verbatim code from resub_mFigHDxAM.ipynb with path substitutions only
## Generates 6-panel Figure 1: a (UKB CCA), b (NHIRD CCA), c (h2), d (rg), e (GWAS FP), f (newhits)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
DATA_DIR <- file.path(BASE_DIR, "data")
FIGURES_DIR <- file.path(BASE_DIR, "figures_output")
dir.create(FIGURES_DIR, showWarnings = FALSE, recursive = TRUE)

###############################################################################
## Cell 1: libraries
###############################################################################
library(yacca)
library(repr)
library(ggplot2)
library(reshape2)
library(cowplot)
library(stringr)

# res <- do.call(rbind.data.frame,
#                lapply(list.files('gres', pattern = "csv$", full.names = T), read.csv))
# res$sim = apply(str_split_fixed(res$X, '_',5)[,-2], 1, paste, collapse='_')

###############################################################################
## Cell 3: default theme
###############################################################################
default_theme <- theme_bw() + theme(text=element_text(size=14))

###############################################################################
## Cell 4: load CCA data
## Original: load('~/data/cca/mice_imputed_cca_final.rdata', verbose=T)
## imp_cca.Rdata contains 'out' (single imputation); wrap as lcca for compat
###############################################################################
load(file.path(DATA_DIR, 'cca', 'imp_cca.Rdata'), verbose=T)
lcca <- list(out)
out <- lcca[[1]]

# str(out)

###############################################################################
## Cell 5: ccdat (hardcoded complete-case canonical correlations)
###############################################################################
ccdat = c(`CV 1` = 0.367035175299488, `CV 2` = 0.517038565488573, `CV 3` = 0.601984317923847,
`CV 4` = 0.697522369281801, `CV 5` = 0.823705718371409, `CV 6` = 0.847231186585211,
`CV 7` = 0.882527689139077, `CV 8` = 0.899659822643449, `CV 9` = 0.911256123643412,
`CV 10` = 0.924392948857031, `CV 11` = 0.93651589132108, `CV 12` = 0.945772218124514,
`CV 13` = 0.949665007373801, `CV 14` = 0.956550832180996, `CV 15` = 0.963552498585247,
`CV 16` = 0.970755140854732, `CV 17` = 0.977729335609696, `CV 18` = 0.983474327171582,
`CV 19` = 0.986492487623242, `CV 20` = 0.988608639968566, `CV 21` = 0.990737600100733,
`CV 22` = 0.993414303176453, `CV 23` = 0.995491432537665, `CV 24` = 0.997001889358986,
`CV 25` = 0.997610775141301, `CV 26` = 0.998155194073544, `CV 27` = 0.998834015621939,
`CV 28` = 0.999256378883214, `CV 29` = 0.999488082189504, `CV 30` = 0.999743284993516,
`CV 31` = 0.999872313961061, `CV 32` = 0.99996705028616, `CV 33` = 0.999995450042255,
`CV 34` = 1)

###############################################################################
## Cell 6
###############################################################################
out <- lcca[[1]]
    tmp  = out$ycrosscorr
# tmp = rbind(tmp, rep(NA,ncol(tmp)))
# rownames(tmp)[nrow(tmp)] <- 'age_at_menarche_mean'
tmp2 <- out$xcrosscorr



###############################################################################
## Cell 8: cca_vars
###############################################################################
cca_vars <-
apply(sapply(lcca,  function(out) {
    tmp  = out$ycrosscorr
tmp = rbind(tmp, rep(NA,ncol(tmp)))
rownames(tmp)[nrow(tmp)] <- 'age_at_menarche_mean'
tmp2 <- out$xcrosscorr
    rownames(tmp2) <- rownames(tmp); colnames(tmp2) <- colnames(tmp)


xdat <- (tmp2 + tmp)/2
xdat[is.na(xdat)] <- tmp2[is.na(xdat)]
# xdat
xdat <- (xdat)
cca_vars <- (out$xvrd/ out$xrd + out$yvrd/ out$yrd)/2
    cca_vars
}), 1, min)
plot(cumsum(cca_vars))

###############################################################################
## Cell 18: dat processing (CCA loadings + redundancy)
###############################################################################
# ?melt.array
dat <- lapply(lcca, function(out) {
tmp  = out$ycrosscorr
tmp = rbind(tmp, rep(NA,ncol(tmp)))

rownames(tmp)[nrow(tmp)] <- 'age_at_menarche_mean'
tmp2 <- out$xcrosscorr
# tmp2 <- tmp2[rownames(tmp), colnames(tmp)]
tmp  = out$ycrosscorr
tmp = rbind(tmp, rep(NA,ncol(tmp)))
rownames(tmp)[nrow(tmp)] <- 'age_at_menarche_mean'
tmp2 <- out$xcrosscorr
    # rownames(tmp2) <- rownames(tmp); colnames(tmp2) <- colnames(tmp)



xdat <- (tmp2 + tmp)/2
xdat[is.na(xdat)] <- tmp2[is.na(xdat)]
# xdat
xdat <- (xdat)
cca_vars <- (out$xvrd/ out$xrd + out$yvrd/ out$yrd)/2


KK=25
KK <- KK<<-14

zz=0
xdat2 <- (xdat[apply(abs(xdat),1,function(x) any(x>.05)),])
pdat <- melt(xdat2[,1:KK],varnames = c('Phenotype','CanonicalVariate'), value.name='Canonical cross-loadings')
zz=0
xdat2 <- (xdat[apply(abs(xdat),1,function(x) any(x>.05)),])
pdat <- melt(xdat2[,1:KK],varnames = c('Phenotype','CanonicalVariate'), value.name='Canonical cross-loadings')



pdat <- pdat[apply(pdat,1,function(x) any(x>.5)),]

pdat$Phenotype <- factor(pdat$Phenotype,
                         levels=names(sort(tapply(abs(pdat$`Canonical cross-loadings`),
                                                  pdat['Phenotype'],function(x) sum(abs(x))))))
pdat$CanonicalVariate = factor(pdat$CanonicalVariate, levels=c('',paste('CV', 1:KK)))
d2 = data.frame(CanonicalVariate=factor(c('',paste('CV', 1:(KK-zz))), levels=c('',paste('CV', 1:(KK-zz)))))
d2$`Relative Canonical\nRedundancy`<-cumsum(c(0,cca_vars[1:(KK-zz)]))
list(pdat,d2)})

###############################################################################
## Cell 19: d2 from dat
###############################################################################
d2 <- do.call(rbind.data.frame, mapply(function(X,i) {X[[2]]$iterate <- i;X[[2]]},
                                         dat, 1:length(dat),SIMPLIFY = F))

###############################################################################
## Cell 20: aggregate + complete cases
###############################################################################
d2 <- aggregate(d2[2], d2[1], median)
d4 <- d3 <- within(d2, Sample <- 'Imputed')
d4[[2]] <- c(0, ccdat[1:(nrow(d3)-1)])
d4$Sample <- 'Complete cases'
d2 <- rbind.data.frame(d3, d4)

###############################################################################
## Cell 22: cv column
###############################################################################
d2$cv <- rep(0:KK,2)


###############################################################################
## Cell 23: sort transform for complete cases
###############################################################################
d2[d2$Sample=='Complete cases', 2] <- c(0,cumsum(sort(diff(c(d2[d2$Sample=='Complete cases', 2])),decreasing = T)))

###############################################################################
## Cell 24: plot_cca_var (all three versions from notebook)
###############################################################################
KK=14

(splot_cca_var <- ggplot(d2, aes(cv, `Relative Canonical\nRedundancy`,
                               linetype=Sample, shape = Sample),
                       ) +
                                                  scale_y_continuous(position = 'left',limits = 0:1)+
                                                  scale_x_continuous(breaks=0:KK,
                                                                     labels=d2$CanonicalVariate[1:(KK+1)],position = 'bottom')+
                                                  geom_hline(col='purple',lty=4,yintercept = .9)+
                                                  geom_hline(col='darkorange',lty=4,yintercept = .95)+
 scale_linetype_manual(values=2:3)+
 xlab('Canonical Variate') +
    geom_point() + geom_line() +
    theme_bw() + theme(#axis.title.y.left = element_blank(),axis.title.x = element_blank(),
                       text=element_text(size=14), legend.position = 'bottom') + theme(legend.key.width=unit(1,'cm')))


(splot_cca_var <- ggplot(d2[1:13,], aes(cv, `Relative Canonical\nRedundancy`),
                       ) +
                                                  scale_y_continuous(position = 'left',limits = 0:1)+
                                                  scale_x_continuous(breaks=0:KK,
                                                                     labels=d2$CanonicalVariate[1:(KK+1)],position = 'bottom')+
                                                  geom_hline(col='purple',lty=3,yintercept = .9)+
                                                  geom_hline(col='darkorange',lty=3,yintercept = .95)+
 xlab('Canonical Variate') +
    geom_point() + geom_line() +
    theme_bw() + theme(#axis.title.y.left = element_blank(),axis.title.x = element_blank(),
                       text=element_text(size=14), legend.position = 'bottom'))


(plot_cca_var <- ggplot(d2[d2$Sample=='Complete cases',], aes(cv, `Relative Canonical\nRedundancy`),
                       ) +
                                                  scale_y_continuous(position = 'left',limits = 0:1)+
                                                  scale_x_continuous(breaks=0:KK,
                                                                     labels=gsub(' ','',d2$CanonicalVariate[1:(KK+1)]),position = 'bottom')+
                                                  geom_hline(col='purple',lty=3,yintercept = .9)+
                                                  geom_hline(col='darkorange',lty=3,yintercept = .95)+
 xlab('Canonical Variate') +
    geom_point() + geom_line() +
    theme_bw() + theme(#axis.title.y.left = element_blank(),axis.title.x = element_blank(),
                       text=element_text(size=14), legend.position = 'bottom'))



###############################################################################
## Cell 26: pdat aggregation
###############################################################################
pdat <- do.call(rbind.data.frame, mapply(function(X,i) {X[[1]]$iterate <- i;X[[1]]},
                                         dat, 1:length(dat),SIMPLIFY = F))
pdat <- aggregate(pdat[3], pdat[1:2], median)


###############################################################################
## Cell 28: Taiwan CCA (tcca + plot_tcca_var)
###############################################################################

tcca <- data.frame(CanonicalVariate = c('', paste0('CV',1:11)),
                   cv=0:11)
tcca$`Relative Canonical\nRedundancy`=
                   c(0, 0.261017159845987, 0.404835764140372, 0.532464280704279, 0.643260247888746,
0.747218762552395, 0.848773522271279, 0.933659192221925, 0.975194233960818,
0.98982819870675, 0.999031285792778, 1)
KK =8
tcca <- tcca[1:9,]
(plot_tcca_var <- ggplot(tcca, aes(cv, `Relative Canonical\nRedundancy`),
                       ) +
                                                  scale_y_continuous(position = 'left',limits = 0:1)+
                                                  scale_x_continuous(breaks=0:KK,
                                                                     labels=tcca$CanonicalVariate[1:(KK+1)],position = 'bottom')+
                                                  geom_hline(col='purple',lty=3,yintercept = .9)+
                                                  geom_hline(col='darkorange',lty=3,yintercept = .95)+
 xlab('Canonical Variate') +
 ylab('') +
    geom_point() + geom_line() +
    theme_bw() + theme(#axis.title.y.left = element_blank(),axis.title.x = element_blank(),
                       text=element_text(size=14), legend.position = 'bottom'))

plot_tcca_var <- plot_tcca_var + theme(plot.margin=unit(c(0,0,0,0),'cm'))

###############################################################################
## Cell 31: pdat2 cleanup
###############################################################################
pdat2 <- pdat
pdat2$Phenotype <- gsub('_earliest','',pdat2$Phenotype)
pdat2$Phenotype <- gsub('_mean','',pdat2$Phenotype)
pdat2$Phenotype <- gsub('_',' ',pdat2$Phenotype)


###############################################################################
## Cell 35: plot_cca_loadings (not used in final figure, but included)
###############################################################################
pdat3 <- pdat2
pdat3 <- pdat3[!grepl('left',pdat3$Phenotype),]

ymax = max(pdat2$`Canonical cross-loadings`)
ymin = min(pdat2$`Canonical cross-loadings`)

(plot_cca_loadings <- ggplot(pdat3, aes( CanonicalVariate, Phenotype,fill=`Canonical cross-loadings`)) +
    geom_tile(color='grey', lwd=.2) + #scale_fill_distiller(palette = 'Spectral') +
                   scale_fill_gradientn(values = c(seq(0,abs(ymin) / (ymax - ymin),length.out = 5),
                                                   seq(abs(ymin) / (ymax - ymin),1,length.out=3)),

                                        ,colors=c(RColorBrewer::brewer.pal(7,'Spectral')[1:4],'#F0F8F8',RColorBrewer::brewer.pal(7,'Spectral')[6:7]))+
    theme_bw() + theme(panel.grid = element_blank(), text=element_text(size=14), legend.position = 'top'))

###############################################################################
## Cell 37: load sim results (2xAM vs 5xAM)
## Original: res <- read.csv('../merged_tabla_redux_results_011024.csv')
###############################################################################
# library(yacca)
library(repr)
library(ggplot2)
library(reshape2)
library(cowplot)
library(stringr)
library(splines)


res <- read.csv(file.path(DATA_DIR, 'sim_results', 'merged_tabla_redux_results_011024.csv')) ## from cleanAllGeneralSims2024

res <- res[res$scenario %in% c('2xAM', '5xAM'),]#,'RM'),]

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



###############################################################################
## Cell 40: quantile_range function
###############################################################################
quantile_range <-
function (x, q = .1)
{
    x <- stats::na.omit(x)
    med <- median(x)
    lower <- quantile(x, q)
    upper <- quantile(x, 1-q)
    data.frame(y = med, ymin = lower, ymax = upper,
        .size = 1)
}

# ggplot(res, aes(as.factor(gen), pgwas_false_positives_0.05/(0.05),
#                 col=as.factor(power))) + facet_wrap(~scenario)+
#     stat_summary(geom='linerange', fun.data = mean_se) + stat_summary(geom='point', fun = mean) +
#     stat_summary(geom='line', aes(as.factor(gen), group=as.factor(power)), fun = mean )

###############################################################################
## Cell 43: melt udat and bdat
###############################################################################
udat <- reshape2::melt(res, id.vars = idvars, measure.vars = mvars_uni)
bdat <- reshape2::melt(res, id.vars = idvars, measure.vars = mvars_biv)

###############################################################################
## Cell 51: plot_h2_2x5
###############################################################################
PAL = RColorBrewer::brewer.pal(3,'Set1')
udat$xAM <- '2-variate xAM'
udat$xAM[udat$args_kmate==5] <- '5-variate xAM'
udat$Quantity <- factor(udat$variable, levels = c('he_h2','h2_true'))

utmp <- udat
utmp$Quantity <- factor(utmp$variable, levels=c('he_h2','h2_true','z'))
utmp<-rbind.data.frame(utmp[!duplicated(udat$gen),], udat)
# utmp[1,] <- NA
utmp$Quantity[1:6] <- 'z'
utmp$value[1:6] <- .5
utmp2 <- utmp
# utmp2$value[1] <- 0#NA
# utmp2
(plot_h2_2x5  <- ggplot(utmp2, aes(gen,value, color=Quantity, lty=xAM, shape=xAM)) + default_theme +
    geom_segment(y = .5, x=0, yend=.5, xend=5, col=PAL[3], lty=1) +
    stat_summary(geom = 'linerange',position=position_dodge(width=.1),
                 fun.data =  function(x) mean_se(x,mult=1.96)) +
    stat_summary(data=utmp2,
                 geom = 'point', position=position_dodge(width=.1),
                                  fun.data =  function(x) mean_se(x,mult=1.96))+

    stat_summary(geom='line', aes(gen,  color=Quantity), fun = mean ) +

    ylab(expression(italic(h)^2)) +
    xlab('Generations of xAM')    +
    scale_color_manual(values=PAL, labels =c(expression(hat(italic(h))[HE]^2),
                                             expression(italic(h)[italic(t)]^2),
                                             expression(italic(h)[0]^2))))

###############################################################################
## Cell 55: plot_rg_2x5
###############################################################################
bdat$xAM <- '2-variate xAM'
bdat$xAM[bdat$args_kmate==5] <- '5-variate xAM'
bdat$Quantity <- bdat$variable
bdat$Quantity <- factor(bdat$variable, levels = c('he_rg','rg_true','z'))
btmp <- rbind.data.frame(bdat[!duplicated(bdat$gen),],bdat)

# utmp[1,] <- NA
btmp$Quantity[1:6] <- 'z'
btmp$value[1:6] <- 0
btmp2 <- btmp
# btmp2$value[1] <- NA

(plot_rg_2x5  <- ggplot(btmp2, aes(gen,value, color=Quantity, lty=xAM, shape=xAM)) + default_theme +
    geom_segment(y = 0, x=0, yend=0,xend=5, col=PAL[3], lty=1) +
    stat_summary(geom = 'linerange',position=position_dodge(width=.1),
                 fun.data =  function(x) mean_se(x,mult=1.96)) +
    stat_summary(data=btmp2,
                 geom = 'point', position=position_dodge(width=.1),
                                  fun.data =  function(x) mean_se(x,mult=1.96))+

    stat_summary(geom='line', aes(gen,  color=Quantity), fun = mean ) +
    # scale_linetype_manual(values=2:3) +
    ylab(expression(italic(r[g]))) +
    xlab('Generations of xAM')    +
    scale_color_manual(values=PAL, labels =c(
                                             expression(hat(italic(r))[HE]),
    expression(italic(r)[score]), expression(italic(r)['𝛽'])))+
    guides(color=guide_legend(order=2), linetype=guide_legend(order=1), shape=guide_legend(order=1)))
                 #

###############################################################################
## Cell 58: plot_2x5 (intermediate multipane, used in some assembly paths)
###############################################################################
ltheme = theme(legend.position = c(.1,.7),
               legend.title = element_blank(),
               legend.key.width=unit(1,'cm'))


suppressWarnings({
    leg <- get_legend(plot_h2_2x5+scale_linetype_discrete()+guides(color=guide_none()) +
                  theme(legend.key.width=unit(1,'cm'),legend.position='top'))
    a1 <- plot_h2_2x5 + guides(linetype='none') + ltheme + theme(axis.title.x = element_blank())
    a2 <- plot_rg_2x5 + guides(linetype='none') + ltheme + theme(legend.position = c(.1,.8))
    aa <- align_plots(a1,a2, axis='rl',align = 'v')
(    plot_2x5 <- plot_grid(rel_heights = c(1,4,7),
              leg,
              aa[[1]],
              aa[[2]],
              ncol=1))
    })

###############################################################################
## Cell 60: reload sim results for GWAS FP (same CSV, re-filter)
## Original: res <- read.csv('../merged_tabla_redux_results_011024.csv')
###############################################################################
# library(yacca)
library(repr)
library(ggplot2)
library(reshape2)
library(cowplot)
library(stringr)
library(splines)


res <- read.csv(file.path(DATA_DIR, 'sim_results', 'merged_tabla_redux_results_011024.csv')) ## from cleanAllGeneralSims2024

res <- res[res$scenario %in% c('2xAM', '5xAM'),]#,'RM'),]

###############################################################################
## Cell 61: PAL
###############################################################################
PAL = RColorBrewer::brewer.pal(3,'Set1')
# names(res)

###############################################################################
## Cell 62: mdat for GWAS FP
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



###############################################################################
## Cell 65: tmpd
###############################################################################
tmpd <- mdat #within(mdat[mdat$GWAS=='Population',], {
#                # T1R[T1R==0] <- (5e-8)*2
#                relative_T1R <- T1R/alpha_bonferoni})
# tmpd$`over(n,m)` = as.factor(1/.8*tmpd$args_n/tmpd$args_m)
# tmpd$novm = paste0('italic(n)/italic(m) == ',tmpd$`over(n,m)`)
# tmpd$novm <- factor(tmpd$novm, levels = levels(as.factor(tmpd$novm))[c(2,1,3)])
tmpd$xAM <-'2-variate'
tmpd$xAM[tmpd$args_kmate==5] <-'5-variate'
tmpd$power = paste(tmpd$power, 'power at 𝛼=0.05')

###############################################################################
## Cell 67: plot_gwas2x5_alt + plot_gwas2x5_alt_v
###############################################################################
# mdat <- reshape2::dcast(molten,seed+gen+args_m+args_n+args_kmate +alpha +GWAS+alpha_bonferoni~ quantity)

# mdat$T1R <- mdat$FalsePositive/(.8)
# mdat$hits <- (mdat$FalsePositive + mdat$TruePositive)
(plot_gwas2x5_alt <-
 ggplot(tmpd[tmpd$GWAS=='Population',],
       aes(gen,relative_T1R, color=xAM,shape=xAM)) + default_theme+
    # geom_alpha_bonferoni*2))+
    # geom_jitter(height = 1)+
     stat_summary(geom='path',  position=position_dodge(width=.1),
                 fun=median
                 ) +
     stat_summary(geom='point',  position=position_dodge(width=.1),
                 fun=median
                 ) +
    # stat(formula=y~ns(x,df =2),method='gam', aes(gen, color=GWAS, lty=xAM), se=F ) +
    # geom_smooth(formula=y~bs(x,degree =3),method='gam', aes(gen, color=GWAS, lty=xAM), se=F ) +
    # stat_summary(geom='path',fun = median, aes(gen,  color=GWAS, lty=xAM)) +
    facet_grid(~power)+ #, labeller = label_parsed) +
    geom_hline(color='grey', lty=3,yintercept = 1) +
    scale_color_manual(values=PAL) +
    scale_y_continuous(breaks = seq(1,2,.25)) +
    geom_hline(color='grey', lty=3,yintercept = 1) +

    ylab(expression(over('Empirical Type-I Error Rate','Theoretical Type-I Error Rate'))) +
    #theme(axis.text.y.right = element_text(color=PAL[3]),
    #     axis.ticks.y.right=element_line(color=PAL[3])) +
    xlab('Generations of xAM') +
     theme(axis.title.x=element_blank())+
     theme(legend.position=c(.5,.9), legend.direction='horizontal', legend.key.width=unit(1,'cm'))
 )

# mdat <- reshape2::dcast(molten,seed+gen+args_m+args_n+args_kmate +alpha +GWAS+alpha_bonferoni~ quantity)

# mdat$T1R <- mdat$FalsePositive/(.8)
# mdat$hits <- (mdat$FalsePositive + mdat$TruePositive)
(plot_gwas2x5_alt_v <-
 ggplot(tmpd[tmpd$GWAS=='Population',],
       aes(gen,relative_T1R, color=xAM,shape=xAM)) + default_theme+
    # geom_alpha_bonferoni*2))+
    # geom_jitter(height = 1)+
     stat_summary(geom='path',  position=position_dodge(width=.1),
                 fun=median
                 ) +
     stat_summary(geom='point',  position=position_dodge(width=.1),
                 fun=median
                 ) +
    # stat(formula=y~ns(x,df =2),method='gam', aes(gen, color=GWAS, lty=xAM), se=F ) +
    # geom_smooth(formula=y~bs(x,degree =3),method='gam', aes(gen, color=GWAS, lty=xAM), se=F ) +
    # stat_summary(geom='path',fun = median, aes(gen,  color=GWAS, lty=xAM)) +
    facet_grid(power~.,as.table = T)+ #, labeller = label_parsed) +
    geom_hline(color='grey', lty=3,yintercept = 1) +
    scale_color_manual(values=PAL) +
    scale_y_continuous(breaks = seq(1,2,.25)) +
    geom_hline(color='grey', lty=3,yintercept = 1) +

    ylab(expression(over('Empirical Type-I Error Rate','Theoretical Type-I Error Rate'))) +
    #theme(axis.text.y.right = element_text(color=PAL[3]),
    #     axis.ticks.y.right=element_line(color=PAL[3])) +
    xlab('Generations of xAM') +
     # theme(axis.title.x=element_blank())+
     theme(legend.position=c(.5,.9), legend.direction='horizontal', legend.key.width=unit(1,'cm'))
 )

###############################################################################
## Cell 77: load newhits
## Original: load('newhits.rdata',verbose = T)
###############################################################################
load(file.path(DATA_DIR, 'sim_results', 'newhits.rdata'),verbose = T)

###############################################################################
## Cell 78: nhdat + nhplot
###############################################################################
nhdat <- melt(newhits, id.vars=c('delta_N','kpheno', 'm_causal'),
              measure.vars=c('delta_ex_on_target','delta_ex_off_target'))

nhdat["Generations of xAM"] <- nhdat$gen
nhdat["Number of phenotypes under xAM"] <- as.factor(nhdat$kpheno)
nhdat["𝑀 causal / phenotype"] <- nhdat$m_causal
nhdat$type <- NA
nhdat$type[nhdat$variable=='delta_ex_off_target'] <- 'Off target'
nhdat$type[nhdat$variable=='delta_ex_on_target'] <- 'On target'
nhdat$type <- factor(nhdat$type, levels=sort(unique(nhdat$type, decreasing = F)))

nhplot <- ggplot(nhdat[nhdat$m_causal==4000,], aes(delta_N,value,fill=type, shape=type, linetype=type)) +
# ggplot(nhdat[nhdat$m_causal==2000 & nhdat$kpheno==5,], aes(delta_N,value,fill=type, shape=type, linetype=type)) +
    # geom_path(position=position_dodge(width=.05)) +
    # geom_point(position=position_dodge(width=.05)) +
geom_bar(stat='identity')+
      theme_bw() + facet_grid(`Number of phenotypes under xAM`~`𝑀 causal / phenotype`,
      # theme_bw() + facet_grid(`𝑀 causal / phenotype`~`Number of phenotypes under xAM`,
                             label=label_both)+
scale_fill_manual(values=RColorBrewer::brewer.pal(name='Set1', 3)[c(1,2)]) +
  theme(
    text = element_text(size = 12),
    plot.title = element_text(hjust = .5, vjust = -8),
    legend.title = element_blank(),
    legend.position = 'top',
    legend.key.width = unit(1.5, 'cm'),
    axis.text.x= element_text(angle=-45, hjust=0),
    axis.title.y.right = element_text(margin = margin(l = 10))  # adds space to the right label
  ) +
    # scale_x_log10(breaks=unique(frac_res$N),labels=comma) +
    # scale_y_log10(breaks=1*10^(0:5),labels=comma) +
    xlab('Incremental GWAS sample size') +
    ylab('Incremental new GWAS hits')


###############################################################################
## Cell 81: final figure assembly
###############################################################################
SS=12
LSIZE=20
WOFFSET=.15
suppressWarnings({

    leg <- get_legend(plot_h2_2x5+scale_linetype_discrete()+guides(color=guide_none()) +
                  theme(legend.key.width=unit(1,'cm'),legend.position='right', legend.direction = 'horizontal')+ theme(text=element_text(size=SS)))
    a1 <- plot_h2_2x5 + guides(linetype=guide_none(), shape=guide_none()) + ltheme +
        theme(axis.title.x = element_blank())+ theme(legend.position = c(WOFFSET,.75))+ theme(text=element_text(size=SS))
    a2 <- plot_rg_2x5 + guides(linetype=guide_none(), shape=guide_none()) + ltheme +
        theme(legend.position = c(WOFFSET,.75))+ theme(text=element_text(size=SS))



plist = align_plots(#plot_cca_loadings + theme(text=element_text(size=SS)),
                    plot_cca_var+xlab('Canonical Variate')+ theme(text=element_text(size=SS)),#+ylab('Rel. Can.\nRedundancy'),
       a1,a2,plot_tcca_var+xlab('Canonical Variate')+ theme(text=element_text(size=SS)),
                    axis = 'ltbr',align = 'vh')

plist2 <- align_plots(plist[[3]],plot_gwas2x5_alt_v +theme(legend.position = c(.5,.62))+xlab('Generations of xAM')+
          ylab(expression('𝛼'[empircal]/'𝛼'[theoretical]))+ theme(axis.title.y=element_text(size=SS+4), text=element_text(size=SS)), nhplot)


lcol = plot_grid(plot_grid(get_legend(plot_h2_2x5+guides(color=guide_none(),shape=guide_none(),linetype=guide_none())),
                           leg,nrow=1,rel_widths=c(1,4)),
                 plist[[2]],plist2[[1]], rel_heights = c(.75,4,4), ncol=1, labels=c('','c','d') , label_size=LSIZE)

ltheme = theme(legend.position = c(.1,.7),
               legend.title = element_blank(),
               legend.key.width=unit(1,'cm'))


suppressWarnings({
    leg <- get_legend(plot_h2_2x5+scale_linetype_discrete()+guides(linetype=guide_legend(),color=guide_none()) +
                  theme(legend.key.width=unit(1,'cm'),legend.position='right')+ theme(text=element_text(size=SS)))
    a1 <- plot_h2_2x5 + guides(linetype=guide_none(), shape=guide_none()) + ltheme +
        theme(axis.title.x = element_blank())+ theme(legend.position = c(WOFFSET,.95))+ theme(text=element_text(size=SS))
    a2 <- plot_rg_2x5 + guides(linetype=guide_none(), shape=guide_none()) + ltheme +
        theme(legend.position = c(WOFFSET,.75))+ theme(text=element_text(size=SS))
    aa <- align_plots(a1,a2,
                      axis='rl',align = 'v')

    brow=plot_grid(lcol,plist2[[2]], nhplot, nrow=1, labels=c('','e','f'),rel_widths=c(3.5,3.2,3.3), label_size=LSIZE)
(    plot_2x5_ <- plot_grid(rel_heights = c(.6,5,7),
              leg,
              # aa[[1]],
              aa[[2]],
                            # aa[[3]],
              ncol=1,
                           labels=c('','d','e')))
    })

# alist = align_plots(plot_2x5, plot_gwas2x5+ theme(legend.position = 'top'),axis = 'rl',align = 'h')
rr1 <- plot_grid(plist[[1]], plist[[4]], rel_widths=c(5,3,2), labels=c('a','b'), label_size=LSIZE)
plot_grid(rr1 , brow,rel_heights=c(2.2,5),
         ncol=1)
# plot_grid(cca_plots,plot_2x5_,plot_gwas2x5,
                           # ncol=3)
    })

###############################################################################
## Cell 82: save PNG
## Original: png('../submission/resub/figures/f1.png', ...)
###############################################################################
png(file.path(FIGURES_DIR, 'fig1.png'), height = 11*.9, width=14*.9, units = 'in', res=240, bg = "white")
    plot_grid(rr1 , brow,rel_heights=c(2.2,5),
         ncol=1)
    dev.off()

## Also save PDF
cairo_pdf(file.path(FIGURES_DIR, 'fig1.pdf'), width = 14*.9, height = 11*.9, bg = "white")
    plot_grid(rr1 , brow,rel_heights=c(2.2,5),
         ncol=1)
    dev.off()

cat("Figure 1 saved to:", file.path(FIGURES_DIR, 'fig1.png'), "\n")
cat("Figure 1 saved to:", file.path(FIGURES_DIR, 'fig1.pdf'), "\n")
