#!/usr/bin/env Rscript
# plot_fig4.R -- Generate Figure 4 (education/height/wealth simulation)
#
# This script is a verbatim port of code cells from
# manu/figure_nb/mFigEdu.ipynb with ONLY the following changes:
#   1. Data path: '~/data/edu_no_CD_LS.01/' -> file.path(BASE_DIR, 'data/edu_sims')
#   2. Image path: 'figures/edu_cdiag_redux.png' -> file.path(BASE_DIR, 'data/cdiags/edu_cdiag_redux.png')
#   3. BASE_DIR setup at top
#   4. ggsave at end
#
# NOTE: The data in data/edu_sims/ comes from a DIFFERENT simulation
# (h2_edu=0) than the manuscript notebook (h2_edu=0.01 from
# ~/data/edu_no_CD_LS.01/). The code will run but produce different
# numerical values than the manuscript figures. This is expected.

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
outdir <- file.path(BASE_DIR, "figures_output")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# ---- Cell 0: libraries + setup ----
## library(yacca)
library(repr)
library(ggplot2)
library(reshape2)
library(cowplot)
library(stringr)
library(splines)
setwd('/home/rsb/Dropbox/ftsim/manu')


default_theme <- theme_bw() + theme(text=element_text(size=14))
default_theme2 <- theme_minimal() + theme(text=element_text(size=14))
PAL <- RColorBrewer::brewer.pal(8,'Set1')


# ---- Cell 2: data load ----
edat <- lapply(list.files(file.path(BASE_DIR, 'data/edu_sims'),
                          pattern='parsed', full.names = T),
               function(x) try(read.csv(x)))
## model_edu_simple_h2edu

# ---- Cell 4: rbind ----
edat <- do.call(rbind.data.frame, edat[sapply(edat,class)=='data.frame'])

# ---- Cell 5: h2_true_wealth ----
edat$h2_true_wealth <- edat$cov_edu.addGen_edu.addGen/ edat$cov_edu.phenotype_edu.phenotype *(1/3)

# ---- Cell 14: h2_true_wealth override ----
edat$h2_true_wealth <- rnorm(nrow(edat),0,1e-6)

# ---- Cell 15: pdat melt for univariate ----
pdat <- melt(edat[edat$h2HE_height < 2,], id.vars=c('seed','gen'),
             measure.vars=c('r2_true_height.pheno_height.addGen',
                            # 'r2_true_height.pheno_edu.addGen',
                            # 'r2_true_height.pheno_total.addGen',
                            'r2_true_edu.pheno_height.addGen',
                            'r2_true_edu.pheno_edu.addGen',
                            # 'r2_true_edu.pheno_total.addGen',
                            'r2_true_wealth.pheno_height.addGen',
                            'r2_true_wealth.pheno_edu.addGen',
                            # 'r2_true_wealth.pheno_total.addGen',
                            #'h2_true_edu','h2_true_height','h2_true_wealth',
                            'h2HE_edu','h2HE_height','h2HE_wealth',
                           'h2_true_edu','h2_true_wealth','h2_true_height'))

pdat$Outcome <- NA
pdat$Outcome[pdat$variable %in% c('r2_true_height.pheno_height.addGen', 'r2_true_height.pheno_edu.addGen',
                                  'r2_true_height.pheno_total.addGen', 'h2HE_height','h2_true_height')] <- 'height'
pdat$Outcome[pdat$variable %in% c('r2_true_edu.pheno_height.addGen', 'r2_true_edu.pheno_edu.addGen',
                                  'r2_true_edu.pheno_total.addGen', 'h2HE_edu','h2_true_edu')] <- 'edu'
pdat$Outcome[pdat$variable %in% c('r2_true_wealth.pheno_height.addGen', 'r2_true_wealth.pheno_edu.addGen',
                                  'r2_true_wealth.pheno_total.addGen', 'h2HE_wealth','h2_true_wealth')] <- 'wealth'

pdat$Quantity <- NA
pdat$Quantity[grepl('^h2HE', pdat$variable)] <- 'h2_HE'
pdat$Quantity[grepl('^h2_true', pdat$variable)] <- 'h2_true'
pdat$Quantity[grepl('height.addGen', pdat$variable, fixed=TRUE)] <- 'r2(y, l_height)'
pdat$Quantity[grepl('edu.addGen', pdat$variable, fixed=TRUE)] <- 'r2(y, l_edu)'
pdat$Quantity[grepl('total.addGen', pdat$variable, fixed=TRUE)] <- 'h2_broad'

# ggplot(pdat, aes(gen, value, color=Quantity)) + stat_summary() +
#     facet_wrap(~Outcome, scales='free') + default_theme +

# ---- Cell 17: pdat cleanup ----
# melt(pdat, id.vars=c('gen','Outcome','Quantity'),
pdat$value[is.nan(pdat$value)] <-NA
pdat <- pdat[!is.na(pdat$value),]


# ---- Cell 19: plot_uni ----
## options(repr.plot.width=12)
options(repr.plot.height=7)


(plot_uni <- ggplot(within(pdat,
                           Outcome <- factor(pdat$Outcome,
                                                  levels=c('height','edu','wealth'))),
     aes(gen,value, color=Quantity, linetype=Quantity)) +
 default_theme2 +
# geom_point() +
    # geom_hline(aes(yintercept = int, color=outcome), lty=1) +
    stat_summary(geom = 'point',fun = median, position=position_dodge(width=.1), lwd=.75) +
    stat_summary(geom = 'path',fun = median, position=position_dodge(width=.1), lwd=.75) +
    scale_linetype_manual(values = c(1:4),
                       labels =c(expression(italic(hat(h))[HE]^2),
                                  expression(italic(h)[true]^2),
                                 expression(italic(R)[italic(G)[edu]]^2),#*','*italic(Y)[edu])),
                                 expression(italic(R)[italic(G)[height]]^2))) +
    scale_color_manual(values = PAL[1:4],
                       labels =c(expression(italic(hat(h))[HE]^2),
                                  expression(italic(h)[true]^2),
                                 expression(italic(R)[italic(G)[edu]]^2),#*','*italic(Y)[edu])),
                                 expression(italic(R)[italic(G)[height]]^2))) +
                 ylab(expression(italic(h)^2)) +
    xlab('Generations of xAM')       +
                 ylab('')+
                 facet_grid(Outcome~., scale='free_y') +
                 theme(legend.key.width=unit(1.5,'cm'),
                       legend.position='left',
                       legend.title = element_blank())#,
                       # strip.text = element_blank())
)

# ---- Cell 28: bivariate molt (initial) ----
mvars <- c(grep('^rg',names(edat), val=T),'corr_edu.addGen_height.addGen')
ivars <- c('gen','X','seed','args_m')
molt <-
reshape2::melt(edat, id.vars = ivars,
               measure.vars = mvars)
molt <- molt[!grepl('rg_true',molt$variable), ]
# molt$variable <- gsub('rg_true','rgtrue', molt$variable)
molt$variable <- gsub('.','_', molt$variable, fixed = T)
molt$variable <- gsub('_phenotype_proband','', molt$variable)
# table(molt$variable)

tmp <- str_split_fixed(molt$variable,'_',3)

molt$outcome <- tmp[,1]
molt$trait_1 <- tmp[,2]
molt$trait_2 <- gsub('addGen_height_addGen', 'height', tmp[,3])
molt$traits <- paste0(molt$trait_1,' / ',molt$trait_2)
# molt2 <- within(molt, trait_1 <- trait_2)
# molt2$trait_2 <- molt$trait_1
# molt <- rbind.data.frame(molt, molt2)
# molt <- molt[molt$outcome!='rgtrue',]
molt$value[is.na(molt$value)]<-0
molt$value[molt$value>5]<-NA

molt <- rbind.data.frame(molt,
                        within(molt[molt$outcome=='corr',], traits <- 'height / wealth'))

# ---- Cell 29: bivariate molt (redefinition) ----
mvars <- c(# "r2_true_edu.pheno_edu.addGen",
           # "r2_true_edu.pheno_height.addGen",
           # "r2_true_edu.pheno_total.addGen",
           # "r2_true_wealth.pheno_edu.addGen",
           # "r2_true_wealth.pheno_height.addGen",
           # "r2_true_wealth.pheno_total.addGen",
           # "r2_true_height.pheno_edu.addGen",
           # "r2_true_height.pheno_height.addGen",
           # "r2_true_height.pheno_total.addGen",
           # 'rbeta_hat_pgwas_edu.phenotype_height.phenotype',
           # 'rbeta_hat_pgwas_edu.phenotype_wealth.phenotype',
           # 'rbeta_hat_pgwas_height.phenotype_wealth.phenotype',
           # 'rbeta_hat_sgwas_edu.phenotype_height.phenotype',
           # 'rbeta_hat_sgwas_edu.phenotype_wealth.phenotype',
           # 'rbeta_hat_sgwas_height.phenotype_wealth.phenotype',
           "rgHE_edu.phenotype.proband_height.phenotype.proband",
           "rgHE_edu.phenotype.proband_wealth.phenotype.proband",
           "rgHE_height.phenotype.proband_wealth.phenotype.proband",
           # "rgHEsib_edu.phenotype.proband_height.phenotype.proband",
           # "rgHEsib_edu.phenotype.proband_wealth.phenotype.proband",
           # "rgHEsib_height.phenotype.proband_wealth.phenotype.proband",
           "corr_edu.addGen_height.addGen",
    "rg_true_edu_height",
"rg_true_edu_wealth", "rg_true_height_wealth"
          )
molt$value[is.na(molt$value)]<-0
molt$value[molt$value>5]<-NA

molt <- rbind.data.frame(molt,
                        within(molt[molt$outcome=='corr',], traits <- 'height / wealth'))

ivars <- c('gen','X','seed','args_m')
molt <-
reshape2::melt(edat, id.vars = ivars,
               measure.vars = mvars)
molt <- molt[!grepl('rg_true',molt$variable), ]
options(digits=2)
molt$variable <- gsub('.','_', molt$variable, fixed = T)
molt$variable <- gsub('_phenotype_proband','', molt$variable)
# table(molt$variable)

tmp <- str_split_fixed(molt$variable,'_',3)

molt$outcome <- tmp[,1]
molt$trait_1 <- tmp[,2]
molt$trait_2 <- gsub('addGen_height_addGen', 'height', tmp[,3])
molt$traits <- paste0(molt$trait_1,' / ',molt$trait_2)
# molt2 <- within(molt, trait_1 <- trait_2)
# molt2$trait_2 <- molt$trait_1
# molt <- rbind.data.frame(molt, molt2)
# molt <- molt[molt$outcome!='rgtrue',]
molt$value[is.na(molt$value)]<-0
molt$value[molt$value>5]<-NA

# ---- Cell 32: plot_biv ----
## options(repr.plot.width=8)
options(repr.plot.height=5)
options(repr.plot.width=12)
options(repr.plot.height=7)


(plot_biv <-
 ggplot(molt[molt$value<5 &# molt$traits !='height / wealth' &
             molt$outcome!='rgHEsib',],
                                aes(gen,value, color=traits, linetype=outcome)) + default_theme2 +
# geom_point() +
    # geom_hline(aes(yintercept = int, color=outcome), lty=1) +
    # stat_summary(geom = 'linerange',position=position_dodge(width=.1),
                 # fun.data=function(x) mean_se(x,1.96)) +
    stat_summary(geom = 'point',fun = median, position=position_dodge(width=.1), lwd=.75) +
    stat_summary(geom = 'path',fun = median, position=position_dodge(width=.1), lwd=.75) +
    ylab(expression(italic(r)[g])) +
    # scale_color_manual(values=PAL[c(1,3:4)])+
    #                  geom_hline(color=PAL[5],yintercept = 0,lty=3) +
    guides(color=guide_legend(title=''))+
    guides(linetype=guide_legend(title=''))+
    # ylab(expression(italic(h)^2)) +
    xlab('Generations of xAM')      +
                 # facet_grid(I(pheno!='height')~., scale='free_y') +
   scale_color_manual(values = PAL <- RColorBrewer::brewer.pal(8,'Set1'))+#,
   #                     labels =c('height',
   #                               'edu',
   #                               'wealth')) +
                     geom_hline(color=PAL[5],yintercept = 0,lty=3) +
    scale_linetype_manual(values = c(1:2),
                       labels =c(expression(italic(r)[score]),
                                 expression(hat(italic(r))[beta]
                                           ))) +
                 guides(linetype=guide_legend(override.aes = aes(color=1)))+
                 theme(legend.key.width=unit(1.5,'cm'),
                       legend.title = element_blank())
                       # strip.text = element_blank())
                 )

# ---- Cell 36: plot_r2 (beta correlations) ----
mvars <- c(
'rbeta_hat_pgwas_edu.phenotype_height.phenotype',
'rbeta_hat_pgwas_height.phenotype_wealth.phenotype',
'rbeta_hat_pgwas_edu.phenotype_wealth.phenotype'
)

ivars <- c('gen','X','seed','args_m')
molt <-
reshape2::melt(edat, id.vars = ivars,
               measure.vars = mvars)
molt$variable <- as.character(molt$variable)
molt$variable[molt$variable =='rbeta_hat_pgwas_edu.phenotype_height.phenotype'] <- 'edu / height'
molt$variable[molt$variable =='rbeta_hat_pgwas_height.phenotype_wealth.phenotype'] <- 'height / wealth'
molt$variable[molt$variable =='rbeta_hat_pgwas_edu.phenotype_wealth.phenotype'] <- 'edu / wealth'

# table(molt$variable)

tmp <- str_split_fixed(molt$variable,'_',3)

molt$outcome <- tmp[,1]
molt$trait_1 <- tmp[,2]
molt$trait_2 <- tmp[,3]
molt$traits <- paste0(molt$trait_1,' / ',molt$trait_2)
# molt2 <- within(molt, trait_1 <- trait_2)
# molt2$trait_2 <- molt$trait_1
# molt <- rbind.data.frame(molt, molt2)
molt <- molt[molt$outcome!='rgtrue',]
molt$value[is.na(molt$value)]<-0


## options(repr.plot.width=8)
options(repr.plot.height=5)
options(repr.plot.width=12)
options(repr.plot.height=7)


(plot_r2 <-
 ggplot(molt[molt$value<5 ,],#& molt$traits !='edu / wealth',],
                                aes(gen,value, color=variable)) + default_theme2 +
# geom_point() +
    # geom_hline(aes(yintercept = int, color=outcome), lty=1) +
    stat_summary(geom = 'point',fun = median, position=position_dodge(width=.1), lwd=.75) +
    stat_summary(geom = 'path',fun = median, position=position_dodge(width=.1), lwd=.75) +
    ylab(expression(italic(r[hat('𝛽')]))) +
    scale_color_manual(values=PAL[c(1:3)])+
                     geom_hline(color=PAL[5],yintercept = 0,lty=3) +
    guides(color=guide_legend(title=''))+
    # guides(linetype=guide_legend(title=''))+
    # ylab(expression(italic(h)^2)) +
    xlab('Generations of xAM')      +
                 theme(legend.key.width=unit(1.5,'cm'),
                       legend.title = element_blank())
                 # facet_grid(I(pheno!='height')~., scale='free_y') +
   # scale_color_manual(values = PAL <- RColorBrewer::brewer.pal(8,'Set1'),
   #                     labels =c('height',
   #                               'edu',
   #                               'wealth')) +
   #  scale_linetype_manual(values = c(1:2),
   #                     labels =c(expression(italic(hat(h))[HE]^2),
   #                               expression(italic(h)[true]^2))) +
   #               guides(linetype=guide_legend(override.aes = aes(color=1)))+
   #               theme(legend.key.width=unit(1.5,'cm'),
   #                     legend.title = element_blank(),
                       # strip.text = element_blank())
                 )

# ---- Cell 47: schematic ----
(plot_diag_edu <- ggdraw() + draw_image(interpolate = T,
                                        magick::image_crop(geometry = '1441x811+50-20',magick::image_read(file.path(BASE_DIR, 'data/cdiags/edu_cdiag_redux.png')))))

# ---- Cell 50: assembly ----
options(repr.plot.height=10)
options(repr.plot.res=300, repr.plot.width=9)
LS=20
suppressWarnings({

aa <- align_plots(axis = 'lr', align='h',
                  plot_uni +theme(
                                 legend.position='top',#c(.5,.5),
                                 # legend.box='horizontal',
                                 legend.key.width = unit(1,'cm')) ,
                  plot_biv#+guides(color=guide_none(),lty=guide_none())
                  +#xlab('') +
                  guides(color=guide_legend(order=2),
                         linetype=guide_legend(order=1,
                                               override.aes=list(color=1)) )+
                  # geom_hline(yintercept = 0, lty=4, color=PAL[5]) +
                  theme(legend.position=c(.663,.285),legend.box = 'horizontal',
                        legend.title = element_blank(),legend.key.width = unit(1,'cm')),
                  plot_r2+theme(legend.position=c(.8,.7),legend.key.width = unit(1,'cm')) +
                  theme(legend.title = element_blank()))

c1 = plot_grid(aa[[1]],ncol=1,labels=c('b'), label_size = LS)
c2 = plot_grid(aa[[2]],aa[[3]],ncol=1,labels=c('c','d'), label_size = LS)
r1 = plot_grid(c1,c2, ncol=2,rel_widths=c(3,3), label_size = LS)
fig4 <- plot_grid(hjust=0,plot_diag_edu,NA, r1, nrow=3, rel_heights = c(3.5,.15,5), labels=c('a'), label_size = LS)
})

# ---- Save ----
outfile <- file.path(outdir, "fig4_edu_sims.png")
ggsave(outfile, fig4, width = 9, height = 10, dpi = 300, bg = "white")
cat("Saved:", outfile, "\n")
cat("Done.\n")
