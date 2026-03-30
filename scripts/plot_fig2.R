## plot_fig2.R — verbatim port of mFigPsych.ipynb cells for Figure 2
## Only changes: file paths (BASE_DIR), ggsave at end

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
FIG_DIR  <- file.path(BASE_DIR, "figures_output")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

## ── Cell 1 ──────────────────────────────────────────────────────────────────
library(yacca)
library(repr)
library(ggplot2)
library(reshape2)
library(cowplot)
library(stringr)
library(splines)

PAL = RColorBrewer::brewer.pal(8,'Set1')

default_theme <- theme_bw() + theme(text=element_text(size=14))
default_theme2 <- theme_minimal() + theme(text=element_text(size=14))

## ── Cell 5 ──────────────────────────────────────────────────────────────────
he <- read.csv(file.path(BASE_DIR, 'data/psych_sims/psychHEres.csv'))
he2 <- read.csv(file.path(BASE_DIR, 'data/psych_sims/psychHEres2way.csv'))
he6 <- read.csv(file.path(BASE_DIR, 'data/psych_sims/psychHEres6way.csv'))
names(he) <- gsub('.phenotype','', names(he))
names(he) <- gsub('.addGen','', names(he))
names(he) <- gsub('.proband','', names(he))

## ── Cell 6 ──────────────────────────────────────────────────────────────────
names(he2) <- gsub('.phenotype','', names(he2))
names(he2) <- gsub('.addGen','', names(he2))
names(he2) <- gsub('.proband','', names(he2))
names(he6) <- gsub('.phenotype','', names(he6))
names(he6) <- gsub('.addGen','', names(he6))
names(he6) <- gsub('.proband','', names(he6))

## ── Cell 20 ─────────────────────────────────────────────────────────────────
bvars = grep('h2HE',grep('_.+_', names(he),val=T), val=T, invert = T)
uvars = grep('h2', names(he),val=T)
ivars <- c('seed','gen','xAM')

bdat <- reshape2::melt(he, measure.vars = bvars, id.vars = ivars)
udat <- reshape2::melt(he, measure.vars = uvars, id.vars = ivars)
bdat <- bdat[!is.na(bdat$value),]
udat <- udat[!is.na(udat$value),]

tmpu <- str_split_fixed(udat$variable, '_', 3)
tmpb <- str_split_fixed(bdat$variable, '_', 3)

udat$outcome <- tmpu[,1]
udat$pheno <-  tmpu[,2]

bdat$outcome <- tmpb[,1]
bdat$pheno_1 <-  tmpb[,2]
bdat$pheno_2 <-  tmpb[,3]

bdat <- bdat[bdat$pheno_1!= bdat$pheno_2, ]

## ── Cell 23 ─────────────────────────────────────────────────────────────────
make_label <- function(number, digits=2) {
    format(round(number,digits), nsmall = digits)#, width=5, justify='right')
    }

## ── Cell 24 ─────────────────────────────────────────────────────────────────
bagg <- aggregate(bdat['value'],bdat[c('gen','xAM','outcome','pheno_1', 'pheno_2')], median)
bagg$se <- aggregate(bdat['value'],bdat[c('gen','xAM','outcome','pheno_1', 'pheno_2')], sd)$value


## ── Cell 28 ─────────────────────────────────────────────────────────────────
library(ggtext)

## ── Cell 31 (plot_rgheat_psych) ─────────────────────────────────────────────
tmpd <- bagg[bagg$outcome!='gamma' & bagg$gen %in% c(0,1,3,5),]

tmpd1 <- tmpd[tmpd$xAM=='2-variate',]
tmpd2 <- tmpd[tmpd$xAM=='6-variate',]
tmpd2$pheno_1 <- tmpd1$pheno_2
tmpd2$pheno_2 <- tmpd1$pheno_1
pdat <- rbind.data.frame(tmpd1,tmpd2)
pdat$label <- paste0(make_label(pdat$value))

# pdat <- rbind.data.frame(pdat[1:16,],pdat)
# pdat$value[1:16] <- NA
# pdat$pheno_1[1:8] <- pdat$pheno_2[1:8] <-'ADHD'

# pdat$pheno_1[9:16] <- pdat$pheno_2[9:16] <-'ADHD'
# # pdat$pheno_1[1:8] <- pdat$pheno_2[1:8] <-'BIP'
# # pdat$pheno_1[9:16] <- pdat$pheno_2[9:16] <-'ANX'
# # pdat$label[1:8] <-' 2x ↘'
# # pdat$label[9:16] <-'↖ 6x '
# pdat$label[1:8] <-''
# # pdat$label[1:8] <-'↘ 2x '
# pdat$label[9:16] <-''
# pdat$label[9:16] <-' 6x ↖'
# pdat$label[1:8] <-'2x ↓'
# pdat$label[9:16] <-'↑ 6x'
nf=.065
ns=2.9
badj=.03
xx=.1
pdat$Outcome <- 'hat(italic(r))[HE]'
pdat$Outcome[pdat$outcome == 'rgtrue'] <- 'italic(r)[score]'
pdat$Outcome <- as.factor(pdat$Outcome)
pdat$Outcome <- factor(pdat$Outcome, levels=rev(levels(pdat$Outcome)))
pdat$Generation <- paste(pdat$gen, 'gen.~xAM', sep='~')
(plot_rgheat_psych <-
ggplot(pdat,
       aes(x=pheno_2, y=pheno_1, fill=value,
           color=(value>.012 & value < .21),
           label=label)) +
    facet_grid(Outcome~Generation,switch='x',labeller = label_parsed) +
     geom_tile(color=1) +
     scale_fill_distiller(palette='Spectral',na.value = 'white') +
    geom_text() +

     #scale_color_distiller(palette='Greys',direction = 0,trans='')
    scale_color_grey(start = 1, end =.1, guide= guide_none(),na.value = 'red') +
    default_theme2+
    scale_x_discrete(position = 'top')  +
     geom_richtext(aes(x='ADHD',y='ADHD',label='6xAM',fontface='bold'),nudge_x = -nf-xx,nudge_y = nf, size=ns,
                   label.padding = grid::unit(rep(0, 4), "pt"),
                   color=1,label.color=NA, fill='white', angle=45) +
     # geom_richtext(aes(x='ADHD',y='ADHD',label='6xAM'),nudge_x = nf,nudge_y = -nf, size=ns,
     #               label.padding = grid::unit(rep(0, 4), "pt"),
     #               color=1,label.color=NA, fill='white', angle=45) +
     # geom_richtext(aes(x='SCZ',y='SCZ',label='2xAM'),nudge_x = -nf,nudge_y = nf, size=ns,
     #               label.padding = grid::unit(rep(0, 4), "pt"),
     #               color=1,label.color=NA, fill='white', angle=45) +
     geom_richtext(aes(x='SCZ',y='SCZ',label='2xAM',fontface='bold'),nudge_x = nf+xx+badj,nudge_y = -nf-badj, size=ns,
                   label.padding = grid::unit(rep(0, 4), "pt"),
                   color=1,label.color=NA, fill='white', angle=45) +
     geom_segment(aes(x=.75,y=.75,xend=6.25,yend=6.25), color='darkgrey', lty=2)+

 theme(axis.title=element_blank(),
                                               legend.position='top',
                                                legend.title = element_blank(),
                                                legend.key.width=unit(1.5,'cm')))


## ── Cell 33 (gamma data prep) ──────────────────────────────────────────────
bdat$traits <- apply(bdat[c('pheno_1','pheno_2')], 1, function(x) paste(sort(x), collapse= ' / '))

med <- aggregate(bdat['value'],bdat[c('gen','xAM','outcome','pheno_1','pheno_2','traits')], median)
med1 <- aggregate(bdat['value'],
                  bdat[c('gen','xAM','outcome','pheno_1','pheno_2','traits')], quantile, .9)
med9 <- aggregate(bdat['value'],
                  bdat[c('gen','xAM','outcome','pheno_1','pheno_2','traits')], quantile, .1)
medsd <- aggregate(bdat['value'],
                  bdat[c('gen','xAM','outcome','pheno_1','pheno_2','traits')], sd)
med$lower <- med1$value
med$upper <- med9$value
med$sd <- medsd$value

gdat <- med[med$outcome=='gamma' & med$gen %in% c(0,1,3,5),]
gdat$traits2 <- apply(gdat[c('pheno_1','pheno_2','gen','xAM')], 1, function(x) paste(sort(x), collapse= ' / '))

gdat <- gdat[!duplicated(gdat$traits2), ]

## ── Cell 49 (plot_gamma) ────────────────────────────────────────────────────
av_string = '\U1D5EE\U1D603\U1D5F2\U1D5FF\U1D5EE\U1D5F4\U1D5F2'
av_string = '\U1D656\U1D66B\U1D65A\U1D667\U1D656\U1D65C\U1D65A'
pldat <- gdat[gdat$gen==5,]
pldat$traits<-as.character(pldat$traits)
                     # levels=c(unique(pldat$traits),'average'))
pldat2 <- pldat[pldat$xAM=='2-variate',]
pldat6 <- pldat[pldat$xAM=='6-variate',]
av2 <- with(pldat2, sum(value/sd^2)/sum(1/sd^2))
sd2 <- with(pldat2, sqrt(1/sum(1/sd^2)))
av6 <- with(pldat6, sum(value/sd^2)/sum(1/sd^2))
sd6 <- with(pldat6, sqrt(1/sum(1/sd^2)))

pldat <- rbind.data.frame(pldat[1:2,],pldat)
pldat$xAM[1:2] <- c('2-variate','6-variate')
pldat$sd[1:2] <- c(sd2,sd6)
pldat$value[1:2] <- c(av2,av6)
pldat$lower[1:2] <- c(av2+qnorm(.05)*sd2,av6+qnorm(.05)*sd6)
pldat$upper[1:2] <- c(av2+qnorm(.95)*sd2,av6+qnorm(.95)*sd6)
pldat$traits[1:2] <- av_string

tmp <- gdat[gdat$gen==5 & gdat$xAM=='6-variate',]

pldat$traits <- factor(pldat$traits,
                       levels=c(av_string,as.character(tmp$traits)[order(tmp$value)]
                                              ))
labs <- as.expression(levels(pldat$traits))

# pdat$Outcome <- 'hat(italic(r))[HE]'
# pdat$Outcome[pdat$outcome == 'rgtrue'] <- 'italic(r)[score]'
(plot_gamma <-
ggplot(pldat,
       aes(x=traits, y=value, ymax=upper, ymin=lower,
           color=xAM,  linewidth=as.factor(traits==av_string))) + coord_flip() +
    # facet_grid(gen,switch='x',labeller = label_parsed) +
     geom_errorbar(position=position_dodge(width=.5)) +
     geom_point(position=position_dodge(width=.5)) +
     geom_hline(yintercept = 1, lty=2) +
      geom_hline(yintercept = 0, lty=3) +
    # geom_text() + #scale_color_distiller(palette='Greys',direction = 0,trans='')
     scale_color_manual(values = PAL)+
    default_theme2+
     scale_linewidth_manual(values=c(.5,1.2))+
     # _discrete(range =c(.5,1.5)) +
     guides(linewidth = guide_none()) +
     ylab(expression(italic(gamma)==italic(r)[xAM]/italic(r)[empirical])) +
     ylab(expression(italic(r)[xAM]/italic(r)[empirical]~~'(5 gen. xAM)')) +
     theme(axis.title.y=element_blank(),
                                               legend.position='top',
                                                legend.title = element_blank(),
                                                legend.key.width=unit(1.5,'cm')))


## ── Cell 53 (prevalence data prep) ─────────────────────────────────────────
pvars = grep('prev',names(he),val=T)
prdat <- reshape2::melt(he, measure.vars = pvars, id.vars = ivars)

tmpr <- str_split_fixed(prdat$variable, '_', 2)
prdat$dx <- tmpr[,2]

## ── Cell 54 (prevalence aggregation) ────────────────────────────────────────
prdat <-aggregate(prdat['value'],prdat[c('seed','gen','xAM','dx')], mean, na.rm=T)

## ── Cell 55 (plot_prev) ────────────────────────────────────────────────────
(plot_prev <- ggplot(prdat, aes(gen, value, color=xAM)) +
    # geom_point(position=position_dodge(width = .2)) +
    stat_summary(fun = median,
                 geom='point',
                 position=position_dodge(width = .2)) +
    stat_summary(fun = median,
                 geom='line',
                 position=position_dodge(width = .2)) +
    stat_summary(fun.max = function(z)  median(z) + sd(z),
                 fun.min = function(z) median(z) - sd(z),
                 fun = median,
                 geom='linerange',
                 position=position_dodge(width = .2)) +
    # stat_smooth(method='gam', formula = y~bs(x, knots=0:5, degree=1), se=F) +
    default_theme2 +
    xlab('Generations xAM') +
    ylab('Prevalence')+
    facet_wrap(~dx, scales='free_y') +
     scale_color_manual(values = PAL))


## ── Cell 65 (final assembly) ───────────────────────────────────────────────
suppressWarnings({

    aa = align_plots(plot_rgheat_psych + theme(legend.position = 'top'),
                     plot_prev + theme(legend.title.align = .5), axis = 'l',align = 'v')
    plot_rgheat_psych_0 = aa[[1]]
    plot_prev_0 = aa[[2]]
    aa = align_plots(plot_gamma + theme(legend.position = 'bottom')+guides(color=guide_none()),
                     plot_rgheat_psych_0, axis = 'b',align = 'h')
    plot_gamma_1 = aa[[1]]
    plot_rgheat_psych_1 = aa[[2]]
    aa = align_plots(plot_gamma_1,plot_prev_0, axis = 'b',align = 'h')
    plot_gamma_2 = aa[[1]]
    plot_prev_1 = aa[[2]]
    c1 = plot_grid(plot_rgheat_psych_1,
                   plot_prev_1, nrow=2,ncol=1, rel_heights=c(6,4),
                   labels=c('a','c'), label_size = 20)
    c2 = plot_grid(ggdraw(), plot_gamma_2, ncol=1, rel_heights=c(.2,9))
    fig2 = plot_grid(c1, c2, nrow=1, rel_widths = c(8,4),labels=c('','b'), label_size = 20)

})

## ── Save ────────────────────────────────────────────────────────────────────
ggsave(file.path(FIG_DIR, "fig2_psych.pdf"),
       fig2, width = 18, height = 12)
ggsave(file.path(FIG_DIR, "fig2_psych.png"),
       fig2, width = 18, height = 12, dpi = 300, bg = "white")

cat("Saved fig2_psych.pdf and fig2_psych.png to", FIG_DIR, "\n")
