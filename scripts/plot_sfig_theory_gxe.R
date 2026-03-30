## plot_sfig_theory_gxe.R
## Generates supplementary figures S8 and S13-S16.
##   S8:  Theory vs observed GWAS false positive calibration
##   S13: h2 across generations, GxE rows x VT cols
##   S14: h2 across generations, VT rows x GxE cols
##   S15: rg across generations with GxE rows
##   S16: PGI cross-phenotype slope with GxE rows
##
## Verbatim port of sFigComplexity_extended.ipynb code cells.

# library(yacca)
library(repr)
library(ggplot2)
library(reshape2)
library(cowplot)
library(stringr)
library(splines)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
FIG_DIR  <- file.path(BASE_DIR, "figures_output")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

default_theme  <- theme_bw() + theme(text = element_text(size = 14))
default_theme2 <- theme_bw() + theme(text = element_text(size = 12))
PAL <- RColorBrewer::brewer.pal(8, 'Set1')

## ── S8: Theory vs observed false positive calibration ──────────────────────────
## From sFigComplexity_extended.ipynb cell 93

monodat <- read.csv(file.path(BASE_DIR, 'data/sim_results/gwas_fp.csv'))

monodat$xAM <- '5-variate'
monodat$xAM[monodat$params=='2xAM'] <- '2-variate'
monodat$`n/m` <- as.factor(monodat$args_subsample/monodat$args_m)

monodat$params <- str_replace_all(monodat$params, '_', ' + ')

monodat$params <- factor(monodat$params, levels= c("2xAM","5xAM","5xAM + gXe", "5xAM + eVT", "5xAM + eVT + gXe",  "5xAM + pVT",
"5xAM + pVT + gXe"))

(plot_fp_multiverse <- ggplot(monodat[
              monodat$gen ==3, ], aes(log10(alpha), log10(FalsePositive),
                color = GWAS, lty = `n/m`)) + geom_line() +
    facet_grid(params~.)  + geom_abline(slope=1,intercept = 0)        +
    default_theme + scale_fill_discrete(guide = guide_none()) +
     theme(legend.position='top', legend.direction = 'horizontal',legend.title.align = .5))

ggsave(file.path(FIG_DIR, "sfig_s8_theory_validation.pdf"),
       plot_fp_multiverse, width = 12, height = 18, bg = "white", dpi = 300)
cat("Saved sfig_s8_theory_validation.pdf\n")

## ── Data loading for S13-S16 ──────────────────────────────────────────────────
## From sFigComplexity_extended.ipynb cells 2, 19, 20, 32

res <- read.csv(file.path(BASE_DIR, 'data/sim_results/merged_tabla_redux_results_0524.csv'))
table(res$scenario)

res$v_tot <- res$h2_true/(1-res$h2_true) * .5 + .5
res$vbeta <- res$vbeta_true*res$v_tot

mvars_uni = c('h2_true','he_h2')
mvars_biv = c('rg_true','he_rg')
grep('^arg',names(res), val=T)
idvars = c('seed','gen','args_kmate','args_rmate', 'args_theta', 'args_phi', 'args_m_causal','power','scenario')

## Cell 19: gdat setup
gdat <- res
gdat$h2_he <- gdat$he_h2

## Cell 20: bias columns
gdat$h2_bias <- with(gdat,h2_he -h2_true)
gdat$rg_bias <- with(gdat,he_rg -rg_true)

## Cell 32: VT/GxE labels
gdat$VT <- NA
gdat$VT[gdat$args_theta==0] <- '0%'
gdat$VT[gdat$args_theta==0.05] <- '5%'
gdat$VT[gdat$args_theta==0.2] <- '20%'
gdat$VT <- factor(gdat$VT, levels=c('0%','5%', '20%'))
gdat$GxE <- NA
gdat$GxE[gdat$args_phi==0] <- '0%'
gdat$GxE[gdat$args_phi==0.05] <- '5%'
gdat$GxE[gdat$args_phi==0.2] <- '20%'
gdat$GxE <- factor(gdat$GxE, levels=c('0%','5%', '20%'))

## ── S13: h2 across generations, GxE as rows ────────────────────────────────────
## From sFigComplexity_extended.ipynb cells 40 + 41

## Cell 40: melt h2 data
pdat <- pdat2 <-
reshape2::melt(gdat, id.vars = c('gen','X','seed','args_m','VT','GxE','scenario','args_theta',
                                 'args_phi'),
               measure.vars = c(#"rbeta_true",
                                "h2_he", "h2_true"
                                ))
pdat$variable <- as.character(pdat$variable)
pdat$variable[pdat2$variable=='rbeta_HE'] <- 'hat(italic(r))[𝛽]'
pdat$variable[pdat2$variable=='rg_true'] <- 'italic(r)[score]'
pdat$variable[pdat2$variable=='h2_he'] <- 'hat(italic(h))^2'
pdat$variable[pdat2$variable=='h2_true'] <- 'italic(h)^2'
pdat$variable <- as.factor(pdat$variable)
pdat$variable <- factor(pdat$variable, levels=rev(levels(pdat$variable)))

## Cell 41: GxE as rows for h2
(plot_biv_multiverse1  <- ggplot(pdat[pdat$GxE=="0%" & grepl('^5xAM',pdat$scenario),],
                                aes(gen,value, color=VT, linetype=GxE)) + default_theme +
    geom_hline(yintercept = .5, color='grey',lty=1) +
    stat_summary(geom = 'point',position=position_dodge(width=.3), fun=median) +
    stat_summary(geom = 'path',position=position_dodge(width=.3), fun=median) +
    ylab(expression(italic(h)^2)) +    xlab('Generations of xAM')      +
  facet_wrap(~variable,drop = T, label='label_parsed', scales='free') +
     default_theme2 +

   scale_color_manual(values = PAL <- RColorBrewer::brewer.pal(8,'Set1')[-(5:7)])
                     )

(plot_biv_multiverse2  <- ggplot(pdat[pdat$GxE=="5%" & grepl('^5xAM',pdat$scenario),],
                                aes(gen,value, color=VT, linetype=GxE)) + default_theme +
    geom_hline(yintercept = .5, color='grey',lty=1) +
    stat_summary(geom = 'point',position=position_dodge(width=.3), fun=median) +
    stat_summary(geom = 'path',position=position_dodge(width=.3), fun=median) +
    ylab(expression(italic(h)^2)) +    xlab('Generations of xAM')      +
  facet_wrap(~variable,drop = T, label='label_parsed', scales='free') +
     default_theme2 + scale_linetype_manual(values=2)+

   scale_color_manual(values = PAL <- RColorBrewer::brewer.pal(8,'Set1')[-(5:7)])
                     )

(plot_biv_multiverse3  <- ggplot(pdat[pdat$GxE=="20%" & grepl('^5xAM',pdat$scenario),],
                                aes(gen,value, color=VT, linetype=GxE)) + default_theme +
    geom_hline(yintercept = .5, color='grey',lty=1) +
    stat_summary(geom = 'point',position=position_dodge(width=.3), fun=median) +
    stat_summary(geom = 'path',position=position_dodge(width=.3), fun=median) +
    ylab(expression(italic(h)^2)) +
    xlab('Generations of xAM')      +
  facet_wrap(~variable,drop = T, label='label_parsed', scales='free') +
     default_theme2 +scale_linetype_manual(values=3)+

   scale_color_manual(values = PAL <- RColorBrewer::brewer.pal(8,'Set1')[-(5:7)])
                     )

plot_s13 <- plot_grid(plot_biv_multiverse1 + xlab(''),
plot_biv_multiverse2 + xlab('')+ theme(strip.text = element_blank()),
plot_biv_multiverse3 + theme(strip.text = element_blank()), nrow=3,ncol=1)

ggsave(file.path(FIG_DIR, "sfig_s13_h2_gxe_grid.pdf"),
       plot_s13, width = 8, height = 9, bg = "white", dpi = 300)
cat("Saved sfig_s13_h2_gxe_grid.pdf\n")

## ── S14: h2 across generations, VT as rows ─────────────────────────────────────
## From sFigComplexity_extended.ipynb cell 42

(plot_biv_multiverse1  <- ggplot(pdat[pdat$VT=="0%" & grepl('^5xAM',pdat$scenario),],
                                aes(gen,value, color=GxE, linetype=VT)) + default_theme +
    geom_hline(yintercept = .5, color='grey',lty=1) +
    stat_summary(geom = 'point',position=position_dodge(width=.3), fun=median) +
    stat_summary(geom = 'path',position=position_dodge(width=.3), fun=median) +
    ylab(expression(italic(h)^2)) +    xlab('Generations of xAM')      +
  facet_wrap(~variable,drop = T, label='label_parsed', scales='free') +
     default_theme2 +

   scale_color_manual(values = PAL <- RColorBrewer::brewer.pal(8,'Set1')[-(5:7)])
                     )

(plot_biv_multiverse2  <- ggplot(pdat[pdat$VT=="5%" & grepl('^5xAM',pdat$scenario),],
                                aes(gen,value, color=GxE, linetype=VT)) + default_theme +
    geom_hline(yintercept = .5, color='grey',lty=1) +
    stat_summary(geom = 'point',position=position_dodge(width=.3), fun=median) +
    stat_summary(geom = 'path',position=position_dodge(width=.3), fun=median) +
    ylab(expression(italic(h)^2)) +    xlab('Generations of xAM')      +
  facet_wrap(~variable,drop = T, label='label_parsed', scales='free') +
     default_theme2 + scale_linetype_manual(values=2)+

   scale_color_manual(values = PAL <- RColorBrewer::brewer.pal(8,'Set1')[-(5:7)])
                     )

(plot_biv_multiverse3  <- ggplot(pdat[pdat$VT=="20%" & grepl('^5xAM',pdat$scenario),],
                                aes(gen,value, color=GxE, linetype=VT)) + default_theme +
    geom_hline(yintercept = .5, color='grey',lty=1) +
    stat_summary(geom = 'point',position=position_dodge(width=.3), fun=median) +
    stat_summary(geom = 'path',position=position_dodge(width=.3), fun=median) +
    ylab(expression(italic(h)^2)) +
    xlab('Generations of xAM')      +
  facet_wrap(~variable,drop = T, label='label_parsed', scales='free') +
     default_theme2 +scale_linetype_manual(values=3)+

   scale_color_manual(values = PAL <- RColorBrewer::brewer.pal(8,'Set1')[-(5:7)])
                     )

plot_s14 <- plot_grid(plot_biv_multiverse1 + xlab(''),
plot_biv_multiverse2 + xlab('')+ theme(strip.text = element_blank()),
plot_biv_multiverse3 + theme(strip.text = element_blank()), nrow=3,ncol=1)

ggsave(file.path(FIG_DIR, "sfig_s14_h2_vt_grid.pdf"),
       plot_s14, width = 8, height = 9, bg = "white", dpi = 300)
cat("Saved sfig_s14_h2_vt_grid.pdf\n")

## ── S15: rg across generations, GxE as rows ────────────────────────────────────
## From sFigComplexity_extended.ipynb cells 36 + 37

## Cell 36: melt rg data
gdat$rbeta_HE <- gdat$he_rg

pdat <- pdat2 <-
reshape2::melt(gdat, id.vars = c('gen','X','seed','args_m','VT','GxE','scenario','args_theta',
                                 'args_phi'),
               measure.vars = c(#"rbeta_true",
                                "rg_true", "rbeta_HE"
                                ))
pdat$variable <- as.character(pdat$variable)
pdat$variable[pdat2$variable=='rbeta_HE'] <- 'hat(italic(r))[𝛽]'
pdat$variable[pdat2$variable=='rg_true'] <- 'italic(r)[score]'
pdat$variable[pdat2$variable=='h2_he'] <- 'hat(italic(h))^2'
pdat$variable[pdat2$variable=='h2_true'] <- 'italic(h)^2'
pdat$variable <- as.factor(pdat$variable)
pdat$variable <- factor(pdat$variable, levels=rev(levels(pdat$variable)))

## Cell 37: GxE as rows for rg
(plot_biv_multiverse1  <- ggplot(pdat[pdat$GxE=="0%" & grepl('^5xAM',pdat$scenario),],
                                aes(gen,value, color=VT, linetype=GxE)) + default_theme +
    geom_hline(yintercept = 0, color='grey',lty=1) +
    stat_summary(geom = 'point',position=position_dodge(width=.3), fun=median) +
    stat_summary(geom = 'path',position=position_dodge(width=.3), fun=median) +
    ylab(expression(italic(r[g]))) +    xlab('Generations of xAM')      +
  facet_wrap(~variable,drop = T, label='label_parsed', scales='free') +
     default_theme2 +

   scale_color_manual(values = PAL <- RColorBrewer::brewer.pal(8,'Set1')[-(5:7)])
                     )

(plot_biv_multiverse2  <- ggplot(pdat[pdat$GxE=="5%" & grepl('^5xAM',pdat$scenario),],
                                aes(gen,value, color=VT, linetype=GxE)) + default_theme +
    geom_hline(yintercept = 0, color='grey',lty=1) +
    stat_summary(geom = 'point',position=position_dodge(width=.3), fun=median) +
    stat_summary(geom = 'path',position=position_dodge(width=.3), fun=median) +
    ylab(expression(italic(r[g]))) +  xlab('Generations of xAM')      +
  facet_wrap(~variable,drop = T, label='label_parsed', scales='free') +
     default_theme2 + scale_linetype_manual(values=2)+

   scale_color_manual(values = PAL <- RColorBrewer::brewer.pal(8,'Set1')[-(5:7)])
                     )

(plot_biv_multiverse3  <- ggplot(pdat[pdat$GxE=="20%" & grepl('^5xAM',pdat$scenario),],
                                aes(gen,value, color=VT, linetype=GxE)) + default_theme +
    geom_hline(yintercept = 0, color='grey',lty=1) +
    stat_summary(geom = 'point',position=position_dodge(width=.3), fun=median) +
    stat_summary(geom = 'path',position=position_dodge(width=.3), fun=median) +
    ylab(expression(italic(r[g]))) +
    xlab('Generations of xAM')      +
  facet_wrap(~variable,drop = T, label='label_parsed', scales='free') +
     default_theme2 +scale_linetype_manual(values=3)+

   scale_color_manual(values = PAL <- RColorBrewer::brewer.pal(8,'Set1')[-(5:7)])
                     )

plot_s15 <- plot_grid(plot_biv_multiverse1 + xlab(''),
plot_biv_multiverse2 + xlab('')+ theme(strip.text = element_blank()),
plot_biv_multiverse3 + theme(strip.text = element_blank()), nrow=3,ncol=1)

ggsave(file.path(FIG_DIR, "sfig_s15_rg_gxe_grid.pdf"),
       plot_s15, width = 8, height = 9, bg = "white", dpi = 300)
cat("Saved sfig_s15_rg_gxe_grid.pdf\n")

## ── S16: PGI cross-phenotype slope, GxE as rows ────────────────────────────────
## From sFigComplexity_extended.ipynb cell 29 (data) + cell 37 pattern

## Cell 29: melt PGI data
pdat <- pdat2 <-
reshape2::melt(gdat, id.vars = c('gen','X','seed','args_m','VT','GxE','scenario','args_theta',
                                 'args_phi'),
               measure.vars = c("rbeta_hat_pgwas", "rbeta_hat_sgwas"
                                ))
pdat$variable <- as.character(pdat$variable)
pdat$variable[pdat2$variable=='rbeta_hat_pgwas'] <- 'Population~GWAS'
pdat$variable[pdat2$variable=='rbeta_hat_sgwas'] <- 'Sibship~GWAS'
pdat$variable <- as.factor(pdat$variable)
pdat$variable <- factor(pdat$variable, levels=rev(levels(pdat$variable)))

## GxE as rows for PGI (same pattern as cell 37)
(plot_biv_multiverse1  <- ggplot(pdat[pdat$GxE=="0%" & grepl('^5xAM',pdat$scenario),],
                                aes(gen,value, color=VT, linetype=GxE)) + default_theme +
    geom_hline(yintercept = 0, color='grey',lty=1) +
    stat_summary(geom = 'point',position=position_dodge(width=.3), fun=median) +
    stat_summary(geom = 'path',position=position_dodge(width=.3), fun=median) +
    ylab(expression(hat(r)[beta])) +    xlab('Generations of xAM')      +
  facet_wrap(~variable,drop = T, label='label_parsed', scales='free') +
     default_theme2 +

   scale_color_manual(values = PAL <- RColorBrewer::brewer.pal(8,'Set1')[-(5:7)])
                     )

(plot_biv_multiverse2  <- ggplot(pdat[pdat$GxE=="5%" & grepl('^5xAM',pdat$scenario),],
                                aes(gen,value, color=VT, linetype=GxE)) + default_theme +
    geom_hline(yintercept = 0, color='grey',lty=1) +
    stat_summary(geom = 'point',position=position_dodge(width=.3), fun=median) +
    stat_summary(geom = 'path',position=position_dodge(width=.3), fun=median) +
    ylab(expression(hat(r)[beta])) +  xlab('Generations of xAM')      +
  facet_wrap(~variable,drop = T, label='label_parsed', scales='free') +
     default_theme2 + scale_linetype_manual(values=2)+

   scale_color_manual(values = PAL <- RColorBrewer::brewer.pal(8,'Set1')[-(5:7)])
                     )

(plot_biv_multiverse3  <- ggplot(pdat[pdat$GxE=="20%" & grepl('^5xAM',pdat$scenario),],
                                aes(gen,value, color=VT, linetype=GxE)) + default_theme +
    geom_hline(yintercept = 0, color='grey',lty=1) +
    stat_summary(geom = 'point',position=position_dodge(width=.3), fun=median) +
    stat_summary(geom = 'path',position=position_dodge(width=.3), fun=median) +
    ylab(expression(hat(r)[beta])) +
    xlab('Generations of xAM')      +
  facet_wrap(~variable,drop = T, label='label_parsed', scales='free') +
     default_theme2 +scale_linetype_manual(values=3)+

   scale_color_manual(values = PAL <- RColorBrewer::brewer.pal(8,'Set1')[-(5:7)])
                     )

plot_s16 <- plot_grid(plot_biv_multiverse1 + xlab(''),
plot_biv_multiverse2 + xlab('')+ theme(strip.text = element_blank()),
plot_biv_multiverse3 + theme(strip.text = element_blank()), nrow=3,ncol=1)

ggsave(file.path(FIG_DIR, "sfig_s16_pgi_gxe_grid.pdf"),
       plot_s16, width = 8, height = 9, bg = "white", dpi = 300)
cat("Saved sfig_s16_pgi_gxe_grid.pdf\n")

cat("Done: plot_sfig_theory_gxe.R\n")
