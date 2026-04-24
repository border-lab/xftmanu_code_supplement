## plot_sfig_cca.R
## Verbatim port of notebook code for supplementary figures S1, S2, S3.
##   S1: Multivariate vs multidimensional mating regimes (synthetic pair plots + CCA)
##        from dimn_plot.ipynb cells 0-4
##   S2: Nonlinear mating CCA (piecewise linear, atan, quadratic)
##        from dimn_plot.ipynb cells 8, 10, 11, 12
##   S3: CCA scree plot from MICE-imputed data
##        from sFig_cca.ipynb cells 1, 2, 4, 6, 15-18

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
FIG_DIR  <- file.path(BASE_DIR, "figures_output")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

## ===================================================================
## S1: Multivariate vs multidimensional mating regimes
## dimn_plot.ipynb cell 0 (libraries / setup)
## ===================================================================

library(yacca)
library(repr)
library(ggplot2)
library(reshape2)
library(cowplot)
library(stringr)
library(splines)
library(MASS)
library(GGally)
library(gridExtra)
library(grid)

default_theme <- theme_bw() + theme(text=element_text(size=12))
default_theme2 <- theme_minimal() + theme(text=element_text(size=12))
PAL <- RColorBrewer::brewer.pal(8,'Set1')

## -------------------------------------------------------------------
## dimn_plot.ipynb cell 1: multivariate, multidimensional (r3)
## -------------------------------------------------------------------
set.seed(1)

a <- rnorm(3)
b <- rnorm(3)
c <- rnorm(3)

W <- a %o%a + b %o%b+ c %o%c

z_cor <- diag(rep(1,3))
Z <- MASS::mvrnorm(1e3, rep(0,3), z_cor)

X1 <-  scale(Z %*% W + MASS::mvrnorm(1e3, rep(0,3), z_cor))
X2 <-  scale(Z %*% W + MASS::mvrnorm(1e3, rep(0,3), z_cor))
ccares <- cca(X1,X2)

C1 <- ccares$canvarx
C2 <- ccares$canvary

Xdat_r3 <- cbind(X1,X2)
colnames(Xdat_r3) <- paste0('Mate ',rep(1:2, each=3),
                        ': pheno ',rep(1:3,times=2))
Cdat_r3 <- cbind(C1,C2)
colnames(Cdat_r3) <- paste0('Mate ',rep(1:2, each=3),
                           ': CV',rep(1:3,times=2))

Xdat_r3 <- as.data.frame(Xdat_r3)
Cdat_r3 <- as.data.frame(Cdat_r3)

## -------------------------------------------------------------------
## dimn_plot.ipynb cell 2: multivariate, unidimensional (r1)
## -------------------------------------------------------------------
set.seed(1)

a <- rnorm(3)
a <- a + min(a)
b <- rnorm(3)
c <- rnorm(3)

W <- a %o%a*1.3# + b %o%b+ c %o%c

z_cor <- diag(rep(1,3))
Z <- MASS::mvrnorm(1e3, rep(0,3), z_cor)

X1 <-  scale(Z %*% W + MASS::mvrnorm(1e3, rep(0,3), z_cor))
X2 <-  scale(Z %*% W + MASS::mvrnorm(1e3, rep(0,3), z_cor))
ccares <- cca(X1,X2)

C1 <- ccares$canvarx
C2 <- ccares$canvary

Xdat_r1 <- cbind(X1,X2)
colnames(Xdat_r1) <- paste0('Mate ',rep(1:2, each=3),
                        ': pheno ',rep(1:3,times=2))
Cdat_r1 <- cbind(C1,C2)
colnames(Cdat_r1) <- paste0('Mate ',rep(1:2, each=3),
                           ': CV',rep(1:3,times=2))

Xdat_r1 <- as.data.frame(Xdat_r1)
Cdat_r1 <- as.data.frame(Cdat_r1)

## -------------------------------------------------------------------
## dimn_plot.ipynb cell 3: univariate, unidimensional (u1)
## -------------------------------------------------------------------
set.seed(1)

a <- rnorm(3)
a <- c(1,0,0)*2
W <- a %o%a# + b %o%b+ c %o%c

z_cor <- diag(rep(1,3))
Z <- MASS::mvrnorm(1e3, rep(0,3), z_cor)

X1 <-  scale(Z %*% W + MASS::mvrnorm(1e3, rep(0,3), z_cor))
X2 <-  scale(Z %*% W + MASS::mvrnorm(1e3, rep(0,3), z_cor))
ccares <- cca(X1,X2)

C1 <- ccares$canvarx
C2 <- ccares$canvary

Xdat_u1 <- cbind(X1,X2)
colnames(Xdat_u1) <- paste0('Mate ',rep(1:2, each=3),
                        ': pheno ',rep(1:3,times=2))
Cdat_u1 <- cbind(C1,C2)
colnames(Cdat_u1) <- paste0('Mate ',rep(1:2, each=3),
                           ': CV ',rep(1:3,times=2))

Xdat_u1 <- as.data.frame(Xdat_u1)
Cdat_u1 <- as.data.frame(Cdat_u1)

## -------------------------------------------------------------------
## dimn_plot.ipynb cell 4: S1 pair plots + composite figure
## -------------------------------------------------------------------
r1x <- grid.grabExpr(print(ggduo(Xdat_r1, names(Xdat_r1)[1:2], names(Xdat_r1)[4:5], mapping=aes(alpha=.01),
      types = list(continuous = "points")) +
    stat_smooth(color=2, method='lm', formula=y~x, se=F) + default_theme+ggtitle('Multivariate, unidimensional')))

r1c <- grid.grabExpr(print(ggduo(Cdat_r1, names(Cdat_r1)[1:2], names(Cdat_r1)[4:5], mapping=aes(alpha=.01),
      types = list(continuous = "points")) +
    stat_smooth(color=4, method='lm', formula=y~x, se=F) + default_theme+ggtitle(' ')))

r3x <- grid.grabExpr(print(ggduo(Xdat_r3, names(Xdat_r3)[1:2], names(Xdat_r3)[4:5], mapping=aes(alpha=.01),
      types = list(continuous = "points")) +
    stat_smooth(color=2, method='lm', formula=y~x, se=F) + default_theme+ggtitle('Multivariate, multidimensional')))

r3c <- grid.grabExpr(print(ggduo(Cdat_r3, names(Cdat_r3)[1:2], names(Cdat_r3)[4:5], mapping=aes(alpha=.01),
      types = list(continuous = "points")) +
    stat_smooth(color=4, method='lm', formula=y~x, se=F) + default_theme+ggtitle(' ')))

u1x <- grid.grabExpr(print(ggduo(Xdat_u1, names(Xdat_u1)[1:2], names(Xdat_r3)[4:5], mapping=aes(alpha=.01),
      types = list(continuous = "points")) +
    stat_smooth(color=2, method='lm', formula=y~x, se=F) + default_theme+ggtitle('Univariate, unidimensional')))

u1c <- grid.grabExpr(print(ggduo(Cdat_u1, names(Cdat_u1)[1:2], names(Cdat_u1)[4:5], mapping=aes(alpha=.01),
      types = list(continuous = "points")) +
    stat_smooth(color=4, method='lm', formula=y~x, se=F) + default_theme+ggtitle(' ')))

options(repr.plot.width=10)
options(repr.plot.height=12)
aa <- align_plots(u1x,u1c,r1x,r1c,r3x,r3c, axis='tblr', align='hv')
sfig_s1 <- plot_grid(aa[[1]],aa[[2]],aa[[3]],aa[[4]],aa[[5]],aa[[6]], ncol=2,
          labels=c('a','','b','','c',''))

ggsave(file.path(FIG_DIR, "sfig_s1_mating_cca.pdf"),
       sfig_s1, width = 10, height = 12, bg = "white")
ggsave(file.path(FIG_DIR, "sfig_s1_mating_cca.png"),
       sfig_s1, width = 10, height = 12, dpi = 300, bg = "white")

## ===================================================================
## S2: Nonlinear mating CCA
## ===================================================================

## -------------------------------------------------------------------
## dimn_plot.ipynb cell 8: piecewise linear (nl1)
## -------------------------------------------------------------------
set.seed(1)

a <- rnorm(3)
a <- c(1,0,0)*2
W <- a %o%a# + b %o%b+ c %o%c

z_cor <- diag(rep(1,3))
Z2<- Z <- MASS::mvrnorm(1.2e3, rep(0,3), z_cor)
Z[,1] <- scale((Z[,1]*(Z[,1]<0)))*2 + scale((Z[,1]*(Z[,1]>=0)))*.15
X1 <-  Z %*% W + MASS::mvrnorm(1.2e3, rep(0,3), z_cor)
X2 <-  Z2 %*% W + MASS::mvrnorm(1.2e3, rep(0,3), z_cor)


ccares <- cca(X1,X2)

C1 <- ccares$canvarx
C2 <- ccares$canvary

Xdat_nl1 <- cbind(X1,X2)
colnames(Xdat_nl1) <- paste0('Mate ',rep(1:2, each=3),
                        ': pheno ',rep(1:3,times=2))
Cdat_nl1 <- cbind(C1,C2)
colnames(Cdat_nl1) <- paste0('Mate ',rep(1:2, each=3),
                           ': CV ',rep(1:3,times=2))

Xdat_nl1 <- as.data.frame(Xdat_nl1)
Cdat_nl1 <- as.data.frame(Cdat_nl1)

## -------------------------------------------------------------------
## dimn_plot.ipynb cell 10: inverse tangent / atan (nl2)
## -------------------------------------------------------------------
set.seed(1)

a <- rnorm(3)
a <- c(1,0,0)*2
W <- a %o%a# + b %o%b+ c %o%c

z_cor <- diag(rep(1,3))
Z2<- Z <- MASS::mvrnorm(1.2e3, rep(0,3), z_cor)
Z[,1] <- scale((atan(Z[,1]*2)))*1.5
X1 <-  Z %*% W + MASS::mvrnorm(1.2e3, rep(0,3), z_cor)
X2 <-  Z2 %*% W + MASS::mvrnorm(1.2e3, rep(0,3), z_cor)


ccares <- cca(X1,X2)

C1 <- ccares$canvarx
C2 <- ccares$canvary

Xdat_nl2 <- cbind(X1,X2)
colnames(Xdat_nl2) <- paste0('Mate ',rep(1:2, each=3),
                        ': pheno ',rep(1:3,times=2))
Cdat_nl2 <- cbind(C1,C2)
colnames(Cdat_nl2) <- paste0('Mate ',rep(1:2, each=3),
                           ': CV ',rep(1:3,times=2))

Xdat_nl2 <- as.data.frame(Xdat_nl2)
Cdat_nl2 <- as.data.frame(Cdat_nl2)

## -------------------------------------------------------------------
## dimn_plot.ipynb cell 11: quadratic (nl3)
## -------------------------------------------------------------------
set.seed(1)

a <- rnorm(3)
a <- c(1,0,0)*2
W <- a %o%a# + b %o%b+ c %o%c

z_cor <- diag(rep(1,3))
Z2<- Z <- MASS::mvrnorm(1.2e3, rep(0,3), z_cor)
Z[,1] <- scale((Z[,1]**2))-5
X1 <-  Z2 %*% W + MASS::mvrnorm(1.2e3, rep(0,3), z_cor)
X2 <-  Z %*% W + MASS::mvrnorm(1.2e3, rep(0,3), z_cor)


ccares <- cca(X1,X2)

C1 <- ccares$canvarx
C2 <- ccares$canvary

Xdat_nl3 <- cbind(X1,X2)
colnames(Xdat_nl3) <- paste0('Mate ',rep(1:2, each=3),
                        ': pheno ',rep(1:3,times=2))
Cdat_nl3 <- cbind(C1,C2)
colnames(Cdat_nl3) <- paste0('Mate ',rep(1:2, each=3),
                           ': CV ',rep(1:3,times=2))

Xdat_nl3 <- as.data.frame(Xdat_nl3)
Cdat_nl3 <- as.data.frame(Cdat_nl3)

## -------------------------------------------------------------------
## dimn_plot.ipynb cell 12: S2 pair plots + composite figure
## -------------------------------------------------------------------
r1x <- grid.grabExpr(print(ggduo(Xdat_nl1, names(Xdat_nl1)[1:2], names(Xdat_nl1)[4:5], mapping=aes(alpha=.01),
      types = list(continuous = "points")) +
    stat_smooth(color=2, method='lm', formula=y~x, se=F) + default_theme))

r1c <- grid.grabExpr(print(ggduo(Cdat_nl1, names(Cdat_nl1)[1:2], names(Cdat_nl1)[4:5], mapping=aes(alpha=.01),
      types = list(continuous = "points")) +
    stat_smooth(color=4, method='lm', formula=y~x, se=F) + default_theme))

r3x <- grid.grabExpr(print(ggduo(Xdat_nl3, names(Xdat_nl3)[1:2], names(Xdat_nl3)[4:5], mapping=aes(alpha=.01),
      types = list(continuous = "points")) +
    stat_smooth(color=2, method='lm', formula=y~x, se=F) + default_theme))

r3c <- grid.grabExpr(print(ggduo(Cdat_nl3, names(Cdat_nl3)[1:2], names(Cdat_nl3)[4:5], mapping=aes(alpha=.01),
      types = list(continuous = "points")) +
    stat_smooth(color=4, method='lm', formula=y~x, se=F) + default_theme))

u1x <- grid.grabExpr(print(ggduo(Xdat_nl2, names(Xdat_nl2)[1:2], names(Xdat_nl2)[4:5], mapping=aes(alpha=.01),
      types = list(continuous = "points")) +
    stat_smooth(color=2, method='lm', formula=y~x, se=F) + default_theme))

u1c <- grid.grabExpr(print(ggduo(Cdat_nl2, names(Cdat_nl2)[1:2], names(Cdat_nl2)[4:5], mapping=aes(alpha=.01),
      types = list(continuous = "points")) +
    stat_smooth(color=4, method='lm', formula=y~x, se=F) + default_theme))

options(repr.plot.width=10)
options(repr.plot.height=12)
aa <- align_plots(u1x,u1c,r1x,r1c,r3x,r3c, axis='tblr', align='hv')
sfig_s2 <- plot_grid(aa[[3]],aa[[4]],aa[[1]],aa[[2]],aa[[5]],aa[[6]], ncol=2,
          labels=c('a','','b','','c',''))

ggsave(file.path(FIG_DIR, "sfig_s2_nonlinear_cca.pdf"),
       sfig_s2, width = 10, height = 12, bg = "white")
ggsave(file.path(FIG_DIR, "sfig_s2_nonlinear_cca.png"),
       sfig_s2, width = 10, height = 12, dpi = 300, bg = "white")

## ===================================================================
## S3: CCA scree plot from MICE-imputed data
## sFig_cca.ipynb cells 4, 6, 15-18
## ===================================================================

## -------------------------------------------------------------------
## sFig_cca.ipynb cell 4: load CCA results
## -------------------------------------------------------------------
load(file.path(BASE_DIR, 'data/cca/mice_imputed_cca_final_rf.rdata'), verbose=T)

## -------------------------------------------------------------------
## sFig_cca.ipynb cell 6: compute relative canonical redundancy
## -------------------------------------------------------------------
cca_vars <-
apply(sapply(lcca,  function(out) {
    tmp  = out$ycrosscorr
tmp = rbind(tmp, rep(NA,ncol(tmp)))
rownames(tmp)[nrow(tmp)] <- 'age_at_menarche_mean'
tmp2 <- out$xcrosscorr
    rownames(tmp2) <- rownames(tmp); colnames(tmp2) <- colnames(tmp)

xdat <- (tmp2 + tmp)/2
xdat[is.na(xdat)] <- tmp2[is.na(xdat)]
# xdat <- (xdat)
cca_vars <- (out$xvrd/ out$xrd + out$yvrd/ out$yrd)/2
    cca_vars
}), 1, min)
plot(cumsum(cca_vars))

## -------------------------------------------------------------------
## sFig_cca.ipynb cell 15: build loadings + scree data across imputations
## -------------------------------------------------------------------
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


KK=12
zz=0
xdat2 <- (xdat[apply(abs(xdat),1,function(x) any(x>.05)),])
pdat <- melt(xdat2[,1:KK],varnames = c('Phenotype','CanonicalVariate'), value.name='Canonical cross-loadings')
KK=12
zz=0
xdat2 <- (xdat[apply(abs(xdat),1,function(x) any(x>.05)),])
pdat <- melt(xdat2[,1:KK],varnames = c('Phenotype','CanonicalVariate'), value.name='Canonical cross-loadings')



pdat <- pdat[apply(pdat,1,function(x) any(x>.5)),]

pdat$Phenotype <- factor(pdat$Phenotype,
                         levels=names(sort(tapply(abs(pdat$`Canonical cross-loadings`),
                                                  pdat['Phenotype'],function(x) sum(abs(x))))))
pdat$CanonicalVariate = factor(pdat$CanonicalVariate, levels=c('',paste('CV', 1:KK)))
d2 <- data.frame(CanonicalVariate=factor(c('',paste('CV', 1:(KK-zz))), levels=c('',paste('CV', 1:(KK-zz)))))
d2$`Relative Canonical\nRedundancy`<-cumsum(c(0,cca_vars[1:(KK-zz)]))
list(pdat,d2)})

## -------------------------------------------------------------------
## sFig_cca.ipynb cell 16: aggregate scree across imputations
## -------------------------------------------------------------------
d2 <- do.call(rbind.data.frame, mapply(function(X,i) {X[[2]]$iterate <- i;X[[2]]},
                                         dat, 1:length(dat),SIMPLIFY = F))

## -------------------------------------------------------------------
## sFig_cca.ipynb cell 17: median across imputations
## -------------------------------------------------------------------
d2 <- aggregate(d2[2], d2[1], median)

## -------------------------------------------------------------------
## sFig_cca.ipynb cell 18: S3 scree plot
## -------------------------------------------------------------------
options(repr.plot.width=8)
options(repr.plot.height=6)
options(repr.plot.res=300)


sfig_s3 <- ggplot(d2, aes(CanonicalVariate, `Relative Canonical\nRedundancy`)) +
                                                  scale_y_continuous(position = 'left',limits = 0:1)+
                                                  scale_x_discrete(position = 'bottom')+
                                                  geom_hline(col='purple',lty=3,yintercept = .9)+
                                                  geom_hline(col='darkorange',lty=3,yintercept = .95)+
     geom_point() + geom_line(group=1) + xlab('Canonical Variate') +
     theme_bw() + theme(#axis.title.y.left = element_blank(),axis.title.x = element_blank(),
                       text=element_text(size=14), legend.position = 'bottom')

ggsave(file.path(FIG_DIR, "sfig_s3_cca_scree.pdf"),
       sfig_s3, width = 8, height = 6, bg = "white")
ggsave(file.path(FIG_DIR, "sfig_s3_cca_scree.png"),
       sfig_s3, width = 8, height = 6, dpi = 300, bg = "white")
