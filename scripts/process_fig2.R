## process_fig2.R
## Loads raw psychiatric simulation data, computes summary statistics,
## and saves processed tables to processed/fig2_*.csv

library(reshape2)
library(stringr)

BASE_DIR <- "/home/rsb/Dropbox/ftsim/round4"
DATA_DIR <- file.path(BASE_DIR, "data", "psych_sims")
OUT_DIR  <- file.path(BASE_DIR, "processed")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

## ── 1. Load raw data ────────────────────────────────────────────────────────

he  <- read.csv(file.path(DATA_DIR, "psychHEres.csv"))
he2 <- read.csv(file.path(DATA_DIR, "psychHEres2way.csv"))
he6 <- read.csv(file.path(DATA_DIR, "psychHEres6way.csv"))
g2  <- read.csv(file.path(DATA_DIR, "psych2way_gwas.csv"))
g6  <- read.csv(file.path(DATA_DIR, "psych6way_gwas.csv"))

## ── 2. Clean column names ───────────────────────────────────────────────────

clean_names <- function(df) {
  names(df) <- gsub(".phenotype", "", names(df), fixed = TRUE)
  names(df) <- gsub(".addGen",    "", names(df), fixed = TRUE)
  names(df) <- gsub(".proband",   "", names(df), fixed = TRUE)
  df
}

he  <- clean_names(he)
he2 <- clean_names(he2)
he6 <- clean_names(he6)

## ── 3. Bivariate HE summaries (bagg) ────────────────────────────────────────
## Melt the combined HE table into long form with outcome / pheno_1 / pheno_2

bvars <- grep("h2HE", grep("_.+_", names(he), val = TRUE), val = TRUE, invert = TRUE)
ivars <- c("seed", "gen", "xAM")

bdat <- melt(he, measure.vars = bvars, id.vars = ivars)
bdat <- bdat[!is.na(bdat$value), ]

tmpb <- str_split_fixed(bdat$variable, "_", 3)
bdat$outcome <- tmpb[, 1]
bdat$pheno_1 <- tmpb[, 2]
bdat$pheno_2 <- tmpb[, 3]

bdat <- bdat[bdat$pheno_1 != bdat$pheno_2, ]

## Aggregate: median and sd across seeds
bagg <- aggregate(bdat["value"],
                  bdat[c("gen", "xAM", "outcome", "pheno_1", "pheno_2")],
                  median)
bagg$se <- aggregate(bdat["value"],
                     bdat[c("gen", "xAM", "outcome", "pheno_1", "pheno_2")],
                     sd)$value

write.csv(bagg, file.path(OUT_DIR, "fig2_bagg.csv"), row.names = FALSE)
cat("Saved fig2_bagg.csv:", nrow(bagg), "rows\n")

## ── 4. Gamma panel data (gdat_gamma) ────────────────────────────────────────
## Need traits column and deduplication for the gamma panel

bdat$traits <- apply(bdat[c("pheno_1", "pheno_2")], 1,
                     function(x) paste(sort(x), collapse = " / "))

med   <- aggregate(bdat["value"],
                   bdat[c("gen", "xAM", "outcome", "pheno_1", "pheno_2", "traits")],
                   median)
med1  <- aggregate(bdat["value"],
                   bdat[c("gen", "xAM", "outcome", "pheno_1", "pheno_2", "traits")],
                   quantile, 0.9)
med9  <- aggregate(bdat["value"],
                   bdat[c("gen", "xAM", "outcome", "pheno_1", "pheno_2", "traits")],
                   quantile, 0.1)
medsd <- aggregate(bdat["value"],
                   bdat[c("gen", "xAM", "outcome", "pheno_1", "pheno_2", "traits")],
                   sd)

med$lower <- med1$value
med$upper <- med9$value
med$sd    <- medsd$value

gdat_gamma <- med[med$outcome == "gamma" & med$gen %in% c(0, 1, 3, 5), ]
gdat_gamma$traits2 <- apply(
  gdat_gamma[c("pheno_1", "pheno_2", "gen", "xAM")], 1,
  function(x) paste(sort(x), collapse = " / ")
)
gdat_gamma <- gdat_gamma[!duplicated(gdat_gamma$traits2), ]

write.csv(gdat_gamma, file.path(OUT_DIR, "fig2_gamma.csv"), row.names = FALSE)
cat("Saved fig2_gamma.csv:", nrow(gdat_gamma), "rows\n")

## ── 5. Weighted-average gamma at gen 5 ──────────────────────────────────────

pldat <- gdat_gamma[gdat_gamma$gen == 5, ]
pldat2 <- pldat[pldat$xAM == "2-variate", ]
pldat6 <- pldat[pldat$xAM == "6-variate", ]

av2 <- with(pldat2, sum(value / sd^2) / sum(1 / sd^2))
sd2 <- with(pldat2, sqrt(1 / sum(1 / sd^2)))
av6 <- with(pldat6, sum(value / sd^2) / sum(1 / sd^2))
sd6 <- with(pldat6, sqrt(1 / sum(1 / sd^2)))

gamma_avg <- data.frame(
  xAM   = c("2-variate", "6-variate"),
  avg   = c(av2, av6),
  sd    = c(sd2, sd6),
  lower = c(av2 + qnorm(.05) * sd2, av6 + qnorm(.05) * sd6),
  upper = c(av2 + qnorm(.95) * sd2, av6 + qnorm(.95) * sd6)
)

write.csv(gamma_avg, file.path(OUT_DIR, "fig2_gamma_avg.csv"), row.names = FALSE)
cat("Saved fig2_gamma_avg.csv\n")

## ── 6. Prevalence data ──────────────────────────────────────────────────────

pvars <- grep("prev", names(he), val = TRUE)
prdat <- melt(he, measure.vars = pvars, id.vars = ivars)
tmpr  <- str_split_fixed(prdat$variable, "_", 2)
prdat$dx <- tmpr[, 2]

prdat <- aggregate(prdat["value"],
                   prdat[c("seed", "gen", "xAM", "dx")],
                   mean, na.rm = TRUE)

write.csv(prdat, file.path(OUT_DIR, "fig2_prevalence.csv"), row.names = FALSE)
cat("Saved fig2_prevalence.csv:", nrow(prdat), "rows\n")

## ── 7. GWAS slope correlations (rbeta) ──────────────────────────────────────

## Aggregate g2 across trait-pairs within seed/gen
gvars <- grepl("^pgwas", names(g2)) | grepl("^sgwas", names(g2)) | grepl("^rbeta", names(g2))
g2agg <- aggregate(g2[gvars],
                   g2[c("seed", "gen", "args_kphen", "args_kmate")],
                   mean, na.rm = TRUE)

## Melt g2 (2-variate)
mvars <- grep("^rbeta_hat", names(g2agg), val = TRUE)
gdat2 <- melt(g2agg, id.vars = c("seed", "gen"), measure.vars = mvars)
gdat2$xAM  <- "2-variate"
gdat2$GWAS <- "Population"
gdat2$GWAS[grepl("sgwas", gdat2$variable)] <- "Sibship"
tmp2 <- str_split_fixed(gdat2$variable, "_", 5)
gdat2$Trait_1 <- str_split_fixed(tmp2[, 4], ".pheno", 2)[, 1]
gdat2$Trait_2 <- str_split_fixed(tmp2[, 5], ".pheno", 2)[, 1]
gdat2 <- gdat2[gdat2$Trait_1 != gdat2$Trait_2, ]

## Melt g6 (6-variate)
mvars6 <- grep("^rbeta_hat", names(g6), val = TRUE)
gdat6 <- melt(g6, id.vars = c("seed", "gen"), measure.vars = mvars6)
gdat6$xAM  <- "6-variate"
gdat6$GWAS <- "Population"
gdat6$GWAS[grepl("sgwas", gdat6$variable)] <- "Sibship"
tmp6 <- str_split_fixed(gdat6$variable, "_", 5)
gdat6$Trait_1 <- str_split_fixed(tmp6[, 4], ".pheno", 2)[, 1]
gdat6$Trait_2 <- str_split_fixed(tmp6[, 5], ".pheno", 2)[, 1]
gdat6 <- gdat6[gdat6$Trait_1 != gdat6$Trait_2, ]

## Combine
gdat_rbeta <- rbind.data.frame(gdat2, gdat6)
gdat_rbeta$pair <- paste(gdat_rbeta$Trait_1, " / ", gdat_rbeta$Trait_2)

## Add sorted trait label for deduplication
gdat_rbeta$traits <- apply(gdat_rbeta[c("Trait_1", "Trait_2")], 1,
                           function(x) paste(sort(x), collapse = " / "))

write.csv(gdat_rbeta, file.path(OUT_DIR, "fig2_rbeta.csv"), row.names = FALSE)
cat("Saved fig2_rbeta.csv:", nrow(gdat_rbeta), "rows\n")

## ── 8. GWAS slope correlation averages at gen 5 ─────────────────────────────

## Summarize pgwas rbeta across all pairs at gen 5
pgwas_sub <- gdat_rbeta[grepl("pgwas", gdat_rbeta$variable) & gdat_rbeta$gen == 5, ]

rbeta_avg <- data.frame(
  xAM = c("2-variate", "6-variate"),
  avg = c(
    mean(pgwas_sub$value[pgwas_sub$xAM == "2-variate"], na.rm = TRUE),
    mean(pgwas_sub$value[pgwas_sub$xAM == "6-variate"], na.rm = TRUE)
  ),
  se = c(
    sd(pgwas_sub$value[pgwas_sub$xAM == "2-variate"], na.rm = TRUE) /
      sqrt(sum(!is.na(pgwas_sub$value[pgwas_sub$xAM == "2-variate"]))),
    sd(pgwas_sub$value[pgwas_sub$xAM == "6-variate"], na.rm = TRUE) /
      sqrt(sum(!is.na(pgwas_sub$value[pgwas_sub$xAM == "6-variate"])))
  )
)

## Also compute using intercept-only regression as in notebook (cell 18)
## The notebook formats pair means, then runs lm(x~1) across pairs at gen 5
## First get pair means (as the notebook does via format(mean(x), digits=2))
pgwas_pair_means <- aggregate(
  pgwas_sub["value"],
  pgwas_sub[c("gen", "pair", "xAM")],
  mean, na.rm = TRUE
)

lm2 <- summary(lm(value ~ 1, data = pgwas_pair_means[pgwas_pair_means$xAM == "2-variate", ]))
lm6 <- summary(lm(value ~ 1, data = pgwas_pair_means[pgwas_pair_means$xAM == "6-variate", ]))

rbeta_lm <- data.frame(
  xAM = c("2-variate", "6-variate"),
  intercept = c(coef(lm2)[1, "Estimate"], coef(lm6)[1, "Estimate"]),
  se        = c(coef(lm2)[1, "Std. Error"], coef(lm6)[1, "Std. Error"])
)

write.csv(rbeta_avg,  file.path(OUT_DIR, "fig2_rbeta_avg.csv"), row.names = FALSE)
write.csv(rbeta_lm,   file.path(OUT_DIR, "fig2_rbeta_lm.csv"),  row.names = FALSE)
cat("Saved fig2_rbeta_avg.csv and fig2_rbeta_lm.csv\n")

## ── 9. Heatmap data (for panel a) ──────────────────────────────────────────

heatdat <- bagg[bagg$outcome != "gamma" & bagg$gen %in% c(0, 1, 3, 5), ]

## Arrange so 2-variate is super-diagonal, 6-variate is sub-diagonal
tmpd1 <- heatdat[heatdat$xAM == "2-variate", ]
tmpd2 <- heatdat[heatdat$xAM == "6-variate", ]
## Swap pheno_1/pheno_2 for 6-variate to place on sub-diagonal
tmpd2_swap       <- tmpd2
tmpd2_swap$pheno_1 <- tmpd2$pheno_2
tmpd2_swap$pheno_2 <- tmpd2$pheno_1

pdat_heat <- rbind.data.frame(tmpd1, tmpd2_swap)
pdat_heat$Outcome <- "hat(italic(r))[HE]"
pdat_heat$Outcome[pdat_heat$outcome == "rgtrue"] <- "italic(r)[score]"
pdat_heat$Outcome <- factor(pdat_heat$Outcome,
                            levels = rev(c("hat(italic(r))[HE]",
                                           "italic(r)[score]")))
pdat_heat$Generation <- paste(pdat_heat$gen, "gen.~xAM", sep = "~")

write.csv(pdat_heat, file.path(OUT_DIR, "fig2_heatmap.csv"), row.names = FALSE)
cat("Saved fig2_heatmap.csv:", nrow(pdat_heat), "rows\n")

cat("\n=== process_fig2.R complete ===\n")
