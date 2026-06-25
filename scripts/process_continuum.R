#!/usr/bin/env Rscript
## process_fig_continuum_r5.R  (ROUND 5 — new sensitivity figure for C1/C4)
## Assembles the within→cross-trait VT-structure continuum and computes the same
## three decomposition tiers as the revised Fig 3 (Direct, Apparent, LDSC).
##
## Structures (total VT per trait held at θ=5%, only cross-trait-ness varies):
##   no VT (θ=0) · uVT same-trait (xvt 0) · weak-½total (xvt .125) ·
##   weak-½per-trait (xvt .5) · strong/constant (xvt 1 = the main-text model)
## Endpoints (no-VT, strong) from round-4 data; intermediates from the Lane xvt runs.
##
## Inputs:  round4 data/vdecomp_lane, data/vdecomp_h2_0_25  (no-VT + strong)
##          ~/data/vdecomp_xvt/results                       (uVT, weak-total, weak-perTrait)
## Output:  round5/fig3_r5/fig_continuum_r5_{h2,rg}_long.csv

library(reshape2)

R4   <- "/home/rsb/Dropbox/ftsim/round4"
XVT  <- path.expand("~/data/vdecomp_xvt/results")
OUT  <- "/home/rsb/Dropbox/ftsim/round4/round5/fig3_r5"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

exact_app <- function(K, h2, rg, cgy, cgyx) {
  tryCatch({
    a <- h2; b <- rg * h2; cc <- cgy; d <- cgyx
    G <- matrix(b, K, K); diag(G) <- a
    C <- matrix(d, K, K); diag(C) <- cc
    M <- crossprod(C, solve(G, C)); dg <- diag(M)
    Rm <- M / sqrt(outer(dg, dg)); c(mean(dg), mean(Rm[upper.tri(Rm)]))
  }, error = function(e) c(NA_real_, NA_real_))
}

## need only these columns; read them from every file and tag with xvt (from filename)
keep <- c("seed","gen","args_kphen","args_rmate","args_theta","args_h2",
          "h2_true","rg_true","cov_g_y","cov_g_y_cross","he_h2","he_rg")

read_one <- function(f) {
  d <- read.csv(f)
  if (!all(keep %in% names(d))) return(NULL)
  d <- d[d$args_kphen == 5, keep]
  if (nrow(d) == 0) return(NULL)
  m <- regmatches(basename(f), regexpr("_xvt([0-9.]+)_", basename(f)))
  d$xvt <- if (length(m)) as.numeric(sub("_xvt([0-9.]+)_", "\\1", m)) else NA_real_
  d
}

files <- c(
  list.files(file.path(R4, "data/vdecomp_lane"),     "_parsed\\.csv$", full.names = TRUE),
  list.files(file.path(R4, "data/vdecomp_h2_0_25"),  "_parsed\\.csv$", full.names = TRUE),
  list.files(XVT,                                    "_parsed\\.csv$", full.names = TRUE))
cat("Reading", length(files), "files...\n")
dat <- do.call(rbind, Filter(Negate(is.null), lapply(files, read_one)))
cat("Rows:", nrow(dat), "\n")

## ── structure + mating labels ──────────────────────────────────────────────
dat$structure <- NA_character_
dat$structure[dat$args_theta == 0]                        <- "no VT"
dat$structure[!is.na(dat$xvt) & dat$xvt == 0]             <- "uVT (same-trait)"
dat$structure[!is.na(dat$xvt) & dat$xvt == 0.125]         <- "weak (½ total)"
dat$structure[!is.na(dat$xvt) & dat$xvt == 0.5]           <- "weak (½ per-trait)"
dat$structure[is.na(dat$xvt) & dat$args_theta == 0.05]    <- "strong (constant)"
dat <- dat[!is.na(dat$structure), ]
dat$structure <- factor(dat$structure,
  levels = c("no VT","uVT (same-trait)","weak (½ total)","weak (½ per-trait)","strong (constant)"))

dat$mating <- ifelse(dat$args_rmate == 0, "RM", "5xAM")
dat$mating <- factor(dat$mating, levels = c("RM","5xAM"))

cat("Counts (gen5, h2=0.5):\n"); print(table(dat$structure[dat$gen==5 & dat$args_h2==0.5],
                                              dat$mating[dat$gen==5 & dat$args_h2==0.5]))

## ── exact apparent ─────────────────────────────────────────────────────────
app <- mapply(exact_app, dat$args_kphen, dat$h2_true, dat$rg_true, dat$cov_g_y, dat$cov_g_y_cross)
dat$apparent_h2 <- app[1, ]; dat$apparent_rg <- app[2, ]
dat$h2_he <- dat$he_h2;       dat$rg_he <- dat$he_rg

## ── long format (Direct, Apparent, LDSC) for h2 and rg ─────────────────────
h2_long <- melt(dat[, c("gen","seed","structure","mating","args_h2","h2_true","apparent_h2","h2_he")],
                id.vars = c("gen","seed","structure","mating","args_h2"),
                variable.name = "quantity", value.name = "h2")
h2_long$quantity <- factor(h2_long$quantity, levels = c("h2_true","apparent_h2","h2_he"),
                           labels = c("Direct h²","Apparent h²","LDSC h²"))
write.csv(h2_long, file.path(OUT, "fig_continuum_r5_h2_long.csv"), row.names = FALSE)

rg_long <- melt(dat[, c("gen","seed","structure","mating","args_h2","rg_true","apparent_rg","rg_he")],
                id.vars = c("gen","seed","structure","mating","args_h2"),
                variable.name = "quantity", value.name = "rg")
rg_long$quantity <- factor(rg_long$quantity, levels = c("rg_true","apparent_rg","rg_he"),
                           labels = c("Direct r_g","Apparent r_g","LDSC r_g"))
write.csv(rg_long, file.path(OUT, "fig_continuum_r5_rg_long.csv"), row.names = FALSE)
cat("Saved fig_continuum_r5_{h2,rg}_long.csv\n=== done ===\n")
