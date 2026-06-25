# Figure & Table → Code Inventory

Every figure and table in the manuscript "Simple models of non-random mating and
environmental transmission bias standard human genetics statistical methods"
(Border et al.) mapped to the script that produces it.

Pipeline convention: `process_*` loads raw simulation/data and writes intermediate
CSVs to `processed/`; `verify_*` prints the in-text numbers for comparison; `plot_*`
reads `processed/` and writes the figure. Simulation **data** are distributed
separately and are not stored in this repository (see `REPRODUCTION_README.md` → Data).

## Main figures

| Figure | Content | Script(s) |
|--------|---------|-----------|
| Fig 1 | Dimensionality of cross-mate correlations (CCA) and its simulation consequences: heritability, genetic correlation, and type-I error under 2× vs 5× xAM; analytic on/off-target associations | `process_fig1.R`, `verify_fig1.R`, `plot_fig1.R` |
| Fig 2 | Psychiatric-disorder simulations: genetic-correlation inflation under multivariate xAM | `process_fig2.R`, `verify_fig2.R`, `plot_fig2.R` |
| Fig 3 | Variance decomposition — direct, apparent, and estimated (LDSC) heritability and genetic correlation across RM / RM+VT / 5×AM / 5×AM+VT, with GWAS false-positive inflation | `process_fig3.R`, `plot_fig3.R` |
| Fig 4 | Education/height/wealth: joint architecture, heritability and PGI R², genetic correlations, GWAS effect correlations | `process_fig4.R`, `verify_fig4.R`, `plot_fig4.R` |

## Extended Data figures

| Figure | Content | Script(s) |
|--------|---------|-----------|
| ED 1 | Genetic-correlation estimates under xAM at cross-mate correlations 0.1 vs 0.2 | `plot_sfig_sensitivity.R` |
| ED 2 | Genetic-correlation estimates under unidimensional xAM, fixed latent correlation | `plot_sfig_sensitivity.R` |
| ED 3 | Theoretical vs observed type-I error rates at off-target loci | `plot_sfig_theory_gxe.R` |
| ED 4 | Variance decomposition at h² = 0.25 (mirrors Fig 3b–c) | `plot_edfig4.R` (data from `process_fig3.R`) |
| ED 5 | Sensitivity of the decomposition to vertical-transmission structure (h² = 0.5) | `process_continuum.R`, `plot_continuum.R` |
| ED 6 | Direct and estimated heritability across successive generations of 5-variate xAM | `process_sfig_theory_gxe.R`, `plot_sfig_theory_gxe.R` |
| ED 7 | Direct-PGI correlation and estimated effect correlation across generations | `process_sfig_theory_gxe.R`, `plot_sfig_theory_gxe.R` |
| ED 8 | Population vs within-sibship GWAS | `process_sfig_sibcomp.R`, `verify_sfig_sibcomp.R`, `plot_sfig_sibcomp.R` |
| ED 9 | Relative difference between LATE and ATE, single variant (linear scale) | `sibdiff.nb` (Mathematica symbolic) |
| ED 10 | Sensitivity of genetic-correlation estimates to within-person (correlated) noise | `process_cn.R`, `verify_cn.R`, `plot_cn.R` |

## Supplementary figures

| Figure | Content | Script(s) |
|--------|---------|-----------|
| S1 | Multivariate vs multidimensional linear mating (synthetic CCA) | `plot_sfig_cca.R` |
| S2 | Canonical correlation analysis of univariate nonlinear mating (synthetic) | `plot_sfig_cca.R` |
| S3 | Cross-mate CCA scree, predictive-mean-matching multiple imputation | `plot_sfig_cca.R` |
| S4 | Cross-mate CCA scree, alternative (random-forest) imputation | `plot_sfig_cca.R` |
| S5 | Incremental on- vs off-target associations as a function of sample size | `plot_sfig_analytic.R` |
| S6 | Type-I error inflation as a function of sample size (off-target, multivariate xAM) | `plot_sfig_analytic.R` |
| S7 | On- vs off-target associations vs sample size after five generations | `plot_sfig_analytic.R` |
| S8 | Expected false-positive rate by off-target effect magnitude and sample size | `plot_sfig_analytic.R` |
| S9 | Direct and estimated heritability across generations (alternative presentation of ED 6) | `process_sfig_theory_gxe.R`, `plot_sfig_theory_gxe.R` |
| S10 | Direct-PGI / estimated-effect correlation across generations (alternative presentation of ED 7) | `process_sfig_theory_gxe.R`, `plot_sfig_theory_gxe.R` |
| S11 | Relative type-I error inflation, within-sibship vs population GWAS | `plot_sfig_sibcomp.R` |
| S12 | Within-sibship vs population GWAS slope correlation, representative vs ascertained samples | ascertainment estimators — data not yet accessible |
| S13 | Bias in within-sibship vs population GWAS slopes, representative vs ascertained samples | ascertainment estimators — data not yet accessible |
| S14 | VT / G×E-induced bias of sibling-difference GWAS-derived PGI | sib-difference estimators — data not yet accessible |
| S15 | True ATE PGI vs true individual-specific PGI | `test_sib_diff` (sib-difference estimators — data not yet accessible) |
| S16 | Computational performance of xftsim (runtime and memory scaling) | `process_benchmarks.R`, `verify_benchmarks.R`, `plot_benchmarks.R` |
| S17 | Decomposition sensitivity to VT structure at h² = 0.25 (companion to ED 5) | `process_continuum.R`, `plot_continuum.R` |
| S18 | Relative difference between LATE and ATE, single variant (log scale) | `sibdiff.nb` (Mathematica symbolic) |

## Tables

| Table | Content | Script(s) |
|-------|---------|-----------|
| Table 1 | xftsim capabilities | static text |
| S1 | Phenotype definitions | `pheno_defn_supplement.r` |
| S2 | Cross-mate canonical correlation analysis results | `process_tables.R`, `verify_tables.R` |
| S3 | Cross-mate correlations (UK Biobank) | `process_tables.R`, `verify_tables.R` |
| S4 | Cross-mate correlations (Taiwan NHIRD) | `nhird_cca.r` |
| S5 | Median true and estimated variance components (generation 0 vs 5) | `process_tables.R`, `verify_tables.R` |
| S6 | Phenotype-agnostic simulation model parameters | static text |
| S7 | VT-structure continuum: variance decomposition at generation 5 | `make_supp_tables.py` |
| S8 | Main-scenario variance decomposition at generation 5 | `make_supp_tables.py` |

## Simulation drivers (Python)

| Script | Produces |
|--------|----------|
| `general_simulations.py` | Core complex-architecture sims feeding Fig 1, Fig 3, and most Extended Data / Supplementary figures |
| `vt_structure_sims.py` | Vertical-transmission-structure continuum sims (Fig 3, ED 5, S17, Tables S7/S8) |
| `compute_apparent.py` | Best-linear-predictor ("apparent") decomposition from saved phenotypes |
| `aggregate_vt.py` | Aggregation of continuum sim replicates across seeds |
| `psychiatric_sims.py` | Psychiatric-disorder sims (Fig 2) |
| `ascertainment_sims.py` | Ascertainment-design sims (Supp S12–S14) |
| `ey_sim.py` | Education/height/wealth sims (Fig 4) |

## Notes

- **Data not yet accessible:** S12–S15 depend on CDEF ascertainment / sibling-difference
  estimator outputs whose source the authors retain but which are not on current machines.
- **Mathematica:** ED 9 / S18 are symbolic computations in `sibdiff.nb`.
- **Fig 4 raw data** (`edu_no_CD_LS.01/`) is not accessible; the plot uses cached
  reference tables from the analysis notebook (see `REPRODUCTION_README.md`).
