# Complete Figure & Analysis Inventory

## Data Availability Summary

### Available Locally (Dropbox)
| File | Location | Used By |
|------|----------|---------|
| merged_tabla_redux_results_011024.csv | ftsim/manu/ | Fig 1 (mFigComplexity), Fig 3 (mFigHDxAM), S5-S16 (sFigComplexityExtra), S17 (mFigSibComp) |
| merged_res_redux_120725.csv | ftsim/ak_resp/ | Revision figs: fig_cn_updated.R, fig_variance_decomp.R |
| corrnoise_res_120925.csv | ftsim/ak_resp/ | fig_cn_updated.R (CN=0.2) |
| corrnoise1_res_120925.csv | ftsim/ak_resp/ | fig_cn_updated.R (CN=0.1) |
| psychHEres.csv | ftsim/manu/simulations/ | Fig 2 (mFigPsych) |
| psychHEres2way.csv | ftsim/manu/simulations/ | Fig 2 |
| psychHEres6way.csv | ftsim/manu/simulations/ | Fig 2 |
| psych2way_gwas.csv | ftsim/manu/simulations/ | Fig 2 |
| psych6way_gwas.csv | ftsim/manu/simulations/ | Fig 2 |
| psych6coloc.csv | ftsim/manu/simulations/ | Fig 2 |
| psych6coloc_trait_wise.csv | ftsim/manu/simulations/ | Fig 2 |
| edu_res/*_parsed.csv (100 files) | ftsim/manu/simulations/edu_res/ | Fig 4 (mFigEdu) |
| cca_table_dat.rdata | ftsim/manu/ | Fig 1a (CCA loadings heatmap) |
| imp_cca.Rdata | ftsim/manu/ | sFig_cca (S3-S4 CCA scree) |
| taiwan_mate_r.csv | ftsim/manu/figure_nb/ | Fig 1b (Taiwan NHIRD CCA) |
| taiwan_mate_se.csv | ftsim/manu/figure_nb/ | Fig 1b |
| taiwan_mate_r_nonan.csv | ftsim/manu/figure_nb/ | Fig 1b |
| danish_xmate.csv | Dropbox/rhog_am/ | Fig 1b (Danish comparison) |
| nhird_cors.csv | round4/xftmanu_code_supplement/ | NHIRD CCA analysis |
| psych_cors.csv | round4/xftmanu_code_supplement/ | Psychiatric sim parameters |
| mathematica_sim.csv | ftsim/manu/ | S8 (theoretical validation) |
| mathematica_sim_prec.csv | ftsim/manu/ | S9-S12 (theoretical predictions) |
| gwas_fp.csv | ftsim/manu/ | S8 (observed GWAS FP rates) |
| gen_sims_gwas_auc.csv | ftsim/manu/ | Supplementary GWAS AUC |
| cdiags/*.svg, *.pdf | ftsim/manu/cdiags/ | Fig 1a, Fig 3a schematic diagrams |
| edu_cdiag_redux.png | ftsim/manu/figures/ | Fig 4a schematic |
| psych_parsed/*.csv | ftsim/manu/psych_parsed/ | Psychiatric parsing intermediate |
| impmice_av_xcor.Rdata | ftsim/manu/ | CCA supplementary |
| impmice_full_xcor.Rdata | ftsim/manu/ | CCA supplementary |
| pcs_cca.Rdata | ftsim/manu/ | CCA supplementary |
| pcs_imp_cca.Rdata | ftsim/manu/ | CCA supplementary |
| xmate_correlations.csv | ftsim/manu/tables/ | Table S3 |

### Recovered from coyote (now in data/)
| File | Local Path | Used By |
|------|-----------|---------|
| corrnoise CN=0.4, 0.8 CSVs (200 files) | data/corrnoise/ | fig_cn_updated.R |
| timing_*.json benchmarks (55 files) | data/benchmarks/ | fig_benchmarks.R |
| timing_all_clean.jsonl | data/timing_all_clean.jsonl | fig_benchmarks_v2.R |
| h2_025 CSVs (200 files) | data/h2_025/ | fig_h2_025.R |
| merged_tabla_redux_results_0524.csv | data/sim_results/ | P46/P48 numbers, S13-S16, Table S5 |

### Not currently accessible (user has location)
| File | Figures | Notes |
|------|---------|-------|
| CDEF ascertainment estimators (testing_CDEF_072724/) | S18-S20 | User knows location |
| CDEF sib-diff estimators (testing_CDEF_080524/) | S23-S24 | User knows location |
| edu_no_CD_LS.01/ raw simulation output | Fig 4 full reproduction | Notebook cache used as reference |

---

## Figure-to-Code Mapping

### Main Figures

#### Figure 1: Dimensionality of cross-mate correlations + simulation consequences
| Panel | Content | Source Notebook | Data Files |
|-------|---------|-----------------|------------|
| a | CCA cross-loadings heatmap (UKB) | mFigComplexity.ipynb → mFigHDxAM.ipynb | cca_table_dat.rdata |
| b | CCA scree plot (UKB + Taiwan NHIRD) | mFigHDxAM.ipynb + taiwan cca.ipynb | cca_table_dat.rdata, taiwan_mate_r.csv, nhird_cors.csv |
| c | True vs estimated h2, 2xAM vs 5xAM | mFigHDxAM.ipynb | merged_tabla_redux_results_011024.csv |
| d | True vs estimated rg, 2xAM vs 5xAM | mFigHDxAM.ipynb | merged_tabla_redux_results_011024.csv |
| e | GWAS type-I error inflation | mFigHDxAM.ipynb | merged_tabla_redux_results_011024.csv |
| f | On-target vs off-target associations (analytic) | mFigNewComplexity.ipynb | mathematica_sim_prec.csv |
**STATUS: ALL DATA AVAILABLE**

#### Figure 2: Psychiatric disorder simulations
| Panel | Content | Source Notebook | Data Files |
|-------|---------|-----------------|------------|
| a | Genetic correlation heatmaps (2way vs 6way) | mFigPsych.ipynb | psychHEres.csv, psychHEres2way.csv, psychHEres6way.csv |
| b | Gamma ratio (rg_xAM / rg_empirical) | mFigPsych.ipynb | same |
| c | Prevalence trajectories | mFigPsych.ipynb | same |
**STATUS: ALL DATA AVAILABLE**

#### Figure 3: xAM + VT + GxE
| Panel | Content | Source Notebook | Data Files |
|-------|---------|-----------------|------------|
| a | Generative model schematic | mFigComplexity.ipynb | cdiags/*.svg |
| b | True vs estimated h2 across scenarios | mFigComplexity.ipynb | merged_tabla_redux_results_011024.csv |
| c | True vs estimated rg across scenarios | mFigComplexity.ipynb | merged_tabla_redux_results_011024.csv |
| d | GWAS false positive inflation | mFigComplexity.ipynb | merged_tabla_redux_results_011024.csv |
**STATUS: ALL DATA AVAILABLE**

#### Figure 4: Education/height/wealth
| Panel | Content | Source Notebook | Data Files |
|-------|---------|-----------------|------------|
| a | Joint architecture schematic | mFigEdu.ipynb | figures/edu_cdiag_redux.png |
| b | h2 and PGI R2 across generations | mFigEdu.ipynb | edu_res/*_parsed.csv (in manu/simulations/edu_res/) |
| c | True vs estimated genetic correlations | mFigEdu.ipynb | same |
| d | GWAS beta correlations | mFigEdu.ipynb | same |
**STATUS: Raw data (edu_no_CD_LS.01) not accessible. Notebook cached outputs used as reference tables. edu_sims/ contains a DIFFERENT model (h2_edu=0) and cannot reproduce manuscript numbers.**

---

### Supplementary Figures

| Figure | Content | Source | Data | Status |
|--------|---------|--------|------|--------|
| S1 | Multivariate vs multidimensional mating (synthetic) | sFig_cca.ipynb | Generated synthetically | COMPLETE (process/verify/plot_sfig_cca.R) |
| S2 | Nonlinear mating CCA (synthetic) | sFig_cca.ipynb | Generated synthetically | COMPLETE |
| S3 | CCA scree (PMM MICE imputed) | sFig_cca.ipynb | mice_imputed_cca_final.rdata | COMPLETE (place in data/cca/ or set CCA_DATA_DIR) |
| S4 | CCA scree (RF miceRanger imputed) | sFig_cca.ipynb | mice_imputed_cca_final_rf.rdata | COMPLETE (place in data/cca/ or set CCA_DATA_DIR) |
| S5 | rg under xAM r=0.1 | sFigComplexityExtra.ipynb | merged_tabla_redux_results_011024.csv | COMPLETE (process/verify/plot_sfig_sensitivity.R) |
| S6 | rg under unidim xAM, fixed latent | sFigComplexityExtra.ipynb / FigSX_reconstruction.R | merged_res_redux_120725.csv | COMPLETE |
| S7 | Sensitivity to correlated noise | fig_cn_updated.R | merged_res_redux_120725.csv + corrnoise CSVs | AVAILABLE (all CN levels recovered from coyote) |
| S8 | Theory vs observed GWAS validation | sFigComplexityExtra.ipynb | gwas_fp.csv, mathematica_sim.csv | COMPLETE (process/verify/plot_sfig_theory_gxe.R) |
| S9 | Incremental on/off-target associations | mFigNewComplexity.ipynb | newhits.rdata | COMPLETE (process/verify/plot_sfig_analytic.R) |
| S10 | Type-I error vs sample size | mFigNewComplexity.ipynb | powerSims2.rdata | COMPLETE |
| S11 | Cumulative on/off-target associations | mFigNewComplexity.ipynb | powerSims3.rdata | COMPLETE |
| S12 | FP rate by effect magnitude | mFigNewComplexity.ipynb | merged_tabla + analytic computation | COMPLETE |
| S13 | h2 under xAM + GxE + VT (GxE rows) | sFigComplexityExtra.ipynb | merged_tabla_redux_results_0524.csv | AVAILABLE (uses 0524 for full 3x3 grid) |
| S14 | h2 under xAM + GxE + VT (VT rows) | sFigComplexityExtra.ipynb | merged_tabla_redux_results_0524.csv | AVAILABLE |
| S15 | rg under xAM + GxE + VT (VT rows) | sFigComplexityExtra.ipynb | merged_tabla_redux_results_0524.csv | AVAILABLE |
| S16 | PGI correlation under xAM + GxE + VT | sFigComplexityExtra.ipynb | merged_tabla_redux_results_0524.csv | AVAILABLE |
| S17 | Population vs sibship GWAS | mFigSibComp.ipynb | merged_tabla_redux_results_011024.csv | COMPLETE (process/verify/plot_sfig_sibcomp.R) |
| S18 | Sibship vs pop GWAS: ascertained (T1E) | sFig_ascertainment_sim_res.ipynb | testing_CDEF_072724/estimators* | PENDING ACCESS |
| S19 | Sibship vs pop GWAS: ascertained (slope corr) | sFig_ascertainment_sim_res.ipynb | testing_CDEF_072724/* | PENDING ACCESS |
| S20 | Sibship vs pop GWAS: ascertained (bias) | sFig_ascertainment_sim_res.ipynb | testing_CDEF_072724/* | PENDING ACCESS |
| S21 | LATE vs ATE single variant (linear) | test_sib_diff.ipynb / sibdiff.nb | Mathematica symbolic computation | AVAILABLE (analytic) |
| S22 | LATE vs ATE single variant (log scale) | test_sib_diff.ipynb / sibdiff.nb | Mathematica symbolic computation | AVAILABLE (analytic) |
| S23 | VT+GxE bias of sib-GWAS PGI | test_sib_diff.ipynb | testing_CDEF_080524/estimators* | PENDING ACCESS |
| S24 | ATE-PGI vs individual-specific PGI | test_sib_diff.ipynb | testing_CDEF_080524/estimators* | PENDING ACCESS |

### Revision-Specific Figures (new for round 4)
| Figure | Content | Source | Data | Status |
|--------|---------|--------|------|--------|
| Variance decomp | h2 decomposition (3 quantities) | fig_variance_decomp.R | merged_res_redux_120725.csv | COMPLETE (*_figR_variance_decomp.R) |
| Benchmarks | Computational scaling | fig_benchmarks_v2.R | timing_all_clean.jsonl | COMPLETE (*_figR_benchmarks.R) |
| h2=0.25 | Low-h2 sensitivity | fig_h2_025.R | data/h2_025/*.csv | COMPLETE (*_figR_h2_025.R) |
| CN updated | Correlated noise extended | fig_cn_updated.R | corrnoise CSVs | COMPLETE (*_figR_cn.R) |

---

### Tables

| Table | Content | Source | Data | Status |
|-------|---------|--------|------|--------|
| Table 1 | xftsim capabilities | Static text | N/A | N/A (text only) |
| Table S1 | Phenotype definitions | pheno_defn_supplement.r | UK Biobank fields | N/A (documentation) |
| Table S2 | CCA results | final_cca.R / cca_table_dat.rdata | cca_table_dat.rdata | AVAILABLE |
| Table S3 | Cross-mate correlations (UKB) | final_cca.R | xmate_correlations.csv | AVAILABLE |
| Table S4 | Cross-mate correlations (NHIRD) | nhird_cca.r | nhird_cors.csv | AVAILABLE |
| Table S5 | Variance components summary | sFigComplexityExtra.ipynb | merged_tabla_redux_results_0524.csv | AVAILABLE (needs 0524 for VT 20% / GxE 20% rows) |
| Table S6 | Simulation parameters | Static text | N/A | N/A (text only) |

---

## Source Notebook to Script Mapping

| Original Notebook | Location | Figures |
|-------------------|----------|---------|
| mFigComplexity.ipynb | ftsim/manu/figure_nb/ | Fig 3 (panels a-d) |
| mFigHDxAM.ipynb | ftsim/manu/figure_nb/ | Fig 1 (panels a-e) |
| mFigNewComplexity.ipynb | ftsim/manu/figure_nb/ | Fig 1f, S9-S12 |
| mFigPsych.ipynb | ftsim/manu/figure_nb/ | Fig 2 (panels a-c) |
| mFigEdu.ipynb | ftsim/manu/figure_nb/ | Fig 4 (panels a-d) |
| sFig_cca.ipynb | ftsim/manu/figure_nb/ | S1-S4 |
| sFigComplexityExtra.ipynb | ftsim/manu/figure_nb/ | S5-S6, S8, S13-S16 |
| mFigSibComp.ipynb | ftsim/manu/figure_nb/ | S17 |
| sFig_ascertainment_sim_res.ipynb | ftsim/manu/figure_nb/ | S18-S20 |
| test_sib_diff.ipynb | ftsim/manu/figure_nb/ | S21-S24 |
| taiwan cca.ipynb | ftsim/manu/figure_nb/ | Fig 1b (NHIRD part) |
| FigSX_reconstruction.R | ftsim/ak_resp/ | S5-S7 (revision versions) |
| fig_cn_updated.R | round4/revision_sims/figures/ | S7 (updated CN) |
| fig_variance_decomp.R | round4/revision_sims/figures/ | Variance decomposition |
| fig_benchmarks_v2.R | round4/revision_sims/figures/ | Benchmark scaling |
| fig_h2_025.R | round4/revision_sims/figures/ | Low-h2 sensitivity |
| sibdiff.nb | xftmanu_code_supplement/ | S21-S22 (symbolic math) |
