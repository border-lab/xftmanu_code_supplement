# Data Manifest

Every data file used by the reproduction scripts, with provenance and structure.

## `data/` root

- `timing_all_clean.jsonl` (14.2KB) 130 records, keys: scenario, seed, n, m, elapsed_seconds, cores

## `data/benchmarks/` (55 files, 6.0KB total)

- 55 JSON files, keys: `scenario, seed, n, m, elapsed_seconds, cores`
- Example: `timing_m1000_seed1.json`

## `data/cca/` (6 files, 104.5MB total)

- `cca_table_dat.rdata` (56KB) — UKB CCA summary: loadings, redundancies, correlations, chi-sq tests
- `imp_cca.Rdata` (27MB) — Full MICE-imputed CCA object (yacca result, loads as `out`)
- `impmice_av_xcor.Rdata` (16KB) — Average cross-sex correlations
- `impmice_full_xcor.Rdata` (46KB) — Full cross-sex correlation matrix
- `pcs_cca.Rdata` (25MB) — CCA on principal components
- `pcs_imp_cca.Rdata` (52MB) — CCA on phenotypes + PCs combined

## `data/cdiags/` (28 files, 654KB total)

Causal diagrams as SVG and PDF. Used by Fig 3 panel a and Fig 4 panel a.

Key files:
- `cropped_diag5xAM.svg` — 5xAM scenario diagram
- `cropped_diag5xAMgXE.svg` — 5xAM + G×E diagram
- `cropped_diag5xAMpVT.svg` — 5xAM + VT diagram
- `cropped_diag5xAMgXEpVT.svg` — 5xAM + VT + G×E diagram
- `edu_cdiag_redux.png` — Education/height/wealth architecture schematic

## `data/corrnoise/` (200 files, 4.4MB total)

Correlated noise simulation results (CN=0.4 and CN=0.8), 50 seeds × 2 CN levels × 2 theta values.

- 200 CSV files, each ~6 rows (gen 0-5) × 183 cols
- Filename pattern: `cn_{seed}_{kmate}_{kphen}_{rmate}_{h2}_{theta}_{phi}_{gamma}_{env}_{m}_{n}_{CN}_pheno_parsed.csv`

## `data/edu_sims/` (101 files, 13.9MB total)

Education/height/wealth simulation results. **WARNING: different model from manuscript** (h2_edu=0 vs manuscript's h2_edu=0.01).

- 101 CSV files, each ~6 rows × 1512 cols
- Filename pattern: `edu_{seed}_800_parsed.csv`

## `data/h2_025/` (200 files, 4.5MB total)

Low-heritability (h2=0.25) sensitivity analysis. 50 seeds × 4 scenarios.

- 200 CSV files, each ~6 rows × 182 cols
- Filename pattern: `sres_{seed}_{kmate}_{kphen}_{rmate}_{h2}_{theta}_{phi}_{gamma}_{env}_{m}_pheno_parsed.csv`

## `data/psych_sims/` (7 files, 5.9MB total)

Psychiatric disorder simulation results for 6 disorders (ADHD, ALC, ANX, BIP, MDD, SCZ).

| File | Rows × Cols | Description |
|------|------------|-------------|
| `psychHEres.csv` | 4704 × 82 | Combined 2-way and 6-way HE regression results |
| `psychHEres2way.csv` | 4410 × 82 | Bivariate xAM HE results (all 15 pairs) |
| `psychHEres6way.csv` | 294 × 85 | 6-variate xAM HE results |
| `psych2way_gwas.csv` | 4500 × 87 | Bivariate GWAS slope estimates |
| `psych6way_gwas.csv` | 300 × 85 | 6-variate GWAS slope estimates |
| `psych6coloc.csv` | 294 × 10 | 6-way colocalization results |
| `psych6coloc_trait_wise.csv` | 1764 × 20 | Trait-wise colocalization |

## `data/sim_results/` (11 files, 348MB total)

Core simulation results. Three merged CSVs from different time periods contain overlapping but distinct scenario sets.

| File | Rows × Cols | Scenarios | Used for |
|------|------------|-----------|----------|
| `merged_tabla_redux_results_011024.csv` | 31500 × 171 | 7 (VT 5% only) | Fig 1, 3 panels b-e |
| `merged_tabla_redux_results_0524.csv` | 65886 × 281 | 12 (VT 5%+20%, GxE 5%+20%) | P46/P48 numbers, S13-S16 |
| `merged_res_redux_120725.csv` | 13824 × 281 | Many incl. CN variants | Revision figs, S5-S6 |
| `corrnoise_res_120925.csv` | 12288 × 233 | CN=0.2 scenarios | S7 (CN baseline) |
| `corrnoise1_res_120925.csv` | 6144 × 209 | CN=0.1 scenarios | S7 (CN baseline) |
| `gwas_fp.csv` | 6510 × 19 | Observed GWAS FP rates | S8 theory validation |
| `mathematica_sim.csv` | 2880 × 16 | Mathematica-derived predictions | S8 theory validation |
| `mathematica_sim_prec.csv` | 4319 × 16 | Higher-precision Mathematica predictions | S8 |
| `newhits.rdata` | (R binary) | Pre-computed incremental hits | Fig 1f, S9 |
| `powerSims2.rdata` | (R binary) | T1E inflation grid | S10 |
| `powerSims3.rdata` | (R binary) | Cumulative hit counts | S11 |

## `data/tables/` (6 files, 1.8MB total)

| File | Rows × Cols | Description |
|------|------------|-------------|
| `complex_sim_res.csv` | 42 × 9 | Simulation parameter summary |
| `pheno_defn.csv` | 46 × 4 | Phenotype definitions |
| `xmate_correlations.csv` | 2025 × 21 | UKB cross-mate Pearson/Spearman correlations with SEs |
| `xmate_cors.csv` | 1035 × 21 | Subset of cross-mate correlations |
| `xmate_maxent.csv` | 1980 × 10 | Maximum entropy mating correlations |
| `xmate_maxent_grad.csv` | 1035 × 15 | Maximum entropy gradient correlations |

## `data/taiwan/` (4 files, 18.5KB total)

| File | Rows × Cols | Description |
|------|------------|-------------|
| `taiwan_mate_r.csv` | 40 × 29 | Taiwan NHIRD cross-mate correlations |
| `taiwan_mate_se.csv` | 28 × 29 | Standard errors for above |
| `taiwan_mate_r_nonan.csv` | 24 × 24 | Cleaned (NaN rows/cols removed) |
| `danish_xmate.csv` | 15 × 11 | Danish cross-mate correlations for comparison |

## Provenance

| Directory | Source | How obtained |
|-----------|--------|-------------|
| `sim_results/` | `ftsim/manu/` and `ftsim/ak_resp/` | Copied from Dropbox |
| `psych_sims/` | `ftsim/manu/simulations/` | Copied from Dropbox |
| `edu_sims/` | `ftsim/manu/simulations/edu_res/` | Copied from Dropbox. **Different model than manuscript** |
| `cca/` | `ftsim/manu/` | Copied from Dropbox |
| `taiwan/` | `ftsim/manu/figure_nb/` and `Dropbox/rhog_am/` | Copied from Dropbox |
| `cdiags/` | `ftsim/manu/cdiags/` and `ftsim/manu/figures/` | Copied from Dropbox |
| `tables/` | `ftsim/manu/tables/` | Copied from Dropbox |
| `corrnoise/` | coyote `/tmp/revision_results/corrnoise/` | scp from coyote |
| `benchmarks/` | coyote `/tmp/revision_results/benchmarks/` | scp from coyote |
| `h2_025/` | coyote `/tmp/revision_results/h2_025/` | scp from coyote |
| `timing_all_clean.jsonl` | coyote `/tmp/` | scp from coyote |

## Large files not in git

See `sim_results/LARGE_FILES.md` for copy instructions.

| File | Size | Source |
|------|------|--------|
| `sim_results/merged_tabla_redux_results_011024.csv` | 70MB | `ftsim/manu/` |
| `sim_results/merged_tabla_redux_results_0524.csv` | 198MB | `ftsim/manu/` |
| `sim_results/merged_res_redux_120725.csv` | 29MB | `ftsim/ak_resp/` |
| `sim_results/corrnoise_res_120925.csv` | 24MB | `ftsim/ak_resp/` |
| `sim_results/corrnoise1_res_120925.csv` | 12MB | `ftsim/ak_resp/` |
| `cca/*.Rdata` (6 files) | ~120MB total | `ftsim/manu/` |
