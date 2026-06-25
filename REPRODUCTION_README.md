# Manuscript Figure Reproduction Suite

Standalone scripts to reproduce every figure and in-text number in the manuscript
"Simple models of non-random mating and environmental transmission bias standard
human genetics statistical methods" (Border et al.).

## Directory Structure

```
round4/
├── data/                  # All input data (copied from various sources)
│   ├── sim_results/       # Merged simulation result CSVs
│   ├── psych_sims/        # Psychiatric disorder simulation results
│   ├── edu_sims/          # Education/height/wealth simulation results (WRONG MODEL - see note)
│   ├── cca/               # CCA Rdata objects (UKB cross-mate analysis)
│   ├── taiwan/            # Taiwan NHIRD correlation data
│   ├── cdiags/            # Causal diagram SVGs and PDFs
│   ├── tables/            # Summary table CSVs
│   ├── corrnoise/         # Correlated noise CN=0.4,0.8 results (from coyote)
│   ├── benchmarks/        # Timing JSON files (from coyote)
│   └── h2_025/            # h2=0.25 sensitivity results (from coyote)
├── scripts/               # All R scripts (process → verify → plot)
├── processed/             # Intermediate CSVs (inspectable, machine-readable)
├── figures_output/        # Generated figures (PDF + PNG)
├── FIGURE_INVENTORY.md    # Detailed figure-to-code mapping
└── README.md              # This file
```

## Three-Layer Architecture

Each figure has three scripts:

1. **`process_*.R`** — Loads raw data, filters, aggregates, saves intermediate CSVs to `processed/`
2. **`verify_*.R`** — Reads processed data, prints every manuscript in-text number for comparison
3. **`plot_*.R`** — Reads ONLY from `processed/`, generates figures to `figures_output/`

This separation means you can inspect `processed/` CSVs directly to verify any number,
and re-run plots without re-processing data.

## Data Sources

### Primary Simulation Results

| File | Size | Scenarios | Used For |
|------|------|-----------|----------|
| `merged_tabla_redux_results_011024.csv` | 70MB | 2xAM, 5xAM, 5xAM+VT, 5xAM+GxE, 5xAM+VT+GxE, RM, RM+VT | Fig 1 (panels c-e), Fig 3 (panels b-d), S5-S6, S17 |
| `merged_tabla_redux_results_0524.csv` | 198MB | Same + VT(20%), GxE(20%), all VT×GxE combos | P46/P48 in-text numbers, S13-S16, Table S5 |
| `merged_res_redux_120725.csv` | 29MB | 5xAM.1, 5xAM.2, +VT, +GxE variants | Revision figs (variance decomp, CN baseline, S5-S6 sensitivity) |

**Key distinction:** The 011024 file has ~750 seeds per scenario with only VT(5%)/GxE(5%).
The 0524 file has ~950 seeds and includes VT(20%)/GxE(20%). The manuscript body text
(P46, P48) uses the 0524 numbers.

### Psychiatric Simulations (Fig 2)

| File | Content |
|------|---------|
| `psychHEres.csv` | Combined 2-way and 6-way HE results |
| `psychHEres2way.csv` | Bivariate xAM HE results |
| `psychHEres6way.csv` | 6-variate xAM HE results |
| `psych2way_gwas.csv` | Bivariate GWAS slope estimates |
| `psych6way_gwas.csv` | 6-variate GWAS slope estimates |

Source: `/home/rsb/Dropbox/ftsim/manu/simulations/`

### Education/Height/Wealth (Fig 4)

**NOTE:** The original simulation data (`~/data/edu_no_CD_LS.01/`, run with `--h2 0.01`)
is not accessible from any current machine. The `edu_sims/` directory contains data from
a DIFFERENT simulation (h2_edu=0, h2_height=0.56) and CANNOT reproduce the manuscript.

Figure 4 uses **hardcoded reference tables** extracted from the notebook's cached output
cells (`mFigEdu.ipynb` cells 18, 22, 31, 39). These are saved as:
- `processed/fig4_manuscript_reference.csv`
- `processed/fig4_manuscript_h2_true_height.csv`
- `processed/fig4_manuscript_rg.csv`
- `processed/fig4_manuscript_rbeta.csv`

To fully reproduce Fig 4 from raw data, re-run the simulation:
```bash
# In xft conda env, with --h2 0.01 (not default 0.05), m=800, n=128000
python edu_experiment_no_CD_linear.py --seed {1..200} --h2 0.01 -m 800
```
Script located at: `ftsim/manu/simulations/edu_experiment_no_CD_linear.py`

### CCA Data (Fig 1a-b)

| File | Content |
|------|---------|
| `cca_table_dat.rdata` | UKB CCA summary (loadings, redundancies, correlations) |
| `imp_cca.Rdata` | Full MICE-imputed CCA object (for S3 scree plot) |
| `taiwan_mate_r.csv` | Taiwan NHIRD cross-mate correlations |
| `nhird_cors.csv` | NHIRD correlation matrix (in xftmanu_code_supplement/) |

### Analytic Prediction Data (S9-S12)

| File | Content |
|------|---------|
| `newhits.rdata` | Pre-computed incremental on/off-target hits (S9, also Fig 1f) |
| `powerSims2.rdata` | T1E inflation grid across gen, M, K, N (S10) |
| `powerSims3.rdata` | Cumulative on/off-target hit counts (S11) |
| `mathematica_sim_prec.csv` | Mathematica-derived analytic predictions |
| `gwas_fp.csv` | Observed GWAS false positive rates (S8 validation) |

Source: `/home/rsb/Dropbox/ftsim/manu/figure_nb/` (Rdata files) and `/home/rsb/Dropbox/ftsim/manu/` (CSVs)

### Revision-Specific Data (from PSC via coyote)

| Directory | Files | Content |
|-----------|-------|---------|
| `corrnoise/` | 200 CSVs | CN=0.4 and CN=0.8 correlated noise sims |
| `benchmarks/` | 55 JSONs | xftsim computational timing data |
| `h2_025/` | 200 CSVs | h2=0.25 sensitivity analysis |
| `timing_all_clean.jsonl` | 1 file | Consolidated timing (130 records) |

## Script Inventory

### Main Figures

| Figure | Process | Verify | Plot | Status |
|--------|---------|--------|------|--------|
| Fig 1 (CCA + 2xAM vs 5xAM) | `process_fig1.R` | `verify_fig1.R` | `plot_fig1.R` | 15/15 numbers verified |
| Fig 2 (psychiatric disorders) | `process_fig2.R` | `verify_fig2.R` | `plot_fig2.R` | 9/9 verified (2xAM/6xAM label swap noted) |
| Fig 3 (xAM + VT + GxE) | `process_fig3.R` | `verify_fig3.R` | `plot_fig3.R` | All verified (uses both 011024 + 0524) |
| Fig 4 (edu/height/wealth) | `process_fig4.R` | `verify_fig4.R` | `plot_fig4.R` | Verified against notebook cache |

### Supplementary Figures

| Supp Fig | Description | Scripts | Output | Status |
|----------|-------------|---------|--------|--------|
| S1 | Multivariate vs multidimensional mating (synthetic) | `*_sfig_cca.R` | `sfig_s1_mating_cca.pdf` | Complete |
| S2 | Nonlinear mating CCA (synthetic) | `*_sfig_cca.R` | `sfig_s2_nonlinear_cca.pdf` | Complete |
| S3 | CCA scree (MICE PMM imputed) | `*_sfig_cca.R` | `sfig_s3_cca_scree.pdf` | Complete |
| S4 | CCA scree (RF miceRanger imputed) | `*_sfig_cca.R` | `sfig_s4_cca_scree_rf.pdf` | Complete (place data in data/cca/ or set CCA_DATA_DIR) |
| S5 | rg under xAM with r=0.1 vs r=0.2 | `*_sfig_sensitivity.R` | `sfig_s5_rg_sensitivity.pdf` | Complete |
| S6 | rg under unidim xAM, fixed latent corr | `*_sfig_sensitivity.R` | `sfig_s6_fixed_latent.pdf` | Complete |
| S7 | Sensitivity to correlated noise (updated) | `*_figR_cn.R` | `figR_cn.pdf` | Complete |
| S8 | Theory vs observed GWAS validation | `*_sfig_theory_gxe.R` | `sfig_s8_theory_validation.pdf` | Complete |
| S9 | Incremental on/off-target associations | `*_sfig_analytic.R` | `sfig_s9_incremental_hits.pdf` | Complete |
| S10 | Type-I error vs sample size | `*_sfig_analytic.R` | `sfig_s10_t1e_inflation.pdf` | Complete |
| S11 | Cumulative on/off-target associations | `*_sfig_analytic.R` | `sfig_s11_cumulative_hits.pdf` | Complete |
| S12 | FP rate by effect magnitude | `*_sfig_analytic.R` | `sfig_s12_persnp_t1e.pdf` | Complete |
| S13 | h2 under GxE×VT grid (GxE rows) | `*_sfig_theory_gxe.R` | `sfig_s13_h2_gxe_grid.pdf` | Complete (uses 0524 data) |
| S14 | h2 under GxE×VT grid (VT rows) | `*_sfig_theory_gxe.R` | `sfig_s14_h2_vt_grid.pdf` | Complete |
| S15 | rg under GxE×VT grid | `*_sfig_theory_gxe.R` | `sfig_s15_rg_gxe_grid.pdf` | Complete |
| S16 | PGI corr under GxE×VT grid | `*_sfig_theory_gxe.R` | `sfig_s16_pgi_gxe_grid.pdf` | Complete |
| S17 | Pop vs sibship GWAS | `*_sfig_sibcomp.R` | `sfig_s17_sibcomp_combined.pdf` | Complete |
| S18 | Sib vs pop GWAS under ascertainment (T1E) | — | — | Data not accessible |
| S19 | Sib vs pop GWAS under ascertainment (slope) | — | — | Data not accessible |
| S20 | Sib vs pop GWAS under ascertainment (bias) | — | — | Data not accessible |
| S21 | LATE vs ATE single variant (linear) | — | — | Mathematica symbolic computation (`sibdiff.nb`) |
| S22 | LATE vs ATE single variant (log scale) | — | — | Same |
| S23 | VT+GxE bias of sib-GWAS PGI | — | — | Data not accessible |
| S24 | ATE-PGI vs individual-specific PGI | — | — | Data not accessible |

`*_` prefix means `process_`, `verify_`, and `plot_` variants all exist.

### Tables

| Table | Description | Scripts | Status |
|-------|-------------|---------|--------|
| Table 1 | xftsim capabilities | — | Static text, no data |
| Table S1 | Phenotype definitions | — | Static text, no data |
| Table S2 | Cross-mate CCA results | `process_tables.R`, `verify_tables.R` | Complete |
| Table S3 | Cross-mate correlations (UKB) | `process_tables.R`, `verify_tables.R` | Complete |
| Table S4 | Cross-mate correlations (NHIRD) | `process_tables.R`, `verify_tables.R` | Complete |
| Table S5 | Variance components summary | `process_tables.R`, `verify_tables.R` | Complete (uses 0524 data) |
| Table S6 | Simulation parameters | — | Static text, no data |

### Revision Figures (new for round 4 reviewer response)

| Figure | Description | Scripts (`*_figR_...`) | Output | Status |
|--------|-------------|----------------------|--------|--------|
| Variance decomp | h2 decomposition: panmictic vs true vs estimated | `*_variance_decomp.R` | `figR_variance_decomp.pdf` | Complete |
| S7 (updated) | Correlated noise sensitivity (5 CN levels) | `*_cn.R` | `figR_cn.pdf` | Complete |
| Benchmarks | xftsim computational scaling | `*_benchmarks.R` | `figR_benchmarks.pdf` | Complete |
| h2=0.25 | Low-heritability sensitivity (mirrors Fig 3b-d) | `*_h2_025.R` | `figR_h2_025.pdf` | Complete |

## Still Missing

### Data not currently accessible

| Item | Figures | Location | Notes |
|------|---------|----------|-------|
| CDEF ascertainment estimators | S18-S20 | Not on local machines or coyote | User knows location; not currently accessible |
| CDEF sib-diff estimators | S23-S24 | Same | Same |
| edu_no_CD_LS.01 raw data | Fig 4 (full reproduction) | Originally ~/data/edu_no_CD_LS.01/ | Notebook cache used instead |

### Figures without scripts yet

| Figure | Blocking issue | What's needed |
|--------|---------------|---------------|
| S18-S20 | Data | Copy CDEF ascertainment estimators into `data/` once accessible |
| S21-S22 | Tooling | Requires Mathematica to run `sibdiff.nb` symbolic computation |
| S23-S24 | Data | Copy CDEF sib-diff estimators into `data/` once accessible |

### Tables without scripts yet

None -- all reproducible tables have scripts.

## Manuscript Reference Figures

Figures as embedded in the manuscript docx have been extracted to `manuscript_figures/`.
`IMAGE_MAP.txt` maps each image file to its figure number. Target dimensions:

| Figure | Dimensions | DPI |
|--------|-----------|-----|
| Fig 1 | 12.6 x 9.9 in | 240 |
| Fig 2 | 18.0 x 12.0 in | 120 |
| Fig 3 | 14.0 x 8.0 in | 300 |
| Fig 4 | 9.0 x 10.0 in | 300 |

## How to Run

```bash
cd /home/rsb/Dropbox/ftsim/round4

# Process all data
for f in scripts/process_*.R; do Rscript "$f"; done

# Verify all numbers
for f in scripts/verify_*.R; do echo "=== $f ==="; Rscript "$f"; done

# Generate all figures
for f in scripts/plot_*.R; do Rscript "$f"; done
```

## Verification Notes

- **Fig 2:** Manuscript P42 appears to swap "bivariate xAM" and "6-variate xAM" labels
  for the MDD/ANX rgtrue values. The actual numbers match if labels are swapped.
- **Fig 4:** Uses hardcoded tables from notebook cache since raw simulation data is
  inaccessible. All numbers match the notebook output exactly.
- **P48:** Uses `merged_tabla_redux_results_0524.csv` (not 011024). Numbers are rg/h2
  *bias* (estimate minus true), not raw estimates.
