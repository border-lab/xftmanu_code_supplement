# Manuscript Figure Reproduction Suite

Scripts to reproduce every figure, table, and in-text number in the manuscript
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
| `merged_res_redux_120725.csv` | 29MB | 5xAM.1, 5xAM.2, +VT, +GxE variants | ED 1/2 sensitivity, ED 10 correlated-noise baseline |

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

### Cluster-computed simulation outputs

| Directory | Files | Content |
|-----------|-------|---------|
| `corrnoise/` | 200 CSVs | CN=0.4 and CN=0.8 correlated noise sims |
| `benchmarks/` | 55 JSONs | xftsim computational timing data |
| `h2_025/` | 200 CSVs | h2=0.25 sensitivity analysis |
| `timing_all_clean.jsonl` | 1 file | Consolidated timing (130 records) |
| VT-structure decomposition sims | per-seed CSVs | Fig 3, ED 4/5, S17, Tables S7/S8 (from `vt_structure_sims.py`; distributed separately) |

## Script Inventory

See **`FIGURE_INVENTORY.md`** for the complete, figure-by-figure map of every main,
Extended Data, and Supplementary figure and table to its producing script(s).

Each figure follows the `process_* -> verify_* -> plot_*` convention where applicable
(`verify_*` prints the in-text numbers; some figures have only `process_*`/`plot_*`).

Availability caveats:
- **S12-S15** require CDEF ascertainment / sibling-difference estimator outputs not on
  current machines (the authors retain the source locations).
- **ED 9 / S18** are symbolic computations in `sibdiff.nb` (require Mathematica).
- **Fig 4** raw simulation data is not accessible; cached notebook reference tables are used.

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
