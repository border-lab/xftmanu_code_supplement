# Large Data Files (not tracked in git)

The following files are required by the process scripts but are too large for git.
They are available in the Dropbox at `ftsim/manu/` and `ftsim/ak_resp/`.

Copy them into this directory (`data/sim_results/`) before running process scripts:

| File | Size | Source |
|------|------|--------|
| `merged_tabla_redux_results_011024.csv` | 70MB | `ftsim/manu/merged_tabla_redux_results_011024.csv` |
| `merged_tabla_redux_results_0524.csv` | 198MB | `ftsim/manu/merged_tabla_redux_results_0524.csv` |
| `merged_res_redux_120725.csv` | 29MB | `ftsim/ak_resp/merged_res_redux_120725.csv` |
| `corrnoise_res_120925.csv` | 24MB | `ftsim/ak_resp/corrnoise_res_120925.csv` |
| `corrnoise1_res_120925.csv` | 12MB | `ftsim/ak_resp/corrnoise1_res_120925.csv` |

## CCA data (also large)

Copy these into `data/cca/`:

| File | Size | Source |
|------|------|--------|
| `cca_table_dat.rdata` | 56KB | `ftsim/manu/cca_table_dat.rdata` |
| `imp_cca.Rdata` | 27MB | `ftsim/manu/imp_cca.Rdata` |
| `impmice_av_xcor.Rdata` | 16KB | `ftsim/manu/impmice_av_xcor.Rdata` |
| `impmice_full_xcor.Rdata` | 46KB | `ftsim/manu/impmice_full_xcor.Rdata` |
| `pcs_cca.Rdata` | 25MB | `ftsim/manu/pcs_cca.Rdata` |
| `pcs_imp_cca.Rdata` | 52MB | `ftsim/manu/pcs_imp_cca.Rdata` |

## Quick setup

```bash
# From the repo root:
cp ~/Dropbox/ftsim/manu/merged_tabla_redux_results_011024.csv data/sim_results/
cp ~/Dropbox/ftsim/manu/merged_tabla_redux_results_0524.csv data/sim_results/
cp ~/Dropbox/ftsim/ak_resp/merged_res_redux_120725.csv data/sim_results/
cp ~/Dropbox/ftsim/ak_resp/corrnoise_res_120925.csv data/sim_results/
cp ~/Dropbox/ftsim/ak_resp/corrnoise1_res_120925.csv data/sim_results/
cp ~/Dropbox/ftsim/manu/cca_table_dat.rdata data/cca/
cp ~/Dropbox/ftsim/manu/imp_cca.Rdata data/cca/
cp ~/Dropbox/ftsim/manu/impmice_av_xcor.Rdata data/cca/
cp ~/Dropbox/ftsim/manu/impmice_full_xcor.Rdata data/cca/
cp ~/Dropbox/ftsim/manu/pcs_cca.Rdata data/cca/
cp ~/Dropbox/ftsim/manu/pcs_imp_cca.Rdata data/cca/
```
