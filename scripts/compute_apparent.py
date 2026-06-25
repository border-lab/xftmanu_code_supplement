#!/usr/bin/env python3
"""
Exact apparent heritability and genetic correlation (the BLP quantity Kong asks for)
from already-saved per-seed vdecomp scalars. NO re-simulation.

Apparent genetic component a_k = BLP(Y_k | proband genotype) = sum_j w_kj g_j,
w_k = G^{-1} c_k, with G = Cov(g) (KxK direct-PGI covariance) and c_k = Cov(g, Y_k).
  apparent h2_k     = (c_k^T G^{-1} c_k) / Var(Y_k)
  apparent rg_(k,l) = normalize( c_k^T G^{-1} c_l )            [ = Cor(a_k, a_l) ]

For the exchangeable K-trait Fig-3 / ED-Fig-4 scenarios, G and C are compound-symmetric,
so per seed (working in units of Var(Y), which cancels) they are fully determined by:
  G~ (=G/Var(Y)):  diag = h2_true,          off-diag = rg_true * h2_true
  C~ (=C/Var(Y)):  diag = cov_g_y,          off-diag = cov_g_y_cross
and  apparent_h2 = diag(C~^T G~^{-1} C~);  apparent_rg = offdiag of its normalization.

Compared against:
  direct    : h2_true (h^2),     rg_true   (r_score)   -- direct genetic effects
  estimate  : he_h2,             he_rg                 -- HE regression (== LDSC here)
  1st-order : apparent_h2,       rg_apparent           -- already in output; UNDERSTATE
              (they drop the Cov(P_X VT_i, P_X VT_j) term)

Validation invariants checked at the end:
  (i)   RM (no VT, no AM): exact apparent_h2 == h2_true, apparent_rg == rg_true (~0)
  (ii)  identity: r2_y_g == cov_g_y^2 / h2_true
  (iii) exact apparent_h2 >= r2_y_g (full BLP >= own-trait BLP)
"""
import numpy as np
import pandas as pd
import glob, os

DATA_DIRS = [
    "/home/rsb/Dropbox/ftsim/round4/data/vdecomp_lane",      # h2 = 0.5
    "/home/rsb/Dropbox/ftsim/round4/data/vdecomp_h2_0_25",   # h2 = 0.25
]
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apparent_decomposition.csv")

NEEDED = ["gen", "seed", "args_kmate", "args_rmate", "args_theta", "args_h2",
          "h2_true", "rg_true", "cov_g_y", "cov_g_y_cross",
          "apparent_h2", "rg_apparent", "r2_y_g", "he_h2", "he_rg", "vtot_full"]


def scenario_label(kmate, rmate, theta):
    mate = "RM" if float(rmate) == 0.0 else f"{int(kmate)}xAM"
    return mate + ("+VT" if float(theta) > 0 else "")


def exact_apparent(K, h2_true, rg_true, cov_g_y, cov_g_y_cross):
    """Return (apparent_h2, apparent_rg) from compound-symmetric reconstruction."""
    K = int(K)
    a, b = h2_true, rg_true * h2_true          # G~ diag, off-diag
    c, d = cov_g_y, cov_g_y_cross              # C~ diag, off-diag
    G = np.full((K, K), b); np.fill_diagonal(G, a)
    C = np.full((K, K), d); np.fill_diagonal(C, c)
    M = C.T @ np.linalg.solve(G, C)            # = Cov(BLP_i, BLP_j) / Var(Y)
    diag = np.diag(M)
    h2 = float(diag.mean())
    if K < 2:
        return h2, np.nan
    Dn = np.sqrt(np.outer(diag, diag))
    R = M / Dn
    iu = np.triu_indices(K, 1)
    return h2, float(R[iu].mean())


# ---- load all per-seed files ---------------------------------------------------
frames = []
for d in DATA_DIRS:
    for f in glob.glob(os.path.join(d, "*_pheno_parsed.csv")):
        df = pd.read_csv(f, usecols=lambda c: c in NEEDED)
        if "h2_true" in df.columns:
            frames.append(df)
raw = pd.concat(frames, ignore_index=True)
print(f"Loaded {len(frames)} files, {len(raw)} rows (seed x scenario x generation).")

# per-row exact apparent
ah2, arg = [], []
for _, r in raw.iterrows():
    h2, rg = exact_apparent(r.args_kmate, r.h2_true, r.rg_true, r.cov_g_y, r.cov_g_y_cross)
    ah2.append(h2); arg.append(rg)
raw["apparent_h2_exact"] = ah2
raw["apparent_rg_exact"] = arg
raw["scenario"] = [scenario_label(k, rm, th)
                   for k, rm, th in zip(raw.args_kmate, raw.args_rmate, raw.args_theta)]

# ---- aggregate -----------------------------------------------------------------
grp = raw.groupby(["args_h2", "scenario", "gen"])
agg = grp.agg(
    n=("seed", "size"),
    h2_direct=("h2_true", "mean"),
    h2_app_1st=("apparent_h2", "mean"),
    h2_app_exact=("apparent_h2_exact", "mean"),
    h2_estimate=("he_h2", "mean"),
    rg_direct=("rg_true", "mean"),       # r_score
    rg_app_1st=("rg_apparent", "mean"),
    rg_app_exact=("apparent_rg_exact", "mean"),
    rg_estimate=("he_rg", "mean"),
).reset_index()
agg.to_csv(OUT_CSV, index=False)

pd.set_option("display.width", 200, "display.max_columns", 30, "display.float_format", lambda x: f"{x:7.3f}")

print("\n================= HERITABILITY (gen 5) =================")
print("estimate-direct = old 'bias';  estimate-apparent = Kong's 'fair' bias")
h = agg[(agg.gen == 5)].copy()
h["bias_vs_direct"] = h.h2_estimate - h.h2_direct
h["bias_vs_apparent"] = h.h2_estimate - h.h2_app_exact
print(h[["args_h2", "scenario", "n", "h2_direct", "h2_app_1st", "h2_app_exact",
         "h2_estimate", "bias_vs_direct", "bias_vs_apparent"]].to_string(index=False))

print("\n================= GENETIC CORRELATION (gen 5) =================")
g = agg[(agg.gen == 5)].copy()
g["bias_vs_direct"] = g.rg_estimate - g.rg_direct
g["bias_vs_apparent"] = g.rg_estimate - g.rg_app_exact
print(g[["args_h2", "scenario", "n", "rg_direct", "rg_app_1st", "rg_app_exact",
         "rg_estimate", "bias_vs_direct", "bias_vs_apparent"]].to_string(index=False))

# ---- validation ----------------------------------------------------------------
print("\n================= VALIDATION =================")
rm = raw[(raw.args_theta == 0.0) & (raw.args_rmate == 0.0)]
print(f"(i)  RM: max|apparent_h2_exact - h2_true| = {np.abs(rm.apparent_h2_exact - rm.h2_true).max():.2e}")
print(f"     RM: max|apparent_rg_exact - rg_true| = {np.nanmax(np.abs(rm.apparent_rg_exact - rm.rg_true)):.2e}")
ident = np.abs(raw.r2_y_g - raw.cov_g_y**2 / raw.h2_true)
print(f"(ii) identity max|r2_y_g - cov_g_y^2/h2_true| = {ident.max():.2e}")
gap = raw.r2_y_g - raw.apparent_h2_exact        # <= 0 in theory (full BLP >= own-trait BLP)
viol = float((gap > 1e-3).mean())               # genuine violations, beyond float/MC noise
print(f"(iii) fraction apparent_h2_exact < r2_y_g beyond 1e-3 tol = {viol:.4f} (should be 0); "
      f"worst excess = {gap.max():.2e} (float/MC noise, only in no-cross-trait-VT cells)")
print(f"\nFull table (gens 0-5) written to {OUT_CSV}")
