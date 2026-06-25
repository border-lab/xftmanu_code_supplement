#!/usr/bin/env python3
"""Aggregate the VT-structure continuum prototype into the H1/H2/H3 + R1/R2/R3 decomposition.
Reads vt_proto/*_xvt*_pheno_parsed.csv, recovers the cross/own ratio from the filename,
computes exact apparent (BLP) per seed via the same compound-symmetric reconstruction as
compute_apparent.py, and prints the gen-5 continuum table for RM and 5xAM.
"""
import numpy as np, pandas as pd, glob, re, os
def exact_apparent(Kk, h2_true, rg_true, cov_g_y, cov_g_y_cross):
    a, b = h2_true, rg_true*h2_true; c, d = cov_g_y, cov_g_y_cross
    G = np.full((Kk,Kk), b); np.fill_diagonal(G, a)
    C = np.full((Kk,Kk), d); np.fill_diagonal(C, c)
    Mx = C.T @ np.linalg.solve(G, C); diag = np.diag(Mx)
    R = Mx/np.sqrt(np.outer(diag,diag)); iu = np.triu_indices(Kk,1)
    return float(diag.mean()), float(R[iu].mean())

RATIO_LABEL = {0.0:"uVT", 0.125:"weak(total-½)", 0.5:"weak(per-trait-½)", 1.0:"strong/constant"}
rows = []
for f in glob.glob(os.path.join(os.path.dirname(__file__), "vt_proto", "*_xvt*_pheno_parsed.csv")):
    m = re.search(r"_xvt([0-9.]+)_pheno", os.path.basename(f))
    if not m:
        continue
    r2 = float(m.group(1))
    df = pd.read_csv(f)
    for _, r in df.iterrows():
        h2a, rga = exact_apparent(int(r.args_kmate), r.h2_true, r.rg_true, r.cov_g_y, r.cov_g_y_cross)
        rows.append(dict(rmate=r.args_rmate, r2=r2, gen=r.gen, seed=r.seed,
                         H1=h2a, H2=r.r2_y_g, H3=r.vbeta_true, h_est=r.he_h2,
                         R1=rga, R2=r.rg_true, R3=r.rbeta_true, r_est=r.he_rg))
D = pd.DataFrame(rows)
g5 = D[D.gen == 5].groupby(["rmate", "r2"]).mean(numeric_only=True).reset_index()
g5["structure"] = g5.r2.map(lambda x: RATIO_LABEL.get(x, f"r2={x}"))
g5 = g5.sort_values(["rmate", "r2"])
nseed = int(D[D.gen == 5].groupby(["rmate","r2"]).size().mean())

pd.set_option("display.float_format", lambda x: f"{x:6.3f}", )
for rm in sorted(g5.rmate.unique()):
    tag = "RM" if rm == 0.0 else f"{int(rm*0+5)}xAM (r={rm})"
    sub = g5[g5.rmate == rm]
    print(f"\n===== {tag}  (gen 5, mean of ~{nseed} seeds; n=16k prototype) =====")
    print("  HERITABILITY        estimate   H1(app)  H2(score) H3(LMM)")
    for _, x in sub.iterrows():
        print(f"   {x.structure:<18} {x.h_est:6.3f}   {x.H1:6.3f}   {x.H2:6.3f}   {x.H3:6.3f}")
    print("  GENETIC CORR        estimate   R1(app)  R2(score) R3(r_beta)")
    for _, x in sub.iterrows():
        print(f"   {x.structure:<18} {x.r_est:6.3f}   {x.R1:6.3f}   {x.R2:6.3f}   {x.R3:6.3f}")
print(f"\n(rows={len(D)}, written from {D.seed.nunique()} seeds x {D.r2.nunique()} ratios x {D.rmate.nunique()} mating regimes)")
