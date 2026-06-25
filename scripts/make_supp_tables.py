#!/usr/bin/env python3
"""
Reproducible generation of the two NEW round-5 supplementary decomposition tables.

  Supplementary Table S7 — VT-structure CONTINUUM decomposition (generation 5).
      Source: fig3_r5/fig_continuum_r5_h2_long.csv  and  fig_continuum_r5_rg_long.csv
              (long format, per-seed; 200 seeds). We report the MEAN over seeds at gen 5
              for each (starting h2, mating regime, VT structure), for the three tiers:
              Direct, Apparent, and LDSC/estimated  (heritability and genetic correlation).

  Supplementary Table S8 — MAIN-SCENARIO decomposition (generation 5).
      Source: apparent_decomposition.csv  (already-aggregated means).
      The manuscript version of the response Tables 1-2, i.e. WITHOUT the response-only
      "predictive" (H2 / R2) columns: just Direct, Apparent (exact BLP), and Estimated.

Run:  python3 make_supp_tables.py        # writes s7_continuum.csv, s8_mainscenario.csv + prints
Outputs are consumed by insert_supp_tables.py to build the Word tables.
"""
import csv, collections

GEN = 5
H2_ORDER = ["0.5", "0.25"]

# ---------- S8: main-scenario (gen 5) ----------
SCEN_ORDER = ["RM", "RM+VT", "5xAM", "5xAM+VT"]
def build_s8():
    rows = list(csv.DictReader(open("apparent_decomposition.csv")))
    out = []
    for h2 in H2_ORDER:
        for sc in SCEN_ORDER:
            r = next((x for x in rows if x["args_h2"]==h2 and x["scenario"]==sc and int(x["gen"])==GEN), None)
            assert r is not None, (h2, sc)
            out.append({
                "h2": h2, "scenario": sc,
                "h_direct": float(r["h2_direct"]), "h_app": float(r["h2_app_exact"]), "h_est": float(r["h2_estimate"]),
                "rg_direct": float(r["rg_direct"]), "rg_app": float(r["rg_app_exact"]), "rg_est": float(r["rg_estimate"]),
            })
    return out

# ---------- S7: continuum (gen 5), mean over seeds ----------
STRUCT_ORDER = ["no VT", "uVT (same-trait)", "weak (½ total)", "weak (½ per-trait)", "strong (constant)"]
MATE_ORDER = ["RM", "5xAM"]
def _mean_by(path, valcol, qmap):
    rows = list(csv.DictReader(open(path)))
    acc = collections.defaultdict(list)
    for x in rows:
        if int(x["gen"]) != GEN: continue
        q = qmap.get(x["quantity"])
        if q is None: continue
        acc[(x["args_h2"], x["mating"], x["structure"], q)].append(float(x[valcol]))
    return {k: sum(v)/len(v) for k, v in acc.items()}
def build_s7():
    hmap = {"Direct h²":"direct", "Apparent h²":"app", "LDSC h²":"est"}
    rmap = {"Direct r_g":"direct", "Apparent r_g":"app", "LDSC r_g":"est"}
    H = _mean_by("fig3_r5/fig_continuum_r5_h2_long.csv", "h2", hmap)
    R = _mean_by("fig3_r5/fig_continuum_r5_rg_long.csv", "rg", rmap)
    out = []
    for h2 in H2_ORDER:
        for mate in MATE_ORDER:
            for st in STRUCT_ORDER:
                out.append({
                    "h2": h2, "mate": mate, "struct": st,
                    "h_direct": H[(h2,mate,st,"direct")], "h_app": H[(h2,mate,st,"app")], "h_est": H[(h2,mate,st,"est")],
                    "rg_direct": R[(h2,mate,st,"direct")], "rg_app": R[(h2,mate,st,"app")], "rg_est": R[(h2,mate,st,"est")],
                })
    return out

if __name__ == "__main__":
    s8 = build_s8(); s7 = build_s7()
    with open("s8_mainscenario.csv","w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(s8[0])); w.writeheader(); w.writerows(s8)
    with open("s7_continuum.csv","w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(s7[0])); w.writeheader(); w.writerows(s7)
    print("=== S8 main-scenario (gen 5) ===")
    for r in s8:
        print(f"  h2={r['h2']:>4} {r['scenario']:<8} h: {r['h_direct']:.3f}/{r['h_app']:.3f}/{r['h_est']:.3f}  rg: {r['rg_direct']:.3f}/{r['rg_app']:.3f}/{r['rg_est']:.3f}")
    print("\n=== S7 continuum (gen 5, mean of 200 seeds) ===")
    for r in s7:
        print(f"  h2={r['h2']:>4} {r['mate']:<5} {r['struct']:<18} h: {r['h_direct']:.3f}/{r['h_app']:.3f}/{r['h_est']:.3f}  rg: {r['rg_direct']:.3f}/{r['rg_app']:.3f}/{r['rg_est']:.3f}")
    print("\nwrote s7_continuum.csv, s8_mainscenario.csv")
