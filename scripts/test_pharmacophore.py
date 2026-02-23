#!/usr/bin/env python3
"""Quick test: run pharmacophore validation on all 100 existing complexes."""
import sys
sys.path.insert(0, "scripts")
from run_vina_docking import validate_pharmacophore
from pathlib import Path
import pandas as pd

cpx = Path("results/vina_docking/complex_pdbs")
prog = pd.read_csv("results/vina_docking/progress.csv")
ok = prog[prog["status"] == "success"].copy()

results = []
for _, row in ok.iterrows():
    nm = row["name"]
    pdb = cpx / f"{nm}_complex.pdb"
    if pdb.exists():
        p = validate_pharmacophore(pdb)
        results.append({
            "name": nm, "vina": row["vina_score"],
            "hbonds": p["n_hbonds"],
            "pscore": p["pharmacophore_score"],
            "His305": p.get("residues", {}).get("His305", {}).get("dist"),
            "His397": p.get("residues", {}).get("His397", {}).get("dist"),
            "Arg274": p.get("residues", {}).get("Arg274", {}).get("dist"),
            "Ser237": p.get("residues", {}).get("Ser237", {}).get("dist"),
        })

df = pd.DataFrame(results)
print("=" * 70)
print("PHARMACOPHORE VALIDATION — All 100 Vina Complexes")
print("=" * 70)
print(f"\nH-bond distribution (of 4 VDR anchors):")
vc = df["hbonds"].value_counts().sort_index()
for k, v in vc.items():
    print(f"  {int(k)} H-bonds: {v} compounds")
print(f"\n  Mean: {df['hbonds'].mean():.2f}")
print(f"  >=1 H-bond: {(df['hbonds'] >= 1).sum()}/100")
print(f"  >=2 H-bonds: {(df['hbonds'] >= 2).sum()}/100")

print("\n" + "=" * 70)
print("Top 15 by Vina Score (with pharmacophore)")
print("=" * 70)
top = df.nsmallest(15, "vina")
for _, r in top.iterrows():
    h305 = f"{r['His305']:.1f}" if pd.notna(r["His305"]) else "N/A"
    h397 = f"{r['His397']:.1f}" if pd.notna(r["His397"]) else "N/A"
    arg = f"{r['Arg274']:.1f}" if pd.notna(r["Arg274"]) else "N/A"
    ser = f"{r['Ser237']:.1f}" if pd.notna(r["Ser237"]) else "N/A"
    print(f"  {r['name']:12s}  Vina={r['vina']:6.1f}  Hb={int(r['hbonds'])}  "
          f"His305={h305:>5s}  His397={h397:>5s}  Arg274={arg:>5s}  Ser237={ser:>5s}")

print("\n" + "=" * 70)
print("Top 10 by Pharmacophore Score (pose quality)")
print("=" * 70)
top_p = df.nlargest(10, "pscore")
for _, r in top_p.iterrows():
    print(f"  {r['name']:12s}  Vina={r['vina']:6.1f}  Hb={int(r['hbonds'])}  "
          f"pscore={r['pscore']:.3f}")

# Correlation
import numpy as np
from scipy.stats import pearsonr, spearmanr
mask = df["pscore"] > 0
if mask.sum() > 5:
    pr, pp = pearsonr(df.loc[mask, "vina"], df.loc[mask, "pscore"])
    sr, sp = spearmanr(df.loc[mask, "vina"], df.loc[mask, "pscore"])
    print(f"\n  Correlation (compounds with >=1 Hbond, n={mask.sum()}):")
    print(f"    Pearson  r={pr:.3f}  p={pp:.3e}")
    print(f"    Spearman r={sr:.3f}  p={sp:.3e}")

# Consensus: strong Vina + good pharmacophore
df["rank_vina"] = df["vina"].rank()
df["rank_pharm"] = df["pscore"].rank(ascending=False)
df["consensus"] = df["rank_vina"] + df["rank_pharm"]
print("\n" + "=" * 70)
print("Consensus Top 10 (Vina rank + Pharmacophore rank)")
print("=" * 70)
cons = df.nsmallest(10, "consensus")
for _, r in cons.iterrows():
    print(f"  {r['name']:12s}  Vina={r['vina']:6.1f}  Hb={int(r['hbonds'])}  "
          f"pscore={r['pscore']:.3f}  consensus_rank={r['consensus']:.0f}")
