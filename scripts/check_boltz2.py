#!/usr/bin/env python3
"""Quick check of Boltz-2 run status."""
import pandas as pd

df = pd.read_csv("results/boltz2_top100/progress.csv")
f = df[df["status"] == "failed"]
print("Failed:")
print(f[["name", "error"]].to_string())

s = df[df["status"] == "success"]
print(f"\nSuccess: {len(s)}/100")
print(f"Confidence: mean={s['confidence_score'].mean():.3f}  "
      f"min={s['confidence_score'].min():.3f}  max={s['confidence_score'].max():.3f}")
print(f"Affinity:   mean={s['affinity'].mean():.2f}  "
      f"min={s['affinity'].min():.2f}  max={s['affinity'].max():.2f}")
print(f"iPTM:       mean={s['ligand_iptm'].mean():.3f}  "
      f"min={s['ligand_iptm'].min():.3f}  max={s['ligand_iptm'].max():.3f}")
