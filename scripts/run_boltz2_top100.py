#!/usr/bin/env python3
"""
Boltz-2 VDR-LBD predictions — production quality for RTX 4090 (16 GB).

Uses the exact 1DB1 crystal construct (259 aa, Chain A):
  - UniProt 118-164 + 216-427 (insertion domain Δ165-215 deleted)
  - Matches the Rochel et al. (2000) construct that co-crystallised
    with 1,25(OH)₂D₃ at 1.0 Å resolution.
  - NOT the full 310-aa contiguous LBD (which includes the disordered
    51-residue insertion domain absent from 1DB1).

Settings (Boltz-2 defaults for production quality):
  - 259-residue 1DB1 construct (~1.4x faster than 310-aa, O(N²) attention)
  - sampling_steps=200 (Boltz-2 default — required for proper diffusion convergence)
  - recycling_steps=3 (Boltz-2 default; affinity module internally uses 5)
  - diffusion_samples=5 (Boltz-2 default for affinity ensemble)
  - All 12 binding pocket residues constrained (incl. His305, His397)
  - MSA cached after first compound (VDR LBD is the same for all)
  - Progress saved after every compound (crash-safe resume)

Estimated time: ~8-12 min/compound × 100 = ~14-20 hours on RTX 4090

Usage:
    python scripts/run_boltz2_top100.py                        # all 100
    python scripts/run_boltz2_top100.py --limit 5              # test 5
    python scripts/run_boltz2_top100.py --sampling-steps 100   # faster
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ===================================================================
# VDR 1DB1 CONSTRUCT SEQUENCE (259 aa, Chain A SEQRES)
# ===================================================================
# Rochel et al. (2000) construct: UniProt 118-164 + 216-427
# (insertion domain Δ165-215 deleted — 51-residue disordered loop
# absent from 1DB1 and all other VDR crystal structures).
#
# Position 1 in this sequence = UniProt 118 (Asp)
# Positions 1-47  = UniProt 118-164  (before insertion domain)
# Positions 48-259 = UniProt 216-427  (after insertion domain)
VDR_LBD_SEQUENCE = (
    "DSLRPKLSEEQQRIIAILLDAHHKTYDPTYSDFCQFRPPVRVNDGGGSV"  # 1-48  (UP 118-164,216)
    "TLELSQLSMLPHLADLVSYSIQKVIGFAKMIPGFRDLTSEDQIVLLKSS"  # 49-98 (UP 217-266)
    "AIEVIMLRSNESFTMDDMSWTCGNQDYKYRVSDVTKAGHSLELIEPLIK"  # 99-148(UP 267-316)
    "FQVGLKKLNLHEEEHVLLMAICIVSPDRPGVQDAALIEAIQDRLSNTLQ"  # 149-198(UP 317-366)
    "TYIRCRHPPPGSHLLYAKMIQKLADLRSLNEEHSKQYRCLSFQPECSMKL"  # 199-248(UP 367-416)
    "TPLVLEVFGNEIS"                                        # 249-259(UP 417-427)
)
assert len(VDR_LBD_SEQUENCE) == 259, f"1DB1 construct must be 259 aa, got {len(VDR_LBD_SEQUENCE)}"

# Binding pocket residues mapped to 1DB1 construct numbering:
#   UniProt pos → 1DB1 pos: UP-117 if UP≤164, else UP-168
#   UniProt: [143,145,227,233,237,269,271,272,274,278,305,397]
# 1DB1 crystal H-bond network (Rochel et al. 2000):
#   25-OH:  His137(305) NE2 + His229(397) NE2
#   1α-OH:  Arg106(274) NH1 + Ser69(237)  OG
#   3β-OH:  Tyr26(143)  OH  + Ser110(278) OG
VDR_LBD_BINDING = [26, 28, 59, 65, 69, 101, 103, 104, 106, 110, 137, 229]

# 6 pharmacophore H-bond anchors (1DB1 construct numbering)
PHARMACOPHORE: dict[int, dict] = {
    137: {"name": "His305",  "atoms": ["NE2"],
          "role": "25-OH acceptor (2.81 Å in 1DB1)"},
    229: {"name": "His397",  "atoms": ["NE2"],
          "role": "25-OH acceptor (2.82 Å in 1DB1)"},
    106: {"name": "Arg274",  "atoms": ["NH1", "NH2", "NE"],
          "role": "1α-OH donor (2.86 Å in 1DB1)"},
    69:  {"name": "Ser237",  "atoms": ["OG"],
          "role": "1α-OH acceptor (2.78 Å in 1DB1)"},
    26:  {"name": "Tyr143",  "atoms": ["OH"],
          "role": "3β-OH acceptor (2.83 Å in 1DB1)"},
    110: {"name": "Ser278",  "atoms": ["OG"],
          "role": "3β-OH acceptor (2.91 Å in 1DB1)"},
}
HBOND_DIST_CUTOFF = 3.2  # Å — tightened for publication-quality scoring

# ===================================================================
ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "pdb" / "top100_g_complexes" / "manifest.csv"
OUTPUT_DIR = ROOT / "results" / "boltz2_top100"


def create_yaml(out_dir: Path, name: str, smiles: str) -> Path:
    """Create Boltz-2 YAML input with VDR-LBD + ligand + pocket + affinity."""
    smi_safe = smiles.replace("'", "''")
    contacts = ", ".join(f"[A, {r}]" for r in VDR_LBD_BINDING)

    yaml_text = f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {VDR_LBD_SEQUENCE}
  - ligand:
      id: B
      smiles: '{smi_safe}'

constraints:
  - pocket:
      binder: B
      contacts: [{contacts}]
      max_distance: 4.5

properties:
  - affinity:
      binder: B
"""
    yaml_path = out_dir / f"{name}.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def run_prediction(
    yaml_path: Path,
    out_dir: Path,
    *,
    sampling_steps: int = 200,
    recycling_steps: int = 3,
    diffusion_samples: int = 5,
    accelerator: str = "gpu",
    timeout: int = 1800,
) -> tuple[bool, str, float]:
    """Run a single Boltz-2 prediction. Returns (success, msg, elapsed_sec)."""
    boltz_exe = shutil.which("boltz")
    if boltz_exe is None:
        venv_boltz = Path(sys.executable).parent / "boltz.exe"
        if venv_boltz.exists():
            boltz_exe = str(venv_boltz)
        else:
            return False, "boltz executable not found", 0.0

    cmd = [
        boltz_exe, "predict",
        str(yaml_path.resolve()),
        "--out_dir", str(out_dir.resolve()),
        "--model", "boltz2",
        "--accelerator", accelerator,
        "--devices", "1",
        "--recycling_steps", str(recycling_steps),
        "--sampling_steps", str(sampling_steps),
        "--diffusion_samples", str(diffusion_samples),
        "--output_format", "pdb",
        "--use_msa_server",
        "--no_kernels",
        "--affinity_mw_correction",
        "--override",
    ]

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, cwd=str(ROOT),
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            err = result.stderr[-500:] if result.stderr else "unknown"
            return False, f"exit {result.returncode} ({elapsed:.0f}s): {err}", elapsed
        return True, f"OK ({elapsed:.0f}s)", elapsed
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s", time.time() - t0
    except Exception as exc:
        return False, str(exc), time.time() - t0


def validate_output(pred_base: Path, name: str) -> dict:
    """Validate Boltz-2 output — check PDB, confidence, affinity, ligand position."""
    info = {
        "valid": True, "confidence_score": None, "ligand_iptm": None,
        "affinity": None, "pdb_path": None, "warnings": [],
    }

    pred_dir = pred_base / f"boltz_results_{name}" / "predictions" / name
    pdb_file = pred_dir / f"{name}_model_0.pdb"

    if not pdb_file.exists():
        info["valid"] = False
        info["warnings"].append("PDB not found")
        return info

    text = pdb_file.read_text()
    if len(text) < 200:
        info["valid"] = False
        info["warnings"].append("PDB empty")
        return info

    info["pdb_path"] = str(pdb_file)
    hetatm = text.count("HETATM")
    if hetatm == 0:
        info["warnings"].append("No HETATM — ligand missing")

    # Confidence
    for cf in pred_dir.glob("confidence_*.json"):
        try:
            d = json.loads(cf.read_text())
            info["confidence_score"] = d.get("confidence_score")
            info["ligand_iptm"] = d.get("ligand_iptm")
        except Exception:
            pass
        break

    # Affinity
    for af in pred_dir.glob("affinity_*.json"):
        try:
            d = json.loads(af.read_text())
            info["affinity"] = d.get("affinity_pred_value")
        except Exception:
            pass
        break

    # Spatial check — is ligand near protein?
    if hetatm > 0:
        prot_xyz, lig_xyz = [], []
        for line in text.splitlines():
            if len(line) >= 54:
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except (ValueError, IndexError):
                    continue
                if line.startswith("ATOM"):
                    prot_xyz.append((x, y, z))
                elif line.startswith("HETATM"):
                    lig_xyz.append((x, y, z))
        if prot_xyz and lig_xyz:
            prot_arr = np.array(prot_xyz)
            lig_cen = np.mean(lig_xyz, axis=0)
            min_d = float(np.linalg.norm(prot_arr - lig_cen, axis=1).min())
            if min_d > 15.0:
                info["warnings"].append(f"Ligand {min_d:.1f}A from protein")

    cs = info.get("confidence_score")
    if cs is not None and cs < 0.3:
        info["warnings"].append(f"Low confidence {cs:.3f}")

    if info["warnings"]:
        info["valid"] = False
    return info


def validate_pharmacophore(pdb_path: Path) -> dict:
    """Score H-bond contacts between ligand polar atoms and 6 VDR anchor residues.

    Identical logic to run_vina_docking.validate_pharmacophore so that both
    pipelines produce directly comparable pharmacophore metrics.
    """
    pharm: dict = {"n_hbonds": 0, "pharmacophore_score": 0.0, "residues": {}}
    key_xyz: dict[tuple[int, str], np.ndarray] = {}
    lig_polar: list[np.ndarray] = []

    text = pdb_path.read_text()
    for line in text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if len(line) < 54:
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except (ValueError, IndexError):
            continue

        if line.startswith("ATOM") and line[21] == "A":
            try:
                rn = int(line[22:26].strip())
            except ValueError:
                continue
            aname = line[12:16].strip()
            if rn in PHARMACOPHORE:
                key_xyz[(rn, aname)] = np.array([x, y, z])
        elif line.startswith("HETATM") and line[21] == "B":
            elem = line[76:78].strip() if len(line) >= 78 else ""
            if not elem:
                elem = line[12:16].strip()[0] if line[12:16].strip() else "C"
            # Handle any 2-char types (standard Boltz PDBs should be fine)
            if len(elem) > 1:
                elem = elem[0]
            if elem in ("O", "N", "S"):
                lig_polar.append(np.array([x, y, z]))

    if not lig_polar or not key_xyz:
        return pharm

    for rn, info in PHARMACOPHORE.items():
        best_d = float("inf")
        for aname in info["atoms"]:
            if (rn, aname) not in key_xyz:
                continue
            for lp in lig_polar:
                d = float(np.linalg.norm(key_xyz[(rn, aname)] - lp))
                if d < best_d:
                    best_d = d

        satisfied = best_d <= HBOND_DIST_CUTOFF
        pharm["residues"][info["name"]] = {
            "dist": round(best_d, 2) if best_d < 999 else None,
            "satisfied": satisfied,
        }
        if satisfied:
            pharm["n_hbonds"] += 1
            pharm["pharmacophore_score"] += max(
                0, (HBOND_DIST_CUTOFF - best_d) / HBOND_DIST_CUTOFF)

    pharm["pharmacophore_score"] = round(pharm["pharmacophore_score"], 3)
    return pharm


def main():
    parser = argparse.ArgumentParser(
        description="Boltz-2 VDR 1DB1 predictions (259-aa Rochel construct)")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    parser.add_argument("--sampling-steps", type=int, default=200,
                        help="Sampling steps (200=default, 100=faster, 50=draft)")
    parser.add_argument("--recycling-steps", type=int, default=3,
                        help="Recycling steps (3=default; affinity uses 5 internally)")
    parser.add_argument("--diffusion-samples", type=int, default=5,
                        help="Diffusion samples for affinity ensemble (5=default)")
    parser.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--timeout", type=int, default=2400,
                        help="Per-compound timeout in seconds (default 2400=40min)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--recompute-pharmacophore", action="store_true",
                        default=False,
                        help="Recompute pharmacophore for existing results")
    args = parser.parse_args()

    # ── Pharmacophore post-processing mode ──────────────────────────
    if args.recompute_pharmacophore:
        out = Path(args.output)
        progress_csv = out / "progress.csv"
        collected_dir = out / "collected_pdbs"
        if not progress_csv.exists():
            print("ERROR: No progress.csv found"); sys.exit(1)
        df_prog = pd.read_csv(progress_csv)
        print(f"Recomputing pharmacophore for {len(df_prog)} Boltz-2 compounds...")
        updated = 0
        for idx, row in df_prog.iterrows():
            name = row["name"]
            pdb_path = collected_dir / f"{name}.pdb"
            if not pdb_path.exists():
                continue
            pharm = validate_pharmacophore(pdb_path)
            df_prog.at[idx, "n_hbonds"] = pharm.get("n_hbonds", 0)
            df_prog.at[idx, "pharmacophore_score"] = pharm.get(
                "pharmacophore_score", 0.0)
            for rn_key, rn_info in PHARMACOPHORE.items():
                col = f"pharm_{rn_info['name']}"
                dist = pharm.get("residues", {}).get(
                    rn_info["name"], {}).get("dist")
                df_prog.at[idx, col] = dist
            updated += 1
        df_prog.to_csv(progress_csv, index=False)
        print(f"Updated {updated} rows in {progress_csv}")
        if "n_hbonds" in df_prog.columns:
            hb = df_prog["n_hbonds"].dropna()
            print(f"Pharmacophore H-bonds (of 6 VDR anchors):")
            print(f"  mean={hb.mean():.1f}  "
                  f"0-hb={int((hb==0).sum())}  1-hb={int((hb==1).sum())}  "
                  f"2-hb={int((hb==2).sum())}  3-hb={int((hb==3).sum())}  "
                  f"4-hb={int((hb==4).sum())}  5-hb={int((hb==5).sum())}  "
                  f"6-hb={int((hb==6).sum())}")
        sys.exit(0)

    df = pd.read_csv(args.manifest)
    if args.start > 0:
        df = df.iloc[args.start:]
    if args.limit is not None:
        df = df.iloc[:args.limit]

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    yaml_dir = out / "yaml_inputs"
    yaml_dir.mkdir(exist_ok=True)
    collected_dir = out / "collected_pdbs"
    collected_dir.mkdir(exist_ok=True)
    progress_csv = out / "progress.csv"

    # Resume from progress + disk scan
    done_names = set()
    if progress_csv.exists():
        prev = pd.read_csv(progress_csv)
        done_names = set(
            prev[prev["status"].isin(["success", "success_warnings"])]["name"]
        )

    # Disk recovery — find completed PDBs not in progress
    for bdir in out.glob("boltz_results_*"):
        bname = bdir.name.replace("boltz_results_", "")
        pdb_p = bdir / "predictions" / bname / f"{bname}_model_0.pdb"
        if pdb_p.exists() and bname not in done_names:
            val = validate_output(out, bname)
            status = "success" if val["valid"] else "success_warnings"
            entry = {
                "name": bname, "source": "", "csv_row": "", "smiles": "",
                "status": status,
                "confidence_score": val.get("confidence_score"),
                "ligand_iptm": val.get("ligand_iptm"),
                "affinity": val.get("affinity"),
                "pdb_path": val.get("pdb_path"),
                "elapsed": 0,
                "warnings": "; ".join(val.get("warnings", [])),
            }
            if progress_csv.exists():
                prev = pd.read_csv(progress_csv)
                prev = pd.concat(
                    [prev, pd.DataFrame([entry])], ignore_index=True)
                prev.to_csv(progress_csv, index=False)
            else:
                pd.DataFrame([entry]).to_csv(progress_csv, index=False)
            done_names.add(bname)
            if val["pdb_path"]:
                shutil.copy2(val["pdb_path"], collected_dir / f"{bname}.pdb")

    total = len(df)
    to_do = sum(
        1 for _, row in df.iterrows()
        if f"{row['source']}_{row['csv_row']}" not in done_names
    )

    print("=" * 70)
    print("BOLTZ-2 VDR 1DB1 PREDICTIONS (Rochel construct)")
    print("=" * 70)
    print(f"Protein: VDR 1DB1 (259 aa, UP 118-164+216-427, {len(VDR_LBD_SEQUENCE)} aa)")
    print(f"Pocket:  {VDR_LBD_BINDING} ({len(VDR_LBD_BINDING)} residues)")
    print(f"Model:   boltz2 | steps={args.sampling_steps}"
          f" | recycling={args.recycling_steps}"
          f" | diff_samples={args.diffusion_samples}")
    print(f"Total:   {total} compounds | Done: {len(done_names)}"
          f" | Remaining: {to_do}")
    print("=" * 70)

    rows: list[dict] = []
    n_ok, n_warn, n_fail, n_skip = 0, 0, 0, 0
    times: list[float] = []
    t_start = time.time()

    for i, (_, row) in enumerate(df.iterrows()):
        src = row["source"]
        csvr = row["csv_row"]
        name = f"{src}_{csvr}"
        tag = f"[{i+1}/{total}]"

        if args.skip_existing and name in done_names:
            n_skip += 1
            continue

        smiles = row["smiles"]
        smi_short = (smiles[:55] + "...") if len(smiles) > 55 else smiles
        print(f"\n{tag} {name}  {smi_short}")

        yaml_path = create_yaml(yaml_dir, name, smiles)

        ok, msg, elapsed = run_prediction(
            yaml_path, out,
            sampling_steps=args.sampling_steps,
            recycling_steps=args.recycling_steps,
            diffusion_samples=args.diffusion_samples,
            accelerator=args.accelerator,
            timeout=args.timeout,
        )
        times.append(elapsed)

        # Retry once on failure with increased timeout (+50%)
        if not ok:
            retry_timeout = int(args.timeout * 1.5)
            print(f"  FAIL: {msg}  — retrying with timeout={retry_timeout}s ...")
            ok, msg, elapsed2 = run_prediction(
                yaml_path, out,
                sampling_steps=args.sampling_steps,
                recycling_steps=args.recycling_steps,
                diffusion_samples=args.diffusion_samples,
                accelerator=args.accelerator,
                timeout=retry_timeout,
            )
            elapsed += elapsed2
            times[-1] = elapsed

        if not ok:
            print(f"  FAIL: {msg}")
            rows.append({
                "name": name, "source": src, "csv_row": csvr,
                "smiles": smiles, "status": "failed", "error": msg,
                "elapsed": elapsed,
            })
            n_fail += 1
        else:
            val = validate_output(out, name)
            status = "success" if val["valid"] else "success_warnings"
            cs = val.get("confidence_score", "?")
            li = val.get("ligand_iptm", "?")
            af = val.get("affinity", "?")
            cs_s = f"{cs:.3f}" if isinstance(cs, float) else str(cs)
            li_s = f"{li:.3f}" if isinstance(li, float) else str(li)
            af_s = f"{af:.2f}" if isinstance(af, float) else str(af)
            print(f"  OK {status}  conf={cs_s}  iptm={li_s}"
                  f"  aff={af_s}  ({elapsed:.0f}s)")

            if val["warnings"]:
                for w in val["warnings"]:
                    print(f"    WARNING: {w}")
                n_warn += 1
            else:
                n_ok += 1

            if val["pdb_path"]:
                shutil.copy2(
                    val["pdb_path"], collected_dir / f"{name}.pdb")

            # Pharmacophore scoring on collected PDB
            pharm = {}
            cpdb = collected_dir / f"{name}.pdb"
            if cpdb.exists():
                pharm = validate_pharmacophore(cpdb)
            n_hb = pharm.get("n_hbonds", 0)
            pscore = pharm.get("pharmacophore_score", 0.0)

            row_data = {
                "name": name, "source": src, "csv_row": csvr,
                "smiles": smiles, "status": status,
                "confidence_score": val.get("confidence_score"),
                "ligand_iptm": val.get("ligand_iptm"),
                "affinity": val.get("affinity"),
                "pdb_path": val.get("pdb_path"),
                "n_hbonds": n_hb,
                "pharmacophore_score": pscore,
                "elapsed": elapsed,
                "warnings": "; ".join(val.get("warnings", [])),
            }
            # Per-residue pharmacophore distances
            for rn_key, rn_info in PHARMACOPHORE.items():
                col = f"pharm_{rn_info['name']}"
                dist = pharm.get("residues", {}).get(
                    rn_info["name"], {}).get("dist")
                row_data[col] = dist
            rows.append(row_data)

        # Save progress + ETA
        batch_df = pd.DataFrame(rows)
        if progress_csv.exists():
            prev = pd.read_csv(progress_csv)
            prev = prev[~prev["name"].isin(batch_df["name"])]
            batch_df = pd.concat([prev, batch_df], ignore_index=True)
        batch_df.to_csv(progress_csv, index=False)

        done_so_far = n_ok + n_warn + n_fail
        if times:
            avg_t = np.mean(times)
            remaining = to_do - done_so_far
            eta_min = remaining * avg_t / 60
            print(f"  ETA: {eta_min:.0f} min ({eta_min/60:.1f}h) | "
                  f"avg={avg_t:.0f}s/cpd | done={done_so_far}/{to_do}")

    # Final summary
    wall = time.time() - t_start
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Success:  {n_ok}  |  Warnings: {n_warn}"
          f"  |  Failed: {n_fail}  |  Skipped: {n_skip}")
    print(f"  Wall time: {wall/60:.1f} min")
    if times:
        print(f"  Avg time:  {np.mean(times):.1f}s/compound")
    print(f"  Output:    {out}")
    print(f"  PDBs:      {collected_dir}")
    print(f"  Progress:  {progress_csv}")

    if progress_csv.exists():
        final = pd.read_csv(progress_csv)
        for col in ["confidence_score", "affinity"]:
            if col in final.columns:
                vals = final[col].dropna()
                if len(vals) > 0:
                    print(f"  {col}: mean={vals.mean():.3f}"
                          f"  min={vals.min():.3f}  max={vals.max():.3f}")


if __name__ == "__main__":
    main()
