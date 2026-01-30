"""
Run Boltz-2 structure predictions for top5/bottom5 G group compounds.

This script:
1. Creates YAML input files for Boltz-2 with VDR protein + ligand
2. Runs Boltz-2 predictions for protein-ligand complexes
3. Saves predicted structures

Usage:
    python scripts/run_boltz2_predictions.py --manifest results/cwra_final/g_group_pdbs_docked/manifest.csv \
        --output results/cwra_final/boltz2_predictions
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# VDR ligand binding domain sequence from 1DB1 (residues 120-423)
VDR_SEQUENCE = """LRPKLSEEQQRIIAILLDAHHKTYDPTYSDFCQFRPPVRVNDGGGSVTLELSQLSMLPHLADLVSYSIQKVIGFAKMIPGFRDLTSEDQIVLLKSSAIEVIMLRSNESFTMDDMSWTCGNQDYKYRVSDVTKAGHSLELIEPLIKFQVGLKKLNLHEEEHVLLMAICIVSPDRPGVQDAALIEAIQDRLSNTLQTYIRCRHPPPLLYAKMIQKLADLRSLNEEHSKQYRCLSFQPECSMKLTPLVLEVFG"""


def create_boltz2_yaml(
    output_path: Path,
    compound_name: str,
    smiles: str,
    protein_sequence: str = VDR_SEQUENCE
) -> Path:
    """Create YAML input file for Boltz-2 prediction."""
    # Use single quotes for SMILES to avoid backslash escape issues
    # SMILES can contain \ characters (e.g., \C for cis double bonds)
    # Use simple short IDs to avoid Boltz-2 truncation issues
    yaml_content = f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {protein_sequence}
  - ligand:
      id: B
      smiles: '{smiles}'
"""
    
    yaml_path = output_path / f"{compound_name}.yaml"
    yaml_path.write_text(yaml_content)
    return yaml_path


def find_boltz_executable() -> str:
    """Find the boltz executable."""
    import shutil
    
    # Try common locations
    possible_paths = [
        shutil.which("boltz"),
        Path.home() / "AppData" / "Roaming" / "Python" / "Python312" / "Scripts" / "boltz.exe",
        Path.home() / "AppData" / "Local" / "Programs" / "Python" / "Python312" / "Scripts" / "boltz.exe",
    ]
    
    for p in possible_paths:
        if p and Path(p).exists():
            return str(p)
    
    # Fallback to running via Python module
    return None


def run_boltz2_prediction(
    yaml_path: Path,
    output_dir: Path,
    model: str = "boltz2",
    devices: int = 1,
    accelerator: str = "gpu",
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    output_format: str = "pdb"
) -> bool:
    """Run Boltz-2 prediction for a single input."""
    
    # Convert paths to absolute strings
    yaml_abs = str(yaml_path.resolve())
    out_abs = str(output_dir.resolve())
    
    # Find boltz executable
    boltz_exe = find_boltz_executable()
    
    if boltz_exe:
        cmd = [
            boltz_exe, "predict",
            yaml_abs,
            "--out_dir", out_abs,
            "--model", model,
            "--devices", str(devices),
            "--accelerator", accelerator,
            "--sampling_steps", str(sampling_steps),
            "--diffusion_samples", str(diffusion_samples),
            "--output_format", output_format,
            "--use_msa_server",  # Use ColabFold server for MSA generation
            "--no_kernels",  # Avoid cuequivariance_ops_torch issues
            "--override"
        ]
    else:
        # Fallback to Python -m approach
        cmd = [
            sys.executable, "-m", "boltz",
            "predict",
            yaml_abs,
            "--out_dir", out_abs,
            "--model", model,
            "--devices", str(devices),
            "--accelerator", accelerator,
            "--sampling_steps", str(sampling_steps),
            "--diffusion_samples", str(diffusion_samples),
            "--output_format", output_format,
            "--use_msa_server",  # Use ColabFold server for MSA generation
            "--no_kernels",  # Avoid cuequivariance_ops_torch issues
            "--override"
        ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 min timeout per prediction
        )
        
        if result.returncode != 0:
            print(f"  Error: {result.stderr[:500]}")
            return False
        return True
        
    except subprocess.TimeoutExpired:
        print("  Timeout after 30 minutes")
        return False
    except Exception as e:
        print(f"  Exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Boltz-2 predictions for G group compounds")
    parser.add_argument("--manifest", "-m", default="results/cwra_final/g_group_pdbs_docked/manifest.csv",
                        help="Input manifest CSV from docking")
    parser.add_argument("--output", "-o", default="results/cwra_final/boltz2_predictions",
                        help="Output directory for Boltz-2 predictions")
    parser.add_argument("--model", default="boltz2", choices=["boltz1", "boltz2"],
                        help="Boltz model to use")
    parser.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu"],
                        help="Accelerator to use")
    parser.add_argument("--sampling-steps", type=int, default=200,
                        help="Number of sampling steps")
    parser.add_argument("--diffusion-samples", type=int, default=1,
                        help="Number of diffusion samples")
    parser.add_argument("--output-format", default="pdb", choices=["pdb", "mmcif"],
                        help="Output format")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip compounds with existing predictions")
    
    args = parser.parse_args()
    
    # Load manifest
    print(f"Loading manifest from {args.manifest}")
    df = pd.read_csv(args.manifest)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_dir = output_dir / "inputs"
    yaml_dir.mkdir(exist_ok=True)
    
    print(f"\nProcessing {len(df)} compounds")
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model}, Accelerator: {args.accelerator}")
    print(f"Sampling steps: {args.sampling_steps}, Diffusion samples: {args.diffusion_samples}")
    print()
    
    results = []
    
    for idx, row in df.iterrows():
        compound_name = f"{row['set']}_{row['group']}_rank{row['rank']:05d}_{row['generator']}"
        compound_name = compound_name.replace("-", "_")  # Clean for filename
        
        print(f"[{idx+1}/{len(df)}] {compound_name}")
        
        # Check if already done (look for output PDB file)
        pred_dir = output_dir / f"boltz_results_{compound_name}"
        pdb_file = pred_dir / "predictions" / compound_name / f"{compound_name}_model_0.pdb"
        if args.skip_existing and pdb_file.exists():
            print("  Skipping (exists)")
            results.append({
                'compound': compound_name,
                'smiles': row['smiles'],
                'status': 'skipped',
                'group': row['group'],
                'set': row['set'],
                'rank': row['rank']
            })
            continue
        
        # Create YAML input
        yaml_path = create_boltz2_yaml(
            yaml_dir,
            compound_name,
            row['smiles']
        )
        print(f"  Created: {yaml_path.name}")
        
        # Run prediction
        print(f"  Running Boltz-2 prediction...")
        success = run_boltz2_prediction(
            yaml_path,
            output_dir,
            model=args.model,
            accelerator=args.accelerator,
            sampling_steps=args.sampling_steps,
            diffusion_samples=args.diffusion_samples,
            output_format=args.output_format
        )
        
        status = 'success' if success else 'failed'
        print(f"  Status: {status}")
        
        results.append({
            'compound': compound_name,
            'smiles': row['smiles'],
            'status': status,
            'group': row['group'],
            'set': row['set'],
            'rank': row['rank']
        })
    
    # Save results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "boltz2_results.csv", index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    success_count = len(results_df[results_df['status'] == 'success'])
    failed_count = len(results_df[results_df['status'] == 'failed'])
    skipped_count = len(results_df[results_df['status'] == 'skipped'])
    
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
