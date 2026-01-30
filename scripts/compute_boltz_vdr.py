#!/usr/bin/env python3
"""
Boltz-2 Affinity Prediction for VDR - Portable Script
======================================================
Run this script on a GPU machine to compute Boltz-2 affinity predictions.

Requirements:
    pip install boltz pandas tqdm rdkit

Usage:
    python compute_boltz_vdr.py --input data/labeled_raw_modalities.csv --start 0 --end 1000

The script will:
1. Read compounds needing boltz_affinity predictions
2. Run Boltz-2 with GPU acceleration
3. Save progress every 20 compounds
4. Update the main CSV file with results
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

# =============================================================================
# CONFIGURATION
# =============================================================================
# VDR sequence (1DB1 Ligand Binding Domain, residues 118-425)
VDR_SEQUENCE = "MEAMAASTSLPDPGDFDRNVPRICGVCGDRATGFHFNAMTCEGCKGFFRRSMKRKALFTCPFNGDCRITKDNRRHCQACRLKRCVDIGMMKEFILTDEEVQRKREMILKRKEEEALKDSLRPKLSEEQQRIIAILLDAHHKTYDPTYSDFCQFRPPVRVNDGGGSHPSRPNSRHTPSFSGDSSSSCSDHCITSSDMMDSSSFSNLDLSEEDSDDPSVTLELSQLSMLPHLADLVSYSIQKVIGFAKMIPGFRDLTSEDQIVLLKSSAIEVIMLRSNESFTMDDMSWTCGNQDYKYRVSDVTKAGHSLELIEPLIKFQVGLKKLNLHEEEHVLLMAICIVSPDRPGVQDAALIEAIQDRLSNTLQTYIRCRHPPPGSHLLYAKMIQKLADLRSLNEEHSKQYRCLSFQPECSMKLTPLVLEVFGNEIS"

# VDR binding site residues for pocket constraint
VDR_BINDING_RESIDUES = [143, 145, 227, 233, 237, 269, 271, 272, 274, 278]

# Boltz executable - adjust path for your system
BOLTZ_EXE = None  # Will be auto-detected


def find_boltz_executable():
    """Find Boltz executable in common locations."""
    import shutil
    
    # Try system PATH first
    boltz = shutil.which("boltz")
    if boltz:
        return boltz
    
    # Try common locations
    locations = [
        os.path.expanduser("~/.local/bin/boltz"),
        os.path.expanduser("~/AppData/Roaming/Python/Python312/Scripts/boltz.exe"),
        os.path.expanduser("~/AppData/Roaming/Python/Python311/Scripts/boltz.exe"),
        "/usr/local/bin/boltz",
    ]
    
    for loc in locations:
        if os.path.exists(loc):
            return loc
    
    # Try python -m boltz
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import boltz; print(boltz.__file__)"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return [sys.executable, "-m", "boltz"]
    except:
        pass
    
    raise FileNotFoundError("Boltz executable not found. Install with: pip install boltz")


def create_yaml(smiles: str, compound_id: str, output_path: str) -> str:
    """Create Boltz YAML input file."""
    smi_escaped = smiles.replace("'", "''")
    
    contacts = ", ".join([f"[A, {r}]" for r in VDR_BINDING_RESIDUES[:6]])
    
    yaml_content = f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {VDR_SEQUENCE}
  - ligand:
      id: B
      smiles: '{smi_escaped}'

constraints:
  - pocket:
      binder: B
      contacts: [{contacts}]
      max_distance: 8.0

properties:
  - affinity:
      binder: B
"""
    
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    
    return output_path


def run_boltz(yaml_path: str, output_dir: str, boltz_exe) -> dict:
    """Run Boltz-2 prediction."""
    
    if isinstance(boltz_exe, list):
        cmd = boltz_exe + ["predict"]
    else:
        cmd = [boltz_exe, "predict"]
    
    cmd += [
        yaml_path,
        "--out_dir", output_dir,
        "--accelerator", "gpu",
        "--devices", "1",
        "--recycling_steps", "1",
        "--sampling_steps", "5",
        "--diffusion_samples_affinity", "1",
        "--no_kernels",  # Required for Windows
        "--num_workers", "0",
        "--use_msa_server",
        "--affinity_mw_correction"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def parse_output(output_dir: str, compound_id: str) -> dict:
    """Parse Boltz output files."""
    result = {
        "affinity_pred_value": np.nan,
        "confidence_score": np.nan
    }
    
    pred_base = Path(output_dir)
    
    # Find predictions directory
    for subdir in pred_base.iterdir():
        if subdir.name.startswith("boltz_results_"):
            pred_dir = subdir / "predictions" / compound_id
            break
    else:
        pred_dir = pred_base / "predictions" / compound_id
    
    if not pred_dir.exists():
        return result
    
    # Read affinity
    for af in pred_dir.glob("affinity_*.json"):
        with open(af, 'r') as f:
            data = json.load(f)
            result["affinity_pred_value"] = data.get("affinity_pred_value", np.nan)
            result["affinity_probability"] = data.get("affinity_probability_binary", np.nan)
        break
    
    # Read confidence
    for cf in pred_dir.glob("confidence_*.json"):
        with open(cf, 'r') as f:
            data = json.load(f)
            result["confidence_score"] = data.get("confidence_score", np.nan)
        break
    
    return result


def process_compounds(
    input_file: str,
    output_file: str = None,
    start_idx: int = 0,
    end_idx: int = None,
    save_every: int = 20
):
    """Process compounds and compute Boltz affinity."""
    
    global BOLTZ_EXE
    if BOLTZ_EXE is None:
        BOLTZ_EXE = find_boltz_executable()
        print(f"Using Boltz: {BOLTZ_EXE}")
    
    # Load data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} compounds")
    
    if output_file is None:
        output_file = input_file
    
    # Find compounds needing prediction
    mask = df['boltz_affinity'].isna() if 'boltz_affinity' in df.columns else pd.Series([True]*len(df))
    to_process = df[mask].copy()
    print(f"Compounds needing prediction: {len(to_process)}")
    
    # Apply range
    if end_idx is None:
        end_idx = len(to_process)
    to_process = to_process.iloc[start_idx:end_idx]
    print(f"Processing range [{start_idx}:{end_idx}]: {len(to_process)} compounds")
    
    if len(to_process) == 0:
        print("Nothing to process!")
        return
    
    # Ensure columns exist
    if 'boltz_affinity' not in df.columns:
        df['boltz_affinity'] = np.nan
    if 'boltz_confidence' not in df.columns:
        df['boltz_confidence'] = np.nan
    
    # Progress tracking
    results = []
    success_count = 0
    fail_count = 0
    start_time = time.time()
    
    pbar = tqdm(to_process.iterrows(), total=len(to_process), desc="Boltz-2")
    
    for i, (idx, row) in enumerate(pbar):
        smiles = row['smiles']
        compound_id = f"cpd_{idx}"
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                yaml_path = os.path.join(tmpdir, f"{compound_id}.yaml")
                create_yaml(smiles, compound_id, yaml_path)
                
                pred_result = run_boltz(yaml_path, tmpdir, BOLTZ_EXE)
                
                if pred_result["success"]:
                    output_data = parse_output(tmpdir, compound_id)
                    success_count += 1
                else:
                    output_data = {"affinity_pred_value": np.nan, "confidence_score": np.nan}
                    fail_count += 1
        except Exception as e:
            output_data = {"affinity_pred_value": np.nan, "confidence_score": np.nan}
            fail_count += 1
        
        # Update dataframe
        if pd.notna(output_data.get("affinity_pred_value")):
            df.at[idx, 'boltz_affinity'] = output_data["affinity_pred_value"]
            df.at[idx, 'boltz_confidence'] = output_data.get("confidence_score")
        
        results.append({
            "df_index": idx,
            "smiles": smiles,
            **output_data
        })
        
        # Update progress bar
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed * 60
        pbar.set_postfix({
            "rate": f"{rate:.1f}/min",
            "ok": success_count,
            "fail": fail_count
        })
        
        # Save periodically
        if (i + 1) % save_every == 0:
            df.to_csv(output_file, index=False)
            pd.DataFrame(results).to_csv(f"boltz_progress_{start_idx}_{end_idx}.csv", index=False)
    
    # Final save
    df.to_csv(output_file, index=False)
    pd.DataFrame(results).to_csv(f"boltz_progress_{start_idx}_{end_idx}.csv", index=False)
    
    # Summary
    print(f"\n{'='*50}")
    print("COMPLETE")
    print(f"{'='*50}")
    print(f"Processed: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Total coverage: {df['boltz_affinity'].notna().sum()}/{len(df)}")


def main():
    parser = argparse.ArgumentParser(description="Compute Boltz-2 affinity for VDR compounds")
    parser.add_argument("--input", "-i", required=True, help="Input CSV with smiles column")
    parser.add_argument("--output", "-o", help="Output CSV (default: update input)")
    parser.add_argument("--start", type=int, default=0, help="Start index in missing compounds")
    parser.add_argument("--end", type=int, help="End index (default: all)")
    parser.add_argument("--save-every", type=int, default=20, help="Save progress every N compounds")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Boltz-2 Affinity Prediction for VDR")
    print("="*60)
    
    process_compounds(
        input_file=args.input,
        output_file=args.output,
        start_idx=args.start,
        end_idx=args.end,
        save_every=args.save_every
    )


if __name__ == "__main__":
    main()
