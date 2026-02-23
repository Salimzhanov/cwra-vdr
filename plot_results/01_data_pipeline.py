#!/usr/bin/env python3
"""
Pipeline Steps:
  1. Load composed modalities + CWRA meta_scores
  2. Optionally exclude newRef_137 (legacy behavior)
  3. Structural classification via RDKit SMARTS
  4. Apply pre-filter (MW > 600 Da or RotB > 15) on generated compounds
  5. Define analysis groups (Reference / Generated / Top 100 / G1-G3)
  6. Compute enrichment statistics (Fisher's exact test)
  7. Compute physicochemical property statistics
  8. Compute nearest-neighbor Tanimoto similarity to reference set
  9. Ring count analysis and per-generator structural features
  10. Export data for visualization (JSON + CSV)

Input files:
  - composed_modalities_with_rdkit.csv  (16,196 compounds × RDKit descriptors)
  - final_selected.csv                   (CWRA meta_scores)

Output files:
  - report_data.json         Full analysis results for report generation
  - report_data_slim.json    Subsampled distributions for Plotly embedding
  - top100_compounds.csv     Top 100 CWRA-selected compounds
  - active_classified.csv    Active pool with structural classifications

Requirements:
  pip install pandas numpy scipy rdkit-pypi
"""

import os
import sys
import json
import warnings
import argparse
import numpy as np
import pandas as pd
from scipy import stats

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# Input paths (adjust to your directory structure)
INPUT_COMPOSED = 'data/composed_modalities_with_rdkit.csv'
INPUT_SELECTED = 'data/final_selected.csv'

# Output paths
OUTPUT_DIR     = 'output'
REPORT_JSON    = os.path.join(OUTPUT_DIR, 'report_data.json')
REPORT_SLIM    = os.path.join(OUTPUT_DIR, 'report_data_slim.json')
TOP100_CSV     = os.path.join(OUTPUT_DIR, 'top100_compounds.csv')
ACTIVE_CSV     = os.path.join(OUTPUT_DIR, 'active_classified.csv')

# Pre-filter thresholds for generated compounds
MW_THRESHOLD   = 600   # Da
ROTB_THRESHOLD = 15

# Morgan fingerprint parameters for Tanimoto similarity
FP_RADIUS = 2
FP_NBITS  = 2048

# NN-Tanimoto sampling (for computational efficiency)
NN_GEN_SAMPLE_SIZE = 2000
NN_PF_SAMPLE_SIZE  = 500
RANDOM_SEED = 42
BASE_REFERENCE_SOURCES = ['initial_370', 'calcitriol']
NEWREF_SOURCE = 'newRef_137'

try:
    MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
        radius=FP_RADIUS, fpSize=FP_NBITS)
except Exception:
    MORGAN_GENERATOR = None

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# STRUCTURAL CLASSIFICATION PATTERNS
# ═══════════════════════════════════════════════════════════════════════

# VDR A-ring: 1,3-dihydroxycyclohex-5-ene with exocyclic methylene
A_RING_PATTERNS = [
    Chem.MolFromSmarts('[OH]C1CC(=C)CC(O)C1'),   # canonical VDR A-ring
    Chem.MolFromSmarts('[OH]C1CC(O)CC(=C)C1'),    # regioisomer
    Chem.MolFromSmarts('OC1CC(O)C(=C)CC1'),       # variant
    Chem.MolFromSmarts('OC1CC(=C)CC(O)C1'),       # variant
]

# Secosteroidal: broken B-ring steroid (6,7-seco triene system)
SECO_PATTERNS = [
    Chem.MolFromSmarts('C/C=C\\C'),   # triene Z-configuration
    Chem.MolFromSmarts('C/C=C/C'),    # triene E-configuration
    Chem.MolFromSmarts('C=CC=CC'),    # conjugated diene
]

# C2-modification: substituent at C2 position of VDR A-ring
# (confers CYP24A1 metabolic stability)
C2_PATTERNS = [
    Chem.MolFromSmarts('[OH]C1C([!H])C(=C)CC(O)C1'),
    Chem.MolFromSmarts('[OH]C1C([!H])C(O)CC(=C)C1'),
    Chem.MolFromSmarts('OC1C([CH2,CH,C])C(=C)CC(O)C1'),
    Chem.MolFromSmarts('OC1C([CH2,CH,C])C(O)CC(=C)C1'),
]

# Five-membered oxygen heterocycle (furan, THF, etc.)
O5_PATTERN = Chem.MolFromSmarts('[O]1~[C]~[C]~[C]~[C]1')


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def classify_compound(smi: str) -> dict:
    """Classify a single compound by VDR-relevant structural features.
    
    Returns dict with boolean flags for:
      - has_a_ring:        VDR A-ring pharmacophore
      - is_secosteroidal:  secosteroidal (broken B-ring) scaffold
      - c2_modified:       C2-substituted A-ring (CYP24A1 stability)
      - pentacyclic:       5+ fused rings (developability concern)
      - has_o5ring:        5-membered O-heterocycle
      - sidechain_cycle:   cyclic side-chain modification
      - n_rings:           total ring count
    """
    result = {
        'has_a_ring': False,
        'is_secosteroidal': False,
        'c2_modified': False,
        'pentacyclic': False,
        'has_o5ring': False,
        'sidechain_cycle': False,
        'n_rings': 0,
    }
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return result

        # A-ring detection
        for pat in A_RING_PATTERNS:
            if pat and mol.HasSubstructMatch(pat):
                result['has_a_ring'] = True
                break

        # Secosteroidal: has A-ring OR has seco-steroid triene + ≥2 OH
        if result['has_a_ring']:
            result['is_secosteroidal'] = True
        else:
            for pat in SECO_PATTERNS:
                if pat and mol.HasSubstructMatch(pat):
                    oh_count = len(mol.GetSubstructMatches(
                        Chem.MolFromSmarts('[OH]')))
                    if oh_count >= 2:
                        result['is_secosteroidal'] = True
                        break

        # C2-modification
        for pat in C2_PATTERNS:
            if pat and mol.HasSubstructMatch(pat):
                result['c2_modified'] = True
                break

        # Ring analysis
        ri = mol.GetRingInfo()
        n_rings = ri.NumRings()
        result['n_rings'] = n_rings
        result['pentacyclic'] = n_rings >= 5

        # O-heterocycle
        if O5_PATTERN and mol.HasSubstructMatch(O5_PATTERN):
            result['has_o5ring'] = True

        # Side-chain cycles
        try:
            core = MurckoScaffold.GetScaffoldForMol(mol)
            core_rings = core.GetRingInfo().NumRings()
            if n_rings > core_rings:
                result['sidechain_cycle'] = True
            elif n_rings >= 2:
                ring_sizes = [len(r) for r in ri.AtomRings()]
                if any(s <= 6 for s in ring_sizes):
                    result['sidechain_cycle'] = True
        except Exception:
            pass

        return result
    except Exception:
        return result


def get_morgan_fp(smi: str):
    """Compute Morgan fingerprint (ECFP4-like) as bit vector."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        if MORGAN_GENERATOR is not None:
            return MORGAN_GENERATOR.GetFingerprint(mol)
        return AllChem.GetMorganFingerprintAsBitVect(
            mol, FP_RADIUS, nBits=FP_NBITS)
    except Exception:
        return None


def compute_nn_tanimoto(smiles_series, ref_fps, ref_indices,
                        exclude_self_indices=None):
    """Compute nearest-neighbor Tanimoto similarity to reference set.
    
    For each compound, finds the maximum Tanimoto similarity to any
    reference compound. For reference compounds, self-identity is
    excluded to measure internal diversity.
    
    Args:
        smiles_series: pd.Series of SMILES strings
        ref_fps: list of reference Morgan fingerprints
        ref_indices: list of reference DataFrame indices
        exclude_self_indices: set of indices to exclude (self-identity)
    
    Returns:
        list of float: NN-Tanimoto values
    """
    results = []
    for idx, smi in smiles_series.items():
        fp = get_morgan_fp(smi)
        if fp is None:
            results.append(np.nan)
            continue
        sims = []
        for ri, rfp in zip(ref_indices, ref_fps):
            if exclude_self_indices and idx == ri:
                continue
            sims.append(DataStructs.TanimotoSimilarity(fp, rfp))
        results.append(max(sims) if sims else np.nan)
    return results


def fisher_exact_test(g1_df, g2_df, cat_col):
    """Two-sided Fisher's exact test for categorical enrichment.
    
    Tests whether `cat_col` is enriched in g1_df vs g2_df.
    
    Returns:
        dict with odds ratio, p-value, significance stars, direction
    """
    a = int(g1_df[cat_col].sum())
    b = len(g1_df) - a
    c = int(g2_df[cat_col].sum())
    d = len(g2_df) - c
    table = [[a, b], [c, d]]
    odds, p = stats.fisher_exact(table)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    direction = '↑ enriched' if odds > 1 else '↓ depleted' if odds < 1 else '—'
    return {
        'odds': round(odds, 3),
        'p': float(p),
        'sig': sig,
        'dir': direction,
    }


def lipinski_pass(row) -> bool:
    """Check Lipinski Rule of Five compliance."""
    try:
        return (row['MW'] <= 500 and row['cLogP'] <= 5 and
                row['HBD'] <= 5 and row['HBA'] <= 10)
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build CWRA structural analysis report data.")
    parser.add_argument(
        '--input-composed',
        default=INPUT_COMPOSED,
        help='Path to composed modalities CSV.'
    )
    parser.add_argument(
        '--input-selected',
        default=INPUT_SELECTED,
        help='Path to final_selected.csv used for CWRA meta_score merge.'
    )
    parser.add_argument(
        '--output-dir',
        default=OUTPUT_DIR,
        help='Directory for report_data outputs (JSON + CSV files).'
    )
    parser.add_argument(
        '--exclude-newref-137',
        action='store_true',
        help='Use legacy behavior: exclude newRef_137 from reference actives.'
    )
    return parser.parse_args()


def main():
    global INPUT_COMPOSED, INPUT_SELECTED, OUTPUT_DIR
    global REPORT_JSON, REPORT_SLIM, TOP100_CSV, ACTIVE_CSV
    args = parse_args()
    INPUT_COMPOSED = args.input_composed
    INPUT_SELECTED = args.input_selected
    OUTPUT_DIR = args.output_dir
    REPORT_JSON = os.path.join(OUTPUT_DIR, 'report_data.json')
    REPORT_SLIM = os.path.join(OUTPUT_DIR, 'report_data_slim.json')
    TOP100_CSV = os.path.join(OUTPUT_DIR, 'top100_compounds.csv')
    ACTIVE_CSV = os.path.join(OUTPUT_DIR, 'active_classified.csv')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    include_newref_137 = not args.exclude_newref_137
    reference_sources = list(BASE_REFERENCE_SOURCES)
    if include_newref_137:
        reference_sources.append(NEWREF_SOURCE)

    print("="* 70)
    print("VDR CWRA STRUCTURAL ANALYSIS")
    print("="* 70)

    # ------------------------------------------------------------------
    # STEP 1: Load data
    # ------------------------------------------------------------------
    print("\n[1/10] Loading data...")
    comp = pd.read_csv(INPUT_COMPOSED)
    sel = pd.read_csv(INPUT_SELECTED)
    print(f"Composed modalities: {len(comp):,} compounds")
    print(f"Final selected (CWRA): {len(sel):,} compounds")

    # ------------------------------------------------------------------
    # STEP 2: Optional newRef_137 exclusion (legacy mode only)
    # ------------------------------------------------------------------
    n_before = len(comp)
    if include_newref_137:
        print("\n[2/10] Keeping newRef_137 as reference actives...")
        n_excluded = 0
        print("Excluded: 0 compounds")
    else:
        print("\n[2/10] Excluding newRef_137 (legacy mode)...")
        comp = comp[comp['source'] != NEWREF_SOURCE].copy()
        n_excluded = n_before - len(comp)
        print(f"Excluded: {n_excluded} compounds ({NEWREF_SOURCE})")
    print(f"Remaining: {len(comp):,}")

    # Merge CWRA meta_score via SMILES
    sel_scores = (sel[['smiles', 'meta_score']]
                  .sort_values('meta_score', ascending=False)
                  .drop_duplicates('smiles', keep='first'))
    df = comp.merge(sel_scores, on='smiles', how='left')
    print(f"meta_score available: {df['meta_score'].notna().sum():,}")

    # ------------------------------------------------------------------
    # STEP 3: Structural classification
    # ------------------------------------------------------------------
    print("\n[3/10] Structural classification (RDKit SMARTS)...")

    # Compute SAScore if missing
    if 'SAScore' not in df.columns:
        from rdkit.Chem import RDConfig
        sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
        import sascorer
        df['SAScore'] = df['smiles'].apply(
            lambda s: sascorer.calculateScore(Chem.MolFromSmiles(s))
            if Chem.MolFromSmiles(s) else np.nan)

    results = df['smiles'].apply(classify_compound)
    res_df = pd.DataFrame(results.tolist())
    for col in res_df.columns:
        df[col] = res_df[col].values
    df['non_steroidal'] = ~df['is_secosteroidal']

    print(f"Secosteroidal:    {df['is_secosteroidal'].sum():>6,} "
          f"({df['is_secosteroidal'].mean()*100:.1f}%)")
    print(f"Non-steroidal:    {df['non_steroidal'].sum():>6,} "
          f"({df['non_steroidal'].mean()*100:.1f}%)")
    print(f"C2-modified:      {df['c2_modified'].sum():>6,} "
          f"({df['c2_modified'].mean()*100:.1f}%)")
    print(f"Pentacyclic:      {df['pentacyclic'].sum():>6,} "
          f"({df['pentacyclic'].mean()*100:.1f}%)")
    print(f"Furan/O-het:      {df['has_o5ring'].sum():>6,} "
          f"({df['has_o5ring'].mean()*100:.1f}%)")
    print(f"Side-chain cyc:   {df['sidechain_cycle'].sum():>6,} "
          f"({df['sidechain_cycle'].mean()*100:.1f}%)")

    # ------------------------------------------------------------------
    # STEP 4: Apply pre-filter on generated compounds
    # ------------------------------------------------------------------
    print("\n[4/10] Applying pre-filter (MW > 600 or RotB > 15)...")
    ref_mask = df['source'].isin(reference_sources)
    gen_mask = df['source'].isin(['G1', 'G2', 'G3'])

    pf_mask = gen_mask & ((df['MW'] > MW_THRESHOLD) |
                          (df['RotB'] > ROTB_THRESHOLD))
    n_prefiltered = pf_mask.sum()

    active_mask = ref_mask | (gen_mask & ~pf_mask)
    df_active = df[active_mask].copy()
    df_prefiltered = df[pf_mask].copy()

    n_total_before = ref_mask.sum() + gen_mask.sum()
    n_active = active_mask.sum()
    n_ref = ref_mask.sum()
    n_gen_active = (gen_mask & ~pf_mask).sum()

    print(f"Pre-filtered: {n_prefiltered:,} compounds")
    print(f"Active pool: {n_active:,} (Ref: {n_ref:,}, Gen: {n_gen_active:,})")

    # ------------------------------------------------------------------
    # STEP 5: Define analysis groups
    # ------------------------------------------------------------------
    print("\n[5/10] Defining analysis groups...")
    ref_active = df_active['source'].isin(reference_sources)
    gen_active = df_active['source'].isin(['G1', 'G2', 'G3'])
    g1_active = df_active['source'] == 'G1'
    g2_active = df_active['source'] == 'G2'
    g3_active = df_active['source'] == 'G3'

    # Top 100: highest meta_score from generated compounds
    df_gen_scored = df_active[gen_active].dropna(subset=['meta_score'])
    top100_idx = df_gen_scored.nlargest(100, 'meta_score').index
    top100_mask = df_active.index.isin(top100_idx)
    top100 = df_active[top100_mask]

    print(f"Reference: {ref_active.sum():,}")
    print(f"All Generated: {gen_active.sum():,}")
    print(f"Top 100: {top100_mask.sum()}")
    print(f"G1: {g1_active.sum():,}  G2: {g2_active.sum():,}  "
          f"G3: {g3_active.sum():,}")
    print(f"Top 100 source: {top100['source'].value_counts().to_dict()}")
    print(f"Top 100 generators: "
          f"{top100['generator'].value_counts().to_dict()}")

    groups = {
        'Reference':     df_active[ref_active],
        'All Generated': df_active[gen_active],
        'Top 100':       top100,
        'G1':            df_active[g1_active],
        'G2':            df_active[g2_active],
        'G3':            df_active[g3_active],
        'Pre-filtered':  df_prefiltered,
    }

    # ------------------------------------------------------------------
    # STEP 6: Enrichment statistics (Fisher's exact test)
    # ------------------------------------------------------------------
    print("\n[6/10] Enrichment statistics...")

    cats = ['is_secosteroidal', 'non_steroidal', 'c2_modified',
            'pentacyclic', 'has_o5ring', 'sidechain_cycle']
    cat_labels = ['Secosteroidal', 'Non-steroidal', 'C2-Modified',
                  'Pentacyclic', 'Furan/O-heterocycle', 'Side-chain Cycles']

    # Percentages and counts per group
    pct = {}
    cnt = {}
    for gname, gdf in groups.items():
        n = len(gdf)
        pct[gname] = {}
        cnt[gname] = {'n': n}
        for cat in cats:
            c = int(gdf[cat].sum())
            cnt[gname][cat] = c
            pct[gname][cat] = round(c / n * 100, 1) if n > 0 else 0

    print("\n  Enrichment progression (Ref → Gen → Top 100):")
    for cat, label in zip(cats, cat_labels):
        print(f" {label:25s}: {pct['Reference'][cat]:5.1f}% → "
              f"{pct['All Generated'][cat]:5.1f}% → "
              f"{pct['Top 100'][cat]:5.1f}%")

    # Fisher's exact tests
    comparisons = [
        ('Top 100 vs All Generated', top100_mask, gen_active),
        ('All Generated vs Reference', gen_active, ref_active),
        ('Top 100 vs Reference',       top100_mask, ref_active),
        ('G2 vs G1',                    g2_active, g1_active),
        ('G3 vs G1',                    g3_active, g1_active),
    ]

    fisher_results = []
    for comp_name, mask1, mask2 in comparisons:
        for cat, label in zip(cats, cat_labels):
            r = fisher_exact_test(df_active[mask1], df_active[mask2], cat)
            fisher_results.append({
                'comp': comp_name, 'cat': label, **r})

    print("\n  Significant Fisher tests:")
    for fr in fisher_results:
        if fr['sig'] != 'ns':
            print(f" {fr['comp']:32s} {fr['cat']:25s} "
                  f"OR={fr['odds']:7.2f}  p={fr['p']:.2e}  {fr['sig']}")

    # ------------------------------------------------------------------
    # STEP 7: Property statistics
    # ------------------------------------------------------------------
    print("\n[7/10] Property statistics...")

    prop_keys = ['vina_score', 'mltle_pKd', 'QED', 'SAScore', 'MW',
                 'cLogP', 'tPSA', 'FractionCSP3', 'HBD', 'HBA', 'RotB']
    prop_labels = ['Vina (kcal/mol)', 'ML-pKd', 'QED', 'SA Score',
                   'MW (Da)', 'cLogP', 'tPSA (Å²)', 'Fsp³',
                   'HBD', 'HBA', 'RotBonds']

    prop_stats = {}
    for gname in ['Reference', 'All Generated', 'Top 100']:
        gdf = groups[gname]
        prop_stats[gname] = {}
        for pk in prop_keys:
            vals = gdf[pk].dropna()
            prop_stats[gname][pk] = {
                'mean': round(float(vals.mean()), 2),
                'std':  round(float(vals.std()), 2),
                'median': round(float(vals.median()), 2),
            }

    for pk, pl in zip(prop_keys[:6], prop_labels[:6]):
        r = prop_stats['Reference'][pk]
        g = prop_stats['All Generated'][pk]
        t = prop_stats['Top 100'][pk]
        print(f"{pl:25s}: Ref {r['mean']:8.2f} → Gen {g['mean']:8.2f} "
              f"→ Top100 {t['mean']:8.2f}")

    # ------------------------------------------------------------------
    # STEP 8: Nearest-neighbor Tanimoto similarity (corrected)
    # ------------------------------------------------------------------
    print("\n[8/10] Nearest-neighbor Tanimoto similarity...")

    # Build reference fingerprint library
    ref_df = df_active[ref_active].copy()
    ref_fps = []
    ref_indices = []
    for idx, smi in ref_df['smiles'].items():
        fp = get_morgan_fp(smi)
        if fp is not None:
            ref_fps.append(fp)
            ref_indices.append(idx)
    print(f"Reference fingerprints: {len(ref_fps)}")

    # Reference self-NN (excluding self-identity)
    print("Computing Reference self-NN (excl. self)...")
    ref_nn = compute_nn_tanimoto(
        ref_df['smiles'], ref_fps, ref_indices,
        exclude_self_indices=set(ref_indices))

    # Generated NN (sampled for efficiency)
    gen_df = df_active[gen_active]
    np.random.seed(RANDOM_SEED)
    gen_sample = (gen_df.sample(NN_GEN_SAMPLE_SIZE, random_state=RANDOM_SEED)
                  if len(gen_df) > NN_GEN_SAMPLE_SIZE else gen_df)
    print(f"Computing Generated NN ({len(gen_sample)} samples)...")
    gen_nn = compute_nn_tanimoto(gen_sample['smiles'], ref_fps, ref_indices)

    # Top 100 NN
    print("Computing Top 100 NN...")
    top100_nn = compute_nn_tanimoto(top100['smiles'], ref_fps, ref_indices)

    # Pre-filtered NN (sampled)
    pf_sample = (df_prefiltered.sample(NN_PF_SAMPLE_SIZE,
                                       random_state=RANDOM_SEED)
                 if len(df_prefiltered) > NN_PF_SAMPLE_SIZE
                 else df_prefiltered)
    print(f"Computing Pre-filtered NN ({len(pf_sample)} samples)...")
    pf_nn = compute_nn_tanimoto(pf_sample['smiles'], ref_fps, ref_indices)

    nn_stats = {}
    for name, values in [('Reference', ref_nn), ('All Generated', gen_nn),
                         ('Top 100', top100_nn), ('Pre-filtered', pf_nn)]:
        clean = [x for x in values if not np.isnan(x)]
        nn_stats[name] = {
            'values': clean,
            'mean':   round(float(np.mean(clean)), 3),
            'std':    round(float(np.std(clean)), 3),
            'median': round(float(np.median(clean)), 3),
        }
        print(f"{name:20s}: mean={nn_stats[name]['mean']:.3f} "
              f"± {nn_stats[name]['std']:.3f}  "
              f"median={nn_stats[name]['median']:.3f}")

    # ------------------------------------------------------------------
    # STEP 9: Ring count & per-generator analysis
    # ------------------------------------------------------------------
    print("\n[9/10] Ring count & per-generator analysis...")

    # Ring count distribution (% per group)
    ring_dist = {}
    for gname in ['Reference', 'All Generated', 'Top 100', 'Pre-filtered']:
        gdf = groups[gname]
        rc = gdf['n_rings'].clip(0, 6)
        ring_dist[gname] = {}
        for r in range(7):
            label = str(r) if r < 6 else '6+'
            ring_dist[gname][label] = (
                round(float((rc == r).sum() / len(gdf) * 100), 1)
                if len(gdf) > 0 else 0)

    # Generator combination analysis for Top 100
    gen_combo_top100 = top100['generator'].value_counts().to_dict()
    source_top100 = top100['source'].value_counts().to_dict()

    # Per-generator structural features
    gen_types = df_active[gen_active]['generator'].unique()
    gen_struct = {}
    for gt in sorted(gen_types):
        gdf = df_active[df_active['generator'] == gt]
        n = len(gdf)
        if n == 0:
            continue
        gen_struct[gt] = {
            'n': n,
            'secosteroidal_pct': round(float(gdf['is_secosteroidal'].sum() / n * 100), 1),
            'c2_pct':            round(float(gdf['c2_modified'].sum() / n * 100), 1),
            'pentacyclic_pct':   round(float(gdf['pentacyclic'].sum() / n * 100), 1),
            'o5ring_pct':        round(float(gdf['has_o5ring'].sum() / n * 100), 1),
            'sidechain_pct':     round(float(gdf['sidechain_cycle'].sum() / n * 100), 1),
        }
        print(f"{gt:35s} (n={n:5d}): "
              f"seco={gen_struct[gt]['secosteroidal_pct']:5.1f}%  "
              f"c2={gen_struct[gt]['c2_pct']:5.1f}%")

    # Lipinski pass rates
    print("\n  Lipinski Ro5 pass rates:")
    for gname in ['Reference', 'All Generated', 'Top 100']:
        lip = groups[gname].apply(lipinski_pass, axis=1).mean() * 100
        print(f" {gname:20s}: {lip:.1f}%")

    # ------------------------------------------------------------------
    # STEP 10: Export data
    # ------------------------------------------------------------------
    print("\n[10/10] Exporting data...")

    # Distribution data for plotting
    plot_data = {}
    for gname in ['Reference', 'All Generated', 'Top 100']:
        gdf = groups[gname]
        vina = gdf['vina_score'].dropna()
        plot_data[gname] = {
            'vina':       vina[(vina >= -15) & (vina <= -3)].tolist(),
            'pkd':        gdf['mltle_pKd'].dropna().tolist(),
            'qed':        gdf['QED'].dropna().tolist(),
            'mw':         gdf['MW'].dropna().clip(upper=800).tolist(),
            'clogp':      gdf['cLogP'].dropna().tolist(),
            'tpsa':       gdf['tPSA'].dropna().clip(0, 250).tolist(),
            'fsp3':       gdf['FractionCSP3'].dropna().tolist(),
            'meta_score': gdf['meta_score'].dropna().tolist(),
        }

    plot_data['nn_tanimoto'] = {k: v['values'] for k, v in nn_stats.items()}
    for gname in ['Reference', 'All Generated', 'Top 100', 'Pre-filtered']:
        plot_data[f'{gname}_rings'] = groups[gname]['n_rings'].clip(0, 7).tolist()

    # Full report data
    report_data = {
        'include_newref_137_as_active': bool(include_newref_137),
        'reference_sources': reference_sources,
        'n_total_before':    int(n_total_before),
        'n_active':          int(n_active),
        'n_ref':             int(ref_active.sum()),
        'n_gen':             int(gen_active.sum()),
        'n_top100':          100,
        'n_prefiltered':     int(n_prefiltered),
        'n_g1':              int(g1_active.sum()),
        'n_g2':              int(g2_active.sum()),
        'n_g3':              int(g3_active.sum()),
        'pct':               pct,
        'cnt':               cnt,
        'fisher':            fisher_results,
        'prop_stats':        prop_stats,
        'nn_tanimoto':       {k: {'mean': v['mean'], 'std': v['std'],
                                   'median': v['median']}
                              for k, v in nn_stats.items()},
        'ring_dist':         ring_dist,
        'gen_combo_top100':  gen_combo_top100,
        'source_top100':     source_top100,
        'gen_struct':        gen_struct,
        'plot_data':         plot_data,
    }

    with open(REPORT_JSON, 'w') as f:
        json.dump(report_data, f, default=str)
    print(f"{REPORT_JSON} "
          f"({os.path.getsize(REPORT_JSON)/1024:.0f} KB)")

    # Slim version: subsample large distributions for Plotly embedding
    import random
    random.seed(RANDOM_SEED)
    slim = json.loads(json.dumps(report_data, default=str))
    pd_slim = slim['plot_data']
    for key in list(pd_slim.keys()):
        if isinstance(pd_slim[key], list) and len(pd_slim[key]) > 2000:
            pd_slim[key] = random.sample(pd_slim[key], 2000)
        elif isinstance(pd_slim[key], dict):
            for sk in pd_slim[key]:
                if isinstance(pd_slim[key][sk], list) and len(pd_slim[key][sk]) > 2000:
                    pd_slim[key][sk] = random.sample(pd_slim[key][sk], 2000)

    with open(REPORT_SLIM, 'w') as f:
        json.dump(slim, f, default=str)
    print(f"{REPORT_SLIM} "
          f"({os.path.getsize(REPORT_SLIM)/1024:.0f} KB)")

    # CSV exports
    top100.to_csv(TOP100_CSV, index=False)
    print(f"{TOP100_CSV} ({len(top100)} rows)")

    df_active.to_csv(ACTIVE_CSV, index=False)
    print(f"{ACTIVE_CSV} ({len(df_active):,} rows)")

    print(f"\n{'='*70}")
    print("Pipeline complete.")
    print(f"Active pool: {n_active:,} (Ref: {n_ref}, Gen: {n_gen_active:,})")
    print(f"Pre-filtered: {n_prefiltered:,}")
    print(f"Top 100: G1={source_top100.get('G1',0)}, "
          f"G2={source_top100.get('G2',0)}, G3={source_top100.get('G3',0)}")
    print(f"NN-Tanimoto: Ref={nn_stats['Reference']['mean']:.3f}, "
          f"Gen={nn_stats['All Generated']['mean']:.3f}, "
          f"Top100={nn_stats['Top 100']['mean']:.3f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
