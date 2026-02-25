#!/usr/bin/env python3
"""
  Run 01_data_pipeline.py first to produce:
    - output/report_data.json
    - output/active_classified.csv
    - output/top100_compounds.csv

Output:
  output/figures/Fig01–Fig18 (25 individual PDF files)

Figure Inventory:
  Fig01  Enrichment heatmap (7 groups × 6 structural categories)
  Fig02  Structural modification grouped bars (Ref / Gen / Top 100)
  Fig03  Two-stage selection dynamics (Ref → Generation → CWRA)
  Fig04a Forest plot: Top 100 vs All Generated (odds ratios)
  Fig04b Forest plot: All Generated vs Reference
  Fig04c Forest plot: Top 100 vs Reference
  Fig04d Forest plot: G2 vs G1 (multi-generator consensus)
  Fig05  CWRA meta_score distributions
  Fig06  Per-generator structural features (horizontal bars)
  Fig07  Secosteroidal vs non-steroidal composition (stacked)
  Fig08a Vina docking score box plots
  Fig08b ML-pKd box plots
  Fig09  Normalized property radar (6 axes)
  Fig10a QED distribution histograms
  Fig10b Molecular weight distribution histograms
  Fig11  Ring count distribution (grouped bars)
  Fig12a Nearest-neighbor Tanimoto bar chart (mean ± std)
  Fig12b Nearest-neighbor Tanimoto violin + box plots
  Fig13a Top 100 generator combination donut chart
  Fig13b Top 100 generation source (G1/G2/G3) donut chart
  Fig14  Top 100 structural features comparison
  Fig15  Combined binding property box plots (Vina + pKd)
  Fig16  Combined QED + MW histograms
  Fig17  Combined cLogP + Fsp³ histograms
  Fig18  Combined NN-Tanimoto (bar + violin)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore', category=FutureWarning)


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# Input paths (from 01_data_pipeline.py)
REPORT_JSON = 'output/report_data.json'
ACTIVE_CSV  = 'output/active_classified.csv'
TOP100_CSV  = 'output/top100_compounds.csv'

# Output
FIG_DIR = 'output/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# ─── Publication rcParams ─────────────────────────────────────────────
plt.rcParams.update({
    # Font
    'font.family':          'sans-serif',
    'font.sans-serif':      ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size':            9,
    # Axes
    'axes.titlesize':       11,
    'axes.titleweight':     'bold',
    'axes.labelsize':       10,
    'axes.labelweight':     'medium',
    'axes.linewidth':       0.8,
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    # Ticks
    'xtick.labelsize':      8.5,
    'ytick.labelsize':      8.5,
    'xtick.major.width':    0.6,
    'ytick.major.width':    0.6,
    'xtick.major.size':     3.5,
    'ytick.major.size':     3.5,
    # Legend
    'legend.fontsize':      8,
    'legend.frameon':       True,
    'legend.framealpha':    0.9,
    'legend.edgecolor':     '#cccccc',
    'legend.handlelength':  1.2,
    # Export
    'figure.dpi':           300,
    'savefig.dpi':          300,
    'savefig.bbox':         'tight',
    'savefig.pad_inches':   0.08,
    'pdf.fonttype':         42,     # TrueType (editable in Illustrator)
    'ps.fonttype':          42,
})

# ─── Color Palette (colorblind-friendly, Nature-style) ────────────────
PAL = {
    'ref':   '#4878A6',     # steel blue — reference compounds
    'gen':   '#73A95C',     # sage green — all generated
    'top':   '#D4654A',     # terracotta — CWRA top 100
    'pf':    '#9B89B3',     # lavender — pre-filtered
    'G1':    '#8FBC8F',     # dark sea green
    'G2':    '#5F9EA0',     # cadet blue
    'G3':    '#D4654A',     # terracotta
    'seco':  '#4878A6',
    'c2':    '#D4654A',
    'penta': '#9B89B3',
    'furan': '#E5A832',     # amber
}

# Per-generator model colors (viridis-inspired)
GEN_PAL = {
    'gmdldr':                    '#4878A6',
    'reinvent':                  '#73A95C',
    'transmol':                  '#E5A832',
    'gmdldr_reinvent':           '#3B7A7A',
    'reinvent_transmol':         '#5F9EA0',
    'gmdldr_transmol':           '#9B89B3',
    'transmol-reinvent-gmdldr':  '#D4654A',
    'reference':                 '#888888',
    'calcitriol':                '#888888',
}

# Structural categories
CATS       = ['is_secosteroidal', 'non_steroidal', 'c2_modified',
              'pentacyclic', 'has_o5ring', 'sidechain_cycle']
CAT_LABELS = ['Secosteroidal', 'Non-steroidal', 'C2-Modified',
              'Pentacyclic', 'Furan/O-heterocycle', 'Side-chain Cycles']
CAT_SHORT  = ['Seco', 'Non-ster', 'C2-Mod', 'Penta', 'Furan/O', 'SC-Cyc']


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def save_fig(fig, name):
    """Save figure as vector PDF and close."""
    path = os.path.join(FIG_DIR, f'{name}.pdf')
    fig.savefig(path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    sz = os.path.getsize(path) / 1024
    print(f"{name}.pdf ({sz:.0f} KB)")


def sig_color(sig_str):
    """Map significance stars to color."""
    return {'***': '#D4654A', '**': '#E5A832',
            '*': '#5F9EA0'}.get(sig_str, '#999999')


def add_grid(ax, axis='y', alpha=0.3):
    """Add subtle grid lines behind data."""
    if axis in ('y', 'both'):
        ax.yaxis.grid(True, alpha=alpha, linewidth=0.5, zorder=0)
    if axis in ('x', 'both'):
        ax.xaxis.grid(True, alpha=alpha, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


# ═══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════

def load_data():
    """Load pipeline outputs."""
    print("Loading data...")
    with open(REPORT_JSON) as f:
        D = json.load(f)

    df = pd.read_csv(ACTIVE_CSV)
    t100 = pd.read_csv(TOP100_CSV)

    assert df[df['source'] == 'newRef_137'].shape[0] == 0, \
        "newRef_137 still present in active dataset!"
    print(f"newRef_137 excluded. Active pool: {len(df):,} compounds")

    return D, df, t100


# ═══════════════════════════════════════════════════════════════════════
# FIGURE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def fig01_enrichment_heatmap(D):
    """Enrichment heatmap: 7 groups × 6 structural categories."""
    pct, cnt = D['pct'], D['cnt']
    groups = ['Reference', 'All Generated', 'Top 100',
              'G1', 'G2', 'G3', 'Pre-filtered']
    z = np.array([[pct[g][c] for c in CATS] for g in groups])
    ylabels = [f"{g} (n={cnt[g]['n']:,})"for g in groups]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cmap = LinearSegmentedColormap.from_list(
        'custom', ['#f7fbff', '#6baed6', '#2171b5', '#D4654A'], N=256)
    im = ax.imshow(z, cmap=cmap, aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(range(len(CAT_SHORT)))
    ax.set_xticklabels(CAT_SHORT, fontsize=9)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8.5)
    ax.xaxis.set_ticks_position('bottom')

    for i in range(len(groups)):
        for j in range(len(CATS)):
            v = z[i, j]
            color = 'white' if v > 50 else 'black'
            ax.text(j, i, f'{v:.1f}%', ha='center', va='center',
                    fontsize=7.5, color=color, fontweight='medium')

    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label('Prevalence (%)', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_title('Structural Feature Prevalence by Compound Group',
                 fontsize=11, fontweight='bold', pad=10)
    fig.tight_layout()
    save_fig(fig, 'Fig01_enrichment_heatmap')


def fig02_structural_bars(D):
    """Grouped bar chart: key structural features across 3 groups."""
    pct = D['pct']
    idx = [0, 2, 3, 4]  # seco, c2, penta, furan
    labels = [CAT_LABELS[i] for i in idx]
    keys = [CATS[i] for i in idx]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(labels))
    w = 0.25
    grps = ['Reference', 'All Generated', 'Top 100']
    cols = [PAL['ref'], PAL['gen'], PAL['top']]
    legs = [f'Reference (n={D["n_ref"]})',
            f'All Generated (n={D["n_gen"]:,})', 'CWRA Top 100']

    for i, (g, c, lab) in enumerate(zip(grps, cols, legs)):
        vals = [pct[g][k] for k in keys]
        bars = ax.bar(x + (i - 1) * w, vals, w, color=c, label=lab,
                      edgecolor='white', linewidth=0.5, zorder=3)
        for bar, v in zip(bars, vals):
            if v > 2:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 1.2, f'{v:.1f}',
                        ha='center', va='bottom', fontsize=6.5,
                        fontweight='medium')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Prevalence (%)', fontsize=10)
    ax.set_ylim(0, 78)
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.92)
    add_grid(ax)
    ax.set_title('Structural Feature Prevalence: '
                 'Reference vs Generated vs CWRA Top 100',
                 fontsize=10, fontweight='bold', pad=10)
    fig.tight_layout()
    save_fig(fig, 'Fig02_structural_modification_bars')


def fig03_two_stage_selection(D):
    """Line chart: structural feature trajectory across selection stages."""
    pct = D['pct']
    stages = ['Reference', 'All Generated', 'Top 100']
    stage_labels = ['Reference\n(Training Set)',
                    'All Generated\n(Post-filter)',
                    'CWRA\nTop 100']
    feats = ['is_secosteroidal', 'c2_modified', 'pentacyclic', 'has_o5ring']
    names = ['Secosteroidal', 'C2-Modified', 'Pentacyclic', 'Furan/O-het']
    cols = [PAL['seco'], PAL['c2'], PAL['penta'], PAL['furan']]
    markers = ['o', 's', 'D', '^']

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for cat, lab, col, mk in zip(feats, names, cols, markers):
        vals = [pct[g][cat] for g in stages]
        ax.plot(range(3), vals, marker=mk, color=col, linewidth=2,
                markersize=7, label=lab, markeredgecolor='white',
                markeredgewidth=0.8, zorder=4)
        for xi, v in enumerate(vals):
            ax.annotate(f'{v:.1f}%', (xi, v), textcoords="offset points",
                        xytext=(0, 9), ha='center', fontsize=7, color=col,
                        fontweight='medium')

    ax.set_xticks(range(3))
    ax.set_xticklabels(stage_labels, fontsize=8.5)
    ax.set_ylabel('Prevalence (%)', fontsize=10)
    ax.set_ylim(-2, 72)
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.92)
    add_grid(ax)

    # Stage arrows
    for xpos, label in [(0.5, 'Generation'), (1.5, 'CWRA Selection')]:
        ax.annotate('', xy=(xpos, 2), xytext=(xpos, -1),
                    arrowprops=dict(arrowstyle='->', color='#999', lw=1))
        ax.text(xpos, -3, label, ha='center', fontsize=7,
                color='#999', style='italic')

    ax.set_title('Two-Stage Selection: '
                 'Reference → Generation → CWRA Recovery',
                 fontsize=10, fontweight='bold', pad=10)
    fig.tight_layout()
    save_fig(fig, 'Fig03_two_stage_selection')


def fig04_forest_plots(D):
    """Forest plots (odds ratio) for 4 group comparisons."""
    fisher = D['fisher']
    comparisons = [
        ('Top 100 vs All Generated', 'CWRA Selection Effect',     '04a'),
        ('All Generated vs Reference', 'Generative Model Drift',  '04b'),
        ('Top 100 vs Reference', 'Net CWRA vs Training Set',      '04c'),
        ('G2 vs G1', 'Multi-Generator Consensus Effect',          '04d'),
    ]

    sig_legend = [
        Patch(facecolor='#D4654A', alpha=0.85, label='p < 0.001 (***)'),
        Patch(facecolor='#E5A832', alpha=0.85, label='p < 0.01 (**)'),
        Patch(facecolor='#5F9EA0', alpha=0.85, label='p < 0.05 (*)'),
        Patch(facecolor='#999999', alpha=0.85, label='ns'),
    ]

    for comp, title, suffix in comparisons:
        fr = [f for f in fisher if f['comp'] == comp]
        if not fr:
            continue

        fig, ax = plt.subplots(figsize=(6, 3))
        for i, f_ in enumerate(fr):
            odds = min(f_['odds'], 30)
            col = sig_color(f_['sig'])
            ax.barh(i, np.log10(max(odds, 0.01)), color=col, height=0.6,
                    edgecolor='white', linewidth=0.5, zorder=3, alpha=0.85)
            or_str = (f"OR={f_['odds']:.2f}"if f_['odds'] < 100
                      else "OR=∞")
            p_str = (f"p={f_['p']:.1e}"if f_['p'] < 0.001
                     else f"p={f_['p']:.3f}")
            ax.text(max(np.log10(max(odds, 0.01)), 0) + 0.08, i,
                    f"{or_str}  {p_str}  {f_['sig']}",
                    va='center', fontsize=7, fontweight='medium',
                    color='#333')

        ax.axvline(x=0, color='#333', linewidth=0.8, linestyle='--',
                   zorder=2, alpha=0.7)
        ax.set_yticks(range(len(fr)))
        ax.set_yticklabels([f['cat'] for f in fr], fontsize=8.5)
        ax.set_xlabel('log₁₀(Odds Ratio)', fontsize=9)
        ax.set_xlim(-2.2, 2)
        ax.invert_yaxis()
        add_grid(ax, axis='x')
        ax.legend(handles=sig_legend, fontsize=6.5, loc='lower right',
                  framealpha=0.9)
        ax.set_title(f'{title}: {comp}', fontsize=10,
                     fontweight='bold', pad=8)
        ax.text(0.02, -0.4, '← Depleted', transform=ax.transAxes,
                fontsize=7, color='#666', style='italic')
        ax.text(0.75, -0.4, 'Enriched →', transform=ax.transAxes,
                fontsize=7, color='#666', style='italic')
        fig.tight_layout()
        save_fig(fig, f'Fig{suffix}_forest_plot_'
                      f'{comp.replace("", "_").lower()}')


def fig05_metascore(D):
    """CWRA meta_score distribution histograms."""
    pd_ = D['plot_data']
    fig, ax = plt.subplots(figsize=(6, 3.8))

    for g, col, lab, alpha in [
        ('All Generated', PAL['gen'],
         f'All Generated (n={D["n_gen"]:,})', 0.45),
        ('Reference', PAL['ref'],
         f'Reference (n={D["n_ref"]})', 0.55),
        ('Top 100', PAL['top'], 'CWRA Top 100', 0.75),
    ]:
        vals = pd_.get(g, {}).get('meta_score', [])
        if vals:
            ax.hist(vals, bins=50, alpha=alpha, color=col, label=lab,
                    edgecolor='white', linewidth=0.3,
                    zorder=3 if g == 'Top 100' else 2)

    ax.set_xlabel('CWRA meta_score', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_xlim(0.15, 0.9)
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.92)
    add_grid(ax)
    ax.set_title('CWRA meta_score Distribution by Group',
                 fontsize=10, fontweight='bold', pad=10)
    fig.tight_layout()
    save_fig(fig, 'Fig05_metascore_distributions')


def fig06_per_generator(D):
    """Horizontal grouped bars: structural features by generator."""
    gs = D['gen_struct']
    gen_order = sorted(
        [g for g in gs if g not in ('reference', 'calcitriol')],
        key=lambda x: gs[x]['n'], reverse=True)
    gen_labels = [f"{g}\n(n={gs[g]['n']:,})"for g in gen_order]

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(gen_order))
    h = 0.2
    feats = [
        ('Secosteroidal', 'secosteroidal_pct', PAL['seco']),
        ('C2-Modified',    'c2_pct',            PAL['c2']),
        ('Pentacyclic',    'pentacyclic_pct',   PAL['penta']),
        ('Furan/O-het',    'o5ring_pct',        PAL['furan']),
    ]

    for i, (lab, key, col) in enumerate(feats):
        vals = [gs[g][key] for g in gen_order]
        bars = ax.barh(y + (i - 1.5) * h, vals, h, color=col, label=lab,
                       edgecolor='white', linewidth=0.5, zorder=3)
        for bar, v in zip(bars, vals):
            if v > 1.5:
                ax.text(bar.get_width() + 0.5,
                        bar.get_y() + bar.get_height()/2,
                        f'{v:.1f}%', va='center', fontsize=6, color='#444')

    ax.set_yticks(y)
    ax.set_yticklabels(gen_labels, fontsize=7.5)
    ax.set_xlabel('Prevalence (%)', fontsize=10)
    ax.set_xlim(0, 82)
    ax.legend(fontsize=7.5, loc='lower right', ncol=2, framealpha=0.92)
    add_grid(ax, axis='x')
    ax.invert_yaxis()
    ax.set_title('Structural Features by Generator / Combination',
                 fontsize=10, fontweight='bold', pad=10)
    fig.tight_layout()
    save_fig(fig, 'Fig06_per_generator_features')

    return gen_order, gen_labels, y


def fig07_seco_stacked(D, gen_order, gen_labels, y):
    """Stacked horizontal bar: secosteroidal vs non-steroidal by generator."""
    gs = D['gen_struct']
    fig, ax = plt.subplots(figsize=(7, 4.5))
    seco = [gs[g]['secosteroidal_pct'] for g in gen_order]
    nonseco = [100 - v for v in seco]

    ax.barh(y, seco, color=PAL['seco'], label='Secosteroidal',
            edgecolor='white', linewidth=0.5, zorder=3)
    ax.barh(y, nonseco, left=seco, color=PAL['gen'],
            label='Non-steroidal', edgecolor='white', linewidth=0.5,
            zorder=3)

    for i, (s, ns) in enumerate(zip(seco, nonseco)):
        if s > 8:
            ax.text(s/2, i, f'{s:.1f}%', ha='center', va='center',
                    fontsize=7, color='white', fontweight='medium')
        if ns > 8:
            ax.text(s + ns/2, i, f'{ns:.1f}%', ha='center', va='center',
                    fontsize=7, color='white', fontweight='medium')

    ax.set_yticks(y)
    ax.set_yticklabels(gen_labels, fontsize=7.5)
    ax.set_xlabel('Composition (%)', fontsize=10)
    ax.set_xlim(0, 100)
    ax.legend(fontsize=8, loc='lower right', framealpha=0.92)
    ax.invert_yaxis()
    ax.set_title('Secosteroidal vs Non-steroidal Composition by Generator',
                 fontsize=10, fontweight='bold', pad=10)
    fig.tight_layout()
    save_fig(fig, 'Fig07_seco_composition_stacked')


def fig08_binding_boxplots(D):
    """Individual box plots for Vina and pKd."""
    pd_ = D['plot_data']
    grps = ['Reference', 'All Generated', 'Top 100']
    cols = [PAL['ref'], PAL['gen'], PAL['top']]

    for prop, ylabel, title_str, ylim, suffix in [
        ('vina', 'Vina Docking Score (kcal/mol)',
         'Vina Docking Score Distribution', (-15, -3), '08a'),
        ('pkd', 'ML-predicted pKd',
         'ML-predicted pKd Distribution', None, '08b'),
    ]:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        data = []
        for g in grps:
            vals = pd_.get(g, {}).get(prop, [])
            if ylim:
                vals = [v for v in vals if ylim[0] <= v <= ylim[1]]
            data.append(vals)

        labels = [f'Reference\n(n={D["n_ref"]})',
                  f'All Generated\n(n={D["n_gen"]:,})',
                  'CWRA\nTop 100']
        bp = ax.boxplot(
            data, labels=labels, patch_artist=True, widths=0.55,
            showfliers=True,
            flierprops=dict(marker='.', markersize=2, alpha=0.3),
            medianprops=dict(color='#333', linewidth=1.5),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8))
        for patch, color in zip(bp['boxes'], cols):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_edgecolor(color)

        # Jitter Top 100 points
        jitter = np.random.normal(0, 0.04, len(data[2]))
        ax.scatter(np.full(len(data[2]), 3) + jitter, data[2],
                   s=8, color=PAL['top'], alpha=0.5, zorder=4,
                   edgecolor='none')

        ax.set_ylabel(ylabel, fontsize=10)
        if ylim:
            ax.set_ylim(ylim)
        add_grid(ax)
        ax.set_title(title_str, fontsize=10, fontweight='bold', pad=10)
        fig.tight_layout()
        save_fig(fig, f'Fig{suffix}_{prop}_boxplot')


def fig09_property_radar(D):
    """Normalized radar chart: 6 drug-likeness / binding axes."""
    ps = D['prop_stats']
    labels = ['QED', 'Fsp³', 'HBD\n(norm)', 'HBA\n(norm)',
              '1−SA/10', '|Vina|/15']
    radar = {}
    for gname in ['Reference', 'All Generated', 'Top 100']:
        p = ps[gname]
        radar[gname] = [
            p['QED']['mean'],
            p['FractionCSP3']['mean'],
            p['HBD']['mean'] / 5,
            p['HBA']['mean'] / 10,
            1 - p['SAScore']['mean'] / 10,
            min(1, abs(p['vina_score']['mean']) / 15),
        ]

    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)

    for gname, col, ls in [
        ('Reference', PAL['ref'], '-'),
        ('All Generated', PAL['gen'], '--'),
        ('Top 100', PAL['top'], '-'),
    ]:
        vals = radar[gname] + [radar[gname][0]]
        ax.plot(angles, vals, color=col, linewidth=2, linestyle=ls,
                label=gname, zorder=4)
        ax.fill(angles, vals, alpha=0.08, color=col)
        for a, v in zip(angles[:-1], radar[gname]):
            ax.annotate(f'{v:.2f}', (a, v), textcoords="offset points",
                        xytext=(0, 5), ha='center', fontsize=6, color=col,
                        fontweight='medium')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=7,
                       color='#888')
    ax.spines['polar'].set_linewidth(0.5)
    ax.grid(linewidth=0.5, alpha=0.4)
    ax.legend(fontsize=8, loc='lower right', bbox_to_anchor=(1.15, -0.05),
              framealpha=0.92)
    ax.set_title('Normalized Property Radar\n(Higher = More Favorable)',
                 fontsize=10, fontweight='bold', pad=20)
    fig.tight_layout()
    save_fig(fig, 'Fig09_property_radar')


def fig10_property_histograms(D):
    """Individual histograms for QED and MW."""
    pd_ = D['plot_data']
    for prop, xlabel, xlim, title_str, suffix, extra in [
        ('qed', 'QED (Quantitative Estimate of Drug-likeness)',
         (0, 1), 'QED Distribution', '10a', None),
        ('mw', 'Molecular Weight (Da)',
         (100, 700), 'Molecular Weight Distribution', '10b', 500),
    ]:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for g, col, alpha, lab in [
            ('All Generated', PAL['gen'], 0.4, 'All Generated'),
            ('Reference', PAL['ref'], 0.5, 'Reference'),
            ('Top 100', PAL['top'], 0.7, 'CWRA Top 100'),
        ]:
            vals = pd_.get(g, {}).get(prop, [])
            if prop == 'mw':
                vals = [v for v in vals if v <= 800]
            ax.hist(vals, bins=35, alpha=alpha, color=col, label=lab,
                    edgecolor='white', linewidth=0.3,
                    zorder=3 if g == 'Top 100' else 2)

        if extra:
            ax.axvline(x=extra, color='#999', linewidth=0.8,
                       linestyle=':', alpha=0.7, label='Ro5 limit')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_xlim(xlim)
        ax.legend(fontsize=7.5,
                  loc='upper left' if prop == 'qed' else 'upper right',
                  framealpha=0.92)
        add_grid(ax)
        ax.set_title(title_str, fontsize=10, fontweight='bold', pad=8)
        fig.tight_layout()
        save_fig(fig, f'Fig{suffix}_{prop}_distribution')


def fig11_ring_count(D):
    """Grouped bar chart: ring count distribution."""
    rd = D['ring_dist']
    ring_labels = ['0', '1', '2', '3', '4', '5', '6+']
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(ring_labels))
    w = 0.2
    grps = ['Reference', 'All Generated', 'Top 100', 'Pre-filtered']
    cols = [PAL['ref'], PAL['gen'], PAL['top'], PAL['pf']]

    for i, (g, col) in enumerate(zip(grps, cols)):
        vals = [rd[g].get(r, 0) for r in ring_labels]
        ax.bar(x + (i - 1.5) * w, vals, w, color=col, label=g,
               edgecolor='white', linewidth=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(ring_labels, fontsize=9)
    ax.set_xlabel('Ring Count', fontsize=10)
    ax.set_ylabel('Percentage (%)', fontsize=10)
    ax.set_ylim(0, 62)
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.92)
    add_grid(ax)
    ax.set_title('Ring Count Distribution by Group',
                 fontsize=10, fontweight='bold', pad=10)
    fig.tight_layout()
    save_fig(fig, 'Fig11_ring_count_distribution')


def fig12_nn_tanimoto(D):
    """NN-Tanimoto: bar chart (mean ± std) and violin + box plots."""
    nn = D['nn_tanimoto']
    pd_ = D['plot_data']
    grps = ['Reference', 'All Generated', 'Top 100', 'Pre-filtered']
    cols = [PAL['ref'], PAL['gen'], PAL['top'], PAL['pf']]
    means = [nn[g]['mean'] for g in grps]
    stds = [nn[g]['std'] for g in grps]
    medians = [nn[g]['median'] for g in grps]
    xlabels = ['Reference\n(self-NN, excl. self)',
               'All Generated', 'CWRA Top 100', 'Pre-filtered']

    # 12a: Bar chart
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(range(4), means, color=cols, edgecolor='white',
                  linewidth=0.5, yerr=stds, capsize=4,
                  error_kw=dict(linewidth=1, capthick=1, color='#444'),
                  zorder=3, alpha=0.85)
    for i, (m, med) in enumerate(zip(means, medians)):
        ax.text(i, m + stds[i] + 0.025, f'{m:.3f}', ha='center',
                va='bottom', fontsize=8, fontweight='bold', color='#333')
        ax.text(i, m - 0.04, f'med={med:.3f}', ha='center', va='top',
                fontsize=6.5, color='white', fontweight='medium')

    ax.set_xticks(range(4))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel('NN-Tanimoto (mean ± std)', fontsize=10)
    ax.set_ylim(0, 1.08)
    add_grid(ax)
    ax.set_title('Nearest-Neighbor Tanimoto Similarity to Reference',
                 fontsize=10, fontweight='bold', pad=10)
    fig.tight_layout()
    save_fig(fig, 'Fig12a_nn_tanimoto_bar')

    # 12b: Violin + box
    nn_data = pd_.get('nn_tanimoto', {})
    violin_data = [nn_data.get(g, [0]) for g in grps]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    parts = ax.violinplot(violin_data, positions=range(4),
                          showmeans=True, showmedians=True,
                          showextrema=False)
    for pc, col in zip(parts['bodies'], cols):
        pc.set_facecolor(col)
        pc.set_alpha(0.5)
        pc.set_edgecolor(col)
        pc.set_linewidth(1)
    parts['cmeans'].set_color('#333')
    parts['cmeans'].set_linewidth(1.5)
    parts['cmedians'].set_color('#333')
    parts['cmedians'].set_linewidth(1)
    parts['cmedians'].set_linestyle('--')

    bp = ax.boxplot(violin_data, positions=range(4), widths=0.15,
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color='#333', linewidth=1.2),
                    whiskerprops=dict(linewidth=0.8, color='#444'),
                    capprops=dict(linewidth=0.8, color='#444'))
    for patch in bp['boxes']:
        patch.set_facecolor('white')
        patch.set_alpha(0.8)

    ax.set_xticks(range(4))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel('NN-Tanimoto', fontsize=10)
    ax.set_ylim(0, 1.08)
    add_grid(ax)
    ax.set_title('NN-Tanimoto Distribution by Group',
                 fontsize=10, fontweight='bold', pad=10)
    fig.tight_layout()
    save_fig(fig, 'Fig12b_nn_tanimoto_violin')


def fig13_top100_composition(D):
    """Donut charts: Top 100 by generator combination and source."""
    gc = D['gen_combo_top100']
    sc = D['source_top100']

    # 13a: Generator combination
    labels_gc = list(gc.keys())
    vals_gc = list(gc.values())
    cols_gc = [GEN_PAL.get(g, '#888') for g in labels_gc]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    wedges, _, autotexts = ax.pie(
        vals_gc, colors=cols_gc, autopct='%1.1f%%', pctdistance=0.78,
        wedgeprops=dict(width=0.55, edgecolor='white', linewidth=1.5),
        textprops=dict(fontsize=8))
    for t in autotexts:
        t.set_fontsize(7.5)
        t.set_fontweight('medium')
    ax.legend(wedges, [f'{l} ({v})' for l, v in zip(labels_gc, vals_gc)],
              fontsize=7, loc='center left', bbox_to_anchor=(0.85, 0.5),
              framealpha=0.92)
    ax.set_title('Top 100: Generator Combination',
                 fontsize=10, fontweight='bold', pad=10)
    fig.tight_layout()
    save_fig(fig, 'Fig13a_top100_gen_combo_pie')

    # 13b: Generation source (G1/G2/G3)
    labels_sc = list(sc.keys())
    vals_sc = list(sc.values())
    cols_sc = [{'G1': PAL['G1'], 'G2': PAL['G2'],
                'G3': PAL['G3']}.get(g, '#888') for g in labels_sc]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    wedges, _, autotexts = ax.pie(
        vals_sc, colors=cols_sc, autopct='%1.1f%%', pctdistance=0.75,
        wedgeprops=dict(width=0.55, edgecolor='white', linewidth=1.5),
        textprops=dict(fontsize=9))
    for t in autotexts:
        t.set_fontsize(8.5)
        t.set_fontweight('bold')
        t.set_color('white')
    ax.legend(wedges, [f'{l} (n={v})' for l, v in zip(labels_sc, vals_sc)],
              fontsize=8, loc='center left', bbox_to_anchor=(0.85, 0.5),
              framealpha=0.92)
    ax.set_title('Top 100: Generation Source (G1/G2/G3)',
                 fontsize=10, fontweight='bold', pad=10)
    fig.tight_layout()
    save_fig(fig, 'Fig13b_top100_source_pie')


def fig14_top100_features(D):
    """Grouped bars: Top 100 structural features vs Ref vs Gen."""
    pct = D['pct']
    comp_cats = CATS[:5]
    comp_labels = CAT_LABELS[:5]
    x = np.arange(len(comp_labels))
    w = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (g, col, lab) in enumerate(zip(
        ['Reference', 'All Generated', 'Top 100'],
        [PAL['ref'], PAL['gen'], PAL['top']],
        ['Reference', 'All Generated', 'CWRA Top 100'],
    )):
        vals = [pct[g][c] for c in comp_cats]
        bars = ax.bar(x + (i - 1) * w, vals, w, color=col, label=lab,
                      edgecolor='white', linewidth=0.5, zorder=3,
                      alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 1.5:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.8, f'{v:.1f}',
                        ha='center', va='bottom', fontsize=6.5,
                        fontweight='medium')

    ax.set_xticks(x)
    ax.set_xticklabels(comp_labels, fontsize=8.5)
    ax.set_ylabel('Prevalence (%)', fontsize=10)
    ax.set_ylim(0, 72)
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.92)
    add_grid(ax)
    ax.set_title('Top 100 Structural Features vs Reference vs All Generated',
                 fontsize=10, fontweight='bold', pad=10)
    fig.tight_layout()
    save_fig(fig, 'Fig14_top100_structural_comparison')


def fig15_combined_binding(D):
    """Side-by-side Vina + pKd box plots."""
    pd_ = D['plot_data']
    grps = ['Reference', 'All Generated', 'Top 100']
    cols = [PAL['ref'], PAL['gen'], PAL['top']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    for ax, prop, ylabel, title_str, ylim in [
        (ax1, 'vina', 'Vina Docking Score (kcal/mol)',
         'Vina Docking Score', (-15, -3)),
        (ax2, 'pkd', 'ML-predicted pKd', 'ML-predicted pKd', None),
    ]:
        data = []
        for g in grps:
            vals = pd_.get(g, {}).get(prop, [])
            if ylim:
                vals = [v for v in vals if ylim[0] <= v <= ylim[1]]
            data.append(vals)

        bp = ax.boxplot(
            data, labels=['Ref', 'All Gen', 'Top 100'],
            patch_artist=True, widths=0.55, showfliers=True,
            flierprops=dict(marker='.', markersize=2, alpha=0.3),
            medianprops=dict(color='#333', linewidth=1.5),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8))
        for patch, color in zip(bp['boxes'], cols):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_edgecolor(color)

        jitter = np.random.normal(0, 0.04, len(data[2]))
        ax.scatter(np.full(len(data[2]), 3) + jitter, data[2],
                   s=6, color=PAL['top'], alpha=0.4, zorder=4,
                   edgecolor='none')
        ax.set_ylabel(ylabel, fontsize=9)
        if ylim:
            ax.set_ylim(ylim)
        add_grid(ax)
        ax.set_title(title_str, fontsize=10, fontweight='bold', pad=8)

    fig.suptitle('Binding Property Distributions',
                 fontsize=11, fontweight='bold', y=1.01)
    fig.tight_layout()
    save_fig(fig, 'Fig15_combined_binding_boxplots')


def fig16_combined_qed_mw(D):
    """Side-by-side QED + MW histograms."""
    pd_ = D['plot_data']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    for ax, prop, xlabel, xlim, title_str, vline in [
        (ax1, 'qed', 'QED', (0, 1), 'QED Distribution', None),
        (ax2, 'mw', 'Molecular Weight (Da)', (100, 700),
         'Molecular Weight Distribution', 500),
    ]:
        for g, col, alpha, lab in [
            ('All Generated', PAL['gen'], 0.4, 'All Generated'),
            ('Reference', PAL['ref'], 0.5, 'Reference'),
            ('Top 100', PAL['top'], 0.7, 'CWRA Top 100'),
        ]:
            vals = pd_.get(g, {}).get(prop, [])
            if prop == 'mw':
                vals = [v for v in vals if v <= 800]
            ax.hist(vals, bins=35, alpha=alpha, color=col, label=lab,
                    edgecolor='white', linewidth=0.3,
                    zorder=3 if g == 'Top 100' else 2)

        if vline:
            ax.axvline(x=vline, color='#999', linewidth=0.8,
                       linestyle=':', alpha=0.7)
            ax.text(vline + 5, ax.get_ylim()[1] * 0.9, 'Ro5',
                    fontsize=7, color='#999')

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_xlim(xlim)
        ax.legend(fontsize=7,
                  loc='upper left' if prop == 'qed' else 'upper right',
                  framealpha=0.92)
        add_grid(ax)
        ax.set_title(title_str, fontsize=10, fontweight='bold', pad=8)

    fig.tight_layout()
    save_fig(fig, 'Fig16_combined_qed_mw')


def fig17_combined_clogp_fsp3(D):
    """Side-by-side cLogP + Fsp³ histograms."""
    pd_ = D['plot_data']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    for ax, prop, xlabel, xlim, title_str, vline in [
        (ax1, 'clogp', 'cLogP', (-2, 12), 'cLogP Distribution', 5),
        (ax2, 'fsp3', 'Fsp³', (0, 1),
         'Fraction sp³ Distribution', None),
    ]:
        for g, col, alpha, lab in [
            ('All Generated', PAL['gen'], 0.4, 'All Generated'),
            ('Reference', PAL['ref'], 0.5, 'Reference'),
            ('Top 100', PAL['top'], 0.7, 'CWRA Top 100'),
        ]:
            vals = pd_.get(g, {}).get(prop, [])
            ax.hist(vals, bins=35, alpha=alpha, color=col, label=lab,
                    edgecolor='white', linewidth=0.3,
                    zorder=3 if g == 'Top 100' else 2)

        if vline:
            ax.axvline(x=vline, color='#999', linewidth=0.8,
                       linestyle=':', alpha=0.7)
            ax.text(vline + 0.15, ax.get_ylim()[1] * 0.9, 'Ro5',
                    fontsize=7, color='#999')

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_xlim(xlim)
        ax.legend(fontsize=7, loc='upper right', framealpha=0.92)
        add_grid(ax)
        ax.set_title(title_str, fontsize=10, fontweight='bold', pad=8)

    fig.tight_layout()
    save_fig(fig, 'Fig17_combined_clogp_fsp3')


def fig18_combined_nn_tanimoto(D):
    """Combined bar + violin for NN-Tanimoto."""
    nn = D['nn_tanimoto']
    pd_ = D['plot_data']
    grps = ['Reference', 'All Generated', 'Top 100', 'Pre-filtered']
    cols = [PAL['ref'], PAL['gen'], PAL['top'], PAL['pf']]
    means = [nn[g]['mean'] for g in grps]
    stds = [nn[g]['std'] for g in grps]
    xlbl = ['Ref\n(self-NN)', 'All Gen', 'Top 100', 'Pre-filt']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # Bar
    ax1.bar(range(4), means, color=cols, edgecolor='white', linewidth=0.5,
            yerr=stds, capsize=4,
            error_kw=dict(linewidth=1, capthick=1, color='#444'),
            zorder=3, alpha=0.85)
    for i, m in enumerate(means):
        ax1.text(i, m + stds[i] + 0.02, f'{m:.3f}', ha='center',
                 va='bottom', fontsize=8, fontweight='bold', color='#333')
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(xlbl, fontsize=8)
    ax1.set_ylabel('NN-Tanimoto (mean ± std)', fontsize=9)
    ax1.set_ylim(0, 1.08)
    add_grid(ax1)
    ax1.set_title('Mean NN-Tanimoto', fontsize=10,
                  fontweight='bold', pad=8)

    # Violin
    nn_data = pd_.get('nn_tanimoto', {})
    vdata = [nn_data.get(g, [0]) for g in grps]

    parts = ax2.violinplot(vdata, positions=range(4), showmeans=True,
                           showmedians=True, showextrema=False)
    for pc, col in zip(parts['bodies'], cols):
        pc.set_facecolor(col)
        pc.set_alpha(0.5)
        pc.set_edgecolor(col)
    parts['cmeans'].set_color('#333')
    parts['cmedians'].set_color('#333')
    parts['cmedians'].set_linestyle('--')

    bp = ax2.boxplot(vdata, positions=range(4), widths=0.12,
                     showfliers=False, patch_artist=True,
                     medianprops=dict(color='#333', linewidth=1),
                     whiskerprops=dict(linewidth=0.6, color='#444'),
                     capprops=dict(linewidth=0.6, color='#444'))
    for patch in bp['boxes']:
        patch.set_facecolor('white')
        patch.set_alpha(0.7)

    ax2.set_xticks(range(4))
    ax2.set_xticklabels(xlbl, fontsize=8)
    ax2.set_ylabel('NN-Tanimoto', fontsize=9)
    ax2.set_ylim(0, 1.08)
    add_grid(ax2)
    ax2.set_title('NN-Tanimoto Distribution', fontsize=10,
                  fontweight='bold', pad=8)

    fig.suptitle('Nearest-Neighbor Tanimoto Similarity to Reference Set',
                 fontsize=11, fontweight='bold', y=1.01)
    fig.tight_layout()
    save_fig(fig, 'Fig18_nn_tanimoto_combined')


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    D, df, t100 = load_data()

    print(f"\nGenerating {25} PDF figures...\n")

    fig01_enrichment_heatmap(D)
    fig02_structural_bars(D)
    fig03_two_stage_selection(D)
    fig04_forest_plots(D)
    fig05_metascore(D)
    gen_order, gen_labels, y = fig06_per_generator(D)
    fig07_seco_stacked(D, gen_order, gen_labels, y)
    fig08_binding_boxplots(D)
    fig09_property_radar(D)
    fig10_property_histograms(D)
    fig11_ring_count(D)
    fig12_nn_tanimoto(D)
    fig13_top100_composition(D)
    fig14_top100_features(D)
    fig15_combined_binding(D)
    fig16_combined_qed_mw(D)
    fig17_combined_clogp_fsp3(D)
    fig18_combined_nn_tanimoto(D)

    # Summary
    n_files = len([f for f in os.listdir(FIG_DIR) if f.endswith('.pdf')])
    total_kb = sum(
        os.path.getsize(os.path.join(FIG_DIR, f))
        for f in os.listdir(FIG_DIR) if f.endswith('.pdf')
    ) / 1024
    print(f"\n{'='*60}")
    print(f"All figures generated in {FIG_DIR}/")
    print(f"Files: {n_files}  |  Total size: {total_kb:.0f} KB")
    print(f"Format: vector PDF, 300 DPI, TrueType fonts")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
