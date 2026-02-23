#!/usr/bin/env python3
"""
VDR CWRA Structural Analysis — Interactive Report

Prerequisites:
  Run 01_data_pipeline.py first to produce output/report_data_slim.json

Output:
  output/vdr_cwra_enrichment_analysis.html (self-contained, ~270 KB)

Sections:
  1. Summary KPIs              8. Per-Generator Analysis
  2. Key Findings              9. Property Profiles
  3. Enrichment Heatmap       10. Ring Count Distribution
  4. Structural Comparison    11. NN-Tanimoto Similarity
  5. Enrichment Ratios        12. Top 100 Composition
  6. Enrichment Data Table    13. Interpretation & Conclusions
  7. Statistical Tests
"""
import json
import os

os.makedirs('output', exist_ok=True)

with open('output/report_data_slim.json') as f:
    D = json.load(f)

pct = D['pct']
cnt = D['cnt']
fisher = D['fisher']
ps = D['prop_stats']
nn = D['nn_tanimoto']
rd = D['ring_dist']
gc = D['gen_combo_top100']
sc = D['source_top100']
gs = D['gen_struct']
pd_ = D['plot_data']

cats = ['is_secosteroidal','non_steroidal','c2_modified','pentacyclic','has_o5ring','sidechain_cycle']
cat_labels = ['Secosteroidal','Non-steroidal','C2-Modified','Pentacyclic','Furan/O-heterocycle','Side-chain Cycles']
cat_short = ['Seco','Non-ster','C2-Mod','Penta','Furan/O','SC-Cyc']

# Modern color palette
C = {
    'ref':  '#5B8DB8',   # steel blue
    'gen':  '#8DB580',   # sage green
    'top':  '#D96B4F',   # terracotta
    'pf':   '#B0A0C0',   # muted lavender
    'G1':   '#A8C8A0',   # light sage
    'G2':   '#6BA3BE',   # teal
    'G3':   '#D96B4F',   # terracotta
    'bg':   '#1a1d23',
    'card': '#22262e',
    'txt':  '#e0e2e8',
    'dim':  '#8890a0',
    'grid': '#2d3240',
    'acc':  '#D96B4F',
}

# Generator combination colors (viridis-inspired)
GC = {
    'gmdldr_reinvent': '#3E8E8D',
    'transmol-reinvent-gmdldr': '#D96B4F',
    'reinvent_transmol': '#6BA3BE',
    'gmdldr_transmol': '#B0A0C0',
    'transmol': '#F0C05A',
}

def fp(p):
    if p < 1e-20: return f"{p:.1e}"
    if p < 0.001: return f"{p:.2e}"
    if p < 0.01: return f"{p:.4f}"
    return f"{p:.3f}"

def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'ns'

# Plotly layout template
layout_tmpl = """{{
  paper_bgcolor: '{bg}', plot_bgcolor: '{card}',
  font: {{color: '{txt}', family: 'Inter, system-ui, sans-serif', size: 12}},
  margin: {{l: 60, r: 30, t: 50, b: 60}},
  xaxis: {{gridcolor: '{grid}', zerolinecolor: '{grid}'}},
  yaxis: {{gridcolor: '{grid}', zerolinecolor: '{grid}'}},
  legend: {{bgcolor: 'rgba(0,0,0,0)', font: {{size: 11}}, itemsizing: 'constant',
            yanchor: 'top', y: -0.18, xanchor: 'center', x: 0.5, orientation: 'h',
            itemwidth: 30, tracegroupgap: 5}},
  hoverlabel: {{bgcolor: '#333', font: {{size: 12}}}},
}}""".format(**C)

# Build HTML
html_parts = []

# ═══════════════════════════════════════════════════════════════════
# HEAD & STYLES
# ═══════════════════════════════════════════════════════════════════
html_parts.append(f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VDR CWRA Structural Enrichment Analysis</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:{C['bg']};color:{C['txt']};font-family:'Inter',system-ui,sans-serif;font-size:14px;line-height:1.6}}
.wrap{{max-width:1240px;margin:0 auto;padding:16px 24px}}
h1{{font-size:1.8em;font-weight:700;margin:32px 0 8px;color:#fff;letter-spacing:-0.02em}}
h2{{font-size:1.35em;font-weight:600;margin:36px 0 16px;padding:10px 0 6px;
    border-bottom:2px solid {C['acc']};color:#fff;letter-spacing:-0.01em}}
h3{{font-size:1.05em;font-weight:500;margin:16px 0 10px;color:{C['dim']}}}
p{{margin:0 0 12px;color:{C['dim']}}}
.sub{{color:{C['dim']};font-size:0.92em;margin-bottom:20px}}
.card{{background:{C['card']};border-radius:10px;padding:20px;margin:12px 0;border:1px solid #2d3240}}
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.g3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}}
.P{{width:100%;height:420px}}
.Ps{{width:100%;height:380px}}
.Pm{{width:100%;height:320px}}
.note{{background:rgba(91,141,184,0.08);border-left:3px solid {C['ref']};padding:14px 18px;margin:12px 0;border-radius:0 8px 8px 0;font-size:0.92em}}
.kpi{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:16px 0}}
.kpi-card{{background:{C['card']};border-radius:8px;padding:14px;text-align:center;border:1px solid #2d3240}}
.kpi-val{{font-size:1.7em;font-weight:700;color:{C['acc']}}}
.kpi-lab{{font-size:0.8em;color:{C['dim']};margin-top:2px}}
table.dtbl{{width:100%;border-collapse:collapse;font-size:0.85em;margin:12px 0}}
table.dtbl th{{background:#1e222a;padding:8px 10px;text-align:left;font-weight:500;border-bottom:2px solid {C['grid']}}}
table.dtbl td{{padding:6px 10px;border-bottom:1px solid {C['grid']}}}
table.dtbl tr:hover{{background:rgba(255,255,255,0.02)}}
table.dtbl tr.sig{{background:rgba(217,107,79,0.06)}}
td.cat{{font-weight:500;color:{C['txt']}}}
nav{{position:sticky;top:0;z-index:100;background:{C['bg']};border-bottom:1px solid {C['grid']};padding:8px 0}}
nav a{{color:{C['dim']};text-decoration:none;font-size:0.8em;padding:4px 10px;border-radius:4px;transition:0.15s}}
nav a:hover{{color:#fff;background:rgba(255,255,255,0.06)}}
.toc{{display:flex;flex-wrap:wrap;gap:4px;max-width:1240px;margin:0 auto;padding:0 24px}}
@media(max-width:768px){{.g2,.g3{{grid-template-columns:1fr}}.wrap{{padding:12px 14px}}}}
</style></head><body>

<nav><div class="toc">
<a href="#s1">1. Summary</a><a href="#s2">2. Key Findings</a><a href="#s3">3. Heatmap</a>
<a href="#s4">4. Modification Comparison</a><a href="#s5">5. Enrichment Ratios</a>
<a href="#s6">6. Data Tables</a><a href="#s7">7. Statistics</a>
<a href="#s8">8. Per-Generation</a><a href="#s9">9. Properties</a>
<a href="#s10">10. Ring Count</a><a href="#s11">11. Tanimoto</a>
<a href="#s12">12. Top 100</a><a href="#s13">13. Conclusions</a>
</div></nav>

<div class="wrap">
<h1>VDR CWRA Structural Enrichment Analysis</h1>
<p class="sub">Corrected analysis · newRef_137 excluded · Pre-filter: MW&gt;600 Da or RotB&gt;15 removed from generated pool · N′ = {D['n_active']:,}</p>
""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 1: SUMMARY
# ═══════════════════════════════════════════════════════════════════
html_parts.append(f"""
<h2 id="s1">1. Summary</h2>
<div class="kpi">
  <div class="kpi-card"><div class="kpi-val">{D['n_active']:,}</div><div class="kpi-lab">Active Pool (N′)</div></div>
  <div class="kpi-card"><div class="kpi-val">{D['n_ref']}</div><div class="kpi-lab">Reference</div></div>
  <div class="kpi-card"><div class="kpi-val">{D['n_gen']:,}</div><div class="kpi-lab">Generated (post-filter)</div></div>
  <div class="kpi-card"><div class="kpi-val">{D['n_top100']}</div><div class="kpi-lab">Top 100 (CWRA)</div></div>
  <div class="kpi-card"><div class="kpi-val">{D['n_prefiltered']:,}</div><div class="kpi-lab">Pre-filtered</div></div>
  <div class="kpi-card"><div class="kpi-val">96%</div><div class="kpi-lab">Top 100 Multi-Gen</div></div>
</div>
<div class="note">
<strong>Pipeline:</strong> 16,059 compounds (excl. newRef_137) → pre-filter MW&gt;600/RotB&gt;15 on generated compounds 
→ {D['n_active']:,} active pool → CWRA meta_score ranking → Top 100 from generated sources (G1: {sc.get('G1',0)}, G2: {sc.get('G2',0)}, G3: {sc.get('G3',0)}).
</div>
""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 2: KEY FINDINGS
# ═══════════════════════════════════════════════════════════════════
html_parts.append(f"""
<h2 id="s2">2. Key Findings</h2>
<div class="card">
<p><strong>CWRA reverses generative drift:</strong> Secosteroidal scaffolds drop from {pct['Reference']['is_secosteroidal']}% (reference) to {pct['All Generated']['is_secosteroidal']}% (all generated) 
then recover to <strong>{pct['Top 100']['is_secosteroidal']}%</strong> in Top 100 — exceeding reference levels.</p>
<p><strong>C2-modifications enriched:</strong> Top 100 achieves {pct['Top 100']['c2_modified']}% C2-modification rate vs {pct['All Generated']['c2_modified']}% in all generated 
(OR=6.08, p=2.6×10⁻⁶) — validates CYP24A1 metabolic stability as CWRA-selected feature.</p>
<p><strong>Multi-generator consensus dominates:</strong> 96% of Top 100 from G2+G3 consensus. G2 compounds show 2.88× enrichment 
in secosteroidal scaffolds vs G1 (p=9.9×10⁻¹¹).</p>
<p><strong>Pentacyclic eliminated:</strong> 0% of Top 100 vs {pct['All Generated']['pentacyclic']}% all generated (p=0.011) — CWRA filters developability-poor scaffolds.</p>
<p><strong>NN-Tanimoto confirms recovery:</strong> Top 100 (0.899) approaches reference self-similarity (0.829) while maintaining structural novelty.</p>
</div>
""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 3: ENRICHMENT HEATMAP
# ═══════════════════════════════════════════════════════════════════
# Build heatmap data
hm_groups = ['Reference', 'All Generated', 'Top 100', 'G1', 'G2', 'G3', 'Pre-filtered']
hm_z = []
hm_text = []
for gi, g in enumerate(hm_groups):
    row_z = []
    row_t = []
    for ci, cat in enumerate(cats):
        v = pct[g][cat]
        c = cnt[g][cat]
        n = cnt[g]['n']
        row_z.append(v)
        row_t.append(f"{cat_labels[ci]}: {v}% ({c}/{n:,})")
    hm_z.append(row_z)
    hm_text.append(row_t)

html_parts.append(f"""
<h2 id="s3">3. Enrichment Heatmap</h2>
<div class="card">
<div id="ch_heatmap" class="P"></div>
</div>
<script>
Plotly.newPlot('ch_heatmap', [{{
  z: {json.dumps(hm_z)},
  x: {json.dumps(cat_short)},
  y: {json.dumps([f"{g} (n={cnt[g]['n']:,})" for g in hm_groups])},
  type: 'heatmap',
  colorscale: [[0,'#1a1d23'],[0.15,'#1e3a4f'],[0.3,'#2a6070'],[0.5,'#3E8E8D'],[0.7,'#6BAE6B'],[0.85,'#D9A84F'],[1,'#D96B4F']],
  text: {json.dumps(hm_text)},
  hoverinfo: 'text',
  texttemplate: '%{{z:.1f}}%',
  textfont: {{size: 11, color: '#fff'}},
  showscale: true,
  colorbar: {{title: '%', titleside: 'right', len: 0.9, thickness: 12, tickfont: {{size: 10}}}},
}}], {{
  ...{layout_tmpl},
  title: {{text: 'Structural Feature Prevalence by Group', font: {{size: 15}}}},
  margin: {{l: 180, r: 80, t: 50, b: 50}},
  xaxis: {{side: 'bottom'}},
  yaxis: {{autorange: 'reversed'}},
}}, {{responsive: true}});
</script>
""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 4: STRUCTURAL MODIFICATION COMPARISON
# ═══════════════════════════════════════════════════════════════════
# Grouped bar chart (skip non_steroidal and sidechain_cycle for cleaner viz)
bar_cats_idx = [0, 2, 3, 4]  # seco, c2, penta, furan
bar_labels = [cat_labels[i] for i in bar_cats_idx]
bar_cats_keys = [cats[i] for i in bar_cats_idx]

html_parts.append(f"""
<h2 id="s4">4. Structural Modification Comparison</h2>
<div class="card">
<p>Grouped bar comparison of key structural features across Reference, All Generated, and CWRA Top 100.
Secosteroidal scaffolds and C2-modifications show the clearest CWRA enrichment pattern.</p>
<div id="ch_bars" class="P"></div>
</div>
<script>
Plotly.newPlot('ch_bars', [
  {{name: 'Reference (n={D["n_ref"]})', x: {json.dumps(bar_labels)}, 
    y: [{','.join(str(pct['Reference'][c]) for c in bar_cats_keys)}], type: 'bar',
    marker: {{color: '{C["ref"]}', line: {{color: 'rgba(255,255,255,0.15)', width: 1}}}},
    text: [{','.join(str(pct['Reference'][c]) for c in bar_cats_keys)}],
    texttemplate: '%{{text:.1f}}%', textposition: 'outside', textfont: {{size: 10}}}},
  {{name: 'All Generated (n={D["n_gen"]:,})', x: {json.dumps(bar_labels)},
    y: [{','.join(str(pct['All Generated'][c]) for c in bar_cats_keys)}], type: 'bar',
    marker: {{color: '{C["gen"]}', line: {{color: 'rgba(255,255,255,0.15)', width: 1}}}},
    text: [{','.join(str(pct['All Generated'][c]) for c in bar_cats_keys)}],
    texttemplate: '%{{text:.1f}}%', textposition: 'outside', textfont: {{size: 10}}}},
  {{name: 'CWRA Top 100', x: {json.dumps(bar_labels)},
    y: [{','.join(str(pct['Top 100'][c]) for c in bar_cats_keys)}], type: 'bar',
    marker: {{color: '{C["top"]}', line: {{color: 'rgba(255,255,255,0.15)', width: 1}}}},
    text: [{','.join(str(pct['Top 100'][c]) for c in bar_cats_keys)}],
    texttemplate: '%{{text:.1f}}%', textposition: 'outside', textfont: {{size: 10}}}},
], {{
  ...{layout_tmpl},
  title: {{text: 'Structural Feature Prevalence: Reference vs Generated vs Top 100', font: {{size: 14}}}},
  barmode: 'group', bargap: 0.2, bargroupgap: 0.08,
  yaxis: {{title: 'Prevalence (%)', range: [0, 80], gridcolor: '{C["grid"]}'}},
  legend: {{yanchor: 'top', y: -0.15, xanchor: 'center', x: 0.5, orientation: 'h'}},
}}, {{responsive: true}});
</script>
""")

# Two-stage selection dynamics (Sankey-like line chart)
two_stage_cats = ['is_secosteroidal', 'c2_modified', 'pentacyclic', 'has_o5ring']
two_stage_labels = ['Secosteroidal', 'C2-Modified', 'Pentacyclic', 'Furan/O-het']
two_stage_colors = ['#5B8DB8', '#D96B4F', '#B0A0C0', '#F0C05A']
stages = ['Reference', 'All Generated', 'Top 100']

html_parts.append(f"""
<div class="card">
<h3>Two-Stage Selection Dynamics</h3>
<p>Tracking structural feature prevalence from Reference → All Generated → Top 100 reveals CWRA's corrective effect on generative drift.</p>
<div id="ch_twostage" class="P"></div>
</div>
<script>
Plotly.newPlot('ch_twostage', [
""")

for i, (cat, label, color) in enumerate(zip(two_stage_cats, two_stage_labels, two_stage_colors)):
    vals = [pct[g][cat] for g in stages]
    html_parts.append(f"""{{name: '{label}', x: ['Reference','All Generated','Top 100'], y: {json.dumps(vals)}, 
    type: 'scatter', mode: 'lines+markers', line: {{color: '{color}', width: 3}}, 
    marker: {{size: 10, color: '{color}', line: {{color: '#fff', width: 1}}}},
    text: [{','.join(f"'{v}%'" for v in vals)}], textposition: 'top center', textfont: {{size: 11}}}},
""")

html_parts.append(f"""
], {{
  ...{layout_tmpl},
  title: {{text: 'Two-Stage Selection: Reference → Generation → CWRA Recovery', font: {{size: 14}}}},
  yaxis: {{title: 'Prevalence (%)', range: [0, 75], gridcolor: '{C["grid"]}'}},
  xaxis: {{type: 'category'}},
  legend: {{yanchor: 'top', y: -0.15, xanchor: 'center', x: 0.5, orientation: 'h'}},
}}, {{responsive: true}});
</script>
""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 5: ENRICHMENT RATIOS & STATISTICAL SIGNIFICANCE
# ═══════════════════════════════════════════════════════════════════
# Build forest-plot style OR charts for each comparison
html_parts.append(f"""
<h2 id="s5">5. Enrichment Ratios &amp; Statistical Significance</h2>
<div class="card">
<p>Odds ratios from Fisher's exact test. Values &gt;1 (right of dashed line) indicate enrichment; &lt;1 indicate depletion.
Bars colored by significance: <span style="color:#D96B4F">★★★</span> p&lt;0.001, <span style="color:#F0C05A">★★</span> p&lt;0.01, <span style="color:#6BA3BE">★</span> p&lt;0.05, <span style="color:{C['dim']}">ns</span>.</p>
</div>
""")

comparisons_to_plot = [
    ('Top 100 vs All Generated', 'CWRA Selection Effect'),
    ('All Generated vs Reference', 'Generative Model Drift'),
    ('Top 100 vs Reference', 'Net CWRA vs Training Set'),
    ('G2 vs G1', 'Multi-Generator Consensus Effect'),
]

for ci, (comp, title) in enumerate(comparisons_to_plot):
    fr = [f for f in fisher if f['comp'] == comp]
    cats_f = [f['cat'] for f in fr]
    ors = [min(f['odds'], 20) for f in fr]  # cap at 20 for viz
    ps_f = [f['p'] for f in fr]
    sigs = [f['sig'] for f in fr]
    dirs = [f['dir'] for f in fr]
    
    colors = []
    for s in sigs:
        if s == '***': colors.append('#D96B4F')
        elif s == '**': colors.append('#F0C05A')
        elif s == '*': colors.append('#6BA3BE')
        else: colors.append(C['dim'])
    
    texts = [f"OR={f['odds']:.2f}, p={fp(f['p'])} {f['sig']}" for f in fr]
    
    html_parts.append(f"""
<div class="card">
<h3>{title}</h3>
<div id="ch_er{ci}" class="Pm"></div>
</div>
<script>
Plotly.newPlot('ch_er{ci}', [{{
  y: {json.dumps(cats_f)}, x: {json.dumps(ors)}, type: 'bar', orientation: 'h',
  marker: {{color: {json.dumps(colors)}, line: {{color: 'rgba(255,255,255,0.2)', width: 1}}}},
  text: {json.dumps(texts)}, hoverinfo: 'text',
  texttemplate: {json.dumps([f"OR={o:.2f} {s}" for o, s in zip(ors, sigs)])},
  textposition: 'outside', textfont: {{size: 10}},
}}], {{
  ...{layout_tmpl},
  title: {{text: '{comp}', font: {{size: 13}}}},
  margin: {{l: 140, r: 100, t: 40, b: 40}},
  xaxis: {{title: 'Odds Ratio', type: 'log', range: [-2, 1.5], gridcolor: '{C["grid"]}'}},
  yaxis: {{autorange: 'reversed'}},
  shapes: [{{type: 'line', x0: 1, x1: 1, y0: -0.5, y1: 5.5, line: {{color: '#fff', width: 1, dash: 'dot'}}}}],
}}, {{responsive: true}});
</script>
""")

# CWRA meta_score distributions
html_parts.append(f"""
<div class="card">
<h3>CWRA meta_score Distributions</h3>
<div id="ch_scores" class="Ps"></div>
</div>
<script>
Plotly.newPlot('ch_scores', [
  {{name: 'Reference', x: {json.dumps(pd_.get('Reference',{}).get('meta_score',[]))}, type: 'histogram',
    opacity: 0.6, marker: {{color: '{C["ref"]}'}}, nbinsx: 50}},
  {{name: 'All Generated', x: {json.dumps(pd_.get('All Generated',{}).get('meta_score',[]))}, type: 'histogram',
    opacity: 0.5, marker: {{color: '{C["gen"]}'}}, nbinsx: 50}},
  {{name: 'Top 100', x: {json.dumps(pd_.get('Top 100',{}).get('meta_score',[]))}, type: 'histogram',
    opacity: 0.7, marker: {{color: '{C["top"]}'}}, nbinsx: 30}},
], {{
  ...{layout_tmpl},
  title: {{text: 'CWRA meta_score Distribution', font: {{size: 14}}}},
  barmode: 'overlay',
  xaxis: {{title: 'meta_score', range: [0.15, 0.9]}},
  yaxis: {{title: 'Count'}},
  legend: {{yanchor: 'top', y: 0.98, xanchor: 'right', x: 0.98, orientation: 'v'}},
}}, {{responsive: true}});
</script>
""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 6: DATA TABLES
# ═══════════════════════════════════════════════════════════════════
groups_for_table = ['Reference','All Generated','Top 100','G1','G2','G3','Pre-filtered']
html_parts.append(f"""
<h2 id="s6">6. Full Enrichment Data</h2>
<div class="card">
<table class="dtbl"><tr><th>Category</th>""")
for g in groups_for_table:
    html_parts.append(f'<th>{g}<br><small>(n={cnt[g]["n"]:,})</small></th>')
html_parts.append('</tr>\n')
for ci, (cat, label) in enumerate(zip(cats, cat_labels)):
    html_parts.append(f'<tr><td class="cat">{label}</td>')
    for g in groups_for_table:
        v = pct[g][cat]
        c = cnt[g][cat]
        n = cnt[g]['n']
        op = min(0.5, v/100)
        html_parts.append(f'<td style="background:rgba(217,107,79,{op:.2f})">{v:.1f}%<br><small>({c}/{n:,})</small></td>')
    html_parts.append('</tr>\n')
html_parts.append('</table></div>')

# ═══════════════════════════════════════════════════════════════════
# SECTION 7: STATISTICAL SIGNIFICANCE TABLE
# ═══════════════════════════════════════════════════════════════════
html_parts.append(f"""
<h2 id="s7">7. Statistical Significance Tests</h2>
<div class="card">
<p>Fisher's exact test (two-sided) for categorical enrichment/depletion.</p>
<table class="dtbl"><tr><th>Comparison</th><th>Category</th><th>Odds Ratio</th><th>p-value</th><th>Sig</th><th>Direction</th></tr>
""")
for f in fisher:
    sig_class = ' class="sig"' if f['sig'] != 'ns' else ''
    or_str = f"{f['odds']:.2f}" if f['odds'] < 100 else "∞"
    html_parts.append(f'<tr{sig_class}><td>{f["comp"]}</td><td class="cat">{f["cat"]}</td><td>{or_str}</td><td>{fp(f["p"])}</td><td>{f["sig"]}</td><td>{f["dir"]}</td></tr>\n')
html_parts.append('</table></div>')

# ═══════════════════════════════════════════════════════════════════
# SECTION 8: PER-GENERATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════
gen_order = sorted(gs.keys(), key=lambda x: gs[x]['n'], reverse=True)
gen_labels = [f"{g} (n={gs[g]['n']:,})" for g in gen_order]
gen_seco = [gs[g]['secosteroidal_pct'] for g in gen_order]
gen_c2 = [gs[g]['c2_pct'] for g in gen_order]
gen_penta = [gs[g]['pentacyclic_pct'] for g in gen_order]
gen_o5 = [gs[g]['o5ring_pct'] for g in gen_order]

html_parts.append(f"""
<h2 id="s8">8. Per-Generation Analysis</h2>
<div class="card">
<p>Structural feature prevalence by individual generator and generator combination.</p>
<div id="ch_gen_multi" class="P"></div>
</div>
<script>
Plotly.newPlot('ch_gen_multi', [
  {{name: 'Secosteroidal', y: {json.dumps(gen_labels)}, x: {json.dumps(gen_seco)}, type: 'bar', orientation: 'h',
    marker: {{color: '#5B8DB8'}}, text: {json.dumps([f"{v:.1f}%" for v in gen_seco])}, textposition: 'outside', textfont: {{size: 10}}}},
  {{name: 'C2-Modified', y: {json.dumps(gen_labels)}, x: {json.dumps(gen_c2)}, type: 'bar', orientation: 'h',
    marker: {{color: '#D96B4F'}}, text: {json.dumps([f"{v:.1f}%" for v in gen_c2])}, textposition: 'outside', textfont: {{size: 10}}}},
  {{name: 'Pentacyclic', y: {json.dumps(gen_labels)}, x: {json.dumps(gen_penta)}, type: 'bar', orientation: 'h',
    marker: {{color: '#B0A0C0'}}, text: {json.dumps([f"{v:.1f}%" for v in gen_penta])}, textposition: 'outside', textfont: {{size: 10}}}},
  {{name: 'Furan/O-het', y: {json.dumps(gen_labels)}, x: {json.dumps(gen_o5)}, type: 'bar', orientation: 'h',
    marker: {{color: '#F0C05A'}}, text: {json.dumps([f"{v:.1f}%" for v in gen_o5])}, textposition: 'outside', textfont: {{size: 10}}}},
], {{
  ...{layout_tmpl},
  title: {{text: 'Structural Features by Generator / Combination', font: {{size: 14}}}},
  barmode: 'group', bargap: 0.15,
  margin: {{l: 230, r: 80, t: 50, b: 50}},
  xaxis: {{title: 'Prevalence (%)', range: [0, 85]}},
  yaxis: {{autorange: 'reversed'}},
  legend: {{yanchor: 'top', y: -0.12, xanchor: 'center', x: 0.5, orientation: 'h'}},
}}, {{responsive: true}});
</script>
""")

# Stacked composition chart
gen_nonseco = [100 - v for v in gen_seco]
html_parts.append(f"""
<div class="card">
<h3>Secosteroidal vs Non-steroidal Composition</h3>
<div id="ch_gen_stack" class="Ps"></div>
</div>
<script>
Plotly.newPlot('ch_gen_stack', [
  {{name: 'Secosteroidal', y: {json.dumps(gen_labels)}, x: {json.dumps(gen_seco)}, type: 'bar', orientation: 'h',
    marker: {{color: '#5B8DB8'}}}},
  {{name: 'Non-steroidal', y: {json.dumps(gen_labels)}, x: {json.dumps(gen_nonseco)}, type: 'bar', orientation: 'h',
    marker: {{color: '#8DB580'}}}},
], {{
  ...{layout_tmpl},
  title: {{text: 'Secosteroidal vs Non-steroidal by Generator', font: {{size: 14}}}},
  barmode: 'stack',
  margin: {{l: 230, r: 40, t: 50, b: 50}},
  xaxis: {{title: 'Composition (%)', range: [0, 100]}},
  yaxis: {{autorange: 'reversed'}},
  legend: {{yanchor: 'top', y: -0.12, xanchor: 'center', x: 0.5, orientation: 'h'}},
}}, {{responsive: true}});
</script>
""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 9: PROPERTY PROFILES
# ═══════════════════════════════════════════════════════════════════
html_parts.append(f"""
<h2 id="s9">9. Property Profiles</h2>
<div class="card">
<p>Physicochemical and binding property distributions. Vina scores trimmed to [−15, −3] for visual clarity; 
MW trimmed at 800 Da.</p>
</div>
""")

# Vina box plot
html_parts.append(f"""
<div class="g2">
<div class="card"><h3>Vina Docking Score (kcal/mol)</h3><div id="ch_vina" class="Ps"></div></div>
<div class="card"><h3>ML-predicted pKd</h3><div id="ch_pkd" class="Ps"></div></div>
</div>
<script>
Plotly.newPlot('ch_vina', [
  {{name: 'Reference', y: {json.dumps(pd_.get('Reference',{}).get('vina',[]))}, type: 'box', 
    marker: {{color: '{C["ref"]}'}}, boxpoints: 'outliers', jitter: 0.3, pointpos: -1.5,
    line: {{color: '{C["ref"]}'}}, fillcolor: 'rgba(91,141,184,0.3)'}},
  {{name: 'All Generated', y: {json.dumps(pd_.get('All Generated',{}).get('vina',[]))}, type: 'box',
    marker: {{color: '{C["gen"]}'}}, boxpoints: false,
    line: {{color: '{C["gen"]}'}}, fillcolor: 'rgba(141,181,128,0.3)'}},
  {{name: 'Top 100', y: {json.dumps(pd_.get('Top 100',{}).get('vina',[]))}, type: 'box',
    marker: {{color: '{C["top"]}'}}, boxpoints: 'all', jitter: 0.3, pointpos: -1.5,
    line: {{color: '{C["top"]}'}}, fillcolor: 'rgba(217,107,79,0.3)'}},
], {{
  ...{layout_tmpl},
  yaxis: {{title: 'Vina Score (kcal/mol)', range: [-15, -3], gridcolor: '{C["grid"]}'}},
  showlegend: false,
}}, {{responsive: true}});

Plotly.newPlot('ch_pkd', [
  {{name: 'Reference', y: {json.dumps(pd_.get('Reference',{}).get('pkd',[]))}, type: 'box',
    marker: {{color: '{C["ref"]}'}}, boxpoints: 'outliers',
    line: {{color: '{C["ref"]}'}}, fillcolor: 'rgba(91,141,184,0.3)'}},
  {{name: 'All Generated', y: {json.dumps(pd_.get('All Generated',{}).get('pkd',[]))}, type: 'box',
    marker: {{color: '{C["gen"]}'}}, boxpoints: false,
    line: {{color: '{C["gen"]}'}}, fillcolor: 'rgba(141,181,128,0.3)'}},
  {{name: 'Top 100', y: {json.dumps(pd_.get('Top 100',{}).get('pkd',[]))}, type: 'box',
    marker: {{color: '{C["top"]}'}}, boxpoints: 'all', jitter: 0.3, pointpos: -1.5,
    line: {{color: '{C["top"]}'}}, fillcolor: 'rgba(217,107,79,0.3)'}},
], {{
  ...{layout_tmpl},
  yaxis: {{title: 'ML-pKd', gridcolor: '{C["grid"]}'}},
  showlegend: false,
}}, {{responsive: true}});
</script>
""")

# Drug-likeness radar
radar_props = ['QED', 'Fsp³', 'HBD/5', 'HBA/10', '1-SA/10']
for gname in ['Reference', 'All Generated', 'Top 100']:
    p = ps[gname]
    # Normalize to 0-1 range for radar
radar_data = {}
for gname in ['Reference', 'All Generated', 'Top 100']:
    p = ps[gname]
    radar_data[gname] = [
        round(p['QED']['mean'], 3),
        round(p['FractionCSP3']['mean'], 3),
        round(p['HBD']['mean'] / 5, 3),
        round(p['HBA']['mean'] / 10, 3),
        round(1 - p['SAScore']['mean'] / 10, 3),
        round(min(1, -p['vina_score']['mean'] / 15), 3),
    ]

radar_cats_js = json.dumps(['QED', 'Fsp³', 'HBD (norm)', 'HBA (norm)', '1−SA/10', 'Vina/15'])

html_parts.append(f"""
<div class="card">
<h3>Normalized Property Radar</h3>
<p>All axes normalized to [0, 1]. Higher = more favorable.</p>
<div id="ch_radar" class="Ps"></div>
</div>
<script>
Plotly.newPlot('ch_radar', [
  {{name: 'Reference', r: {json.dumps(radar_data['Reference'] + [radar_data['Reference'][0]])},
    theta: {radar_cats_js}.concat([{radar_cats_js}[0]]), type: 'scatterpolar', fill: 'toself',
    fillcolor: 'rgba(91,141,184,0.15)', line: {{color: '{C["ref"]}', width: 2}},
    marker: {{size: 5, color: '{C["ref"]}'}}}},
  {{name: 'All Generated', r: {json.dumps(radar_data['All Generated'] + [radar_data['All Generated'][0]])},
    theta: {radar_cats_js}.concat([{radar_cats_js}[0]]), type: 'scatterpolar', fill: 'toself',
    fillcolor: 'rgba(141,181,128,0.15)', line: {{color: '{C["gen"]}', width: 2}},
    marker: {{size: 5, color: '{C["gen"]}'}}}},
  {{name: 'Top 100', r: {json.dumps(radar_data['Top 100'] + [radar_data['Top 100'][0]])},
    theta: {radar_cats_js}.concat([{radar_cats_js}[0]]), type: 'scatterpolar', fill: 'toself',
    fillcolor: 'rgba(217,107,79,0.15)', line: {{color: '{C["top"]}', width: 2}},
    marker: {{size: 5, color: '{C["top"]}'}}}},
], {{
  ...{layout_tmpl},
  polar: {{
    bgcolor: '{C["card"]}',
    radialaxis: {{visible: true, range: [0, 1], gridcolor: '{C["grid"]}', tickfont: {{size: 9}}}},
    angularaxis: {{gridcolor: '{C["grid"]}', tickfont: {{size: 11}}}},
  }},
  legend: {{yanchor: 'top', y: -0.08, xanchor: 'center', x: 0.5, orientation: 'h'}},
}}, {{responsive: true}});
</script>
""")

# QED and MW distributions
html_parts.append(f"""
<div class="g2">
<div class="card"><h3>QED Distribution</h3><div id="ch_qed" class="Ps"></div></div>
<div class="card"><h3>Molecular Weight</h3><div id="ch_mw" class="Ps"></div></div>
</div>
<script>
Plotly.newPlot('ch_qed', [
  {{name: 'Reference', x: {json.dumps(pd_.get('Reference',{}).get('qed',[]))}, type: 'histogram',
    opacity: 0.55, marker: {{color: '{C["ref"]}'}}, nbinsx: 30}},
  {{name: 'All Generated', x: {json.dumps(pd_.get('All Generated',{}).get('qed',[]))}, type: 'histogram',
    opacity: 0.4, marker: {{color: '{C["gen"]}'}}, nbinsx: 30}},
  {{name: 'Top 100', x: {json.dumps(pd_.get('Top 100',{}).get('qed',[]))}, type: 'histogram',
    opacity: 0.7, marker: {{color: '{C["top"]}'}}, nbinsx: 20}},
], {{
  ...{layout_tmpl}, barmode: 'overlay',
  xaxis: {{title: 'QED', range: [0, 1]}}, yaxis: {{title: 'Count'}},
  legend: {{yanchor: 'top', y: 0.98, xanchor: 'right', x: 0.98, orientation: 'v'}},
}}, {{responsive: true}});

Plotly.newPlot('ch_mw', [
  {{name: 'Reference', x: {json.dumps(pd_.get('Reference',{}).get('mw',[]))}, type: 'histogram',
    opacity: 0.55, marker: {{color: '{C["ref"]}'}}, nbinsx: 30}},
  {{name: 'All Generated', x: {json.dumps(pd_.get('All Generated',{}).get('mw',[]))}, type: 'histogram',
    opacity: 0.4, marker: {{color: '{C["gen"]}'}}, nbinsx: 30}},
  {{name: 'Top 100', x: {json.dumps(pd_.get('Top 100',{}).get('mw',[]))}, type: 'histogram',
    opacity: 0.7, marker: {{color: '{C["top"]}'}}, nbinsx: 20}},
], {{
  ...{layout_tmpl}, barmode: 'overlay',
  xaxis: {{title: 'MW (Da)', range: [100, 700]}}, yaxis: {{title: 'Count'}},
  legend: {{yanchor: 'top', y: 0.98, xanchor: 'right', x: 0.98, orientation: 'v'}},
}}, {{responsive: true}});
</script>
""")

# Property summary table
html_parts.append(f"""
<div class="card">
<h3>Property Summary Table</h3>
<table class="dtbl"><tr><th>Property</th><th>Reference ({D['n_ref']})</th><th>All Generated ({D['n_gen']:,})</th><th>Top 100</th></tr>
""")
prop_keys = ['vina_score','mltle_pKd','QED','SAScore','MW','cLogP','tPSA','FractionCSP3','HBD','HBA','RotB']
prop_labels_tbl = ['Vina (kcal/mol)','ML-pKd','QED','SA Score','MW (Da)','cLogP','tPSA (Å²)','Fsp³','HBD','HBA','RotBonds']
for pk, pl in zip(prop_keys, prop_labels_tbl):
    html_parts.append(f'<tr><td class="cat">{pl}</td>')
    for g in ['Reference','All Generated','Top 100']:
        m = ps[g][pk]['mean']
        s = ps[g][pk]['std']
        html_parts.append(f'<td>{m} <small>± {s}</small></td>')
    html_parts.append('</tr>\n')
html_parts.append('</table></div>')

# ═══════════════════════════════════════════════════════════════════
# SECTION 10: RING COUNT DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════
ring_labels = ['0','1','2','3','4','5','6+']

html_parts.append(f"""
<h2 id="s10">10. Ring Count Distribution</h2>
<div class="card">
<p>Distribution of ring count across groups. Top 100 strongly favors 3-ring systems, with complete elimination of 0-, 1-, and 5+-ring scaffolds.</p>
<div id="ch_ring" class="P"></div>
</div>
<script>
Plotly.newPlot('ch_ring', [
  {{name: 'Reference', x: {json.dumps(ring_labels)}, y: {json.dumps([rd['Reference'].get(r,0) for r in ring_labels])},
    type: 'bar', marker: {{color: '{C["ref"]}'}}}},
  {{name: 'All Generated', x: {json.dumps(ring_labels)}, y: {json.dumps([rd['All Generated'].get(r,0) for r in ring_labels])},
    type: 'bar', marker: {{color: '{C["gen"]}'}}}},
  {{name: 'Top 100', x: {json.dumps(ring_labels)}, y: {json.dumps([rd['Top 100'].get(r,0) for r in ring_labels])},
    type: 'bar', marker: {{color: '{C["top"]}'}}}},
  {{name: 'Pre-filtered', x: {json.dumps(ring_labels)}, y: {json.dumps([rd['Pre-filtered'].get(r,0) for r in ring_labels])},
    type: 'bar', marker: {{color: '{C["pf"]}'}}}},
], {{
  ...{layout_tmpl},
  title: {{text: 'Ring Count Distribution by Group', font: {{size: 14}}}},
  barmode: 'group', bargap: 0.2, bargroupgap: 0.05,
  xaxis: {{title: 'Ring Count'}},
  yaxis: {{title: 'Percentage (%)', range: [0, 65]}},
  legend: {{yanchor: 'top', y: -0.15, xanchor: 'center', x: 0.5, orientation: 'h'}},
}}, {{responsive: true}});
</script>
""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 11: NEAREST-NEIGHBOR TANIMOTO SIMILARITY
# ═══════════════════════════════════════════════════════════════════
nn_groups = ['Reference', 'All Generated', 'Top 100', 'Pre-filtered']
nn_means = [nn[g]['mean'] for g in nn_groups]
nn_stds = [nn[g]['std'] for g in nn_groups]
nn_colors_bar = [C['ref'], C['gen'], C['top'], C['pf']]

html_parts.append(f"""
<h2 id="s11">11. Nearest-Neighbor Tanimoto Similarity to Reference</h2>
<div class="card">
<p><strong>Method:</strong> For each compound, the maximum Tanimoto similarity (Morgan FP, radius 2, 2048 bits) to any reference compound 
is computed. For reference compounds, self-identity is excluded. This measures proximity to the nearest known VDR active.</p>
</div>
<div class="g2">
<div class="card">
<h3>Mean NN-Tanimoto</h3>
<div id="ch_nn_bar" class="Ps"></div>
</div>
<div class="card">
<h3>NN-Tanimoto Distribution</h3>
<div id="ch_nn_violin" class="Ps"></div>
</div>
</div>
<script>
Plotly.newPlot('ch_nn_bar', [{{
  x: {json.dumps(nn_groups)}, y: {json.dumps(nn_means)},
  error_y: {{type: 'data', array: {json.dumps(nn_stds)}, visible: true, color: '#fff', thickness: 1.5}},
  type: 'bar',
  marker: {{color: {json.dumps(nn_colors_bar)}, line: {{color: 'rgba(255,255,255,0.2)', width: 1}}}},
  text: {json.dumps([f"{v:.3f}" for v in nn_means])},
  texttemplate: '%{{text}}', textposition: 'outside', textfont: {{size: 12, color: '#fff'}},
}}], {{
  ...{layout_tmpl},
  yaxis: {{title: 'NN-Tanimoto (mean ± std)', range: [0, 1.05], gridcolor: '{C["grid"]}'}},
  showlegend: false,
}}, {{responsive: true}});

Plotly.newPlot('ch_nn_violin', [
""")

nn_plot_data = pd_.get('nn_tanimoto', {})
for g, color in zip(nn_groups, nn_colors_bar):
    vals = nn_plot_data.get(g, [])
    if vals:
        html_parts.append(f"""{{name: '{g}', y: {json.dumps(vals[:500])}, type: 'violin', box: {{visible: true}},
    meanline: {{visible: true}}, line: {{color: '{color}'}}, fillcolor: '{color}33',
    points: false, spanmode: 'soft'}},
""")

html_parts.append(f"""
], {{
  ...{layout_tmpl},
  yaxis: {{title: 'NN-Tanimoto', range: [0, 1.05], gridcolor: '{C["grid"]}'}},
  violingap: 0.3, violinmode: 'group',
  showlegend: false,
}}, {{responsive: true}});
</script>

<div class="note">
<strong>Interpretation:</strong> Reference self-NN ({nn['Reference']['mean']}) reflects high internal structural similarity within known VDR actives (excluding self-identity).
Top 100 ({nn['Top 100']['mean']}) closely approaches reference, confirming CWRA selects compounds structurally 
proximal to the VDR pharmacophore. All Generated ({nn['All Generated']['mean']}) and Pre-filtered ({nn['Pre-filtered']['mean']}) show 
substantially lower similarity, validating both the CWRA and pre-filter stages.
</div>
""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 12: TOP 100 COMPOSITION
# ═══════════════════════════════════════════════════════════════════
# Generator combo pie
gc_labels = list(gc.keys())
gc_values = list(gc.values())
gc_colors = [GC.get(g, '#888') for g in gc_labels]

# Source pie
sc_labels = list(sc.keys())
sc_values = list(sc.values())
sc_colors_map = {'G1': C['G1'], 'G2': C['G2'], 'G3': C['G3']}
sc_colors = [sc_colors_map.get(g, '#888') for g in sc_labels]

html_parts.append(f"""
<h2 id="s12">12. Top 100 Composition</h2>
<div class="card">
<p>Breakdown of Top 100 CWRA-selected compounds by generator combination and generation source.
Multi-generator consensus (G2 + G3) accounts for <strong>{sc.get('G2',0) + sc.get('G3',0)}%</strong> of Top 100.</p>
</div>
<div class="g2">
<div class="card">
<h3>By Generator Combination</h3>
<div id="ch_pie_gen" class="Ps"></div>
</div>
<div class="card">
<h3>By Generation Source</h3>
<div id="ch_pie_src" class="Ps"></div>
</div>
</div>
<script>
Plotly.newPlot('ch_pie_gen', [{{
  labels: {json.dumps(gc_labels)}, values: {json.dumps(gc_values)},
  type: 'pie', hole: 0.4,
  marker: {{colors: {json.dumps(gc_colors)}, line: {{color: '{C["bg"]}', width: 2}}}},
  textinfo: 'label+percent', textposition: 'outside',
  textfont: {{size: 10, color: '{C["txt"]}'}},
  outsidetextfont: {{size: 10}},
  pull: [0.02, 0.02, 0.02, 0.02, 0.02],
}}], {{
  ...{layout_tmpl},
  margin: {{l: 20, r: 20, t: 30, b: 30}},
  showlegend: false,
}}, {{responsive: true}});

Plotly.newPlot('ch_pie_src', [{{
  labels: {json.dumps(sc_labels)}, values: {json.dumps(sc_values)},
  type: 'pie', hole: 0.4,
  marker: {{colors: {json.dumps(sc_colors)}, line: {{color: '{C["bg"]}', width: 2}}}},
  textinfo: 'label+value+percent', textposition: 'inside',
  textfont: {{size: 12, color: '#fff'}},
}}], {{
  ...{layout_tmpl},
  margin: {{l: 20, r: 20, t: 30, b: 30}},
  showlegend: false,
}}, {{responsive: true}});
</script>
""")

# Top 100 structural features comparison
html_parts.append(f"""
<div class="card">
<h3>Top 100 Structural Features vs Reference vs All Generated</h3>
<div id="ch_top100_struct" class="Ps"></div>
</div>
<script>
Plotly.newPlot('ch_top100_struct', [
  {{name: 'Reference', x: {json.dumps(cat_labels[:5])},
    y: [{','.join(str(pct['Reference'][c]) for c in cats[:5])}], type: 'bar',
    marker: {{color: '{C["ref"]}', opacity: 0.7}}}},
  {{name: 'All Generated', x: {json.dumps(cat_labels[:5])},
    y: [{','.join(str(pct['All Generated'][c]) for c in cats[:5])}], type: 'bar',
    marker: {{color: '{C["gen"]}', opacity: 0.7}}}},
  {{name: 'Top 100', x: {json.dumps(cat_labels[:5])},
    y: [{','.join(str(pct['Top 100'][c]) for c in cats[:5])}], type: 'bar',
    marker: {{color: '{C["top"]}'}}}},
], {{
  ...{layout_tmpl},
  barmode: 'group', bargap: 0.15,
  yaxis: {{title: 'Prevalence (%)', range: [0, 75]}},
  legend: {{yanchor: 'top', y: -0.15, xanchor: 'center', x: 0.5, orientation: 'h'}},
}}, {{responsive: true}});
</script>
""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 13: INTERPRETATION & CONCLUSIONS
# ═══════════════════════════════════════════════════════════════════
html_parts.append(f"""
<h2 id="s13">13. Interpretation &amp; Conclusions</h2>
<div class="card">
<h3>1. CWRA Reverses Generative Model Drift</h3>
<p>Deep generative models shift output heavily toward non-steroidal scaffolds ({pct['All Generated']['non_steroidal']}% vs {pct['Reference']['non_steroidal']}% reference). 
CWRA meta_score composite ranking reverses this drift: Top 100 recovers {pct['Top 100']['is_secosteroidal']}% secosteroidal content — 
exceeding reference ({pct['Reference']['is_secosteroidal']}%) — demonstrating effective pharmacophoric recovery without explicit structural filtering.</p>

<h3>2. Multi-Generator Consensus is Critical</h3>
<p>{sc.get('G2',0) + sc.get('G3',0)}% of Top 100 from G2/G3 multi-generator consensus. G2 compounds show 2.88× enrichment 
in secosteroidal scaffolds vs G1 (p=9.9×10⁻¹¹). Consensus generation naturally preserves pharmacophoric features 
that individual generators tend to lose during exploration.</p>

<h3>3. C2-Modifications Enriched</h3>
<p>Top 100 achieves {pct['Top 100']['c2_modified']}% C2-modification rate (vs {pct['All Generated']['c2_modified']}% all generated, OR=6.08, p=2.6×10⁻⁶). 
Exceeds reference rate ({pct['Reference']['c2_modified']}%). Validates CYP24A1 metabolic stability as a CWRA-selected feature.</p>

<h3>4. Pre-filter Validation</h3>
<p>The {D['n_prefiltered']:,} pre-filtered compounds show the lowest NN-Tanimoto to reference ({nn['Pre-filtered']['mean']}), 
confirming that MW&gt;600/RotB&gt;15 outliers are structurally dissimilar to the VDR pharmacophore. Pre-filtering prevents 
score compression from bifunctional degraders and PEG-linked molecules.</p>

<h3>5. Pentacyclic Scaffolds Eliminated</h3>
<p>0% of Top 100 vs {pct['All Generated']['pentacyclic']}% all generated (p=0.011). Pentacyclic compounds carry high MW (~560 Da) 
and low QED (~0.31) — CWRA effectively selects against these developability-poor scaffolds.</p>

<h3>6. Property Balance</h3>
<p>Top 100 maintains strong binding (Vina {ps['Top 100']['vina_score']['mean']}, pKd {ps['Top 100']['mltle_pKd']['mean']}) with moderate drug-likeness 
(QED {ps['Top 100']['QED']['mean']}). Higher Fsp³ ({ps['Top 100']['FractionCSP3']['mean']}) indicates greater 3D character — favorable for VDR pocket complementarity.</p>
</div>
""")

# Close HTML
html_parts.append("""
</div><!-- end wrap -->
</body></html>""")

# Write file
html = '\n'.join(html_parts)
out_path = 'output/vdr_cwra_enrichment_analysis.html'
with open(out_path, 'w') as f:
    f.write(html)

print(f"Report written: {len(html):,} chars → {out_path}")
import os
print(f"Size: {os.path.getsize(out_path)/1024:.0f} KB")
