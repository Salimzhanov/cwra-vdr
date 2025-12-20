#!/usr/bin/env python3
"""
Optimized script to produce weighted rank consensus score bar chart
for VDR ligand candidate ranking visualization.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('test_full_ranking.csv')

# Get top 20 by cwra_rank
top20 = df.nsmallest(20, 'cwra_rank')

# Map source to origin
origin_map = {
    'G1': 'unique_G1',
    'G2': 'overlap_G2',
    'G3': 'overlap_G3',
    'initial_370': '370_initial',
    'calcitriol': 'Calcitriol'
}

# Prepare data
data = []
for idx, row in top20.iterrows():
    compound = row[' ']
    origin = origin_map.get(row['source'], row['source'])
    consensus_score = row['cwra_score']
    rank = int(row['cwra_rank'])
    data.append({
        'compound': compound,
        'origin': origin,
        'consensus_score': consensus_score,
        'rank': rank
    })

df_plot = pd.DataFrame(data)

# Color palette matching the original figure
color_map = {
    'unique_G1': '#9ED8DB',      # Light cyan - G1 generative
    'overlap_G2': '#4A90A4',     # Medium blue - G2 overlap
    'overlap_G3': '#1A3A5C',     # Dark navy - G3 overlap
    '370_initial': '#7CB950',    # Green - Initial set
    'Calcitriol': '#E07B73'      # Coral/salmon - Calcitriol
}

# Create y-axis labels with origin in parentheses
df_plot['label'] = df_plot.apply(
    lambda x: f"{x['compound']}\n({x['origin']})" if x['compound'] != 'calcitriol' else 'Calcitriol',
    axis=1
)

# Assign colors based on origin
df_plot['color'] = df_plot['origin'].map(color_map)

# Create figure
fig, ax = plt.subplots(figsize=(10, 12))

# Plot horizontal bars (reversed order so #1 is at top)
y_positions = np.arange(len(df_plot))[::-1]
bars = ax.barh(y_positions, df_plot['consensus_score'], color=df_plot['color'], height=0.7, edgecolor='none')

# Set y-axis labels
ax.set_yticks(y_positions)
ax.set_yticklabels(df_plot['label'], fontsize=9)

# Add rank labels at end of each bar
for i, (pos, score, rank) in enumerate(zip(y_positions, df_plot['consensus_score'], df_plot['rank'])):
    label_color = 'white' if df_plot.iloc[i]['origin'] == 'Calcitriol' else 'black'
    ax.text(score + 0.003, pos, f'#{rank}', va='center', ha='left', fontsize=9, 
            fontweight='bold' if df_plot.iloc[i]['origin'] == 'Calcitriol' else 'normal', color=label_color)

# Formatting
ax.set_xlabel('Consensus score', fontsize=11)
ax.set_xlim(0, df_plot['consensus_score'].max() + 0.01)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
ax.grid(axis='x', linestyle='--', alpha=0.3, color='gray')
ax.set_axisbelow(True)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Create legend
legend_labels = ['G1 – generative', 'G2 – overlap', 'G3 – overlap', 'Initial set', 'Calcitriol']
legend_colors = ['#9ED8DB', '#4A90A4', '#1A3A5C', '#7CB950', '#E07B73']
legend_handles = [plt.Rectangle((0,0), 1, 1, facecolor=c, edgecolor='none') for c in legend_colors]

ax.legend(legend_handles, legend_labels, title='Dataset origin', loc='upper right',
          frameon=True, fancybox=False, edgecolor='gray', fontsize=9, title_fontsize=10)

plt.tight_layout()
plt.savefig('weighted_rank_bar.pdf', dpi=300, bbox_inches='tight')
plt.savefig('weighted_rank_bar.png', dpi=300, bbox_inches='tight')
print("Plot saved to weighted_rank_bar.pdf and weighted_rank_bar.png")