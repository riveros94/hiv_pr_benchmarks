import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
from scipy.spatial import ConvexHull

# Set up fonts with fallbacks
def setup_fonts():
    """Configure fonts with fallbacks for different operating systems"""
    # List of preferred sans-serif fonts in order of preference
    preferred_fonts = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 
                      'sans-serif']
    
    # Find the first available font
    available_font = None
    for font in preferred_fonts:
        try:
            if any(f for f in fm.fontManager.ttflist if font.lower() in f.name.lower()):
                available_font = font
                break
        except:
            continue
    
    # Set the font
    if available_font:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [available_font] + preferred_fonts
    else:
        # If none of our preferred fonts are found, use the system default sans-serif
        plt.rcParams['font.family'] = 'sans-serif'

# Set figure DPI to meet journal requirements (300 for color)
DPI = 300


def create_combined_publication_plots(data):
    # Convert NFV to logFC and create class
    data['logFC'] = np.log2(data['NFV'])
    data['NFV_class'] = data['NFV'].apply(lambda x: 0 if x < 3 else 1)
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
    # Common settings for all subplots
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(1)
            spine.set_color('black')
        ax.set_xlabel('t-SNE Component 1', fontsize=22)
        ax.set_ylabel('t-SNE Component 2', fontsize=22)
    
    # 1. Fold Change Plot (subplot a)
    scatter1 = ax1.scatter(data['tsne_1'], data['tsne_2'], 
                          c=data['logFC'], 
                          cmap='plasma_r', 
                          s=80,
                          alpha=0.8)
    cbar = plt.colorbar(scatter1, ax=ax1)
    cbar.set_label('Fold Change (log₂)', size=18)
    cbar.ax.tick_params(labelsize=20)
    ax1.text(-0.1, -0.1, 'a', transform=ax1.transAxes, 
             fontsize=24, fontweight='bold')
    
    # 2. NFV Classes Plot (subplot b)
    colors = {0: '#1f77b4', 1: '#d62728'}
    labels = {0: 'Susceptible', 1: 'Resistant'}
    
    for label in data['NFV_class'].unique():
        subset = data[data['NFV_class'] == label]
        ax2.scatter(subset['tsne_1'], subset['tsne_2'], 
                   c=colors[label], 
                   label=labels[label], 
                   s=80, 
                   alpha=0.8)
    
    ax2.legend(title='NFV phenotype', 
              title_fontsize=22, 
              fontsize=17, 
              loc='lower left',
              frameon=True,
              edgecolor='black',
              fancybox=False)
    ax2.text(-0.1, -0.1, 'b', transform=ax2.transAxes, 
             fontsize=24, fontweight='bold')
    
    # 3. Clusters Plot (subplot c)
    colors = {0: '#1f77b4', 1: '#d62728', 2: '#2ca02c', 
             3: '#ff7f0e', 4: '#9467bd'}
    
    for cluster in sorted(data['cluster'].unique()):
        subset = data[data['cluster'] == cluster]
        ax3.scatter(subset['tsne_1'], subset['tsne_2'], 
                   c=colors[cluster], 
                   label=f'Cluster {int(cluster)+1}',  # Fixed: changed 'cluster' parameter to 'label' 
                   s=80, 
                   alpha=0.7)
    
    ax3.legend(fontsize=17, 
              loc='lower left',
              frameon=True,
              edgecolor='black',
              fancybox=False)
    ax3.text(-0.1, -0.1, 'c', transform=ax3.transAxes, 
             fontsize=24, fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('combined_tsne_plots.png', dpi=300, bbox_inches='tight')
    plt.close()


# Example usage:
data = pd.read_csv('hiv_benchmarks/data/inhouse_data/pi_sequence_cluster.csv')
tsne = pd.read_csv('hiv_benchmarks/data/inhouse_data/clustering/seq_labels_tsne_2d_kmeans.csv')
joined_data = pd.concat([data, tsne[['tsne_1', 'tsne_2']]], axis=1)
joined_data = joined_data[joined_data['NFV'].notna()]
joined_data['logFC'] = np.log2(joined_data['NFV'])
joined_data['NFV_class'] = joined_data['NFV'].apply(lambda x: 0 if x < 3 else 1)
create_combined_publication_plots(joined_data)
