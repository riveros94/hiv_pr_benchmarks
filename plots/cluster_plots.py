import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from matplotlib.patches import Patch

def create_combined_figure(data, train, test_data, output_filename='combined_figure.png'):
    """
    Create a combined figure with four subplots:
    A: Fold change plot
    B: NFV classes plot
    C: Shaded cluster plot
    D: Cluster proportions
    
    Parameters:
    -----------
    data : pandas DataFrame
        Contains the full dataset with columns: tsne_1, tsne_2, label, NFV, NFV_class, logFC, cluster
    train : pandas DataFrame
        Training dataset with 'cluster' column
    test_data : pandas DataFrame
        Test dataset (filtered from data) with all necessary columns
    output_filename : str
        Name of the output file
    """
    # Setup for publication quality
    DPI = 300
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Create figure with four subplots in a 2x2 grid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot A: Fold change plot
    scatter = ax1.scatter(data['tsne_1'], data['tsne_2'], 
                          c=data['logFC'], cmap='plasma_r', 
                          s=90, alpha=0.7)
    ax1.set_xlabel('t-SNE Component 1', fontsize=22)
    ax1.set_ylabel('t-SNE Component 2', fontsize=22)
    ax1.tick_params(axis='both', which='major', labelsize=22)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Fold Change (logFC)', size=22)
    cbar.ax.tick_params(labelsize=22)
    
    # Plot B: NFV classes plot
    colors = {0: 'blue', 1: 'red'}
    labels = {0: 'Susceptible', 1: 'Resistant'}
    
    for label in sorted(data['NFV_class'].unique()):
        subset = data[data['NFV_class'] == label]
        ax2.scatter(subset['tsne_1'], subset['tsne_2'], 
                    c=colors[label], label=labels[label], 
                    s=90, alpha=0.7)
    
    ax2.set_xlabel('t-SNE Component 1', fontsize=22)
    ax2.set_ylabel('t-SNE Component 2', fontsize=22)
    ax2.tick_params(axis='both', which='major', labelsize=22)
    ax2.legend(title='NFV phenotype', title_fontsize=22, 
              fontsize=20, loc='lower left')
    
    # Function for cluster shading
    def add_cluster_shading(ax, data, cluster_label, color, alpha=0.2):
        subset = data[data['cluster'] == cluster_label]
        points = subset[['tsne_1', 'tsne_2']].values
        if len(points) >= 3:
            hull = ConvexHull(points)
            vertices = hull.vertices
            polygon = plt.Polygon(points[vertices], color=color, alpha=alpha, edgecolor='none')
            ax.add_patch(polygon)
    
    # Color definitions
    cluster_colors = {0: '#1f77b4', 1: '#d62728', 2: '#2ca02c', 
                     3: '#ff7f0e', 4: '#9467bd'}
    phenotype_colors = {0: '#1f77b4', 1: '#d62728'}
    
    # Plot C: Shaded Cluster Plot
    for label in sorted(data['cluster'].unique()):
        add_cluster_shading(ax3, data, label, cluster_colors[label])
    
    # Plot all points in grey
    ax3.scatter(data['tsne_1'], data['tsne_2'], 
                c='grey', s=90, alpha=0.8)
    
    # Plot test points colored by NFV_class
    for label in sorted(test_data['NFV_class'].unique()):
        subset = test_data[test_data['NFV_class'] == label]
        ax3.scatter(subset['tsne_1'], subset['tsne_2'], 
                    c=phenotype_colors[label], s=90, alpha=0.8)
    
    cluster_patches = [
        Patch(color=cluster_colors[label], 
              label=f'Cluster {int(label) + 1}')
        for label in sorted(data['cluster'].unique())
    ]
    
    ax3.legend(handles=cluster_patches, 
              fontsize=20,
              loc='lower left',
              frameon=True,
              edgecolor='black',
              fancybox=False)
    
    ax3.set_xlabel('t-SNE Component 1', fontsize=22)
    ax3.set_ylabel('t-SNE Component 2', fontsize=22)
    ax3.tick_params(axis='both', which='major', labelsize=22)
    
    # Plot D: Cluster Proportions
    def calculate_label_proportion(dataset_data):
        proportions = dataset_data['cluster'].value_counts(normalize=True)
        proportions.index = proportions.index + 1  # Add 1 to make clusters start at 1
        return proportions.sort_index()
    
    train_cluster_proportion = calculate_label_proportion(train)
    test_cluster_proportion = calculate_label_proportion(test_data)
    
    proportions = pd.DataFrame({
        'Train dataset': train_cluster_proportion,
        'Test dataset': test_cluster_proportion
    }).fillna(0)
    
    bar_width = 0.4
    x = range(len(proportions.index))
    
    ax4.bar([i - bar_width/2 for i in x], 
            proportions['Train dataset'], 
            bar_width, 
            label='Train dataset',
            color='#2ecc71',
            alpha=1)
    
    ax4.bar([i + bar_width/2 for i in x], 
            proportions['Test dataset'], 
            bar_width, 
            label='Test dataset',
            color='#9b59b6',
            alpha=1)
    
    ax4.set_xlabel('Cluster Labels', fontsize=22)
    ax4.set_ylabel('Proportion', fontsize=22)
    ax4.set_ylim(0, 0.30)
    ax4.set_xticks(x)
    ax4.set_xticklabels(proportions.index, fontsize=22)
    ax4.tick_params(axis='y', labelsize=22)
    
    # Common settings for all plots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(1)
            spine.set_color('black')
    
    # Remove top and right spines for bar plot
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    ax4.legend(fontsize=20,
              loc='upper left',
              frameon=True,
              edgecolor='black',
              fancybox=False)
    
    # Add subplot labels
    ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, 
             fontsize=24, fontweight='bold')
    ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, 
             fontsize=24, fontweight='bold')
    ax3.text(-0.1, 1.05, 'C', transform=ax3.transAxes, 
             fontsize=24, fontweight='bold')
    ax4.text(-0.1, 1.05, 'D', transform=ax4.transAxes, 
             fontsize=24, fontweight='bold')
    
    # Save combined plot
    plt.tight_layout()
    plt.savefig(output_filename, dpi=DPI, bbox_inches='tight')
    plt.close()
    
# Example usage:
data = pd.read_csv('hiv_benchmarks/data/inhouse_data/pi_sequence_cluster.csv')
tsne = pd.read_csv('hiv_benchmarks/data/inhouse_data/clustering/seq_labels_tsne_2d_kmeans.csv')
joined_data = pd.concat([data, tsne[['tsne_1', 'tsne_2']]], axis=1)
joined_data = joined_data[joined_data['NFV'].notna()]
joined_data['logFC'] = np.log2(joined_data['NFV'])
joined_data['NFV_class'] = joined_data['NFV'].apply(lambda x: 0 if x < 3 else 1)

train = pd.read_csv('hiv_benchmarks/models/inhouse/rosetta_lr/NFVtrain.csv')
test = pd.read_csv('hiv_benchmarks/models/inhouse/rosetta_lr/NFVtest.csv')
common_ids = set(test['SeqID']).intersection(set(joined_data['SeqID']))
test_data = joined_data[joined_data['SeqID'].isin(common_ids)]

create_combined_figure(joined_data, train, test_data)
