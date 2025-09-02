#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 20:10:44 2025

@author: rocio
"""

import pandas as pd
import numpy as np
from scipy.stats import f_oneway, kruskal, levene
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns

def test_cluster_resistance_association(data):
    """
    Tests if sequence-based clusters are associated with resistance levels.
    """ 
    results = {}
    
    cluster_groups = []
    cluster_ids = sorted(data['cluster_id'].unique())
    
    for cluster_id in cluster_ids:
        cluster_data = data[data['cluster_id'] == cluster_id]['log_NFV']
        cluster_groups.append(cluster_data)
    
    # 1. ANOVA 
    f_stat, anova_p = f_oneway(*cluster_groups)
    results['anova'] = {'F': f_stat, 'p': anova_p}
    
    # 2. Kruskal-Wallis 
    h_stat, kw_p = kruskal(*cluster_groups)
    results['kruskal_wallis'] = {'H': h_stat, 'p': kw_p}
    
    # 3. Test for homogeneity of variances
    levene_stat, levene_p = levene(*cluster_groups)
    results['levene'] = {'W': levene_stat, 'p': levene_p}
    
    # 4. Estatísticas descritivas
    stats_by_cluster = []
    for i, cluster_id in enumerate(cluster_ids):
        cluster_data = cluster_groups[i]
        stats_by_cluster.append({
            'cluster': cluster_id,
            'n': len(cluster_data),
            'mean_log_fc': np.mean(cluster_data),
            'std_log_fc': np.std(cluster_data),
            'median_log_fc': np.median(cluster_data)
        })
    
    results['descriptive'] = pd.DataFrame(stats_by_cluster)
    
    return results


data = pd.read_csv('clustering.csv')
data['log_NFV'] = np.log10(data['NFV'])

results = test_cluster_resistance_association(data)

print("TESTING: Do sequence-based clusters differ in resistance levels?")
print("="*60)
print(f"ANOVA: F = {results['anova']['F']:.5f}, p = {results['anova']['p']:.2e}")  
print(f"Kruskal-Wallis: H = {results['kruskal_wallis']['H']:.5f}, p = {results['kruskal_wallis']['p']:.2e}")
print(f"Levene test: W = {results['levene']['W']:.5f}, p = {results['levene']['p']:.2e}")
print("\nDescriptive statistics:")
print(results['descriptive'])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Box plot
sns.boxplot(data=data, x='cluster_id', y='log_NFV', ax=axes[0])
axes[0].set_title('Log(Fold Change) by Cluster')

# Violin plot
sns.violinplot(data=data, x='cluster_id', y='log_NFV', ax=axes[1])
axes[1].set_title('Distribution of Log(FC) by Cluster')

# Proportion of resistant sequences per cluster
resistance_prop = data.groupby('cluster_id')['resistant'].mean()
resistance_prop.plot(kind='bar', ax=axes[2])
axes[2].set_title('Proportion of Resistant Sequences by Cluster')
axes[2].set_ylabel('Proportion Resistant')

plt.tight_layout()
plt.show()

print("\nResistance statistics by cluster:")
resistance_stats = data.groupby('cluster_id').agg({
    'resistant': ['count', 'sum', 'mean'],
    'log_NFV': ['mean', 'std']
}).round(3)
print(resistance_stats)
