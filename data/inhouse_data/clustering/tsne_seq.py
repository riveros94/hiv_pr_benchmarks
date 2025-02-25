import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def kMeansRes(scaled_data, k, alpha_k=0.02):
    '''
    Parameters 
    ----------
    scaled_data: matrix 
        scaled data. rows are samples and columns are features for clustering
    k: int
        current k for applying KMeans
    alpha_k: float
        manually tuned factor that gives penalty to the number of clusters
    Returns 
    -------
    scaled_inertia: float
        scaled inertia value for current k           
    '''
    
    inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
    # fit k-means
    kmeans = KMeans(n_clusters=k, random_state=42).fit(scaled_data)
    scaled_inertia = kmeans.inertia_ / inertia_o + alpha_k * k
    return scaled_inertia

def chooseBestKforKMeans(X, k_range):
    ans = []
    for k in k_range:
        scaled_inertia = kMeansRes(X, k)
        ans.append((k, scaled_inertia))
    results = pd.DataFrame(ans, columns = ['k','Scaled Inertia']).set_index('k')
    best_k = results.idxmin()[0]
    return best_k

def cluster_analysis(data, num_dimensions):
    results = []
    perplexity_range = range(5, 51, 5)  # De 5 a 50, de 5 em 5

    for perplexity in perplexity_range:
        tsne = TSNE(n_components=num_dimensions, perplexity=perplexity, random_state=42)
        X = tsne.fit_transform(data)

        k_range = range(2, 11)
        
        best_k = chooseBestKforKMeans(X, k_range)
        
        model = KMeans(n_clusters=best_k)
        model.fit(X)
        labels = model.labels_
        silhouette_avg = silhouette_score(X, labels)
        
        # Armazenar os resultados para esta configuração de perplexidade e melhor k
        results.append({
            'perplexity': perplexity,
            'best_k': best_k,
            'silhouette_score': silhouette_avg
        })
    results = pd.DataFrame(results)
    
    return results

def run_tsne_and_kmeans(data, perplexity, k, num_components):
    # Executa t-SNE
    tsne = TSNE(n_components=num_components, perplexity=perplexity, random_state=42)
    tsne_data = tsne.fit_transform(data)

    # Executa KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(tsne_data)
    labels = kmeans.labels_

    # Cria um DataFrame com os dados transformados pelo t-SNE e os rótulos do KMeans
    columns = ['tsne_' + str(i) for i in range(1, num_components + 1)]
    result_df = pd.DataFrame(np.column_stack([tsne_data, labels]), columns=columns + ['cluster'])

    return result_df

os.chdir('/home/rocio/Documents/Doutorado_projeto/artigo_benchmark/hiv_benchmarks/data/inhouse_data/clustering')
data = pd.read_csv('alignment_matrix.csv', header=None)
two_comp = cluster_analysis(data, 2)

plt.figure(figsize=(10, 6))  # Sets the figure size
plt.plot(two_comp['perplexity'], two_comp['silhouette_score'], marker='o', linestyle='-', color='b')
plt.title('Perplexity vs. Silhouette Score - 2 Components')  # Graph title
plt.xlabel('Perplexity')  # X-axis label
plt.ylabel('Silhouette Score')  # Y-axis label
plt.grid(True)  # Adds a grid to the graph for better readability
for index, row in two_comp.iterrows():
    plt.text(row['perplexity'], row['silhouette_score'], f" k={row['best_k']}", fontsize=9)
plt.show()

# Best combination
kmeans_two = run_tsne_and_kmeans(data, perplexity=35, k=5, num_components=2)
kmeans_two.to_csv('seq_labels_tsne_2d_kmeans.csv', index=False)

# Plota os pontos coloridos de acordo com os rótulos
plt.figure(figsize=(8, 6))
plt.scatter(kmeans_two['tsne_1'], kmeans_two['tsne_2'], c=kmeans_two['label'], cmap='viridis', s=50, alpha=0.5)
plt.title('t-SNE + KMeans')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Cluster')
plt.show()

# General data - CLASSIFICATION
sequence_data = pd.read_csv('../pi_sequences_classification.csv')
sequence_data = sequence_data.iloc[1:].reset_index(drop=True)
sequence_data['cluster'] = kmeans_two['cluster']
sequence_data.to_csv('../pi_sequence_cluster_classification.csv', index=False)

# General data
sequence_data = pd.read_csv('../pi_final_dataset.csv')
sequence_data = sequence_data.iloc[1:].reset_index(drop=True)
sequence_data['cluster'] = kmeans_two['cluster']
sequence_data.to_csv('../pi_sequence_cluster.csv', index=False)
