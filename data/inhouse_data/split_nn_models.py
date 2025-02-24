
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(df, cluster_column, test_ratio=0.2, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    total_samples = len(df)
    test_samples = int(total_samples * test_ratio)
    clusters = df[cluster_column].unique()
    test_samples_per_cluster = test_samples // len(clusters)
    test_df = pd.DataFrame(columns=df.columns)
    for cluster in clusters:
        cluster_samples = df[df[cluster_column] == cluster]
        test_samples_cluster = cluster_samples.sample(test_samples_per_cluster, random_state=seed)
        test_df = pd.concat([test_df, test_samples_cluster], axis=0)
    train_df = df.drop(test_df.index)
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return train_df, test_df



def split_and_save_fasta_dataset(df, drug, path="", test_size=0.2, seed=13):
    train_df, test_df = split_dataset(df, cluster_column= 'cluster', test_ratio=0.2, seed=seed)
    
    def generate_fasta(dataset):
        fasta = ""
        for index, row in dataset.iterrows():
            seq_id = row['SeqID']
            sequence = row['Sequence']
            drug_value = int(row[drug])
            fasta += f">{seq_id}_{drug_value}\n{sequence}\n"
        return fasta
    
    train_fasta = generate_fasta(train_df)
    test_fasta = generate_fasta(test_df)
    drug_lowercase = drug.lower()
    
    # Include path in file names
    train_filename = f'{path}/{drug_lowercase}_train.fasta' if path else f'{drug_lowercase}_train.fasta'
    test_filename = f'{path}/{drug_lowercase}_test.fasta' if path else f'{drug_lowercase}_test.fasta'
    
    with open(train_filename, 'w') as train_file:
        train_file.write(train_fasta)
    with open(test_filename, 'w') as test_file:
        test_file.write(test_fasta)
  
    return train_fasta, test_fasta

# Load pi_dataset
os.chdir('/home/rocio/Documents/Doutorado_projeto/artigo_benchmark/hiv_benchmarks/data/inhouse_data/')
data = pd.read_csv("pi_sequence_cluster_classification.csv")
path = 'nn_models'

# FPV
drug = 'FPV'
dataset = data.dropna(subset=[drug])
train_fasta, test_fasta = split_and_save_fasta_dataset(dataset, drug,)

# ATV
drug = 'ATV'
dataset = data.dropna(subset=[drug])
train_fasta, test_fasta = split_and_save_fasta_dataset(dataset, drug)

# SQV
drug = 'SQV'
dataset = data.dropna(subset=[drug])
train_fasta, test_fasta = split_and_save_fasta_dataset(dataset, drug)

# IDV
drug = 'IDV'
dataset = data.dropna(subset=[drug])
train_fasta, test_fasta = split_and_save_fasta_dataset(dataset, drug)

# LPV
drug = 'LPV'
dataset = data.dropna(subset=[drug])
train_fasta, test_fasta = split_and_save_fasta_dataset(dataset, drug)

# NFV
drug = 'NFV'
dataset = data.dropna(subset=[drug])
train_fasta, test_fasta = split_and_save_fasta_dataset(dataset, drug)

# TPV
drug = 'TPV'
dataset = data.dropna(subset=[drug])
train_fasta, test_fasta = split_and_save_fasta_dataset(dataset, drug)

# DRV
drug = 'DRV'
dataset = data.dropna(subset=[drug])
train_fasta, test_fasta = split_and_save_fasta_dataset(dataset, drug)


