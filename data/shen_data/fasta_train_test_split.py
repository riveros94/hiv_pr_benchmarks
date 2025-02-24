import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_and_save_fasta_dataset(df, drug, path="", test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
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
    train_filename = f'{path}{drug_lowercase}_train.fasta' if path else f'{drug_lowercase}_train.fasta'
    test_filename = f'{path}{drug_lowercase}_test.fasta' if path else f'{drug_lowercase}_test.fasta'
    
    with open(train_filename, 'w') as train_file:
        train_file.write(train_fasta)
    with open(test_filename, 'w') as test_file:
        test_file.write(test_fasta)
  
    return train_fasta, test_fasta

# Load pi_dataset
os.chdir('/home/rocio/Documents/Doutorado_projeto/artigo_benchmark/hiv_benchmarks/data/shen_data/')
data = pd.read_csv("pi_sequences_classification.csv")
path = 'nn_models/'
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


