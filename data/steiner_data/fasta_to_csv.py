import os
import pandas as pd
from sklearn.model_selection import train_test_split

def read_fasta(file_path):
    sequences = {'SeqID': [], 'Class': [], 'Sequence': []}
    caracteres_invalidos = {'X', '*', '.', '~'}
    
    with open(file_path, 'r') as file:
        current_id = None
        current_class = None
        current_sequence = ''
        
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_id and not any(char in current_sequence for char in caracteres_invalidos):
                    sequences['SeqID'].append(current_id)
                    sequences['Class'].append(current_class)
                    sequences['Sequence'].append(current_sequence)
                current_id, current_class = line[1:].split('_')
                current_sequence = ''
            else:
                current_sequence += line
        
        if current_id and not any(char in current_sequence for char in caracteres_invalidos):
            sequences['SeqID'].append(current_id)
            sequences['Class'].append(current_class)
            sequences['Sequence'].append(current_sequence)
    
    return sequences

def process_and_split_fasta(file_path, drug, test_size=0.2, random_state=42, path=''):
    # Lê o arquivo FASTA e converte para DataFrame
    sequences = read_fasta(file_path)
    df = pd.DataFrame(sequences)
    
    # Divide em treino e teste
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    def generate_fasta(dataset):
        fasta = ""
        for index, row in dataset.iterrows():
            seq_id = row['SeqID']
            sequence = row['Sequence']
            drug_value = int(row['Class'])  # Assumindo que 'Class' é a coluna com os valores do drug
            fasta += f">{seq_id}_{drug_value}\n{sequence}\n"
        return fasta
    
    # Gera os arquivos FASTA para treino e teste
    train_fasta = generate_fasta(train_df)
    test_fasta = generate_fasta(test_df)
    drug_lowercase = drug.lower()
    
    # Define os nomes dos arquivos incluindo o caminho
    train_filename = f'{path}/{drug_lowercase}_train.fasta' if path else f'{drug_lowercase}_train.fasta'
    test_filename = f'{path}/{drug_lowercase}_test.fasta' if path else f'{drug_lowercase}_test.fasta'
    
    # Salva os arquivos
    with open(train_filename, 'w') as train_file:
        train_file.write(train_fasta)
    with open(test_filename, 'w') as test_file:
        test_file.write(test_fasta)
    
    return df, train_df, test_df

# Exemplo de uso:
# train_df, test_df = process_and_split_fasta('input.fasta', 'DrugName', test_size=0.2, random_state=42, path='output/')

os.chdir('/home/rocio/Documents/Doutorado_projeto/artigo_benchmark/hiv_benchmarks/data/steiner_data/')

# ATV
filtrated_fasta_data, train_df, test_df = process_and_split_fasta('atv.fasta', drug = 'ATV', test_size=0.2, random_state=42, path='nn_models')
filtrated_fasta_data.to_csv('atv_filtrated.csv', index=False)

# DRV
filtrated_fasta_data, train_df, test_df = process_and_split_fasta('drv.fasta', drug = 'DRV', test_size=0.2, random_state=42, path='nn_models')
filtrated_fasta_data.to_csv('drv_filtrated.csv', index=False)

# FPV
filtrated_fasta_data, train_df, test_df = process_and_split_fasta('fpv.fasta', drug = 'FPV', test_size=0.2, random_state=42, path='nn_models')
filtrated_fasta_data.to_csv('fpv_filtrated.csv', index=False)

# IDV
filtrated_fasta_data, train_df, test_df = process_and_split_fasta('idv.fasta', drug = 'IDV', test_size=0.2, random_state=42, path='nn_models')
filtrated_fasta_data.to_csv('idv_filtrated.csv', index=False)

# LPV
filtrated_fasta_data, train_df, test_df = process_and_split_fasta('lpv.fasta', drug = 'LPV', test_size=0.2, random_state=42, path='nn_models')
filtrated_fasta_data.to_csv('lpv_filtrated.csv', index=False)

# NFV
filtrated_fasta_data, train_df, test_df = process_and_split_fasta('nfv.fasta', drug = 'NFV', test_size=0.2, random_state=42, path='nn_models')
filtrated_fasta_data.to_csv('nfv_filtrated.csv', index=False)

# SQV
filtrated_fasta_data, train_df, test_df = process_and_split_fasta('sqv.fasta', drug = 'SQV', test_size=0.2, random_state=42, path='nn_models')
filtrated_fasta_data.to_csv('sqv_filtrated.csv', index=False)

# TPV
filtrated_fasta_data, train_df, test_df = process_and_split_fasta('tpv.fasta', drug = 'TPV', test_size=0.2, random_state=42, path='nn_models')
filtrated_fasta_data.to_csv('tpv_filtrated.csv', index=False)
