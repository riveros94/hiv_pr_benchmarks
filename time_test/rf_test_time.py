#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 21:30:07 2025

@author: rocio
"""

import os
import time
import pickle
import random
import numpy as np
import pandas as pd
from Bio import PDB
from scipy.spatial import Delaunay
from itertools import combinations_with_replacement


def read_fasta(file_path):
    """
    Reads a FASTA file and converts it to a pandas DataFrame.
    Treats the entire header as the sequence ID and doesn't 
    attempt to parse the header format.
    
    Parameters:
    - file_path (str): Path to the FASTA file.
    
    Returns:
    - DataFrame: Contains SeqID and Sequence columns.
    """
    sequences = {'SeqID': [], 'Sequence': []}
    caracteres_invalidos = {'X', '*', '.', '~'}
    
    with open(file_path, 'r') as file:
        current_id = None
        current_sequence = ''
        
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_id and not any(char in current_sequence for char in caracteres_invalidos):
                    sequences['SeqID'].append(current_id)
                    sequences['Sequence'].append(current_sequence)
                
                # Get the full header as ID (removing the '>' character)
                current_id = line[1:]
                current_sequence = ''
            else:
                current_sequence += line
        
        # Add the last sequence
        if current_id and not any(char in current_sequence for char in caracteres_invalidos):
            sequences['SeqID'].append(current_id)
            sequences['Sequence'].append(current_sequence)
    
    # Convert to DataFrame
    return pd.DataFrame(sequences)


# Função para calcular a distância entre dois pontos tridimensionais
def calculate_distance(point1, point2):
    """
    Calcula a distância euclidiana entre dois pontos tridimensionais.

    Parâmetros:
    - point1 (array): As coordenadas do primeiro ponto.
    - point2 (array): As coordenadas do segundo ponto.

    Retorna:
    - float: A distância euclidiana entre os dois pontos.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Função para obter o tipo de aminoácido correspondente a um índice
def get_amino_acid_type(index, amino_acids):
    """
    Retorna o tipo de aminoácido correspondente a um índice na lista de aminoácidos.

    Parâmetros:
    - index (int): O índice do aminoácido na lista.
    - amino_acids (list): A lista de tipos de aminoácidos.

    Retorna:
    - str: O tipo de aminoácido correspondente ao índice.
    """
    return amino_acids[index]

# Função para ler as coordenadas e tipos de aminoácidos de um arquivo PDB
def read_pdb_coordinates(file_path):
    """
    Lê as coordenadas dos átomos de carbono alfa (Cα) e os tipos de aminoácidos de um arquivo PDB.

    Parâmetros:
    - file_path (str): O caminho para o arquivo PDB.

    Retorna:
    - numpy.array: As coordenadas dos átomos de carbono alfa (Cα).
    - list: Os tipos de aminoácidos presentes no arquivo.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", file_path)

    coordinates = []
    amino_acids = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    atom = residue["CA"]
                    coords = atom.get_coord()
                    coordinates.append(coords)
                    amino_acids.append(residue.get_resname()) 

    return np.array(coordinates), amino_acids

# Função para atualizar a distância entre os aminoácidos no dataframe
def update_distance(df, amino_acid1, amino_acid2, distance):
    """
    Atualiza a distância entre dois aminoácidos no dataframe.

    Parâmetros:
    - df (DataFrame): O dataframe contendo as distâncias entre os aminoácidos.
    - amino_acid1 (str): O tipo do primeiro aminoácido.
    - amino_acid2 (str): O tipo do segundo aminoácido.
    - distance (float): A distância entre os dois aminoácidos.
    """
    for col in df.columns:
        if amino_acid1 + amino_acid2 == col or amino_acid2 + amino_acid1 == col:
            df[col] += distance

# Função para construir o dataframe com as distâncias entre os aminoácidos
def construct_df(simplices, coordinates, aminoacids):
    """
    Constrói um dataframe contendo as distâncias entre os aminoácidos.

    Parâmetros:
    - simplices (array): Os triângulos na triangulação.
    - coordinates (array): As coordenadas dos átomos de carbono alfa (Cα).
    - aminoacids (list): Os tipos de aminoácidos presentes.

    Retorna:
    - DataFrame: O dataframe contendo as distâncias entre os aminoácidos.
    """
    aminoacidos = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    combinacoes = list(combinations_with_replacement(aminoacidos,2))
    df = pd.DataFrame(0, columns=[f'{a[0]}{a[1]}' for a in combinacoes], index=[0])

    for i, simplex in enumerate(simplices):
        index1, index2, index3, index4 = simplex

        amino_acid1 = get_amino_acid_type(index1, aminoacids)
        amino_acid2 = get_amino_acid_type(index2, aminoacids)
        amino_acid3 = get_amino_acid_type(index3, aminoacids)
        amino_acid4 = get_amino_acid_type(index4, aminoacids)

        distance1_2 = calculate_distance(coordinates[index1], coordinates[index2])
        update_distance(df, amino_acid1, amino_acid2, distance1_2)

        distance1_3 = calculate_distance(coordinates[index1], coordinates[index3])
        update_distance(df, amino_acid1, amino_acid3, distance1_3)

        distance1_4 = calculate_distance(coordinates[index1], coordinates[index4])
        update_distance(df, amino_acid1, amino_acid4, distance1_4)
        
        distance2_3 = calculate_distance(coordinates[index2], coordinates[index3])
        update_distance(df, amino_acid2, amino_acid3, distance2_3)

        distance2_4 = calculate_distance(coordinates[index2], coordinates[index4])
        update_distance(df, amino_acid2, amino_acid4, distance2_4)
        
        distance3_4 = calculate_distance(coordinates[index3], coordinates[index4])
        update_distance(df, amino_acid3, amino_acid4, distance3_4)
    
    return df

def process_structures(pdb_path, sequence_df):
    """
    Process PDB file and sequence dataframe.

    Args:
        pdb_path (str): Path to PDB file
        sequence_df (pd.DataFrame): DataFrame with sequences

    Returns:
        pd.DataFrame: Combined data
    """
    one_letter_code = {
        'PRO': 'P', 'GLN': 'Q', 'ILE': 'I', 'THR': 'T', 'LEU': 'L',
        'TRP': 'W', 'LYS': 'K', 'ARG': 'R', 'VAL': 'V', 'ALA': 'A',
        'ASP': 'D', 'GLY': 'G', 'GLU': 'E', 'MET': 'M', 'SER': 'S',
        'HIS': 'H', 'ASN': 'N', 'TYR': 'Y', 'PHE': 'F', 'CYS': 'C'
    }

    coordinates, amino_acids = read_pdb_coordinates(pdb_path)
    amino_acids = [one_letter_code.get(code, 'X') for code in amino_acids]  # Handle unknown codes

    triangulation = Delaunay(coordinates)
    simplices = triangulation.simplices
    sample = construct_df(simplices, coordinates, amino_acids)
    
    sequences = list(sequence_df['Sequence'])
    for seq in sequences:
        sequence_list = list(seq)
        sequence_list *= 2  # Duplicates the sequence
        sample = pd.concat([sample, construct_df(simplices, coordinates, sequence_list)], axis=0)

    sample = sample.iloc[1:]  # Remove the first row (template)
    sample.reset_index(drop=True, inplace=True)
    
    return sample
   
def make_prediction(model_pkl, data):
    with open(model_pkl, 'rb') as file:
        model = pickle.load(file)
    
    feature_names = model.feature_names_in_
    X_input = data.loc[:, feature_names]
    
    prediction = model.predict(X_input)
    return prediction

def predict_hiv_resistance(model, fasta_path, drug, pdb_path, n_runs=100, seed=42, output_file="predictions.txt"):
    """
    Predict HIV resistance using a model, FASTA file, and PDB structure.
    
    Parameters:
    - model (str): Path to the pickled model.
    - fasta_path (str): Path to the FASTA file.
    - drug (str): Drug name.
    - pdb_path (str): Path to the PDB file.
    - n_runs (int): Number of runs for timing (default 100).
    - seed (int): Random seed.
    - output_file (str): Output file path.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    times = []
    for i in range(n_runs):
        start_time = time.time()
        dataframe = read_fasta(fasta_path)
        triang = process_structures(pdb_path, dataframe)
        prediction = make_prediction(model, triang)
        end_time = time.time()
        times.append(end_time - start_time)
    mean_time = np.mean(times)
    std_time = np.std(times)
    result = f"Prediction for drug {drug}: {prediction}\n"
    result += f"Number of runs: {n_runs}\n"
    result += f"Mean execution time: {mean_time:.4f} seconds\n"
    result += f"Standard deviation: {std_time:.4f} seconds\n"
    result += f"Seed used: {seed}\n"
    result += f"Individual run times: {', '.join([f'{t:.4f}' for t in times])}\n"
    result += "\nSequence Information:\n"
    for i, (seq_id, sequence) in enumerate(zip(dataframe['SeqID'], dataframe['Sequence'])):
        result += f"Sequence {i+1}: {seq_id}\n"
        result += f"Length: {len(sequence)}\n"
    
    print(result)  # Console output
    with open(output_file, 'a') as f:
        f.write(result + '\n')

# In-house test
fasta_path = 'seq_test.fasta'
pdb_path = '3oxc_edited.pdb'
predict_hiv_resistance(model='2NFV_fold_rf_model.pkl', 
                          fasta_path=fasta_path,
                          drug='NFV', 
                          pdb_path=pdb_path, 
                          output_file='inhouse_rf_prediction.txt')

# Steiner test
predict_hiv_resistance(model='3_fold_rf_model.pkl',
                          fasta_path=fasta_path, 
                          drug='NFV', 
                          pdb_path=pdb_path, 
                          output_file='steiner_rf_prediction.txt')

# Shen test
predict_hiv_resistance(model='1_fold_rf_model.pkl', 
                          fasta_path=fasta_path,
                          drug='NFV', 
                          pdb_path=pdb_path, 
                          output_file='shen_knn_prediction.txt')

